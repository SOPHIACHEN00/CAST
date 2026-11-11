"""Submitted-and-Simulated WEF (S2-WEF) dynamic free-rider detector.

Implements Nakamura et al.'s round-wise pipeline: simulate a server WEF from
consecutive global models, combine similarity gamma and WEF deviation, robustly
standardize them, apply Ward clustering, and validate the suspicious cluster by
score-wise majority voting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch

from .wef import WEFConfig, WEFDetector


@dataclass(frozen=True)
class S2WEFConfig:
    weight_from_end: int = 2
    gamma_multiplier: float = 1.5
    deviation_epsilon: float = 0.05
    silhouette_threshold: float = 0.30
    merge_jump_threshold: float = 0.90
    numerical_eps: float = 1e-12

    def validate(self) -> None:
        if self.weight_from_end < 1:
            raise ValueError("weight_from_end must be at least 1")
        if self.gamma_multiplier < 0:
            raise ValueError("gamma_multiplier must be non-negative")
        if self.deviation_epsilon < 0:
            raise ValueError("deviation_epsilon must be non-negative")
        if not -1.0 <= self.silhouette_threshold <= 1.0:
            raise ValueError("silhouette_threshold must be in [-1, 1]")
        if self.merge_jump_threshold < 0:
            raise ValueError("merge_jump_threshold must be non-negative")
        if self.numerical_eps <= 0:
            raise ValueError("numerical_eps must be positive")


class S2WEFDetector:
    """Paper-faithful round-wise S2-WEF detector."""

    @staticmethod
    def _select_update_matrix(
        update: List[torch.Tensor], weight_from_end: int
    ) -> torch.Tensor:
        matrices = [tensor.detach() for tensor in update if tensor.ndim >= 2]
        if len(matrices) < weight_from_end:
            raise ValueError(
                f"S2-WEF requested matrix {weight_from_end} from the end, "
                f"but the global update exposes only {len(matrices)} matrices"
            )
        return matrices[-weight_from_end]

    @staticmethod
    def simulate_wef_matrix(
        previous_round_global_update: List[torch.Tensor] | None,
        reference_shape: torch.Size,
        local_iterations: int,
        config: S2WEFConfig,
    ) -> torch.Tensor:
        """Implement Eqs. (5)-(6) from one consecutive global-model delta."""
        if local_iterations < 1:
            raise ValueError("local_iterations must be at least 1")
        if previous_round_global_update is None:
            return torch.zeros(reference_shape, dtype=torch.float32)
        selected = S2WEFDetector._select_update_matrix(
            previous_round_global_update, config.weight_from_end
        ).cpu()
        if tuple(selected.shape) != tuple(reference_shape):
            raise ValueError("Simulated and submitted WEF matrices have different shapes")
        absolute_change = torch.abs(selected)
        threshold = torch.mean(absolute_change)
        return (absolute_change > threshold).to(torch.float32) * local_iterations

    @staticmethod
    def _robust_z(values: np.ndarray, eps: float) -> np.ndarray:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return (values - median) / (mad + eps)

    @staticmethod
    def _ward_two_clusters(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the Ward two-cluster cut and all merge heights, dependency-free."""
        clusters = {index: [index] for index in range(len(points))}
        next_cluster_id = len(points)
        merge_heights: list[float] = []
        two_cluster_members: list[list[int]] | None = None
        while len(clusters) > 1:
            cluster_ids = sorted(clusters)
            best_pair = None
            best_distance = float("inf")
            for left_pos, left_id in enumerate(cluster_ids[:-1]):
                left_members = clusters[left_id]
                left_centroid = points[left_members].mean(axis=0)
                for right_id in cluster_ids[left_pos + 1:]:
                    right_members = clusters[right_id]
                    right_centroid = points[right_members].mean(axis=0)
                    scale = np.sqrt(
                        2.0 * len(left_members) * len(right_members)
                        / (len(left_members) + len(right_members))
                    )
                    distance = float(scale * np.linalg.norm(left_centroid - right_centroid))
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (left_id, right_id)
            if best_pair is None:
                raise RuntimeError("Ward clustering could not select a merge")
            left_id, right_id = best_pair
            merged = clusters.pop(left_id) + clusters.pop(right_id)
            clusters[next_cluster_id] = merged
            next_cluster_id += 1
            merge_heights.append(best_distance)
            if len(clusters) == 2:
                two_cluster_members = list(clusters.values())

        labels = np.zeros(len(points), dtype=int)
        if two_cluster_members is not None:
            for label, members in enumerate(two_cluster_members):
                labels[members] = label
        return labels, np.asarray(merge_heights, dtype=float)

    @staticmethod
    def _silhouette(points: np.ndarray, labels: np.ndarray) -> float:
        pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        values: list[float] = []
        for index, label in enumerate(labels):
            same = np.flatnonzero(labels == label)
            other_labels = [value for value in np.unique(labels) if value != label]
            if len(same) <= 1:
                values.append(0.0)
                continue
            same_without_self = same[same != index]
            within = float(pairwise[index, same_without_self].mean())
            between = min(
                float(pairwise[index, labels == other_label].mean())
                for other_label in other_labels
            )
            denominator = max(within, between)
            values.append(0.0 if denominator == 0.0 else (between - within) / denominator)
        return float(np.mean(values))

    @staticmethod
    def detect_round(
        client_wef_matrices: Dict[int, torch.Tensor],
        simulated_wef_matrix: torch.Tensor,
        config: S2WEFConfig | None = None,
    ) -> Dict[str, object]:
        cfg = config or S2WEFConfig()
        cfg.validate()
        client_ids = sorted(client_wef_matrices)
        if not client_ids:
            return {
                "client_scores": {}, "gamma_scores": {}, "deviation_scores": {},
                "suspicious_clients": [], "cluster_count": 1,
                "config": asdict(cfg),
            }

        flattened = torch.stack([
            client_wef_matrices[cid].detach().reshape(-1).cpu().to(torch.float64)
            for cid in client_ids
        ])
        simulated = simulated_wef_matrix.detach().reshape(-1).cpu().to(torch.float64)
        if flattened.shape[1] != simulated.numel():
            raise ValueError("Submitted and simulated WEF matrices must have equal size")
        if not torch.isfinite(flattened).all() or not torch.isfinite(simulated).all():
            raise ValueError("S2-WEF matrices contain NaN or infinite values")

        simulated_norm = torch.linalg.vector_norm(simulated)
        client_norms = torch.linalg.vector_norm(flattened, dim=1)
        cosine_denominator = client_norms * simulated_norm
        cosine = torch.where(
            cosine_denominator > cfg.numerical_eps,
            (flattened @ simulated) / cosine_denominator.clamp_min(cfg.numerical_eps),
            torch.zeros_like(client_norms),
        )
        l1_distance = torch.sum(torch.abs(flattened - simulated), dim=1)
        gamma = cosine / l1_distance.clamp_min(cfg.numerical_eps)

        deviation_result = WEFDetector.compute_round_scores(
            client_wef_matrices,
            WEFConfig(
                reputation_epsilon=cfg.deviation_epsilon,
                weight_from_end=cfg.weight_from_end,
                numerical_eps=cfg.numerical_eps,
            ),
        )
        deviation = np.asarray(
            [deviation_result["client_scores"][cid] for cid in client_ids], dtype=float
        )
        gamma_values = gamma.numpy()
        z_gamma = S2WEFDetector._robust_z(gamma_values, cfg.numerical_eps)
        z_deviation = S2WEFDetector._robust_z(deviation, cfg.numerical_eps)
        points = np.column_stack([z_gamma, z_deviation])

        labels = np.zeros(len(client_ids), dtype=int)
        cluster_count = 1
        silhouette = None
        merge_jump = None
        suspicious_label = None
        if len(client_ids) >= 3 and not np.allclose(points, points[0]):
            tentative_labels, distances = S2WEFDetector._ward_two_clusters(points)
            if len(np.unique(tentative_labels)) == 2:
                silhouette = S2WEFDetector._silhouette(points, tentative_labels)
                if len(distances) >= 2:
                    merge_jump = float(distances[-1] / (distances[-2] + cfg.numerical_eps))
                if (
                    silhouette >= cfg.silhouette_threshold
                    and merge_jump is not None
                    and merge_jump >= cfg.merge_jump_threshold
                ):
                    labels = tentative_labels
                    cluster_count = 2
                    centroid_norms = {
                        label: float(np.linalg.norm(points[labels == label].mean(axis=0)))
                        for label in np.unique(labels)
                    }
                    suspicious_label = max(centroid_norms, key=centroid_norms.get)

        gamma_threshold = float(cfg.gamma_multiplier * np.median(gamma_values))
        deviation_threshold = float(np.max(deviation) - cfg.deviation_epsilon)
        gamma_flags = gamma_values > gamma_threshold
        deviation_flags = deviation > deviation_threshold
        suspicious_clients: list[int] = []
        gamma_vote = 0.0
        deviation_vote = 0.0
        tentative_suspicious_clients: list[int] = []
        if cluster_count == 2 and suspicious_label is not None:
            mask = labels == suspicious_label
            tentative_suspicious_clients = [
                cid for cid, selected in zip(client_ids, mask) if selected
            ]
            gamma_vote = float(np.mean(gamma_flags[mask]))
            deviation_vote = float(np.mean(deviation_flags[mask]))
            if gamma_vote >= 0.5 or deviation_vote >= 0.5:
                suspicious_clients = tentative_suspicious_clients

        def keyed(values: np.ndarray) -> Dict[int, float]:
            return {cid: float(values[index]) for index, cid in enumerate(client_ids)}

        return {
            "client_scores": keyed(np.linalg.norm(points, axis=1)),
            "gamma_scores": keyed(gamma_values),
            "deviation_scores": keyed(deviation),
            "cosine_to_simulated": keyed(cosine.numpy()),
            "l1_to_simulated": keyed(l1_distance.numpy()),
            "robust_z_gamma": keyed(z_gamma),
            "robust_z_deviation": keyed(z_deviation),
            "cluster_labels": {cid: int(labels[i]) for i, cid in enumerate(client_ids)},
            "tentative_suspicious_clients": tentative_suspicious_clients,
            "suspicious_clients": suspicious_clients,
            "cluster_count": cluster_count,
            "silhouette_score": silhouette,
            "merge_jump_ratio": merge_jump,
            "gamma_threshold": gamma_threshold,
            "deviation_threshold": deviation_threshold,
            "gamma_flags": {cid: bool(gamma_flags[i]) for i, cid in enumerate(client_ids)},
            "deviation_flags": {cid: bool(deviation_flags[i]) for i, cid in enumerate(client_ids)},
            "gamma_vote": gamma_vote,
            "deviation_vote": deviation_vote,
            "simulated_wef_sum": float(simulated.sum()),
            "wef_matrix_sums": deviation_result["wef_matrix_sums"],
            "config": asdict(cfg),
            "weights": asdict(cfg),
        }

    @staticmethod
    def detect_free_riders_from_history(
        round_wef_matrices: List[List[torch.Tensor]],
        round_global_gradients: List[List[torch.Tensor]],
        local_iterations: int,
        config: S2WEFConfig | None = None,
    ) -> Dict[str, object]:
        cfg = config or S2WEFConfig()
        cfg.validate()
        if len(round_global_gradients) != len(round_wef_matrices):
            raise ValueError("S2-WEF requires aligned WEF and global-gradient histories")
        records: list[Dict[str, object]] = []
        suspicious_counts: Dict[int, int] = {}
        for round_index, matrices in enumerate(round_wef_matrices):
            if not matrices:
                raise ValueError("Each S2-WEF round must contain client matrices")

            # S2-WEF needs two previously broadcast global models.  The first
            # recorded client round has only the initial global model and no
            # valid pairwise difference, so the paper-faithful action is to
            # defer detection until the next round rather than use a zero
            # simulated pattern as evidence.
            if round_index == 0:
                records.append(
                    {
                        "round_idx": round_index,
                        "client_scores": {},
                        "gamma_scores": {},
                        "deviation_scores": {},
                        "suspicious_clients": [],
                        "cluster_count": 1,
                        "silhouette_score": None,
                        "merge_jump_ratio": None,
                        "gamma_vote": 0.0,
                        "deviation_vote": 0.0,
                        "simulated_wef_sum": 0.0,
                        "detection_skipped": True,
                        "config": asdict(cfg),
                        "weights": asdict(cfg),
                    }
                )
                continue

            simulated = S2WEFDetector.simulate_wef_matrix(
                round_global_gradients[round_index - 1],
                matrices[0].shape,
                local_iterations,
                cfg,
            )
            result = S2WEFDetector.detect_round(
                {client_id: matrix for client_id, matrix in enumerate(matrices)},
                simulated,
                cfg,
            )
            result["round_idx"] = round_index
            result["detection_skipped"] = False
            records.append(result)
            for client_id in result["suspicious_clients"]:
                suspicious_counts[client_id] = suspicious_counts.get(client_id, 0) + 1

        last_suspicious = records[-1]["suspicious_clients"] if records else []
        return {
            "suspicious_clients": list(last_suspicious),
            "suspicious_counts": suspicious_counts,
            "round_records": records,
            "decision_threshold": None,
            "effective_rounds": len(records),
            "config": asdict(cfg),
        }

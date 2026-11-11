import math
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans


class EDSCalculator:
    #Reimplementation of the EDS detector from Detecting Free-Riders in Federated Learning using an Ensemble of Similarity Distance Metrics (Arisdakessian 2024):
    #1. using submitted client updates from every round
    #2. computing pairwise ED / MD / CD between clients
    #3. log-normalize and combine them into EDS
    #4. cluster per-round client scores with KMeans(k=2)
    #5. mark clients that repeatedly fall into the high-distance cluster

    @staticmethod
    def flatten_gradient(gradient: List[torch.Tensor]) -> np.ndarray:
        return np.concatenate([g.detach().reshape(-1).cpu().numpy() for g in gradient])

    @staticmethod
    def _resolve_weights(
        weights: Optional[Dict[str, float]],
        ed_values: Optional[np.ndarray] = None,
        md_values: Optional[np.ndarray] = None,
        cd_values: Optional[np.ndarray] = None,
        dynamic_weighting: bool = False,
        eps: float = 1e-12,
    ) -> Dict[str, float]:
        if weights is None:
            weights = {
                "euclidean": 1.0 / 3.0,
                "manhattan": 1.0 / 3.0,
                "chebyshev": 1.0 / 3.0,
            }

        if not dynamic_weighting or ed_values is None or md_values is None or cd_values is None:
            return weights

        ed_mean = float(np.mean(ed_values))
        md_mean = float(np.mean(md_values))
        cd_mean = float(np.mean(cd_values))
        inv = np.array([
            1.0 / (ed_mean + eps),
            1.0 / (md_mean + eps),
            1.0 / (cd_mean + eps),
        ])
        inv = inv / inv.sum()
        return {
            "euclidean": float(inv[0]),
            "manhattan": float(inv[1]),
            "chebyshev": float(inv[2]),
        }

    @staticmethod
    def compute_round_scores(
        client_gradients: Dict[int, List[torch.Tensor]],
        weights: Optional[Dict[str, float]] = None,
        dynamic_weighting: bool = False,
    ) -> Dict[str, object]:
        client_ids = sorted(client_gradients.keys())
        num_clients = len(client_ids)
        if num_clients < 2:
            return {
                "client_scores": {cid: 0.0 for cid in client_ids},
                "pairwise_eds": np.zeros((num_clients, num_clients), dtype=np.float64),
                "weights": EDSCalculator._resolve_weights(weights),
            }

        flat_grads = [EDSCalculator.flatten_gradient(client_gradients[cid]) for cid in client_ids]
        mat = np.stack(flat_grads, axis=0)

        diff = mat[:, None, :] - mat[None, :, :]
        ed = np.linalg.norm(diff, axis=2)
        md = np.abs(diff).sum(axis=2)
        cd = np.abs(diff).max(axis=2)

        ed_log = np.log1p(ed)
        md_log = np.log1p(md)
        cd_log = np.log1p(cd)

        resolved_weights = EDSCalculator._resolve_weights(
            weights=weights,
            ed_values=ed_log[np.triu_indices(num_clients, 1)],
            md_values=md_log[np.triu_indices(num_clients, 1)],
            cd_values=cd_log[np.triu_indices(num_clients, 1)],
            dynamic_weighting=dynamic_weighting,
        )

        pairwise_eds = (
            resolved_weights["euclidean"] * ed_log
            + resolved_weights["manhattan"] * md_log
            + resolved_weights["chebyshev"] * cd_log
        )

        client_scores: Dict[int, float] = {}
        for idx, cid in enumerate(client_ids):
            mask = np.ones(num_clients, dtype=bool)
            mask[idx] = False
            client_scores[cid] = float(pairwise_eds[idx, mask].mean()) if mask.any() else 0.0

        return {
            "client_scores": client_scores,
            "pairwise_eds": pairwise_eds,
            "weights": resolved_weights,
        }

    @staticmethod
    def detect_round_free_riders(
        client_scores: Dict[int, float],
        n_clusters: int = 2,
        random_state: int = 42,
    ) -> Dict[str, object]:
        client_ids = list(client_scores.keys())
        scores = np.array([client_scores[cid] for cid in client_ids], dtype=np.float64)

        if len(client_ids) < n_clusters or np.allclose(scores, scores[0]):
            labels = np.zeros(len(client_ids), dtype=int)
            suspicious = []
            cluster_centers = np.array([scores.mean() if len(scores) > 0 else 0.0], dtype=np.float64)
            return {
                "labels": {cid: int(labels[i]) for i, cid in enumerate(client_ids)},
                "suspicious_clients": suspicious,
                "cluster_centers": cluster_centers.tolist(),
            }

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scores.reshape(-1, 1))
        centers = kmeans.cluster_centers_.reshape(-1)
        high_score_cluster = int(np.argmax(centers))
        suspicious = [cid for cid, label in zip(client_ids, labels) if int(label) == high_score_cluster]

        return {
            "labels": {cid: int(labels[i]) for i, cid in enumerate(client_ids)},
            "suspicious_clients": suspicious,
            "cluster_centers": centers.tolist(),
        }

    @staticmethod
    def detect_free_riders_from_history(
        round_client_gradients: List[List[List[torch.Tensor]]],
        weights: Optional[Dict[str, float]] = None,
        dynamic_weighting: bool = False,
        warmup_rounds: int = 0,
        persistence_ratio: float = 0.5,
        min_suspicious_rounds: int = 2,
        min_rounds_after_warmup: int = 1,
    ) -> Dict[str, object]:
        round_records = []
        suspicious_counts: Dict[int, int] = {}

        for round_idx, round_grads in enumerate(round_client_gradients):
            if round_idx < warmup_rounds:
                continue

            client_gradients = {cid: grad for cid, grad in enumerate(round_grads)}
            round_scores = EDSCalculator.compute_round_scores(
                client_gradients=client_gradients,
                weights=weights,
                dynamic_weighting=dynamic_weighting,
            )
            round_detection = EDSCalculator.detect_round_free_riders(round_scores["client_scores"])

            for cid in round_detection["suspicious_clients"]:
                suspicious_counts[cid] = suspicious_counts.get(cid, 0) + 1

            round_records.append(
                {
                    "round_idx": round_idx,
                    "client_scores": round_scores["client_scores"],
                    "suspicious_clients": round_detection["suspicious_clients"],
                    "cluster_centers": round_detection["cluster_centers"],
                    "weights": round_scores["weights"],
                }
            )

        effective_rounds = len(round_records)
        if effective_rounds < min_rounds_after_warmup:
            return {
                "suspicious_clients": [],
                "suspicious_counts": suspicious_counts,
                "round_records": round_records,
                "decision_threshold": None,
                "effective_rounds": effective_rounds,
            }

        threshold = max(min_suspicious_rounds, int(math.ceil(effective_rounds * persistence_ratio)))
        final_suspicious = sorted(
            cid for cid, count in suspicious_counts.items()
            if count >= threshold
        )

        return {
            "suspicious_clients": final_suspicious,
            "suspicious_counts": suspicious_counts,
            "round_records": round_records,
            "decision_threshold": threshold,
            "effective_rounds": effective_rounds,
        }

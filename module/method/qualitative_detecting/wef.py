"""Weight Evolving Frequency (WEF) free-rider detector.

The client-side tracker implements Eqs. (5)-(7) from Chen et al.: it counts,
after every local epoch, which penultimate-layer weights changed by more than
that epoch's mean absolute change. The server-side detector implements the
three WEF-Matrix metrics and the reputation threshold from Eqs. (8)-(11).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class WEFConfig:
    reputation_epsilon: float = 0.05
    weight_from_end: int = 2
    numerical_eps: float = 1e-12

    def validate(self) -> None:
        if self.reputation_epsilon < 0:
            raise ValueError("reputation_epsilon must be non-negative")
        if self.weight_from_end < 1:
            raise ValueError("weight_from_end must be at least 1")
        if self.numerical_eps <= 0:
            raise ValueError("numerical_eps must be positive")


class WEFClientTracker:
    """Client-side WEF collector with cumulative and round-local snapshots."""

    def __init__(self, weight_from_end: int = 2) -> None:
        if weight_from_end < 1:
            raise ValueError("weight_from_end must be at least 1")
        self.weight_from_end = weight_from_end
        self.parameter_name: str | None = None
        self.frequency: torch.Tensor | None = None
        self.round_frequency: torch.Tensor | None = None
        self.previous_weight: torch.Tensor | None = None
        self.last_dynamic_threshold: float | None = None
        self.observation_count = 0

    def _select_weight(self, model: nn.Module) -> tuple[str, torch.Tensor]:
        matrix_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.ndim >= 2
        ]
        if len(matrix_parameters) < self.weight_from_end:
            raise ValueError(
                f"WEF requested matrix parameter {self.weight_from_end} from the end, "
                f"but model exposes only {len(matrix_parameters)} matrix-shaped parameters"
            )
        return matrix_parameters[-self.weight_from_end]

    def initialize_from_model(self, model: nn.Module) -> None:
        name, weight = self._select_weight(model)
        self.parameter_name = name
        self.frequency = torch.zeros_like(
            weight.detach(),
            dtype=torch.float32,
            device=weight.device,
        )
        self.round_frequency = torch.zeros_like(self.frequency)
        self.previous_weight = weight.detach().clone()

    def begin_round(self, model: nn.Module) -> None:
        name, weight = self._select_weight(model)
        if self.frequency is None:
            self.initialize_from_model(model)
            return
        if name != self.parameter_name or tuple(weight.shape) != tuple(self.frequency.shape):
            raise ValueError("WEF selected parameter changed across training rounds")
        self.frequency = self.frequency.to(weight.device)
        self.round_frequency = torch.zeros_like(self.frequency, device=weight.device)
        self.previous_weight = weight.detach().clone()

    def observe_model(self, model: nn.Module) -> float:
        name, weight = self._select_weight(model)
        if self.frequency is None or self.round_frequency is None or self.previous_weight is None:
            self.initialize_from_model(model)
            self.last_dynamic_threshold = 0.0
            return 0.0
        if name != self.parameter_name or tuple(weight.shape) != tuple(self.frequency.shape):
            raise ValueError("WEF selected parameter changed during local training")

        current_weight = weight.detach()
        previous_weight = self.previous_weight.to(current_weight.device)
        absolute_change = torch.abs(current_weight - previous_weight)
        dynamic_threshold = torch.mean(absolute_change)
        self.frequency = self.frequency.to(current_weight.device)
        self.round_frequency = self.round_frequency.to(current_weight.device)
        increment = (absolute_change > dynamic_threshold).to(self.frequency.dtype)
        self.frequency.add_(increment)
        self.round_frequency.add_(increment)
        self.previous_weight = current_weight.clone()
        self.last_dynamic_threshold = float(dynamic_threshold.detach().cpu())
        self.observation_count += 1
        return self.last_dynamic_threshold

    def snapshot(self) -> torch.Tensor:
        if self.frequency is None:
            raise RuntimeError("WEF tracker has not been initialized")
        return self.frequency.detach().clone().cpu()

    def round_snapshot(self) -> torch.Tensor:
        """Return only the WEF counts collected in the current FL round."""
        if self.round_frequency is None:
            raise RuntimeError("WEF tracker has not been initialized")
        return self.round_frequency.detach().clone().cpu()

    def set_round_counterfeit_from_update(
        self,
        submitted_update: List[torch.Tensor],
        local_iterations: int,
        accumulate: bool = False,
    ) -> None:
        """Construct a one-step counterfeit WEF matrix.

        S2-WEF consumes a round-local matrix, so its default is intentionally
        non-cumulative.  Original WEF-Defense accumulates the uploaded matrix
        across rounds; callers using that path must set ``accumulate=True``.
        """
        if local_iterations < 1:
            raise ValueError("local_iterations must be at least 1")
        matrix_updates = [tensor for tensor in submitted_update if tensor.ndim >= 2]
        if len(matrix_updates) < self.weight_from_end:
            raise ValueError(
                f"WEF requested update matrix {self.weight_from_end} from the end, "
                f"but the submitted update exposes only {len(matrix_updates)} matrices"
            )
        selected = matrix_updates[-self.weight_from_end].detach()
        absolute_change = torch.abs(selected)
        dynamic_threshold = torch.mean(absolute_change)
        counterfeit_round = (
            (absolute_change > dynamic_threshold).to(torch.float32) * local_iterations
        )
        self.round_frequency = counterfeit_round
        if accumulate:
            if self.frequency is None:
                self.frequency = torch.zeros_like(counterfeit_round)
            self.frequency = self.frequency.to(counterfeit_round.device)
            self.frequency.add_(counterfeit_round.to(self.frequency.dtype))
        self.last_dynamic_threshold = float(dynamic_threshold.cpu())


class WEFDetector:
    """Server-side WEF-Matrix separation for cadence qualitative output."""

    @staticmethod
    def _normalize_deviation(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
        deviations = np.abs(values - float(np.mean(values)))
        denominator = float(np.sum(deviations))
        if denominator <= eps:
            return deviations, np.zeros_like(deviations)
        return deviations, deviations / denominator

    @staticmethod
    def compute_round_scores(
        client_wef_matrices: Dict[int, torch.Tensor],
        config: WEFConfig | None = None,
    ) -> Dict[str, object]:
        resolved_config = config or WEFConfig()
        resolved_config.validate()
        client_ids = sorted(client_wef_matrices)
        if not client_ids:
            return {
                "client_scores": {},
                "euclidean_distances": {},
                "cosine_similarities": {},
                "average_frequencies": {},
                "wef_matrix_sums": {},
                "suspicious_clients": [],
                "reputation_threshold": float("inf"),
                "config": asdict(resolved_config),
            }

        flattened = [
            client_wef_matrices[client_id].detach().reshape(-1).cpu().to(torch.float64)
            for client_id in client_ids
        ]
        sizes = {int(matrix.numel()) for matrix in flattened}
        if len(sizes) != 1:
            raise ValueError("All clients must upload WEF matrices with the same size")
        matrix = torch.stack(flattened, dim=0)
        if not torch.isfinite(matrix).all():
            raise ValueError("WEF matrices contain NaN or infinite values")

        num_clients = len(client_ids)
        pairwise_difference = matrix[:, None, :] - matrix[None, :, :]
        pairwise_euclidean = torch.linalg.vector_norm(pairwise_difference, dim=2)
        if num_clients > 1:
            euclidean = pairwise_euclidean.sum(dim=1) / (num_clients - 1)
        else:
            euclidean = torch.zeros(1, dtype=matrix.dtype)

        norms = torch.linalg.vector_norm(matrix, dim=1)
        denominator = norms[:, None] * norms[None, :]
        cosine_matrix = torch.where(
            denominator > resolved_config.numerical_eps,
            (matrix @ matrix.transpose(0, 1)) / denominator.clamp_min(
                resolved_config.numerical_eps
            ),
            torch.zeros_like(denominator),
        )
        if num_clients > 1:
            cosine = (
                cosine_matrix.sum(dim=1) - torch.diagonal(cosine_matrix)
            ) / (num_clients - 1)
        else:
            cosine = torch.zeros(1, dtype=matrix.dtype)
        average_frequency = matrix.mean(dim=1)
        matrix_sums = matrix.sum(dim=1)

        euclidean_values = euclidean.numpy()
        cosine_values = cosine.numpy()
        average_values = average_frequency.numpy()
        delta_distance, normalized_distance = WEFDetector._normalize_deviation(
            euclidean_values,
            resolved_config.numerical_eps,
        )
        delta_cosine, normalized_cosine = WEFDetector._normalize_deviation(
            cosine_values,
            resolved_config.numerical_eps,
        )
        delta_average, normalized_average = WEFDetector._normalize_deviation(
            average_values,
            resolved_config.numerical_eps,
        )
        deviation_scores = normalized_distance + normalized_cosine + normalized_average

        if np.allclose(deviation_scores, 0.0, atol=resolved_config.numerical_eps):
            reputation_threshold = float("inf")
            suspicious_clients: list[int] = []
        else:
            reputation_threshold = float(
                np.max(deviation_scores) - resolved_config.reputation_epsilon
            )
            suspicious_clients = [
                client_id
                for client_id, score in zip(client_ids, deviation_scores)
                if float(score) > reputation_threshold
            ]

        def as_client_dict(values: np.ndarray) -> Dict[int, float]:
            return {
                client_id: float(values[index])
                for index, client_id in enumerate(client_ids)
            }

        return {
            "client_scores": as_client_dict(deviation_scores),
            "euclidean_distances": as_client_dict(euclidean_values),
            "cosine_similarities": as_client_dict(cosine_values),
            "average_frequencies": as_client_dict(average_values),
            "wef_matrix_sums": as_client_dict(matrix_sums.numpy()),
            "distance_deviations": as_client_dict(delta_distance),
            "cosine_deviations": as_client_dict(delta_cosine),
            "average_deviations": as_client_dict(delta_average),
            "suspicious_clients": suspicious_clients,
            "reputation_threshold": reputation_threshold,
            "config": asdict(resolved_config),
        }

    @staticmethod
    def detect_free_riders_from_history(
        round_wef_matrices: List[List[torch.Tensor]],
        config: WEFConfig | None = None,
        warmup_rounds: int = 0,
        min_rounds_after_warmup: int = 1,
    ) -> Dict[str, object]:
        resolved_config = config or WEFConfig()
        resolved_config.validate()
        round_records: list[Dict[str, object]] = []
        suspicious_counts: Dict[int, int] = {}

        for round_index, round_matrices in enumerate(round_wef_matrices):
            if round_index < warmup_rounds:
                continue
            score_result = WEFDetector.compute_round_scores(
                client_wef_matrices={
                    client_id: matrix
                    for client_id, matrix in enumerate(round_matrices)
                },
                config=resolved_config,
            )
            for client_id in score_result["suspicious_clients"]:
                suspicious_counts[client_id] = suspicious_counts.get(client_id, 0) + 1

            round_records.append(
                {
                    "round_idx": round_index,
                    "client_scores": score_result["client_scores"],
                    "euclidean_distances": score_result["euclidean_distances"],
                    "cosine_similarities": score_result["cosine_similarities"],
                    "average_frequencies": score_result["average_frequencies"],
                    "wef_matrix_sums": score_result["wef_matrix_sums"],
                    "suspicious_clients": score_result["suspicious_clients"],
                    "reputation_threshold": score_result["reputation_threshold"],
                    "weights": score_result["config"],
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
                "config": asdict(resolved_config),
            }

        last_record = round_records[-1]
        return {
            "suspicious_clients": list(last_record["suspicious_clients"]),
            "suspicious_counts": suspicious_counts,
            "round_records": round_records,
            "decision_threshold": last_record["reputation_threshold"],
            "effective_rounds": effective_rounds,
            "config": asdict(resolved_config),
        }

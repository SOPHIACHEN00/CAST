"""Archived v1 STD-DAGMM detector from Lin, Du, and Liu (2019).

Each FL round is treated independently. Client gradient tensors are flattened,
passed through a deep autoencoder, and represented for density estimation by
the learned embedding, relative Euclidean reconstruction error, cosine
reconstruction similarity, and the standard deviation of the submitted update.
The estimation network and Gaussian-mixture energy follow the DAGMM objective.

This module preserves the original cadence v21 implementation, including its
KMeans/MAD per-round labels and 50% cross-round persistence decision.  The
canonical ``std_dagmm`` module now uses score-first AUC/rank evaluation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class STDDAGMMConfig:
    hidden_dims: tuple[int, ...] = (60, 30, 10)
    latent_dim: int = 1
    estimation_hidden_dim: int = 10
    n_components: int = 2
    epochs: int = 100
    learning_rate: float = 1e-3
    lambda_energy: float = 0.1
    lambda_cov_diag: float = 0.005
    covariance_eps: float = 1e-4
    # Paper-faithful default; the shared RMS normalization is an optional
    # engineering variant exposed by the caller.
    normalize_input: bool = False
    min_cluster_separation: float = 0.5
    energy_mad_scale: float = 3.0
    random_state: int = 42

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be at least 1")
        if self.estimation_hidden_dim < 1:
            raise ValueError("estimation_hidden_dim must be at least 1")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.lambda_energy < 0 or self.lambda_cov_diag < 0:
            raise ValueError("loss weights must be non-negative")
        if self.covariance_eps <= 0:
            raise ValueError("covariance_eps must be positive")
        if self.min_cluster_separation < 0:
            raise ValueError("min_cluster_separation must be non-negative")
        if self.energy_mad_scale < 0:
            raise ValueError("energy_mad_scale must be non-negative")


def _linear_stack(
    dimensions: Sequence[int],
    activate_last: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(in_features, out_features))
        if activate_last or index < len(dimensions) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _resolve_hidden_dims(
    input_dim: int,
    requested: Sequence[int],
    latent_dim: int,
) -> tuple[int, ...]:
    """Keep the compression network decreasing for small gradient vectors."""
    resolved: list[int] = []
    previous = input_dim
    for requested_width in requested:
        width = min(int(requested_width), max(latent_dim + 1, previous // 2))
        if latent_dim < width < previous:
            resolved.append(width)
            previous = width
    if not resolved and input_dim > latent_dim + 1:
        resolved.append(max(latent_dim + 1, input_dim // 2))
    return tuple(resolved)


class _STDDAGMMNetwork(nn.Module):
    def __init__(self, input_dim: int, config: STDDAGMMConfig) -> None:
        super().__init__()
        hidden_dims = _resolve_hidden_dims(input_dim, config.hidden_dims, config.latent_dim)
        encoder_dims = (input_dim, *hidden_dims, config.latent_dim)
        decoder_dims = (config.latent_dim, *reversed(hidden_dims), input_dim)

        self.encoder = _linear_stack(encoder_dims, activate_last=False)
        self.decoder = _linear_stack(decoder_dims, activate_last=False)

        estimation_input_dim = config.latent_dim + 3
        self.estimation = nn.Sequential(
            nn.Linear(estimation_input_dim, config.estimation_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.estimation_hidden_dim, config.n_components),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        eps: float,
        update_std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)

        relative_euclidean = torch.linalg.vector_norm(
            x - reconstruction,
            dim=1,
            keepdim=True,
        ) / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(eps)
        cosine_similarity = F.cosine_similarity(
            x,
            reconstruction,
            dim=1,
            eps=eps,
        ).unsqueeze(1)
        if update_std is None:
            update_std = torch.std(x, dim=1, unbiased=False, keepdim=True)

        density_features = torch.cat(
            [embedding, relative_euclidean, cosine_similarity, update_std],
            dim=1,
        )
        memberships = self.estimation(density_features)
        return reconstruction, density_features, memberships, update_std


class STDDAGMMDetector:
    """Round-local STD-DAGMM scorer with cadence-compatible history output."""

    @staticmethod
    def flatten_gradient(
        gradient: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not gradient:
            raise ValueError("Each client gradient must contain at least one tensor")
        return torch.cat(
            [tensor.detach().reshape(-1).to(device=device, dtype=torch.float32) for tensor in gradient]
        )

    @staticmethod
    def _estimate_gmm(
        features: torch.Tensor,
        memberships: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        membership_mass = memberships.sum(dim=0).clamp_min(eps)
        phi = membership_mass / memberships.shape[0]
        means = memberships.transpose(0, 1) @ features
        means = means / membership_mass.unsqueeze(1)

        centered = features.unsqueeze(1) - means.unsqueeze(0)
        covariance = torch.einsum(
            "nk,nkd,nke->kde",
            memberships,
            centered,
            centered,
        )
        covariance = covariance / membership_mass[:, None, None]
        return phi, means, covariance

    @staticmethod
    def _sample_energy(
        features: torch.Tensor,
        phi: torch.Tensor,
        means: torch.Tensor,
        covariance: torch.Tensor,
        covariance_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_dim = features.shape[1]
        identity = torch.eye(feature_dim, device=features.device, dtype=features.dtype)
        regularized_covariance = covariance + covariance_eps * identity.unsqueeze(0)

        centered = features.unsqueeze(1) - means.unsqueeze(0)
        solved = torch.linalg.solve(
            regularized_covariance.unsqueeze(0),
            centered.unsqueeze(-1),
        ).squeeze(-1)
        mahalanobis = torch.sum(centered * solved, dim=2)

        sign, log_determinant = torch.linalg.slogdet(regularized_covariance)
        if torch.any(sign <= 0):
            raise RuntimeError("STD-DAGMM covariance is not positive definite")

        log_component_probability = (
            torch.log(phi.clamp_min(covariance_eps)).unsqueeze(0)
            - 0.5
            * (
                feature_dim * math.log(2.0 * math.pi)
                + log_determinant.unsqueeze(0)
                + mahalanobis
            )
        )
        energy = -torch.logsumexp(log_component_probability, dim=1)

        covariance_diagonal = torch.diagonal(
            regularized_covariance,
            dim1=1,
            dim2=2,
        )
        covariance_penalty = torch.sum(1.0 / covariance_diagonal.clamp_min(covariance_eps))
        return energy, covariance_penalty

    @staticmethod
    def fit_score_round(
        client_gradients: Dict[int, List[torch.Tensor]],
        config: STDDAGMMConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> Dict[str, object]:
        resolved_config = config or STDDAGMMConfig()
        resolved_config.validate()
        resolved_device = torch.device(device)

        client_ids = sorted(client_gradients)
        if len(client_ids) < 2:
            return {
                "client_scores": {client_id: 0.0 for client_id in client_ids},
                "client_stds": {client_id: 0.0 for client_id in client_ids},
                "final_loss": 0.0,
                "input_scale": 1.0,
                "config": asdict(resolved_config),
            }

        flattened = [
            STDDAGMMDetector.flatten_gradient(client_gradients[client_id], resolved_device)
            for client_id in client_ids
        ]
        input_lengths = {int(vector.numel()) for vector in flattened}
        if len(input_lengths) != 1:
            raise ValueError("All clients must submit gradients with the same flattened size")

        inputs = torch.stack(flattened, dim=0)
        if not torch.isfinite(inputs).all():
            raise ValueError("Client gradients contain NaN or infinite values")

        raw_stds = torch.std(inputs, dim=1, unbiased=False)
        input_scale_tensor = torch.sqrt(torch.mean(inputs.square())).clamp_min(
            resolved_config.covariance_eps
        )
        model_inputs = inputs / input_scale_tensor if resolved_config.normalize_input else inputs

        torch.manual_seed(resolved_config.random_state)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(resolved_config.random_state)

        network = _STDDAGMMNetwork(model_inputs.shape[1], resolved_config).to(resolved_device)
        optimizer = torch.optim.Adam(network.parameters(), lr=resolved_config.learning_rate)

        final_loss = float("nan")
        for _ in range(resolved_config.epochs):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            reconstruction, features, memberships, _ = network(
                model_inputs,
                resolved_config.covariance_eps,
                update_std=raw_stds.unsqueeze(1),
            )
            phi, means, covariance = STDDAGMMDetector._estimate_gmm(
                features,
                memberships,
                resolved_config.covariance_eps,
            )
            energy, covariance_penalty = STDDAGMMDetector._sample_energy(
                features,
                phi,
                means,
                covariance,
                resolved_config.covariance_eps,
            )
            reconstruction_loss = F.mse_loss(reconstruction, model_inputs)
            loss = (
                reconstruction_loss
                + resolved_config.lambda_energy * energy.mean()
                + resolved_config.lambda_cov_diag * covariance_penalty
            )
            if not torch.isfinite(loss):
                raise RuntimeError("STD-DAGMM training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())

        network.eval()
        with torch.no_grad():
            _, features, memberships, _ = network(
                model_inputs,
                resolved_config.covariance_eps,
                update_std=raw_stds.unsqueeze(1),
            )
            phi, means, covariance = STDDAGMMDetector._estimate_gmm(
                features,
                memberships,
                resolved_config.covariance_eps,
            )
            energies, _ = STDDAGMMDetector._sample_energy(
                features,
                phi,
                means,
                covariance,
                resolved_config.covariance_eps,
            )

        energy_values = energies.detach().cpu().numpy().astype(np.float64)
        std_values = raw_stds.detach().cpu().numpy().astype(np.float64)
        return {
            "client_scores": {
                client_id: float(energy_values[index])
                for index, client_id in enumerate(client_ids)
            },
            "client_stds": {
                client_id: float(std_values[index])
                for index, client_id in enumerate(client_ids)
            },
            "final_loss": final_loss,
            "input_scale": float(input_scale_tensor.detach().cpu()),
            "config": asdict(resolved_config),
        }

    @staticmethod
    def detect_round_free_riders(
        client_scores: Dict[int, float],
        min_cluster_separation: float = 0.5,
        energy_mad_scale: float = 3.0,
        random_state: int = 42,
    ) -> Dict[str, object]:
        from sklearn.cluster import KMeans

        client_ids = list(client_scores)
        energies = np.asarray([client_scores[client_id] for client_id in client_ids], dtype=np.float64)
        num_clients = len(client_ids)

        if num_clients < 3 or np.allclose(energies, energies[0]):
            return {
                "labels": {client_id: 0 for client_id in client_ids},
                "suspicious_clients": [],
                "cluster_centers": [float(energies.mean()) if num_clients else 0.0],
                "cluster_separation": 0.0,
                "energy_threshold": float("inf"),
            }

        kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(energies.reshape(-1, 1))
        centers = kmeans.cluster_centers_.reshape(-1)
        high_energy_cluster = int(np.argmax(centers))
        high_energy_indices = np.flatnonzero(labels == high_energy_cluster)
        separation = float(
            abs(centers[0] - centers[1]) / (float(np.std(energies)) + 1e-12)
        )
        median_energy = float(np.median(energies))
        median_absolute_deviation = float(np.median(np.abs(energies - median_energy)))
        robust_scale = max(1.4826 * median_absolute_deviation, 1e-12)
        energy_threshold = median_energy + energy_mad_scale * robust_scale

        # Lin et al. identify clients whose energy is higher than the majority.
        # The paper does not specify a numerical threshold; enforce its benign-
        # majority assumption and require a configurable, non-trivial split.
        is_strict_minority = len(high_energy_indices) * 2 < num_clients
        accept_cluster = is_strict_minority and separation >= min_cluster_separation
        suspicious = []
        if accept_cluster:
            suspicious = [
                client_ids[index]
                for index in high_energy_indices
                if energies[index] > energy_threshold
            ]

        return {
            "labels": {client_id: int(labels[index]) for index, client_id in enumerate(client_ids)},
            "suspicious_clients": suspicious,
            "cluster_centers": [float(center) for center in centers],
            "cluster_separation": separation,
            "energy_threshold": energy_threshold,
        }

    @staticmethod
    def detect_free_riders_from_history(
        round_client_gradients: List[List[List[torch.Tensor]]],
        config: STDDAGMMConfig | None = None,
        device: torch.device | str = "cpu",
        warmup_rounds: int = 0,
        persistence_ratio: float = 0.5,
        min_suspicious_rounds: int = 2,
        min_rounds_after_warmup: int = 1,
    ) -> Dict[str, object]:
        resolved_config = config or STDDAGMMConfig()
        resolved_config.validate()
        if not 0.0 <= persistence_ratio <= 1.0:
            raise ValueError("persistence_ratio must be in [0, 1]")

        round_records: list[Dict[str, object]] = []
        suspicious_counts: Dict[int, int] = {}

        for round_index, round_gradients in enumerate(round_client_gradients):
            if round_index < warmup_rounds:
                continue

            score_result = STDDAGMMDetector.fit_score_round(
                client_gradients={
                    client_id: gradient
                    for client_id, gradient in enumerate(round_gradients)
                },
                config=STDDAGMMConfig(
                    **{
                        **asdict(resolved_config),
                        "random_state": resolved_config.random_state + round_index,
                    }
                ),
                device=device,
            )
            round_detection = STDDAGMMDetector.detect_round_free_riders(
                client_scores=score_result["client_scores"],
                min_cluster_separation=resolved_config.min_cluster_separation,
                energy_mad_scale=resolved_config.energy_mad_scale,
                random_state=resolved_config.random_state + round_index,
            )

            for client_id in round_detection["suspicious_clients"]:
                suspicious_counts[client_id] = suspicious_counts.get(client_id, 0) + 1

            round_records.append(
                {
                    "round_idx": round_index,
                    "client_scores": score_result["client_scores"],
                    "client_stds": score_result["client_stds"],
                    "suspicious_clients": round_detection["suspicious_clients"],
                    "cluster_centers": round_detection["cluster_centers"],
                    "cluster_separation": round_detection["cluster_separation"],
                    "energy_threshold": round_detection["energy_threshold"],
                    "final_loss": score_result["final_loss"],
                    "input_scale": score_result["input_scale"],
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

        decision_threshold = max(
            min_suspicious_rounds,
            int(math.ceil(effective_rounds * persistence_ratio)),
        )
        suspicious_clients = sorted(
            client_id
            for client_id, count in suspicious_counts.items()
            if count >= decision_threshold
        )
        return {
            "suspicious_clients": suspicious_clients,
            "suspicious_counts": suspicious_counts,
            "round_records": round_records,
            "decision_threshold": decision_threshold,
            "effective_rounds": effective_rounds,
            "config": asdict(resolved_config),
        }

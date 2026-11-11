"""Delta-DAGMM free-rider detector from Huang et al. (2022).

Cadence stores each submitted client update as the difference between the
client's locally updated parameters and the current global parameters.  That is
already the paper's Delta-theta sample, so this module applies only the paper's
affine sample treatment before fitting the compression and estimation networks.
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
class DeltaDAGMMConfig:
    hidden_dims: tuple[int, ...] = (60, 30, 10)
    latent_dim: int = 1
    estimation_hidden_dim: int = 10
    n_components: int = 2
    epochs: int = 100
    learning_rate: float = 1e-3
    lambda_energy: float = 0.1
    lambda_cov_diag: float = 0.005
    covariance_eps: float = 1e-4
    linear_scale: float = 1.0
    linear_offset: float = 0.0
    output_layer_only: bool = True
    # Huang et al. specify the affine treatment x = k*Delta_theta + b, but
    # not a subsequent round-wise RMS normalization. Keep the paper path as
    # the default; normalization remains an opt-in numerical variant.
    normalize_input: bool = False
    threshold_multiplier: float = 1.08
    # The paper's decision is the literal E_i > W * E_avg comparison.
    signed_energy_safe_threshold: bool = False
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
        if not math.isfinite(self.linear_scale) or self.linear_scale == 0:
            raise ValueError("linear_scale must be finite and non-zero")
        if not math.isfinite(self.linear_offset):
            raise ValueError("linear_offset must be finite")
        if self.threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be positive")


def _linear_stack(dimensions: Sequence[int], activate_last: bool) -> nn.Sequential:
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


class _DeltaDAGMMNetwork(nn.Module):
    def __init__(self, input_dim: int, config: DeltaDAGMMConfig) -> None:
        super().__init__()
        hidden_dims = _resolve_hidden_dims(input_dim, config.hidden_dims, config.latent_dim)
        self.encoder = _linear_stack(
            (input_dim, *hidden_dims, config.latent_dim),
            activate_last=False,
        )
        self.decoder = _linear_stack(
            (config.latent_dim, *reversed(hidden_dims), input_dim),
            activate_last=False,
        )

        # Z = [Z_c, Z_r, Z_avg]. DAGMM's Z_r consists of relative
        # Euclidean reconstruction distance and cosine similarity.
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.encoder(x)
        reconstruction = self.decoder(code)
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
        input_mean = torch.mean(x, dim=1, keepdim=True)
        density_features = torch.cat(
            [code, relative_euclidean, cosine_similarity, input_mean],
            dim=1,
        )
        memberships = self.estimation(density_features)
        return reconstruction, density_features, memberships, input_mean


class DeltaDAGMMDetector:
    """Paper-aligned round detector with cadence-compatible history output."""

    @staticmethod
    def flatten_update(
        update: List[torch.Tensor],
        device: torch.device,
        output_layer_only: bool = True,
    ) -> torch.Tensor:
        if not update:
            raise ValueError("Each client update must contain at least one tensor")

        selected = update
        if output_layer_only:
            weight_tensors = [tensor for tensor in update if tensor.ndim >= 2]
            if not weight_tensors:
                raise ValueError(
                    "output_layer_only requires at least one matrix-shaped parameter update"
                )
            selected = [weight_tensors[-1]]

        return torch.cat(
            [tensor.detach().reshape(-1).to(device=device, dtype=torch.float32) for tensor in selected]
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
            raise RuntimeError("Delta-DAGMM covariance is not positive definite")

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
        client_updates: Dict[int, List[torch.Tensor]],
        config: DeltaDAGMMConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> Dict[str, object]:
        resolved_config = config or DeltaDAGMMConfig()
        resolved_config.validate()
        resolved_device = torch.device(device)
        client_ids = sorted(client_updates)

        if len(client_ids) < 2:
            return {
                "client_scores": {client_id: 0.0 for client_id in client_ids},
                "client_means": {client_id: 0.0 for client_id in client_ids},
                "final_loss": 0.0,
                "input_scale": 1.0,
                "config": asdict(resolved_config),
            }

        flattened = [
            DeltaDAGMMDetector.flatten_update(
                client_updates[client_id],
                resolved_device,
                output_layer_only=resolved_config.output_layer_only,
            )
            for client_id in client_ids
        ]
        input_lengths = {int(vector.numel()) for vector in flattened}
        if len(input_lengths) != 1:
            raise ValueError("All clients must submit updates with the same selected size")

        deltas = torch.stack(flattened, dim=0)
        if not torch.isfinite(deltas).all():
            raise ValueError("Client updates contain NaN or infinite values")

        # Eq. (5): x = k * (theta_i(t) - theta(t)) + b. Cadence's input is
        # already theta_i(t) - theta(t), so do not subtract model parameters here.
        treated_inputs = (
            resolved_config.linear_scale * deltas + resolved_config.linear_offset
        )
        input_scale_tensor = torch.sqrt(torch.mean(treated_inputs.square())).clamp_min(
            resolved_config.covariance_eps
        )
        model_inputs = (
            treated_inputs / input_scale_tensor
            if resolved_config.normalize_input
            else treated_inputs
        )
        treated_means = torch.mean(treated_inputs, dim=1)

        torch.manual_seed(resolved_config.random_state)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(resolved_config.random_state)

        network = _DeltaDAGMMNetwork(model_inputs.shape[1], resolved_config).to(resolved_device)
        optimizer = torch.optim.Adam(network.parameters(), lr=resolved_config.learning_rate)
        final_loss = float("nan")

        for _ in range(resolved_config.epochs):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            reconstruction, features, memberships, _ = network(
                model_inputs,
                resolved_config.covariance_eps,
            )
            phi, means, covariance = DeltaDAGMMDetector._estimate_gmm(
                features,
                memberships,
                resolved_config.covariance_eps,
            )
            energy, covariance_penalty = DeltaDAGMMDetector._sample_energy(
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
                raise RuntimeError("Delta-DAGMM training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())

        network.eval()
        with torch.no_grad():
            _, features, memberships, _ = network(
                model_inputs,
                resolved_config.covariance_eps,
            )
            phi, means, covariance = DeltaDAGMMDetector._estimate_gmm(
                features,
                memberships,
                resolved_config.covariance_eps,
            )
            energies, _ = DeltaDAGMMDetector._sample_energy(
                features,
                phi,
                means,
                covariance,
                resolved_config.covariance_eps,
            )

        energy_values = energies.detach().cpu().numpy().astype(np.float64)
        mean_values = treated_means.detach().cpu().numpy().astype(np.float64)
        return {
            "client_scores": {
                client_id: float(energy_values[index])
                for index, client_id in enumerate(client_ids)
            },
            "client_means": {
                client_id: float(mean_values[index])
                for index, client_id in enumerate(client_ids)
            },
            "final_loss": final_loss,
            "input_scale": float(input_scale_tensor.detach().cpu()),
            "config": asdict(resolved_config),
        }

    @staticmethod
    def detect_round_free_riders(
        client_scores: Dict[int, float],
        threshold_multiplier: float = 1.08,
        signed_energy_safe_threshold: bool = False,
    ) -> Dict[str, object]:
        if threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be positive")
        client_ids = list(client_scores)
        energies = np.asarray(
            [client_scores[client_id] for client_id in client_ids],
            dtype=np.float64,
        )
        if energies.size == 0:
            mean_energy = 0.0
            energy_threshold = float("inf")
            suspicious: list[int] = []
        else:
            mean_energy = float(np.mean(energies))
            if signed_energy_safe_threshold and mean_energy < 0:
                # The paper assumes a positive energy scale. A continuous GMM
                # density can exceed one, making negative log density negative;
                # literal multiplication would then lower the threshold and
                # invert the intended "8% above average" comparison.
                energy_threshold = mean_energy + (
                    threshold_multiplier - 1.0
                ) * abs(mean_energy)
            else:
                energy_threshold = mean_energy * threshold_multiplier
            suspicious = [
                client_id
                for client_id, energy in zip(client_ids, energies)
                if float(energy) > energy_threshold
            ]

        return {
            "labels": {
                client_id: int(client_id in suspicious) for client_id in client_ids
            },
            "suspicious_clients": suspicious,
            "mean_energy": mean_energy,
            "energy_threshold": energy_threshold,
            "signed_energy_adjusted": bool(
                signed_energy_safe_threshold and mean_energy < 0
            ),
        }

    @staticmethod
    def detect_free_riders_from_history(
        round_client_gradients: List[List[List[torch.Tensor]]],
        config: DeltaDAGMMConfig | None = None,
        device: torch.device | str = "cpu",
        warmup_rounds: int = 0,
        persistence_ratio: float = 2.0 / 3.0,
        min_suspicious_rounds: int = 1,
        min_rounds_after_warmup: int = 1,
    ) -> Dict[str, object]:
        resolved_config = config or DeltaDAGMMConfig()
        resolved_config.validate()
        if not 0.0 <= persistence_ratio <= 1.0:
            raise ValueError("persistence_ratio must be in [0, 1]")

        round_records: list[Dict[str, object]] = []
        suspicious_counts: Dict[int, int] = {}
        for round_index, round_updates in enumerate(round_client_gradients):
            if round_index < warmup_rounds:
                continue

            round_config = DeltaDAGMMConfig(
                **{
                    **asdict(resolved_config),
                    "random_state": resolved_config.random_state + round_index,
                }
            )
            score_result = DeltaDAGMMDetector.fit_score_round(
                client_updates={
                    client_id: update
                    for client_id, update in enumerate(round_updates)
                },
                config=round_config,
                device=device,
            )
            round_detection = DeltaDAGMMDetector.detect_round_free_riders(
                client_scores=score_result["client_scores"],
                threshold_multiplier=resolved_config.threshold_multiplier,
                signed_energy_safe_threshold=resolved_config.signed_energy_safe_threshold,
            )
            for client_id in round_detection["suspicious_clients"]:
                suspicious_counts[client_id] = suspicious_counts.get(client_id, 0) + 1

            round_records.append(
                {
                    "round_idx": round_index,
                    "client_scores": score_result["client_scores"],
                    "client_means": score_result["client_means"],
                    "suspicious_clients": round_detection["suspicious_clients"],
                    "mean_energy": round_detection["mean_energy"],
                    "energy_threshold": round_detection["energy_threshold"],
                    "signed_energy_adjusted": round_detection["signed_energy_adjusted"],
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

        # The paper defines F = ceil(2/3 * number_of_rounds) and says that a
        # client is finally a free-rider when detected for more than F rounds.
        # Therefore the first admissible count is F + 1.
        persistence_floor = int(math.ceil(effective_rounds * persistence_ratio)) + 1
        decision_threshold = max(min_suspicious_rounds, persistence_floor)
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

"""Sample-weighted server aggregation shared by FL and last-round coalitions.

Client uploads are parameter deltas (local minus broadcast), so returned steps
are ADDED to the server model. Local training is deliberately unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class AggregationConfig:
    method: str = 'fedavg'
    server_lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 0.001

    def __post_init__(self) -> None:
        method = str(self.method).lower()
        if method == 'median':
            method = 'weighted_median'
        if method not in {'fedavg', 'fedadam', 'weighted_median'}:
            raise ValueError(f'Unknown FL aggregation method: {self.method}')
        object.__setattr__(self, 'method', method)
        for name in ('server_lr', 'beta1', 'beta2', 'tau'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if self.server_lr <= 0 or self.tau <= 0:
            raise ValueError('server_lr and tau must be positive')
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError('server beta1 and beta2 must be in [0, 1)')

    @classmethod
    def from_args(cls, args: Any) -> AggregationConfig:
        return cls(method=getattr(args, 'fl_aggregate', 'fedavg'),
                   server_lr=getattr(args, 'server_lr', 0.01),
                   beta1=getattr(args, 'server_beta1', 0.9),
                   beta2=getattr(args, 'server_beta2', 0.99),
                   tau=getattr(args, 'server_tau', 0.001))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def output_subdirectory(self) -> str:
        """Keep old FedAvg paths; separate new algorithms and Adam settings."""
        if self.method == 'fedavg':
            return ''
        if self.method == 'weighted_median':
            return self.method
        return (f'fedadam_lr{self.server_lr}_b1{self.beta1}'
                f'_b2{self.beta2}_tau{self.tau}')


def add_aggregation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--fl_aggregate', '--aggregation_method', dest='fl_aggregate',
        choices=['fedavg', 'fedadam', 'weighted_median', 'median'], default='fedavg',
        help='Sample-weighted server rule. median is an alias for weighted_median; default preserves FedAvg.')
    parser.add_argument('--server_lr', type=float, default=0.01,
        help='FedAdam server learning rate; independent of the unchanged local optimizer.')
    parser.add_argument('--server_beta1', type=float, default=0.9)
    parser.add_argument('--server_beta2', type=float, default=0.99)
    parser.add_argument('--server_tau', type=float, default=0.001,
        help='FedAdam denominator offset; initial second moment is tau squared, without bias correction.')


class ServerAggregator:
    """One run's server state. A restored instance owns copies of all moments.

    ``weights`` are normalized non-negative sample shares. FedAvg intentionally
    uses the existing multiply-then-add arithmetic, without re-normalizing them.
    Weighted median sorts unscaled coordinate values and accumulates their sample
    weights; it is NOT the median of weight-multiplied updates.
    """

    def __init__(self, config: AggregationConfig, *, state: Mapping[str, Any] | None = None):
        self.config = config
        self.rounds = 0
        self.m: list[torch.Tensor] | None = None
        self.v: list[torch.Tensor] | None = None
        if state is not None:
            if dict(state.get('config', {})) != config.to_dict():
                raise ValueError('Server state and aggregation configuration do not match')
            rounds = state.get('rounds')
            if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
                raise ValueError('Invalid server state round count')
            self.rounds = rounds
            self.m = self._clone_tensors(state.get('m'))
            self.v = self._clone_tensors(state.get('v'))
            if (self.m is None) != (self.v is None):
                raise ValueError('Incomplete FedAdam moment state')
            if config.method == 'fedadam' and rounds > 0 and self.m is None:
                raise ValueError('Missing FedAdam moment state after training rounds')

    @staticmethod
    def _clone_tensors(values: Sequence[torch.Tensor] | None) -> list[torch.Tensor] | None:
        return None if values is None else [value.detach().clone() for value in values]

    def state_dict(self) -> dict[str, Any]:
        return dict(config=self.config.to_dict(), rounds=self.rounds,
                    m=self._clone_tensors(self.m), v=self._clone_tensors(self.v))

    @torch.no_grad()
    def step(self, updates: Sequence[Sequence[torch.Tensor]],
             weights: Sequence[Any], parameters: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not updates or len(updates) != len(weights) or not parameters:
            raise ValueError('Nonempty updates, matching weights and model parameters are required')
        shares = torch.as_tensor(weights, dtype=torch.float64, device=parameters[0].device)
        if shares.ndim != 1 or not bool(torch.isfinite(shares).all()) or bool((shares < 0).any()):
            raise ValueError('Sample weights must be a finite non-negative vector')
        total = float(shares.sum())
        if total <= 0 or not math.isclose(total, 1., rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError('Sample weights must be normalized and sum to one')
        for update in updates:
            if len(update) != len(parameters):
                raise ValueError('Client update has the wrong number of model parameters')
            for delta, parameter in zip(update, parameters):
                if delta.shape != parameter.shape or delta.device != parameter.device or delta.dtype != parameter.dtype:
                    raise ValueError('Client update shape/device/dtype differs from the server parameter')
                if not bool(torch.isfinite(delta).all()):
                    raise ValueError('Client update contains nonfinite values')

        if self.config.method == 'weighted_median':
            result = self._weighted_median(updates, shares, parameters)
        else:
            result = [torch.zeros_like(parameter) for parameter in parameters]
            for update, weight in zip(updates, weights):
                for aggregate, delta in zip(result, update):
                    aggregate += delta * weight
            if self.config.method == 'fedadam':
                result = self._adam_step(result)
        self.rounds += 1
        return result

    def _adam_step(self, delta: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        config = self.config
        if self.m is None:
            self.m = [torch.zeros_like(value) for value in delta]
            self.v = [torch.full_like(value, config.tau ** 2) for value in delta]
        if len(self.m) != len(delta) or len(self.v) != len(delta):
            raise ValueError('FedAdam state does not match the model parameters')
        result = []
        for index, value in enumerate(delta):
            old_m, old_v = self.m[index], self.v[index]
            if old_m.shape != value.shape or old_v.shape != value.shape:
                raise ValueError('FedAdam state shape does not match the model')
            old_m = old_m.to(device=value.device, dtype=value.dtype)
            old_v = old_v.to(device=value.device, dtype=value.dtype)
            if not bool(torch.isfinite(old_m).all() & torch.isfinite(old_v).all()) or bool((old_v < 0).any()):
                raise ValueError('FedAdam moment state must be finite, with non-negative second moment')
            self.m[index] = config.beta1 * old_m + (1 - config.beta1) * value
            self.v[index] = config.beta2 * old_v + (1 - config.beta2) * value.square()
            result.append(config.server_lr * self.m[index] / (self.v[index].sqrt() + config.tau))
        return result

    @staticmethod
    def _weighted_median(updates: Sequence[Sequence[torch.Tensor]], shares: torch.Tensor,
                         parameters: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        active = [i for i, weight in enumerate(shares) if float(weight) > 0]
        positive_weights = shares[active]
        half = positive_weights.sum() / 2
        # Only arithmetic roundoff is tolerated, not a float32-scale weight band.
        tolerance = 8 * torch.finfo(torch.float64).eps * float(positive_weights.sum()) * len(active)
        result = []
        for parameter_index, parameter in enumerate(parameters):
            output = torch.empty_like(parameter).reshape(-1)
            flat_updates = [updates[i][parameter_index].reshape(-1) for i in active]
            # Bound temporary GPU memory for large models during coalition enumeration.
            for start in range(0, output.numel(), 65536):
                end = min(start + 65536, output.numel())
                values = torch.stack([value[start:end] for value in flat_updates])
                ordered, indices = values.sort(dim=0, stable=True)
                cumulative = positive_weights[indices].cumsum(dim=0)
                lower = (cumulative < half - tolerance).sum(dim=0).clamp(max=len(active) - 1)
                lower_value = ordered.gather(0, lower.unsqueeze(0)).squeeze(0)
                mass = cumulative.gather(0, lower.unsqueeze(0)).squeeze(0)
                upper = (lower + 1).clamp(max=len(active) - 1)
                upper_value = ordered.gather(0, upper.unsqueeze(0)).squeeze(0)
                is_tie = (mass - half).abs() <= tolerance
                output[start:end] = torch.where(is_tie, lower_value / 2 + upper_value / 2, lower_value)
            result.append(output.reshape(parameter.shape))
        return result

import math
import time
from numbers import Integral

import numpy as np
import torch

from module.method.quantitative_detecting.coalition_value import (
    AttackAwareCoalitionValue,
)


class SamplingShapleyValue(AttackAwareCoalitionValue):
    """Truncated Monte Carlo Shapley estimator over random permutations."""

    name = "SamplingShapleyValue"

    def _resolve_num_permutations(self, num_permutations: int | None) -> int:
        if self.num_parts < 1:
            raise ValueError("sampling_shapley requires at least one client")
        if num_permutations is None:
            sample_budget = min(
                2 ** self.num_parts,
                math.log(self.num_parts) * self.num_parts ** 2,
            )
            return max(1, round(sample_budget / self.num_parts))
        if not isinstance(num_permutations, Integral) or isinstance(
            num_permutations,
            bool,
        ):
            raise ValueError("num_permutations must be a positive integer")
        if int(num_permutations) < 1:
            raise ValueError("num_permutations must be at least 1")
        return int(num_permutations)

    def get_contributions(
        self,
        num_permutations: int | None = None,
        seed: int | None = None,
        tolerance: float = 1e-6,
    ):
        effective_num_permutations = self._resolve_num_permutations(
            num_permutations
        )
        resolved_tolerance = float(tolerance)
        if not math.isfinite(resolved_tolerance) or resolved_tolerance < 0.0:
            raise ValueError("tolerance must be a finite non-negative number")
        if seed is not None and (
            not isinstance(seed, Integral) or isinstance(seed, bool)
        ):
            raise ValueError("seed must be an integer or None")

        self.requested_num_permutations = num_permutations
        self.effective_num_permutations = effective_num_permutations
        self.sampling_seed = None if seed is None else int(seed)
        self.tolerance = resolved_tolerance

        rng = np.random.default_rng(self.sampling_seed)
        all_clients = set(range(self.num_parts))
        started_at = time.time()

        for value_index, value_function in enumerate(self.value_functions):
            grand_value = float(
                self.evaluate_subset_world(all_clients, value_function)
            )
            empty_value = float(
                self.evaluate_subset_world(set(), value_function)
            )
            marginal_totals = np.zeros(self.num_parts, dtype=float)

            for _ in range(effective_num_permutations):
                coalition = set()
                previous_value = empty_value
                for sampled_client_id in rng.permutation(self.num_parts):
                    if abs(grand_value - previous_value) < resolved_tolerance:
                        break
                    client_id = int(sampled_client_id)
                    coalition.add(client_id)
                    current_value = float(
                        self.evaluate_subset_world(coalition, value_function)
                    )
                    marginal_totals[client_id] += current_value - previous_value
                    previous_value = current_value

            self.contributions[value_index] = (
                marginal_totals / effective_num_permutations
            )

        device = getattr(self.model, "device", torch.device("cpu"))
        if "cuda" in str(device):
            torch.cuda.synchronize()

        elapsed_seconds = time.time() - started_at
        return self.contributions.tolist(), elapsed_seconds

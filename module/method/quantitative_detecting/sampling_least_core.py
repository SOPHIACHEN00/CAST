import math
import time
from numbers import Integral

import numpy as np
import torch
from scipy.optimize import linprog

from module.method.quantitative_detecting.coalition_value import (
    AttackAwareCoalitionValue,
)


class SamplingLeastCore(AttackAwareCoalitionValue):
    """Least Core allocation constrained by a sampled set of coalitions."""

    name = "SamplingLeastCore"

    def _resolve_num_coalitions(self, num_coalitions: int | None) -> int:
        if self.num_parts < 1:
            raise ValueError("sampling_least_core requires at least one client")
        total_coalitions = 2 ** self.num_parts
        if num_coalitions is None:
            # Empty and grand coalitions are mandatory constraints.  The
            # attacker-removed world in the full client matrix can have one
            # client, for which n**2 is 1 although there are two coalitions.
            return max(2, min(total_coalitions, self.num_parts ** 2))
        if not isinstance(num_coalitions, Integral) or isinstance(
            num_coalitions,
            bool,
        ):
            raise ValueError("num_coalitions must be an integer")
        resolved_count = int(num_coalitions)
        if not 2 <= resolved_count <= total_coalitions:
            raise ValueError(
                "num_coalitions must be between 2 and "
                f"{total_coalitions} for {self.num_parts} clients"
            )
        return resolved_count

    def _sample_coalitions(
        self,
        num_coalitions: int,
        rng: np.random.Generator,
    ) -> tuple[frozenset[int], ...]:
        grand_mask = (1 << self.num_parts) - 1
        interior_count = num_coalitions - 2
        if interior_count > 0:
            interior_masks = rng.choice(
                np.arange(1, grand_mask, dtype=np.int64),
                size=interior_count,
                replace=False,
            )
            selected_masks = [0]
            selected_masks.extend(sorted(int(mask) for mask in interior_masks))
            selected_masks.append(grand_mask)
        else:
            selected_masks = [0, grand_mask]

        return tuple(
            frozenset(
                client_id
                for client_id in range(self.num_parts)
                if mask & (1 << client_id)
            )
            for mask in selected_masks
        )

    def _solve_value_function(
        self,
        value_function: str,
        coalitions: tuple[frozenset[int], ...],
    ) -> tuple[np.ndarray, float]:
        coalition_values = [
            float(self.evaluate_subset_world(set(coalition), value_function))
            for coalition in coalitions
        ]
        grand_value = coalition_values[-1]

        objective = np.concatenate(
            [np.zeros(self.num_parts, dtype=float), np.array([1.0])]
        )
        inequality_rows = []
        inequality_values = []
        for coalition, coalition_value in zip(coalitions, coalition_values):
            row = np.zeros(self.num_parts + 1, dtype=float)
            for client_id in coalition:
                row[client_id] = -1.0
            row[-1] = -1.0
            inequality_rows.append(row)
            inequality_values.append(-coalition_value)

        equality_rows = np.array(
            [[1.0] * self.num_parts + [0.0]],
            dtype=float,
        )
        equality_values = np.array([grand_value], dtype=float)
        allocation_bound = float(self.num_parts)
        bounds = [
            (-allocation_bound, allocation_bound)
            for _ in range(self.num_parts)
        ]
        bounds.append((0.0, None))

        result = linprog(
            objective,
            A_ub=np.asarray(inequality_rows, dtype=float),
            b_ub=np.asarray(inequality_values, dtype=float),
            A_eq=equality_rows,
            b_eq=equality_values,
            bounds=bounds,
            method="highs",
        )
        solution = getattr(result, "x", None)
        if (
            not getattr(result, "success", False)
            or solution is None
            or len(solution) != self.num_parts + 1
            or not np.all(np.isfinite(solution))
        ):
            message = getattr(result, "message", "unknown solver failure")
            raise RuntimeError(f"Sampling Least Core LP failed: {message}")

        allocation = np.asarray(solution[:-1], dtype=float)
        epsilon = float(solution[-1])
        if not math.isfinite(epsilon):
            raise RuntimeError("Sampling Least Core LP returned a non-finite epsilon")
        return allocation, epsilon

    def get_contributions(
        self,
        num_coalitions: int | None = None,
        seed: int | None = None,
    ):
        effective_num_coalitions = self._resolve_num_coalitions(num_coalitions)
        if seed is not None and (
            not isinstance(seed, Integral) or isinstance(seed, bool)
        ):
            raise ValueError("seed must be an integer or None")

        self.requested_num_coalitions = num_coalitions
        self.effective_num_coalitions = effective_num_coalitions
        self.sampling_seed = None if seed is None else int(seed)

        started_at = time.time()
        rng = np.random.default_rng(self.sampling_seed)
        self.selected_coalitions = self._sample_coalitions(
            effective_num_coalitions,
            rng,
        )
        self.least_core_epsilons = []

        for value_index, value_function in enumerate(self.value_functions):
            allocation, epsilon = self._solve_value_function(
                value_function,
                self.selected_coalitions,
            )
            self.contributions[value_index] = allocation
            self.least_core_epsilons.append(epsilon)

        self.least_core_epsilon = (
            self.least_core_epsilons[0]
            if len(self.least_core_epsilons) == 1
            else list(self.least_core_epsilons)
        )

        device = getattr(self.model, "device", torch.device("cpu"))
        if "cuda" in str(device):
            torch.cuda.synchronize()

        elapsed_seconds = time.time() - started_at
        return self.contributions.tolist(), elapsed_seconds

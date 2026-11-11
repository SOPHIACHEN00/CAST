import csv
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


QUANTITATIVE_DIAGNOSTIC_FIELDS = [
    "Dataset",
    "AttackMethod",
    "AttackRegime",
    "Alpha",
    "ClientNum",
    "AttackerID",
    "TrialID",
    "DetectingMethod",
    "World",
    "ResolvedSeed",
    "RequestedSamplingBudget",
    "EffectiveSamplingBudget",
    "SamplingBudgetUnit",
    "Tolerance",
    "LeastCoreEpsilon",
    "ElapsedSeconds",
]


def _require_integer(value, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def resolve_sampling_seed(
    explicit_seed: int | None,
    experiment_seed: int,
    trial_id: int,
) -> int:
    """Resolve an estimator-local seed without changing global RNG state."""
    resolved_experiment_seed = _require_integer(experiment_seed, "experiment_seed")
    resolved_trial_id = _require_integer(trial_id, "trial_id")
    if explicit_seed is not None:
        return _require_integer(explicit_seed, "sampling seed")
    return resolved_experiment_seed + resolved_trial_id * 100003


def build_sampling_call_kwargs(
    method: str,
    *,
    num_parts: int,
    experiment_seed: int,
    trial_id: int,
    shapley_num_permutations: int | None,
    shapley_seed: int | None,
    shapley_tolerance: float,
    least_core_num_coalitions: int | None,
    least_core_seed: int | None,
) -> dict[str, Any]:
    """Build only the keyword arguments accepted by the selected estimator."""
    normalized_method = str(method).lower()
    if normalized_method == "shapley":
        return {}

    resolved_num_parts = _require_integer(num_parts, "num_parts")
    if resolved_num_parts < 1:
        raise ValueError("num_parts must be at least 1")

    if normalized_method == "sampling_shapley":
        if shapley_num_permutations is not None:
            resolved_permutations = _require_integer(
                shapley_num_permutations,
                "sampling_shapley_num_permutations",
            )
            if resolved_permutations < 1:
                raise ValueError(
                    "sampling_shapley_num_permutations must be at least 1"
                )
        else:
            resolved_permutations = None

        resolved_tolerance = float(shapley_tolerance)
        if not math.isfinite(resolved_tolerance) or resolved_tolerance < 0.0:
            raise ValueError(
                "sampling_shapley_tolerance must be a finite non-negative number"
            )
        return {
            "num_permutations": resolved_permutations,
            "seed": resolve_sampling_seed(
                shapley_seed,
                experiment_seed,
                trial_id,
            ),
            "tolerance": resolved_tolerance,
        }

    if normalized_method == "sampling_least_core":
        total_coalitions = 2 ** resolved_num_parts
        if least_core_num_coalitions is not None:
            resolved_coalitions = _require_integer(
                least_core_num_coalitions,
                "sampling_least_core_num_coalitions",
            )
            if resolved_coalitions < 2:
                raise ValueError(
                    "sampling_least_core_num_coalitions must be at least 2"
                )
            if resolved_coalitions > total_coalitions:
                raise ValueError(
                    "sampling_least_core_num_coalitions may be at most "
                    f"{total_coalitions} for {resolved_num_parts} clients"
                )
        else:
            resolved_coalitions = None
        return {
            "num_coalitions": resolved_coalitions,
            "seed": resolve_sampling_seed(
                least_core_seed,
                experiment_seed,
                trial_id,
            ),
        }

    raise ValueError(f"Method '{method}' is not a quantitative estimator")


def extract_realized_fake_rounds(
    round_activity: Sequence[Mapping[str, Any]],
) -> frozenset[int]:
    """Extract the exact fake-round set recorded by an FL run."""
    return frozenset(
        int(activity["round_idx"])
        for activity in round_activity
        if bool(activity.get("is_fake_round", False))
    )


def sample_probabilistic_fake_rounds(
    *,
    eligible_rounds,
    num_rounds: int,
    attack_probability: float,
    seed: int,
) -> tuple[frozenset[int], int]:
    """Sample the fixed-size fake-round set used by one lazy FL run."""
    resolved_num_rounds = _require_integer(num_rounds, "num_rounds")
    if resolved_num_rounds < 0:
        raise ValueError("num_rounds must be non-negative")
    resolved_probability = float(attack_probability)
    if not math.isfinite(resolved_probability) or not 0.0 <= resolved_probability <= 1.0:
        raise ValueError("attack_probability must be a finite number in [0, 1]")
    resolved_seed = _require_integer(seed, "seed")
    resolved_eligible_rounds = np.asarray(
        sorted(set(int(round_num) for round_num in eligible_rounds)),
        dtype=int,
    )
    if any(
        round_num < 0 or round_num >= resolved_num_rounds
        for round_num in resolved_eligible_rounds
    ):
        raise ValueError("eligible_rounds must lie within the configured FL rounds")

    fake_budget = int(round(resolved_probability * resolved_num_rounds))
    fake_budget = max(
        0,
        min(int(resolved_eligible_rounds.size), fake_budget),
    )
    if fake_budget == 0:
        return frozenset(), 0

    rng = np.random.default_rng(resolved_seed)
    sampled = rng.choice(
        resolved_eligible_rounds,
        size=fake_budget,
        replace=False,
    )
    return frozenset(int(round_num) for round_num in sampled), fake_budget


def publish_realized_fake_rounds(
    output: set[int] | None,
    fake_rounds,
) -> None:
    """Replace a caller-owned schedule output with one run's realized rounds."""
    if output is None:
        return
    output.clear()
    output.update(int(round_num) for round_num in fake_rounds)


def build_quantitative_diagnostic_row(
    *,
    metadata: Mapping[str, Any],
    world: str,
    estimator,
    requested_budget: int | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build one sampled-estimator diagnostics row for an evaluated world."""
    method = str(metadata["DetectingMethod"]).lower()
    if method == "sampling_shapley":
        effective_budget = int(estimator.effective_num_permutations)
        budget_unit = "permutations"
        tolerance = float(estimator.tolerance)
        least_core_epsilon = "N/A"
    elif method == "sampling_least_core":
        effective_budget = int(estimator.effective_num_coalitions)
        budget_unit = "coalitions"
        tolerance = "N/A"
        least_core_epsilon = float(estimator.least_core_epsilon)
    else:
        raise ValueError(
            f"Diagnostics are only defined for sampled methods, got '{method}'"
        )

    return {
        "Dataset": metadata["Dataset"],
        "AttackMethod": metadata["AttackMethod"],
        "AttackRegime": metadata["AttackRegime"],
        "Alpha": metadata["Alpha"],
        "ClientNum": metadata["ClientNum"],
        "AttackerID": metadata["AttackerID"],
        "TrialID": metadata["TrialID"],
        "DetectingMethod": method,
        "World": str(world),
        "ResolvedSeed": int(estimator.sampling_seed),
        "RequestedSamplingBudget": (
            "auto" if requested_budget is None else int(requested_budget)
        ),
        "EffectiveSamplingBudget": effective_budget,
        "SamplingBudgetUnit": budget_unit,
        "Tolerance": tolerance,
        "LeastCoreEpsilon": least_core_epsilon,
        "ElapsedSeconds": float(elapsed_seconds),
    }


def write_quantitative_diagnostics(
    output_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write one complete sampled-method diagnostic artifact."""
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=QUANTITATIVE_DIAGNOSTIC_FIELDS,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

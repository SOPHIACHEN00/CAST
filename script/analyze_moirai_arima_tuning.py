#!/usr/bin/env python3
"""Read-only diagnostics for Moirai/ARIMA tuning campaigns.

The experiment launchers keep each task isolated.  This analyzer joins the
task manifest/status/log with its Accuracy/Detecting/Time CSV files and emits
one task-level table plus a parameter-group aggregate.  The input tree is
never modified.  Files are written only when ``--output-dir`` is explicit.

Parameter groups are ordered by execution/forecast stability.  Accuracy gaps
and attacker contributions are reported for diagnosis, but are deliberately
excluded from the ordering so tuning does not simply maximize attack damage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "1.1"
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FLOAT_RE = re.compile(FLOAT_PATTERN)
KEY_VALUE_RE = re.compile(r"^([A-Za-z0-9_.-]+)=(.*)$")
FIT_STATS_RE = re.compile(
    rf"success=(\d+)\s+constant=(\d+)\s+failed=(\d+)"
)
FORECAST_STATS_RE = re.compile(
    rf"success=(\d+)\s+constant=(\d+)\s+fallback=(\d+)\s+failed=(\d+)"
)
MAE_RE = re.compile(rf"\bmae=({FLOAT_PATTERN})\b", re.IGNORECASE)
EMA_RE = re.compile(rf"\bema=({FLOAT_PATTERN})\b", re.IGNORECASE)
RESIDUAL_NORM_RE = re.compile(
    rf"\bresidual_norm=({FLOAT_PATTERN})\b", re.IGNORECASE
)
ROUND_RE = re.compile(r"\bround=(\d+)\b", re.IGNORECASE)
FALLBACK_REASON_RE = re.compile(r"\[FALLBACK\].*?\breason=([^\s]+)")

IDENTITY_COLUMNS = (
    "dataset",
    "method",
    "alpha",
    "client_num",
    "scenario",
    "gap",
    "probability",
    "trial",
)

# These are model/readiness/tuning knobs.  Protocol identity (alpha/client/
# lazy schedule) stays in task identity and is intentionally not part of a
# parameter-group definition.
TUNING_PREFIXES = ("arima_", "moirai_", "layerwise_")
TUNING_EXACT = {
    "lambda_h",
    "random_round",
    "use_hessian",
}

AGGREGATE_METRICS = (
    "status_runtime_seconds",
    "runtime_total_sec",
    "forecast_coverage",
    "alignment_coverage",
    "forecast_first_round",
    "fit_usable_fraction",
    "forecast_usable_fraction",
    "mae_first",
    "mae_last",
    "mae_mean",
    "ema_last",
    "reset_count",
    "periodic_refit_count",
    "residual_update_count",
    "residual_decay_count",
    "residual_norm_mean",
    "residual_norm_last",
    "residual_norm_max",
    "accuracy_with_auc",
    "accuracy_without_auc",
    "accuracy_auc_gap",
    "accuracy_final_gap",
    "accuracy_mean_gap",
    "attacker_contribution_raw",
    "attacker_contribution_minmax",
)


@dataclass
class TaskLocation:
    """Paths belonging to one isolated experiment task."""

    label: str
    run_root: Optional[Path] = None
    manifest_path: Optional[Path] = None
    results_root: Optional[Path] = None


def _read_text(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "none", "nan", "null"}:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _pvariance(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return statistics.pvariance(clean) if len(clean) > 1 else 0.0


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator else None


def _parse_manifest(path: Path) -> Tuple[Dict[str, str], List[str]]:
    values: Dict[str, str] = {}
    argv: List[str] = []
    in_argv = False
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if line == "command_argv:":
            in_argv = True
            continue
        if in_argv:
            if line:
                argv.append(line)
            continue
        match = KEY_VALUE_RE.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values, argv


def _parse_command_options(argv: Sequence[str]) -> Dict[str, str]:
    options: Dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        option = token[2:]
        if "=" in option:
            name, value = option.split("=", 1)
            options[name.replace("-", "_")] = value
            index += 1
            continue
        name = option.replace("-", "_")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            options[name] = argv[index + 1]
            index += 2
        else:
            options[name] = "true"
            index += 1
    return options


def _parse_status(text: str) -> Dict[str, Any]:
    """Parse the launcher's process/protocol status without conflating them."""

    raw = text.strip()
    if not raw:
        return {
            "status": "missing",
            "status_raw": "",
            "status_process_success": False,
            "status_protocol": None,
            "status_protocol_pass": False,
            "status_success": False,
        }

    tokens = raw.split()
    process_success = tokens[0].lower() == "success"
    protocol: Optional[str] = None
    fields: Dict[str, str] = {}
    for token in tokens[1:]:
        if token.startswith("protocol_"):
            protocol = token[len("protocol_") :].lower()
        elif "=" in token:
            name, value = token.split("=", 1)
            fields[name] = value

    protocol_pass = protocol == "pass"
    strict_success = process_success and protocol_pass
    if strict_success:
        normalized_status = "success"
    elif process_success and protocol is not None:
        normalized_status = f"protocol_{protocol}"
    else:
        normalized_status = tokens[0].lower()

    parsed: Dict[str, Any] = {
        "status": normalized_status,
        "status_raw": raw,
        "status_process_success": process_success,
        "status_protocol": protocol,
        "status_protocol_pass": protocol_pass,
        "status_success": strict_success,
        "status_runtime_seconds": _as_float(fields.get("runtime_seconds")),
        "status_rounds": _as_int(fields.get("rounds")),
        "status_honest": _as_int(fields.get("honest")),
        "status_fake": _as_int(fields.get("fake")),
        "status_hessian": _as_int(fields.get("hessian")),
        "status_exit_code": _as_int(fields.get("exit_code")),
    }
    return parsed


def _is_tuning_parameter(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return normalized.startswith(TUNING_PREFIXES) or normalized in TUNING_EXACT


def _resolve_manifest_results(
    input_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, str],
) -> Optional[Path]:
    configured = manifest.get("results_root")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_dir():
            return configured_path.resolve()

    task_label = manifest.get("task_label", manifest_path.parent.name)
    # If the analyzer was pointed at a Results campaign, this direct child is
    # the safest relocation fallback for a copied/moved manifest.
    direct = input_root / task_label
    if direct.is_dir():
        return direct.resolve()

    # Conventional launcher layout:
    #   PROJECT/.matrix_runs/CAMPAIGN/experiments/TASK
    #   PROJECT/Results/CAMPAIGN/TASK
    run_root = manifest_path.parent
    if run_root.parent.name == "experiments":
        campaign_root = run_root.parent.parent
        project_root = campaign_root.parent.parent
        inferred = project_root / "Results" / campaign_root.name / task_label
        if inferred.is_dir():
            return inferred.resolve()
    return None


def _result_roots_below(root: Path) -> List[Path]:
    result_roots = set()
    patterns = (
        "global_accuracy_log_trial*.csv",
        "combined_contributions_trial*.csv",
        "attacking_time_log_trial*.csv",
    )
    for pattern in patterns:
        for csv_path in root.rglob(pattern):
            # CSV lives in Accuracy_*/Detecting_*/Time_*; its parent is the
            # isolated task Results root.
            result_roots.add(csv_path.parent.parent.resolve())
    return sorted(result_roots, key=lambda path: str(path))


def _infer_run_root(results_root: Path) -> Optional[Path]:
    # Expected Results shape is PROJECT/Results/CAMPAIGN/TASK.
    if results_root.parent.parent.name != "Results":
        return None
    project_root = results_root.parent.parent.parent
    campaign = results_root.parent.name
    task = results_root.name
    for container in (".matrix_runs", ".tuning_runs", ".pilot_runs", ".campaign_runs"):
        candidate = project_root / container / campaign / "experiments" / task
        if candidate.is_dir():
            return candidate.resolve()
    return None


def discover_tasks(root: Path) -> List[TaskLocation]:
    """Discover task manifests and/or isolated Results directories."""

    tasks: Dict[str, TaskLocation] = {}
    for manifest_path in root.rglob("manifest.txt"):
        manifest, _ = _parse_manifest(manifest_path)
        if not (
            "task_label" in manifest
            or "dataset" in manifest
            or "attack_method" in manifest
            or "results_root" in manifest
        ):
            continue
        run_root = manifest_path.parent.resolve()
        results_root = _resolve_manifest_results(root, manifest_path, manifest)
        label = manifest.get("task_label", run_root.name)
        key = str(results_root or run_root)
        tasks[key] = TaskLocation(
            label=label,
            run_root=run_root,
            manifest_path=manifest_path.resolve(),
            results_root=results_root,
        )

    for results_root in _result_roots_below(root):
        key = str(results_root)
        if key in tasks:
            continue
        run_root = _infer_run_root(results_root)
        manifest_path = None
        if run_root is not None and (run_root / "manifest.txt").is_file():
            manifest_path = (run_root / "manifest.txt").resolve()
        tasks[key] = TaskLocation(
            label=results_root.name,
            run_root=run_root,
            manifest_path=manifest_path,
            results_root=results_root,
        )

    # A direct task path may have no CSV yet (failed/incomplete task).  Keep it
    # visible when its manifest was found above.  A completely unrelated path
    # correctly yields zero tasks.
    return sorted(tasks.values(), key=lambda item: item.label)


def _read_csv_rows(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
        except (OSError, csv.Error):
            # A malformed/missing artifact is represented through absent
            # metrics.  The task remains in the summary for failure triage.
            continue
    return rows


def _first_nonempty(rows: Sequence[Mapping[str, str]], column: str) -> Optional[str]:
    for row in rows:
        value = row.get(column)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _trapezoid(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    if not points:
        return None
    ordered = sorted(points)
    if len(ordered) == 1:
        return ordered[0][1]
    return sum(
        (x1 - x0) * (y0 + y1) / 2.0
        for (x0, y0), (x1, y1) in zip(ordered, ordered[1:])
    )


def _accuracy_metrics(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    with_by_round: Dict[float, float] = {}
    without_by_round: Dict[float, float] = {}
    for row in rows:
        round_number = _as_float(row.get("Round"))
        if round_number is None:
            continue
        with_value = _as_float(row.get("WithAttackerAccuracy"))
        without_value = _as_float(row.get("WithoutAttackerAccuracy"))
        if with_value is not None:
            with_by_round[round_number] = with_value
        if without_value is not None:
            without_by_round[round_number] = without_value

    with_points = sorted(with_by_round.items())
    without_points = sorted(without_by_round.items())
    with_auc = _trapezoid(with_points)
    without_auc = _trapezoid(without_points)
    common_rounds = sorted(set(with_by_round) & set(without_by_round))
    gaps = [with_by_round[idx] - without_by_round[idx] for idx in common_rounds]
    return {
        "accuracy_round_count": len(set(with_by_round) | set(without_by_round)),
        "accuracy_with_auc": with_auc,
        "accuracy_without_auc": without_auc,
        "accuracy_auc_gap": (
            with_auc - without_auc
            if with_auc is not None and without_auc is not None
            else None
        ),
        "accuracy_with_final": with_points[-1][1] if with_points else None,
        "accuracy_without_final": without_points[-1][1] if without_points else None,
        "accuracy_final_gap": gaps[-1] if gaps else None,
        "accuracy_mean_gap": _mean(gaps),
    }


def _contribution_metrics(
    rows: Sequence[Mapping[str, str]],
    attacker_id: Optional[int],
    client_num: Optional[int],
) -> Dict[str, Any]:
    attacker_row: Optional[Mapping[str, str]] = None
    for row in rows:
        row_attacker = _as_int(row.get("AttackerID"))
        row_client = _as_int(row.get("ClientID"))
        expected = attacker_id if attacker_id is not None else row_attacker
        if expected is not None and row_client == expected:
            attacker_row = row
    raw = _as_float(attacker_row.get("WithAttacker")) if attacker_row else None
    minmax = (
        _as_float(attacker_row.get("WithAttackerMinMax"))
        if attacker_row
        else None
    )
    return {
        "attacker_contribution_raw": raw,
        "attacker_contribution_minmax": minmax,
        "two_client_minmax_degenerate": bool(client_num == 2),
        # This stays false even for N>2: MinMax is descriptive, never the
        # primary tuning criterion in this analyzer.
        "minmax_primary_metric": False,
    }


def _time_metrics(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    with_values = [
        value
        for value in (_as_float(row.get("WithAttackerSec")) for row in rows)
        if value is not None
    ]
    without_values = [
        value
        for value in (_as_float(row.get("WithoutAttackerSec")) for row in rows)
        if value is not None
    ]
    with_sec = sum(with_values) if with_values else None
    without_sec = sum(without_values) if without_values else None
    total = None
    if with_sec is not None or without_sec is not None:
        total = float(with_sec or 0.0) + float(without_sec or 0.0)
    return {
        "runtime_with_attacker_sec": with_sec,
        "runtime_without_attacker_sec": without_sec,
        "runtime_total_sec": total,
    }


def _log_metrics(text: str) -> Dict[str, Any]:
    forecast_events = 0
    fallback_events = 0
    aligned_events = 0
    reset_events = 0
    periodic_events = 0
    periodic_fit_events = 0
    fit_events = 0
    fit_success = fit_constant = fit_failed = 0
    forecast_success = forecast_constant = forecast_fallback = forecast_failed = 0
    residual_updates = 0
    residual_decays = 0
    forecast_rounds: List[int] = []
    mae_values: List[float] = []
    ema_values: List[float] = []
    residual_norm_values: List[float] = []
    fallback_reasons: Counter[str] = Counter()

    for line in text.splitlines():
        is_forecast = "[FORECAST]" in line or "USING MOIRAI PREDICTION" in line
        if is_forecast:
            forecast_events += 1
            round_match = ROUND_RE.search(line)
            if round_match:
                forecast_rounds.append(int(round_match.group(1)))
        if "[FALLBACK]" in line:
            fallback_events += 1
            reason_match = FALLBACK_REASON_RE.search(line)
            if reason_match:
                fallback_reasons[reason_match.group(1)] += 1

        if "[FIT]" in line:
            stats_match = FIT_STATS_RE.search(line)
            if stats_match:
                fit_events += 1
                success, constant, failed = map(int, stats_match.groups())
                fit_success += success
                fit_constant += constant
                fit_failed += failed
            if "reason=periodic@" in line:
                periodic_fit_events += 1

        if is_forecast:
            stats_match = FORECAST_STATS_RE.search(line)
            if stats_match:
                success, constant, fallback, failed = map(int, stats_match.groups())
                forecast_success += success
                forecast_constant += constant
                forecast_fallback += fallback
                forecast_failed += failed

        if "[OBSERVE]" in line:
            mae_match = MAE_RE.search(line)
            if mae_match:
                mae_values.append(float(mae_match.group(1)))
                # Numeric MAE is emitted only after an exact/aligned pending
                # prediction was consumed.  This also covers layerwise reset
                # lines that omit an explicit alignment= token.
                aligned_events += 1
            ema_match = EMA_RE.search(line)
            if ema_match:
                ema_values.append(float(ema_match.group(1)))
            residual_norm_match = RESIDUAL_NORM_RE.search(line)
            if residual_norm_match:
                residual_norm_values.append(float(residual_norm_match.group(1)))
            if re.search(r"\breset=1\b", line):
                reset_events += 1
            if re.search(r"\bperiodic_refit=1\b", line):
                periodic_events += 1
            if re.search(r"\bresidual_updated=1\b", line):
                residual_updates += 1
            if re.search(r"\bdecay_applied=1\b", line):
                residual_decays += 1

    # Older logs may record a periodic FIT but not the OBSERVE flag.  Do not
    # double count newer logs that contain both representations.
    periodic_refits = periodic_events if periodic_events else periodic_fit_events
    forecast_denominator = forecast_events + fallback_events
    fit_denominator = fit_success + fit_constant + fit_failed
    forecast_dimension_denominator = (
        forecast_success
        + forecast_constant
        + forecast_fallback
        + forecast_failed
    )
    return {
        "forecast_event_count": forecast_events,
        "forecast_first_round": forecast_rounds[0] if forecast_rounds else None,
        "forecast_last_round": forecast_rounds[-1] if forecast_rounds else None,
        "fallback_event_count": fallback_events,
        "forecast_coverage": _ratio(forecast_events, forecast_denominator),
        "aligned_count": aligned_events,
        "alignment_coverage": _ratio(aligned_events, forecast_events),
        "reset_count": reset_events,
        "periodic_refit_count": periodic_refits,
        "fit_count": fit_events,
        "fit_success_count": fit_success,
        "fit_constant_count": fit_constant,
        "fit_failed_count": fit_failed,
        "fit_usable_fraction": _ratio(fit_success + fit_constant, fit_denominator),
        "forecast_success_count": forecast_success,
        "forecast_constant_count": forecast_constant,
        "forecast_fallback_count": forecast_fallback,
        "forecast_failed_count": forecast_failed,
        "forecast_usable_fraction": _ratio(
            forecast_success + forecast_constant,
            forecast_dimension_denominator,
        ),
        "mae_count": len(mae_values),
        "mae_first": mae_values[0] if mae_values else None,
        "mae_last": mae_values[-1] if mae_values else None,
        "mae_mean": _mean(mae_values),
        "ema_last": ema_values[-1] if ema_values else None,
        "residual_update_count": residual_updates,
        "residual_decay_count": residual_decays,
        "residual_norm_count": len(residual_norm_values),
        "residual_norm_first": (
            residual_norm_values[0] if residual_norm_values else None
        ),
        "residual_norm_last": (
            residual_norm_values[-1] if residual_norm_values else None
        ),
        "residual_norm_mean": _mean(residual_norm_values),
        "residual_norm_max": (
            max(residual_norm_values) if residual_norm_values else None
        ),
        "fallback_reason_counts": json.dumps(
            dict(sorted(fallback_reasons.items())),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _artifact_paths(results_root: Optional[Path], kind: str) -> List[Path]:
    if results_root is None or not results_root.is_dir():
        return []
    patterns = {
        "accuracy": "Accuracy_*/global_accuracy_log_trial*.csv",
        "contribution": "Detecting_*/combined_contributions_trial*.csv",
        "time": "Time_*/attacking_time_log_trial*.csv",
    }
    return list(results_root.glob(patterns[kind]))


def analyze_task(location: TaskLocation) -> Dict[str, Any]:
    manifest: Dict[str, str] = {}
    argv: List[str] = []
    if location.manifest_path is not None and location.manifest_path.is_file():
        manifest, argv = _parse_manifest(location.manifest_path)
    options = _parse_command_options(argv)

    accuracy_rows = _read_csv_rows(_artifact_paths(location.results_root, "accuracy"))
    contribution_rows = _read_csv_rows(
        _artifact_paths(location.results_root, "contribution")
    )
    time_rows = _read_csv_rows(_artifact_paths(location.results_root, "time"))
    all_rows: List[Mapping[str, str]] = accuracy_rows + contribution_rows + time_rows

    def choose(*values: Any) -> Optional[str]:
        for value in values:
            if value is not None and str(value).strip() != "":
                return str(value).strip()
        return None

    dataset = choose(manifest.get("dataset"), options.get("dataset"), _first_nonempty(all_rows, "Dataset"))
    method = choose(
        manifest.get("attack_method"),
        manifest.get("method"),
        options.get("attack_method"),
        _first_nonempty(all_rows, "AttackMethod"),
    )
    alpha = choose(manifest.get("alpha"), options.get("alpha"), _first_nonempty(all_rows, "Alpha"))
    client_num = _as_int(
        choose(
            manifest.get("client_num"),
            options.get("client_num"),
            _first_nonempty(all_rows, "ClientNum"),
        )
    )
    attacker_id = _as_int(
        choose(
            manifest.get("attacker_id"),
            options.get("attacker_id"),
            _first_nonempty(all_rows, "AttackerID"),
        )
    )
    scenario = choose(
        manifest.get("scenario"),
        options.get("lazy_schedule"),
        _first_nonempty(all_rows, "LazySchedule"),
    )
    gap = choose(
        manifest.get("gap"),
        options.get("lazy_honest_gap"),
        _first_nonempty(all_rows, "LazyHonestGap"),
    )
    paired_probability = choose(
        manifest.get("paired_probability"),
        manifest.get("probability"),
    )
    attack_probability = choose(
        options.get("attack_probability"),
        _first_nonempty(all_rows, "AttackProbability"),
    )
    probability = paired_probability or attack_probability
    trial = choose(
        manifest.get("trial_id"),
        options.get("trial_id"),
        _first_nonempty(all_rows, "TrialID"),
    )

    run_root = location.run_root
    status_path = run_root / "status.txt" if run_root is not None else None
    log_path = run_root / "logs" / "run.log" if run_root is not None else None
    status_text = _read_text(status_path).strip()
    status_fields = _parse_status(status_text)
    if not status_text and (accuracy_rows or contribution_rows or time_rows):
        status_fields["status"] = "artifacts_present_status_unknown"

    task: Dict[str, Any] = {
        "task_label": location.label,
        "pilot_profile_set": manifest.get("pilot_profile_set"),
        "variant_kind": manifest.get("variant_kind"),
        "profile_name": manifest.get("profile_name"),
        "dataset": dataset,
        "method": method,
        "alpha": alpha,
        "client_num": client_num,
        "scenario": scenario,
        "gap": gap,
        "probability": probability,
        "attack_probability": attack_probability,
        "trial": trial,
        "attacker_id": attacker_id,
        "run_root": str(run_root) if run_root is not None else None,
        "results_root": (
            str(location.results_root) if location.results_root is not None else None
        ),
        "manifest_path": (
            str(location.manifest_path)
            if location.manifest_path is not None
            else None
        ),
        "status_path": str(status_path) if status_path is not None else None,
        "log_path": str(log_path) if log_path is not None else None,
    }
    task.update(status_fields)

    # Record every explicitly supplied model/readiness tuning parameter.
    tuning_values: Dict[str, str] = {}
    for name, value in options.items():
        if _is_tuning_parameter(name):
            tuning_values[name] = value
    for name, value in manifest.items():
        normalized = name.lower().replace("-", "_")
        if _is_tuning_parameter(normalized):
            tuning_values.setdefault(normalized, value)
        elif normalized.startswith(("tuning_", "param_")):
            tuning_values.setdefault(normalized, value)
    for name, value in sorted(tuning_values.items()):
        task[f"param_{name}"] = value

    task.update(_log_metrics(_read_text(log_path)))
    task.update(_time_metrics(time_rows))
    task.update(_accuracy_metrics(accuracy_rows))
    task.update(_contribution_metrics(contribution_rows, attacker_id, client_num))
    return task


def _parameter_group_key(task: Mapping[str, Any], parameter_columns: Sequence[str]) -> Tuple[Any, ...]:
    return (
        task.get("dataset"),
        task.get("method"),
        *(task.get(column) for column in parameter_columns),
    )


def aggregate_parameter_groups(tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    parameter_columns = sorted(
        {
            column
            for task in tasks
            for column in task
            if column.startswith("param_")
        }
    )
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[_parameter_group_key(task, parameter_columns)].append(task)

    aggregates: List[Dict[str, Any]] = []
    for key, group_tasks in grouped.items():
        identity = {
            "dataset": key[0],
            "method": key[1],
        }
        identity.update(dict(zip(parameter_columns, key[2:])))
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
        result: Dict[str, Any] = {
            "parameter_group_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
            **identity,
            "task_count": len(group_tasks),
            "successful_task_count": sum(bool(task.get("status_success")) for task in group_tasks),
            "status_success_rate": _mean(
                1.0 if task.get("status_success") else 0.0 for task in group_tasks
            ),
            "two_client_task_count": sum(
                bool(task.get("two_client_minmax_degenerate")) for task in group_tasks
            ),
            "minmax_primary_metric": False,
        }

        for metric in AGGREGATE_METRICS:
            numeric = [_as_float(task.get(metric)) for task in group_tasks]
            result[f"{metric}_observed_count"] = sum(value is not None for value in numeric)
            result[f"{metric}_mean"] = _mean(numeric)
            result[f"{metric}_variance"] = _pvariance(numeric)

        forecast_total = sum(int(task.get("forecast_event_count") or 0) for task in group_tasks)
        fallback_total = sum(int(task.get("fallback_event_count") or 0) for task in group_tasks)
        aligned_total = sum(int(task.get("aligned_count") or 0) for task in group_tasks)
        fit_success = sum(int(task.get("fit_success_count") or 0) for task in group_tasks)
        fit_constant = sum(int(task.get("fit_constant_count") or 0) for task in group_tasks)
        fit_failed = sum(int(task.get("fit_failed_count") or 0) for task in group_tasks)
        forecast_success = sum(int(task.get("forecast_success_count") or 0) for task in group_tasks)
        forecast_constant = sum(int(task.get("forecast_constant_count") or 0) for task in group_tasks)
        forecast_fallback = sum(int(task.get("forecast_fallback_count") or 0) for task in group_tasks)
        forecast_failed = sum(int(task.get("forecast_failed_count") or 0) for task in group_tasks)

        result.update(
            {
                "forecast_event_total": forecast_total,
                "fallback_event_total": fallback_total,
                "forecast_coverage_pooled": _ratio(
                    forecast_total, forecast_total + fallback_total
                ),
                "aligned_total": aligned_total,
                "alignment_coverage_pooled": _ratio(aligned_total, forecast_total),
                "fit_usable_fraction_pooled": _ratio(
                    fit_success + fit_constant,
                    fit_success + fit_constant + fit_failed,
                ),
                "forecast_usable_fraction_pooled": _ratio(
                    forecast_success + forecast_constant,
                    forecast_success
                    + forecast_constant
                    + forecast_fallback
                    + forecast_failed,
                ),
            }
        )
        aggregates.append(result)

    # Stability-first diagnostic order.  No accuracy/contribution/attack-damage
    # field participates in this ordering.
    def stability_key(group: Mapping[str, Any]) -> Tuple[float, float, float, float, str]:
        success_rate = _as_float(group.get("status_success_rate")) or 0.0
        forecast_coverage = _as_float(group.get("forecast_coverage_pooled")) or 0.0
        fit_usable = _as_float(group.get("fit_usable_fraction_pooled")) or 0.0
        reset_mean = _as_float(group.get("reset_count_mean")) or 0.0
        return (-success_rate, -forecast_coverage, -fit_usable, reset_mean, str(group.get("parameter_group_id")))

    aggregates.sort(key=stability_key)
    for rank, aggregate in enumerate(aggregates, 1):
        aggregate["diagnostic_priority_rank"] = rank
    return aggregates


def analyze(root: Path) -> Dict[str, Any]:
    locations = discover_tasks(root)
    tasks = [analyze_task(location) for location in locations]
    groups = aggregate_parameter_groups(tasks)
    return {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(root),
        "task_count": len(tasks),
        "parameter_group_count": len(groups),
        "selection_principle": (
            "diagnostic_priority_rank uses run success, forecast coverage, fit usability, "
            "and reset stability; accuracy/contribution gaps are descriptive only"
        ),
        "minmax_warning": (
            "Two-client MinMax contribution collapses to endpoint values and is degenerate. "
            "MinMax is never used as the primary tuning metric; use raw attacker contribution."
        ),
        "tasks": tasks,
        "parameter_groups": groups,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted({column for row in rows for column in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def write_outputs(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        output_dir / "task_summary.csv",
        output_dir / "parameter_group_summary.csv",
        output_dir / "tuning_summary.json",
    )
    collisions = [path for path in targets if path.exists()]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"refusing to overwrite existing analysis output: {joined}")
    _write_csv(targets[0], report["tasks"])
    _write_csv(targets[1], report["parameter_groups"])
    with targets[2].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_fixture_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="moirai_arima_analyzer_") as tmp:
        root = Path(tmp)
        campaign = root / ".matrix_runs" / "pilot"
        task_root = campaign / "experiments" / "task_000001_tictactoe_moirai_online"
        results = root / "Results" / "pilot" / task_root.name
        (task_root / "logs").mkdir(parents=True)
        results.mkdir(parents=True)
        manifest = "\n".join(
            [
                "task_number=1",
                f"task_label={task_root.name}",
                "dataset=tictactoe",
                "alpha=100",
                "client_num=2",
                "attacker_id=1",
                "pilot_profile_set=confirm7",
                "variant_kind=moirai_online",
                "profile_name=confirm_lr01",
                "attack_method=moirai_online",
                "trial_id=7",
                "scenario=fixed",
                "gap=2",
                "paired_probability=0.50",
                f"results_root={results}",
                "command_argv:",
                "  python",
                "  FLCE_updated.py",
                "  --moirai_tune_lr",
                "  0.05",
                "  --moirai_adapt_interval",
                "  2",
                "  --use_hessian",
                "  --lambda_h",
                "  0.9",
            ]
        )
        (task_root / "manifest.txt").write_text(manifest + "\n", encoding="utf-8")
        (task_root / "status.txt").write_text(
            "success protocol_pass runtime_seconds=10 rounds=50 "
            "honest=25 fake=25 hessian=50\n",
            encoding="utf-8",
        )
        (task_root / "logs" / "run.log").write_text(
            "\n".join(
                [
                    "[Moirai][OnlineTune][FALLBACK] round=9 mode=avg reason=warmup",
                    "[Moirai][OnlineTune] round=11 USING MOIRAI PREDICTION",
                    "[Moirai][OnlineTune][OBSERVE] round=11 pred_round=11 alignment=aligned mae=2.0e-2 ema=2.0e-2 residual_updated=1 residual_updates=1 decay_applied=0 residual_norm=1.0e-1",
                    "[Moirai][OnlineTune] round=13 USING MOIRAI PREDICTION",
                    "[Moirai][OnlineTune][OBSERVE] round=13 pred_round=13 alignment=aligned mae=1.0e-2 ema=1.9e-2 residual_updated=1 residual_updates=2 decay_applied=1 residual_norm=2.0e-1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        common = [
            "Dataset",
            "AttackMethod",
            "Alpha",
            "ClientNum",
            "AttackerID",
            "TrialID",
        ]
        _write_fixture_csv(
            results / "Accuracy_quantitative_shapley" / "global_accuracy_log_trial7.csv",
            common + ["Round", "WithAttackerAccuracy", "WithoutAttackerAccuracy"],
            [
                ["tictactoe", "moirai_online", 100, 2, 1, 7, 0, 0.80, 0.82],
                ["tictactoe", "moirai_online", 100, 2, 1, 7, 1, 0.84, 0.83],
            ],
        )
        _write_fixture_csv(
            results / "Detecting_quantitative_shapley" / "combined_contributions_trial7.csv",
            common + ["ClientID", "WithAttacker", "WithAttackerMinMax"],
            [
                ["tictactoe", "moirai_online", 100, 2, 1, 7, 0, 0.1, 0.0],
                ["tictactoe", "moirai_online", 100, 2, 1, 7, 1, 0.3, 1.0],
            ],
        )
        _write_fixture_csv(
            results / "Time_quantitative_shapley" / "attacking_time_log_trial7.csv",
            common + ["WithAttackerSec", "WithoutAttackerSec"],
            [["tictactoe", "moirai_online", 100, 2, 1, 7, 1.25, 0.75]],
        )

        # A second method/profile in the same campaign verifies mixed-family
        # discovery and, critically, that a process exit of zero does not turn
        # a failed protocol audit into a successful task.
        failed_task_root = (
            campaign / "experiments" / "task_000002_mnist_arima_adaptive"
        )
        failed_results = root / "Results" / "pilot" / failed_task_root.name
        (failed_task_root / "logs").mkdir(parents=True)
        failed_results.mkdir(parents=True)
        failed_manifest = "\n".join(
            [
                "task_number=2",
                f"task_label={failed_task_root.name}",
                "pilot_profile_set=confirm7",
                "dataset=mnist",
                "alpha=100000",
                "client_num=4",
                "attacker_id=1",
                "variant_kind=arima_adaptive",
                "profile_name=confirm_i10",
                "attack_method=arima_adaptive_mnist_imdb_layerwise",
                "trial_id=0",
                "scenario=fixed",
                "gap=2",
                "paired_probability=0.50",
                f"results_root={failed_results}",
                "command_argv:",
                "  python",
                "  FLCE_updated.py",
                "  --arima_retrain_interval",
                "  10",
                "  --use_hessian",
            ]
        )
        (failed_task_root / "manifest.txt").write_text(
            failed_manifest + "\n", encoding="utf-8"
        )
        (failed_task_root / "status.txt").write_text(
            "success protocol_fail runtime_seconds=5 rounds=49 "
            "honest=25 fake=24 hessian=49\n",
            encoding="utf-8",
        )
        (failed_task_root / "logs" / "run.log").write_text(
            "[ARIMA][LAYERWISE][ADAPT][FORECAST] round=11 "
            "success=8 constant=0 fallback=0 failed=0\n",
            encoding="utf-8",
        )

        report = analyze(campaign)
        assert report["task_count"] == 2
        tasks_by_profile = {task["profile_name"]: task for task in report["tasks"]}
        task = tasks_by_profile["confirm_lr01"]
        assert task["pilot_profile_set"] == "confirm7"
        assert task["variant_kind"] == "moirai_online"
        assert task["status_success"] is True
        assert task["status_protocol_pass"] is True
        assert task["status_runtime_seconds"] == 10.0
        assert task["status_rounds"] == 50
        assert task["status_honest"] == 25
        assert task["status_fake"] == 25
        assert task["status_hessian"] == 50
        assert task["forecast_event_count"] == 2
        assert task["forecast_first_round"] == 11
        assert task["forecast_last_round"] == 13
        assert task["fallback_event_count"] == 1
        assert task["aligned_count"] == 2
        assert task["residual_update_count"] == 2
        assert task["residual_decay_count"] == 1
        assert math.isclose(task["residual_norm_last"], 0.2)
        assert math.isclose(task["residual_norm_max"], 0.2)
        assert math.isclose(task["mae_first"], 0.02)
        assert math.isclose(task["mae_last"], 0.01)
        assert math.isclose(task["runtime_total_sec"], 2.0)
        assert math.isclose(task["attacker_contribution_raw"], 0.3)
        assert task["two_client_minmax_degenerate"] is True
        assert task["param_moirai_tune_lr"] == "0.05"
        failed_task = tasks_by_profile["confirm_i10"]
        assert failed_task["variant_kind"] == "arima_adaptive"
        assert failed_task["status"] == "protocol_fail"
        assert failed_task["status_process_success"] is True
        assert failed_task["status_protocol_pass"] is False
        assert failed_task["status_success"] is False
        assert failed_task["status_rounds"] == 49
        assert failed_task["forecast_first_round"] == 11
        successful_group = next(
            group
            for group in report["parameter_groups"]
            if group["method"] == "moirai_online"
        )
        assert successful_group["diagnostic_priority_rank"] == 1

        # A Results campaign root must resolve back to the same isolated run
        # root so status/log diagnostics are not lost.
        results_only_report = analyze(root / "Results" / "pilot")
        assert results_only_report["task_count"] == 1
        assert results_only_report["tasks"][0]["status"] == "success"
        assert results_only_report["tasks"][0]["residual_update_count"] == 2

        arima_log = _log_metrics(
            "\n".join(
                [
                    "[ARIMA][ADAPT][FIT] reason=start generation=1 success=7 constant=2 failed=1",
                    "[ARIMA][ADAPT][FORECAST] round=10 success=7 constant=2 fallback=1 failed=1",
                    "[ARIMA][ADAPT][OBSERVE] round=10 pred_round=10 alignment=aligned mae=3e-2 ema=3e-2 periodic_refit=1",
                ]
            )
        )
        assert arima_log["fit_success_count"] == 7
        assert arima_log["fit_constant_count"] == 2
        assert arima_log["fit_failed_count"] == 1
        assert arima_log["forecast_fallback_count"] == 1
        assert math.isclose(arima_log["forecast_usable_fraction"], 9.0 / 11.0)
        assert arima_log["periodic_refit_count"] == 1

        output = root / "analysis"
        write_outputs(output, report)
        assert (output / "task_summary.csv").is_file()
        assert (output / "parameter_group_summary.csv").is_file()
        assert (output / "tuning_summary.json").is_file()
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        help="campaign root or isolated Results/campaign root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "explicit directory for task_summary.csv, parameter_group_summary.csv, "
            "and tuning_summary.json; omitted means stdout only"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a synthetic fixture test without reading an experiment tree",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.root is None:
        raise SystemExit("root is required unless --self-test is used")
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"input root is not a directory: {root}")
    report = analyze(root)
    if args.output_dir is not None:
        write_outputs(args.output_dir.expanduser().resolve(), report)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

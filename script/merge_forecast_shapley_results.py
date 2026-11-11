#!/usr/bin/env python3
"""Merge isolated forecast/Shapley task outputs into download-friendly CSVs.

The source Results tree is treated as read-only.  Only complete tasks are
included in the merged metric tables; incomplete or inconsistent tasks remain
visible in the audit files.  The tool uses only the Python standard library so
it can run inside the existing ``flce_attack`` environment without installing
extra packages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


DATASETS = ("tictactoe", "adult", "dota2", "mnist", "imdb")
CAMPAIGN_RE = re.compile(
    r"^forecast_(?P<dataset>tictactoe|adult|dota2|mnist|imdb)_"
    r"(?P<mode>pure|lazy)_shapley_.+$"
)

TASK_TAIL_TEMPLATE = (
    r"^task_(?P<task_id>\d+)_{dataset}_{mode}_(?P<variant>.+)_"
    r"a(?P<alpha>[^_]+)_n(?P<client_num>\d+)_"
    r"(?P<schedule>pure|fixed|probabilistic)_g(?P<gap>[^_]+)_"
    r"p(?P<probability>[^_]+)_trial(?P<trial_id>\d+)$"
)

COMMON_REQUIRED = {
    "Dataset",
    "AttackMethod",
    "Alpha",
    "ClientNum",
    "AttackerID",
    "TrialID",
    "UseAttack",
    "DetectingType",
    "DetectingMethod",
    "UseHessian",
    "UseLazy",
    "LazySchedule",
    "AttackProbability",
    "LazyHonestGap",
    "LazyHonestPhase",
    "LazyFakePolicy",
}

ARTIFACT_SPECS = {
    "accuracy": {
        "pattern": "Accuracy_quantitative_shapley/global_accuracy_log_trial*.csv",
        "required": COMMON_REQUIRED
        | {"Round", "WithAttackerAccuracy", "WithoutAttackerAccuracy"},
        "row_key": "Round",
        "output_dir": "accuracy",
        "output_file": "accuracy_all_tasks.csv",
    },
    "contribution": {
        "pattern": "Detecting_quantitative_shapley/combined_contributions_trial*.csv",
        "required": COMMON_REQUIRED
        | {
            "ClientID",
            "WithAttacker",
            "WithoutAttacker",
            "WithAttackerMinMax",
            "WithoutAttackerMinMax",
        },
        "row_key": "ClientID",
        "output_dir": "contribution",
        "output_file": "contribution_all_tasks.csv",
    },
    "time": {
        "pattern": "Time_quantitative_shapley/attacking_time_log_trial*.csv",
        "required": COMMON_REQUIRED | {"WithAttackerSec", "WithoutAttackerSec"},
        "row_key": None,
        "output_dir": "time",
        "output_file": "time_all_tasks.csv",
    },
}

IDENTITY_COLUMNS = (
    "Dataset",
    "AttackMethod",
    "Alpha",
    "ClientNum",
    "AttackerID",
    "TrialID",
    "UseAttack",
    "DetectingType",
    "DetectingMethod",
    "UseHessian",
    "UseLazy",
    "LazySchedule",
    "AttackProbability",
    "LazyHonestGap",
    "LazyHonestPhase",
    "LazyFakePolicy",
)

METHOD_VARIANTS = {
    "arima": ("arima", "offline", False, "arima_offline"),
    "arima_adaptive": ("arima", "adaptive", False, "arima_adaptive"),
    "moirai": ("moirai", "offline", False, "moirai_offline"),
    "moirai_online": ("moirai", "online", False, "moirai_online"),
    "arima_mnist_imdb_layerwise": (
        "arima",
        "offline",
        True,
        "arima_offline",
    ),
    "arima_adaptive_mnist_imdb_layerwise": (
        "arima",
        "adaptive",
        True,
        "arima_adaptive",
    ),
    "moirai_mnist_imdb_layerwise": (
        "moirai",
        "offline",
        True,
        "moirai_offline",
    ),
    "moirai_online_mnist_imdb_layerwise": (
        "moirai",
        "online",
        True,
        "moirai_online",
    ),
}

SOURCE_COLUMNS = (
    "SourceTaskUID",
    "SourceCampaign",
    "SourceMode",
    "SourceTask",
    "SourceTaskID",
    "SourceVariant",
    "ConfiguredProbability",
    "LauncherStatus",
    "MethodFamily",
    "MethodMode",
    "Layerwise",
    "LogicalTaskKey",
    "SourceFile",
)

INVENTORY_CONFIG_KEYS = (
    "dataset",
    "alpha",
    "client_num",
    "attacker_id",
    "attack_method",
    "trial_id",
    "num_epoch",
    "num_local_epochs",
    "random_round",
    "use_attack",
    "use_hessian",
    "lambda_h",
    "use_lazy",
    "lazy_schedule",
    "attack_probability",
    "lazy_honest_gap",
    "lazy_honest_phase",
    "lazy_fake_policy",
    "arima_var_window",
    "arima_var_threshold",
    "arima_err_threshold",
    "arima_min_rounds_after_start",
    "arima_retrain_interval",
    "arima_err_ema_beta",
    "arima_trim_history",
    "moirai_online_tuning",
    "moirai_tune_lr",
    "moirai_tune_steps",
    "moirai_tune_start_round",
    "moirai_adapt_interval",
    "moirai_residual_decay",
    "moirai_residual_clip",
    "moirai_err_ema_beta",
    "moirai_err_threshold",
    "moirai_min_rounds_after_start",
    "moirai_trim_history",
    "layerwise_selection_metric",
    "layerwise_selection_fraction",
    "layerwise_selection_top_n",
    "layerwise_pca_threshold",
    "layerwise_pca_components",
)

LOGICAL_KEY_FIELDS = (
    "dataset",
    "alpha",
    "client_num",
    "attacker_id",
    "attack_method",
    "trial_id",
    "use_attack",
    "use_hessian",
    "lambda_h",
    "use_lazy",
    "lazy_schedule",
    "lazy_honest_gap",
    "lazy_honest_phase",
    "lazy_fake_policy",
    "random_round",
    "num_epoch",
    "num_local_epochs",
    "arima_var_window",
    "arima_var_threshold",
    "arima_err_threshold",
    "arima_min_rounds_after_start",
    "arima_retrain_interval",
    "arima_err_ema_beta",
    "arima_trim_history",
    "moirai_online_tuning",
    "moirai_tune_lr",
    "moirai_tune_steps",
    "moirai_tune_start_round",
    "moirai_adapt_interval",
    "moirai_residual_decay",
    "moirai_residual_clip",
    "moirai_err_ema_beta",
    "moirai_err_threshold",
    "moirai_min_rounds_after_start",
    "moirai_trim_history",
    "layerwise_selection_metric",
    "layerwise_selection_fraction",
    "layerwise_selection_top_n",
    "layerwise_pca_threshold",
    "layerwise_pca_components",
)


@dataclass
class ArtifactInfo:
    kind: str
    path: Optional[Path] = None
    header: List[str] = field(default_factory=list)
    row_count: int = 0
    first_row: Dict[str, str] = field(default_factory=dict)
    errors: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class TaskRecord:
    campaign: str
    dataset: str
    mode: str
    task_dir: Path
    task_label: str
    task_id: str = ""
    variant: str = ""
    configured_probability: str = ""
    launcher_status: str = "unavailable"
    launcher_manifest: Dict[str, str] = field(default_factory=dict)
    manifest_path: Optional[Path] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    cli_args: Dict[str, Any] = field(default_factory=dict)
    attack_parameters: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, ArtifactInfo] = field(default_factory=dict)
    errors: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[Tuple[str, str]] = field(default_factory=list)
    logical_task_key: str = ""
    duplicate_logical_count: int = 1

    @property
    def uid(self) -> str:
        return f"{self.campaign}/{self.task_label}"

    @property
    def complete(self) -> bool:
        return not self.errors and all(
            kind in self.artifacts and not self.artifacts[kind].errors
            for kind in ARTIFACT_SPECS
        )


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _normalized(value: Any) -> str:
    text = _display(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered
    if lowered in {"none", "null", "n/a", "na", ""}:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text.lower()
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "-0.0"} else normalized


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _decode_probability(value: str) -> str:
    if value.lower() in {"na", "n/a", ""}:
        return ""
    return value.replace("p", ".")


def _relative(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_launcher_manifest(path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw_line.strip() == "command_argv:":
                break
            if "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            parsed[key.strip()] = value.strip()
    except OSError:
        return {}
    return parsed


def _task_name_fallback(task: TaskRecord) -> Dict[str, str]:
    pattern = re.compile(
        TASK_TAIL_TEMPLATE.format(
            dataset=re.escape(task.dataset),
            mode=re.escape(task.mode),
        )
    )
    match = pattern.match(task.task_label)
    return match.groupdict() if match else {}


def _inspect_csv(
    path: Path,
    kind: str,
    settle_seconds: int,
) -> ArtifactInfo:
    info = ArtifactInfo(kind=kind, path=path)
    spec = ARTIFACT_SPECS[kind]
    try:
        stat = path.stat()
        if settle_seconds > 0 and time.time() - stat.st_mtime < settle_seconds:
            info.errors.append(
                (
                    "file_not_settled",
                    f"modified less than {settle_seconds}s ago: {path.name}",
                )
            )
            return info
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                info.errors.append(("empty_csv", "CSV has no header"))
                return info
            info.header = [column.strip() for column in header]
            if any(not column for column in info.header):
                info.errors.append(("empty_header_name", "CSV contains an empty header"))
            duplicate_headers = sorted(
                column for column, count in Counter(info.header).items() if count > 1
            )
            if duplicate_headers:
                info.errors.append(
                    ("duplicate_headers", ",".join(duplicate_headers))
                )
            missing = sorted(set(spec["required"]) - set(info.header))
            if missing:
                info.errors.append(("missing_headers", ",".join(missing)))
            if info.errors:
                return info

            constants: Dict[str, str] = {}
            row_keys: List[str] = []
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(info.header):
                    info.errors.append(
                        (
                            "malformed_row",
                            f"line {line_number}: expected {len(info.header)} fields, got {len(values)}",
                        )
                    )
                    continue
                row = dict(zip(info.header, values))
                if not info.first_row:
                    info.first_row = row
                    constants = {column: row.get(column, "") for column in IDENTITY_COLUMNS}
                else:
                    for column, expected in constants.items():
                        if row.get(column, "") != expected:
                            info.errors.append(
                                (
                                    "identity_changed_within_csv",
                                    f"line {line_number}: {column}",
                                )
                            )
                row_key = spec["row_key"]
                if row_key is not None:
                    row_keys.append(row.get(str(row_key), ""))
                info.row_count += 1

            if len(row_keys) != len(set(row_keys)):
                info.errors.append(("duplicate_row_keys", str(spec["row_key"])))
    except (OSError, csv.Error, UnicodeError) as exc:
        info.errors.append(("csv_read_error", str(exc)))
    return info


def _method_metadata(method: str) -> Tuple[str, str, bool, str]:
    return METHOD_VARIANTS.get(method, ("unknown", "unknown", False, "unknown"))


def _value_from_task(task: TaskRecord, cli_key: str) -> Any:
    if cli_key in task.cli_args:
        return task.cli_args[cli_key]
    aliases = {
        "dataset": "Dataset",
        "alpha": "Alpha",
        "client_num": "ClientNum",
        "attacker_id": "AttackerID",
        "attack_method": "AttackMethod",
        "trial_id": "TrialID",
        "use_attack": "UseAttack",
        "use_hessian": "UseHessian",
        "use_lazy": "UseLazy",
        "lazy_schedule": "LazySchedule",
        "attack_probability": "AttackProbability",
        "lazy_honest_gap": "LazyHonestGap",
        "lazy_honest_phase": "LazyHonestPhase",
        "lazy_fake_policy": "LazyFakePolicy",
    }
    csv_column = aliases.get(cli_key)
    if csv_column:
        for kind in ("accuracy", "contribution", "time"):
            artifact = task.artifacts.get(kind)
            if artifact and artifact.first_row.get(csv_column, "") != "":
                return artifact.first_row[csv_column]
    return ""


def _csv_identity_value(task: TaskRecord, column: str) -> str:
    for kind in ("accuracy", "contribution", "time"):
        artifact = task.artifacts.get(kind)
        if artifact and artifact.first_row and column in artifact.first_row:
            return artifact.first_row[column]
    return ""


def _check_source_values(
    task: TaskRecord,
    field_name: str,
    sources: Mapping[str, Any],
) -> None:
    normalized = {
        source: _normalized(value)
        for source, value in sources.items()
        if _normalized(value) != ""
    }
    if len(set(normalized.values())) <= 1:
        return
    rendered = ", ".join(
        f"{source}={value!r}" for source, value in sorted(normalized.items())
    )
    task.errors.append(("source_identity_mismatch", f"{field_name}: {rendered}"))


def _cross_check_task_sources(
    task: TaskRecord,
    fallback: Mapping[str, str],
) -> None:
    cli_to_csv = {
        "dataset": "Dataset",
        "alpha": "Alpha",
        "client_num": "ClientNum",
        "attacker_id": "AttackerID",
        "attack_method": "AttackMethod",
        "trial_id": "TrialID",
        "use_attack": "UseAttack",
        "use_hessian": "UseHessian",
        "use_lazy": "UseLazy",
        "lazy_schedule": "LazySchedule",
        "attack_probability": "AttackProbability",
        "lazy_honest_gap": "LazyHonestGap",
        "lazy_honest_phase": "LazyHonestPhase",
        "lazy_fake_policy": "LazyFakePolicy",
    }
    for cli_key, csv_column in cli_to_csv.items():
        _check_source_values(
            task,
            cli_key,
            {
                "provenance_cli": task.cli_args.get(cli_key, ""),
                "csv": _csv_identity_value(task, csv_column),
            },
        )

    _check_source_values(
        task,
        "dataset",
        {
            "campaign": task.dataset,
            "provenance_cli": task.cli_args.get("dataset", ""),
            "csv": _csv_identity_value(task, "Dataset"),
        },
    )
    _check_source_values(
        task,
        "alpha",
        {
            "task_label": fallback.get("alpha", ""),
            "provenance_cli": task.cli_args.get("alpha", ""),
            "csv": _csv_identity_value(task, "Alpha"),
        },
    )
    _check_source_values(
        task,
        "client_num",
        {
            "task_label": fallback.get("client_num", ""),
            "provenance_cli": task.cli_args.get("client_num", ""),
            "csv": _csv_identity_value(task, "ClientNum"),
        },
    )
    _check_source_values(
        task,
        "attack_method",
        {
            "launcher_manifest": task.launcher_manifest.get("attack_method", ""),
            "provenance_cli": task.cli_args.get("attack_method", ""),
            "csv": _csv_identity_value(task, "AttackMethod"),
        },
    )
    _check_source_values(
        task,
        "trial_id",
        {
            "task_label": fallback.get("trial_id", ""),
            "launcher_manifest": task.launcher_manifest.get("trial", ""),
            "provenance_cli": task.cli_args.get("trial_id", ""),
            "csv": _csv_identity_value(task, "TrialID"),
        },
    )
    expected_lazy = task.mode == "lazy"
    _check_source_values(
        task,
        "use_lazy",
        {
            "campaign": expected_lazy,
            "provenance_cli": task.cli_args.get("use_lazy", ""),
            "csv": _csv_identity_value(task, "UseLazy"),
        },
    )

    method = _display(task.cli_args.get("attack_method", "")) or _csv_identity_value(
        task, "AttackMethod"
    )
    _, _, _, derived_variant = _method_metadata(method)
    _check_source_values(
        task,
        "variant",
        {
            "task_label": fallback.get("variant", ""),
            "launcher_manifest": task.launcher_manifest.get("variant", ""),
            "derived_from_method": derived_variant if derived_variant != "unknown" else "",
        },
    )

    if task.mode == "pure":
        _check_source_values(
            task,
            "pure_schedule_label",
            {
                "task_label": fallback.get("schedule", ""),
                "launcher_manifest": task.launcher_manifest.get("schedule", ""),
            },
        )
        return

    _check_source_values(
        task,
        "lazy_schedule",
        {
            "task_label": fallback.get("schedule", ""),
            "launcher_manifest": task.launcher_manifest.get("schedule", ""),
            "provenance_cli": task.cli_args.get("lazy_schedule", ""),
            "csv": _csv_identity_value(task, "LazySchedule"),
        },
    )
    _check_source_values(
        task,
        "lazy_honest_gap",
        {
            "task_label": fallback.get("gap", ""),
            "launcher_manifest": task.launcher_manifest.get("gap", ""),
            "provenance_cli": task.cli_args.get("lazy_honest_gap", ""),
            "csv": _csv_identity_value(task, "LazyHonestGap"),
        },
    )
    label_probability = _decode_probability(fallback.get("probability", ""))
    launcher_probability = task.launcher_manifest.get("probability", "")
    _check_source_values(
        task,
        "configured_probability",
        {
            "task_label": label_probability,
            "launcher_manifest": launcher_probability,
            "merged_value": task.configured_probability,
        },
    )
    schedule = _normalized(task.cli_args.get("lazy_schedule", ""))
    if schedule == "probabilistic":
        _check_source_values(
            task,
            "probabilistic_attack_probability",
            {
                "configured_probability": task.configured_probability,
                "provenance_cli": task.cli_args.get("attack_probability", ""),
                "csv": _csv_identity_value(task, "AttackProbability"),
            },
        )


def _logical_key(task: TaskRecord) -> str:
    payload = {
        key: _normalized(_value_from_task(task, key)) for key in LOGICAL_KEY_FIELDS
    }
    payload["mode"] = task.mode
    payload["variant"] = task.variant
    payload["configured_probability"] = _normalized(task.configured_probability)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _load_task(
    results_root: Path,
    campaign_dir: Path,
    dataset: str,
    mode: str,
    task_dir: Path,
    runs_root: Optional[Path],
    settle_seconds: int,
) -> TaskRecord:
    task = TaskRecord(
        campaign=campaign_dir.name,
        dataset=dataset,
        mode=mode,
        task_dir=task_dir,
        task_label=task_dir.name,
    )

    launcher_task_dir: Optional[Path] = None
    launcher_campaign_available = False
    if runs_root is not None:
        launcher_campaign_dir = runs_root / campaign_dir.name
        launcher_campaign_available = launcher_campaign_dir.is_dir()
        candidate = launcher_campaign_dir / "tasks" / task_dir.name
        if candidate.is_dir():
            launcher_task_dir = candidate
            launcher_manifest_path = candidate / "manifest.txt"
            if launcher_manifest_path.is_file():
                task.launcher_manifest = _read_launcher_manifest(launcher_manifest_path)
            status_path = candidate / "status.txt"
            if status_path.is_file():
                task.launcher_status = status_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            else:
                task.errors.append(
                    ("launcher_status_missing", "matching launcher task has no status.txt")
                )
        elif launcher_campaign_available:
            task.errors.append(
                (
                    "launcher_task_missing",
                    "campaign exists in .shapley_runs but matching task directory is absent",
                )
            )

    fallback = _task_name_fallback(task)
    task.task_id = task.launcher_manifest.get("task_id", fallback.get("task_id", ""))
    task.variant = task.launcher_manifest.get("variant", fallback.get("variant", ""))
    task.configured_probability = task.launcher_manifest.get(
        "probability", _decode_probability(fallback.get("probability", ""))
    )
    if mode == "pure":
        task.configured_probability = ""

    manifest_paths = sorted(task_dir.glob("manifests/run_manifest_*.json"))
    if len(manifest_paths) != 1:
        task.errors.append(
            (
                "manifest_count",
                f"expected 1 provenance manifest, found {len(manifest_paths)}",
            )
        )
    else:
        task.manifest_path = manifest_paths[0]
        try:
            with manifest_paths[0].open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("manifest root is not an object")
            task.manifest = payload
            cli_args = payload.get("cli_args", {})
            attack_parameters = payload.get("attack_parameters", {})
            if not isinstance(cli_args, dict):
                task.errors.append(("manifest_cli_args", "cli_args is not an object"))
            else:
                task.cli_args = cli_args
            if isinstance(attack_parameters, dict):
                task.attack_parameters = attack_parameters
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            task.errors.append(("manifest_read_error", str(exc)))

    for kind, spec in ARTIFACT_SPECS.items():
        paths = sorted(task_dir.glob(str(spec["pattern"])))
        if len(paths) != 1:
            task.errors.append(
                (
                    f"{kind}_file_count",
                    f"expected 1 {kind} CSV, found {len(paths)}",
                )
            )
            task.artifacts[kind] = ArtifactInfo(kind=kind)
            continue
        task.artifacts[kind] = _inspect_csv(paths[0], kind, settle_seconds)

    client_num = _as_int(_value_from_task(task, "client_num"))
    num_epoch = _as_int(_value_from_task(task, "num_epoch")) or 50
    expected_rows = {
        "accuracy": num_epoch,
        "contribution": client_num,
        "time": 1,
    }
    for kind, expected in expected_rows.items():
        artifact = task.artifacts[kind]
        if expected is not None and artifact.row_count != expected:
            artifact.errors.append(
                (
                    "row_count_mismatch",
                    f"expected {expected}, found {artifact.row_count}",
                )
            )

    accuracy = task.artifacts["accuracy"]
    if not accuracy.errors:
        rounds: List[int] = []
        try:
            with accuracy.path.open("r", encoding="utf-8-sig", newline="") as handle:  # type: ignore[union-attr]
                for row in csv.DictReader(handle):
                    rounds.append(int(row["Round"]))
            if sorted(rounds) != list(range(num_epoch)):
                accuracy.errors.append(
                    ("round_coverage", f"expected rounds 0..{num_epoch - 1}")
                )
        except (OSError, ValueError, KeyError, csv.Error) as exc:
            accuracy.errors.append(("round_read_error", str(exc)))

    contribution = task.artifacts["contribution"]
    if client_num is not None and not contribution.errors:
        client_ids: List[int] = []
        try:
            with contribution.path.open("r", encoding="utf-8-sig", newline="") as handle:  # type: ignore[union-attr]
                for row in csv.DictReader(handle):
                    client_ids.append(int(row["ClientID"]))
            if sorted(client_ids) != list(range(client_num)):
                contribution.errors.append(
                    ("client_coverage", f"expected ClientID 0..{client_num - 1}")
                )
        except (OSError, ValueError, KeyError, csv.Error) as exc:
            contribution.errors.append(("client_read_error", str(exc)))

    first_rows = [
        artifact.first_row
        for artifact in task.artifacts.values()
        if artifact.first_row
    ]
    for column in IDENTITY_COLUMNS:
        values = {_normalized(row.get(column, "")) for row in first_rows}
        if len(values) > 1:
            task.errors.append(
                ("cross_artifact_identity_mismatch", f"{column}: {sorted(values)}")
            )

    if not fallback:
        task.errors.append(
            ("task_label_parse", "task directory name does not match launcher schema")
        )
    _cross_check_task_sources(task, fallback)

    method = _display(_value_from_task(task, "attack_method"))
    family, method_mode, layerwise, derived_variant = _method_metadata(method)
    if not task.variant:
        task.variant = derived_variant
    if method not in METHOD_VARIANTS:
        task.warnings.append(("unknown_method", method))
    if task.launcher_status == "unavailable":
        task.warnings.append(
            ("launcher_status_unavailable", "no matching .shapley_runs task status")
        )
    elif not task.launcher_status.startswith("success protocol_pass"):
        task.errors.append(("launcher_status_not_success", task.launcher_status))

    task.logical_task_key = _logical_key(task)
    return task


def discover_tasks(
    results_root: Path,
    runs_root: Optional[Path],
    datasets: Sequence[str],
    settle_seconds: int,
    require_full_matrix: bool,
) -> Tuple[List[TaskRecord], List[Tuple[str, str, str]]]:
    tasks: List[TaskRecord] = []
    global_issues: List[Tuple[str, str, str]] = []
    selected = set(datasets)
    discovered_modes: set[Tuple[str, str]] = set()
    for campaign_dir in sorted(results_root.iterdir(), key=lambda path: path.name):
        if not campaign_dir.is_dir():
            continue
        match = CAMPAIGN_RE.match(campaign_dir.name)
        if not match:
            continue
        dataset = match.group("dataset")
        mode = match.group("mode")
        if dataset not in selected:
            continue
        discovered_modes.add((dataset, mode))
        task_dirs = sorted(
            (
                child
                for child in campaign_dir.iterdir()
                if child.is_dir() and child.name.startswith("task_")
            ),
            key=lambda path: path.name,
        )
        if not task_dirs:
            global_issues.append(
                (campaign_dir.name, "campaign_has_no_tasks", "no task_* directories")
            )
        if runs_root is not None:
            launcher_campaign_dir = runs_root / campaign_dir.name
            if launcher_campaign_dir.is_dir():
                tasks_file = launcher_campaign_dir / "tasks.tsv"
                if tasks_file.is_file():
                    try:
                        planned_count = sum(
                            bool(line.strip())
                            for line in tasks_file.read_text(
                                encoding="utf-8", errors="replace"
                            ).splitlines()
                        )
                        if planned_count != len(task_dirs):
                            global_issues.append(
                                (
                                    campaign_dir.name,
                                    "campaign_task_count_mismatch",
                                    f"tasks.tsv planned {planned_count}, Results contains {len(task_dirs)} task directories",
                                )
                            )
                    except OSError as exc:
                        global_issues.append(
                            (campaign_dir.name, "tasks_plan_read_error", str(exc))
                        )
                else:
                    global_issues.append(
                        (
                            campaign_dir.name,
                            "tasks_plan_missing",
                            "matching .shapley_runs campaign has no tasks.tsv",
                        )
                    )
                summary_file = launcher_campaign_dir / "launcher_summary.txt"
                if summary_file.is_file():
                    launcher_summary = _read_launcher_manifest(summary_file)
                    planned_from_summary = _as_int(launcher_summary.get("total_tasks"))
                    if (
                        planned_from_summary is not None
                        and planned_from_summary != len(task_dirs)
                    ):
                        global_issues.append(
                            (
                                campaign_dir.name,
                                "launcher_summary_task_count_mismatch",
                                f"launcher summary planned {planned_from_summary}, Results contains {len(task_dirs)} task directories",
                            )
                        )
            else:
                global_issues.append(
                    (
                        campaign_dir.name,
                        "launcher_campaign_missing",
                        "Results campaign has no matching .shapley_runs campaign",
                    )
                )
        for task_dir in task_dirs:
            tasks.append(
                _load_task(
                    results_root,
                    campaign_dir,
                    dataset,
                    mode,
                    task_dir,
                    runs_root,
                    settle_seconds,
                )
            )

    if require_full_matrix:
        for dataset in sorted(selected):
            for mode in ("pure", "lazy"):
                if (dataset, mode) not in discovered_modes:
                    global_issues.append(
                        (
                            dataset,
                            "dataset_mode_missing",
                            f"no forecast_{dataset}_{mode}_shapley_* campaign found",
                        )
                    )

    duplicate_groups: Dict[str, List[TaskRecord]] = defaultdict(list)
    for task in tasks:
        duplicate_groups[task.logical_task_key].append(task)
    for group in duplicate_groups.values():
        count = len(group)
        for task in group:
            task.duplicate_logical_count = count
            if count > 1:
                task.warnings.append(
                    (
                        "duplicate_logical_task",
                        f"{count} physical tasks share logical key {task.logical_task_key}",
                    )
                )
    return tasks, global_issues


def _source_metadata(task: TaskRecord, kind: str) -> Dict[str, str]:
    method = _display(_value_from_task(task, "attack_method"))
    family, method_mode, layerwise, _ = _method_metadata(method)
    artifact = task.artifacts[kind]
    return {
        "SourceTaskUID": task.uid,
        "SourceCampaign": task.campaign,
        "SourceMode": task.mode,
        "SourceTask": task.task_label,
        "SourceTaskID": task.task_id,
        "SourceVariant": task.variant,
        "ConfiguredProbability": task.configured_probability,
        "LauncherStatus": task.launcher_status,
        "MethodFamily": family,
        "MethodMode": method_mode,
        "Layerwise": "true" if layerwise else "false",
        "LogicalTaskKey": task.logical_task_key,
        "SourceFile": artifact.path.name if artifact.path else "",
    }


def _iter_artifact_rows(artifact: ArtifactInfo) -> Iterator[Dict[str, str]]:
    if artifact.path is None:
        return
    with artifact.path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _union_headers(tasks: Sequence[TaskRecord], dataset: str, kind: str) -> List[str]:
    headers: List[str] = []
    seen = set()
    for task in tasks:
        if task.dataset != dataset or not task.complete:
            continue
        for column in task.artifacts[kind].header:
            if column not in seen:
                seen.add(column)
                headers.append(column)
    return list(SOURCE_COLUMNS) + headers


def _inventory_row(task: TaskRecord, results_root: Path) -> Dict[str, str]:
    method = _display(_value_from_task(task, "attack_method"))
    family, method_mode, layerwise, _ = _method_metadata(method)
    row: Dict[str, str] = {
        "SourceTaskUID": task.uid,
        "Campaign": task.campaign,
        "Mode": task.mode,
        "TaskLabel": task.task_label,
        "TaskID": task.task_id,
        "Variant": task.variant,
        "ConfiguredProbability": task.configured_probability,
        "LauncherStatus": task.launcher_status,
        "MethodFamily": family,
        "MethodMode": method_mode,
        "Layerwise": "true" if layerwise else "false",
        "LogicalTaskKey": task.logical_task_key,
        "DuplicateLogicalCount": str(task.duplicate_logical_count),
        "MergeStatus": "merged" if task.complete else "skipped_incomplete",
        "ErrorCount": str(len(task.errors) + sum(len(a.errors) for a in task.artifacts.values())),
        "WarningCount": str(len(task.warnings)),
        "ManifestFile": _relative(task.manifest_path, results_root),
        "CreatedAtUTC": _display(task.manifest.get("created_at_utc")),
        "EntrySHA256": _display(task.manifest.get("entry_sha256")),
        "AttackModuleSHA256": _display(task.manifest.get("attack_module_sha256")),
        "AccuracyRows": str(task.artifacts.get("accuracy", ArtifactInfo("accuracy")).row_count),
        "ContributionRows": str(task.artifacts.get("contribution", ArtifactInfo("contribution")).row_count),
        "TimeRows": str(task.artifacts.get("time", ArtifactInfo("time")).row_count),
    }
    for key in INVENTORY_CONFIG_KEYS:
        row[key] = _display(_value_from_task(task, key))
    row["CliArgsJSON"] = json.dumps(
        task.cli_args, sort_keys=True, separators=(",", ":"), default=str
    )
    row["AttackParametersJSON"] = json.dumps(
        task.attack_parameters, sort_keys=True, separators=(",", ":"), default=str
    )
    return row


def _issue_rows(
    tasks: Sequence[TaskRecord],
    global_issues: Sequence[Tuple[str, str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for campaign, code, message in global_issues:
        rows.append(
            {
                "Severity": "error",
                "SourceTaskUID": campaign,
                "Code": code,
                "Message": message,
            }
        )
    for task in tasks:
        for code, message in task.errors:
            rows.append(
                {
                    "Severity": "error",
                    "SourceTaskUID": task.uid,
                    "Code": code,
                    "Message": message,
                }
            )
        for artifact in task.artifacts.values():
            for code, message in artifact.errors:
                rows.append(
                    {
                        "Severity": "error",
                        "SourceTaskUID": task.uid,
                        "Code": f"{artifact.kind}:{code}",
                        "Message": message,
                    }
                )
        for code, message in task.warnings:
            rows.append(
                {
                    "Severity": "warning",
                    "SourceTaskUID": task.uid,
                    "Code": code,
                    "Message": message,
                }
            )
    return rows


def summarize(
    tasks: Sequence[TaskRecord],
    global_issues: Sequence[Tuple[str, str, str]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "campaign_count": len({task.campaign for task in tasks}),
        "task_count": len(tasks),
        "complete_task_count": sum(task.complete for task in tasks),
        "skipped_task_count": sum(not task.complete for task in tasks),
        "warning_count": sum(len(task.warnings) for task in tasks),
        "error_count": len(global_issues)
        + sum(len(task.errors) for task in tasks)
        + sum(len(artifact.errors) for task in tasks for artifact in task.artifacts.values()),
        "duplicate_logical_group_count": len(
            {
                task.logical_task_key
                for task in tasks
                if task.duplicate_logical_count > 1
            }
        ),
        "datasets": {},
        "reference_single_trial_complete_matrix": {
            "tasks_per_dataset": 576,
            "accuracy_rows_per_dataset": 28800,
            "contribution_rows_per_dataset": 2880,
            "time_rows_per_dataset": 576,
        },
    }
    for dataset in DATASETS:
        dataset_tasks = [task for task in tasks if task.dataset == dataset]
        complete = [task for task in dataset_tasks if task.complete]
        if not dataset_tasks:
            continue
        summary["datasets"][dataset] = {
            "campaigns": len({task.campaign for task in dataset_tasks}),
            "tasks": len(dataset_tasks),
            "complete_tasks": len(complete),
            "skipped_tasks": len(dataset_tasks) - len(complete),
            "pure_tasks": sum(task.mode == "pure" for task in complete),
            "lazy_tasks": sum(task.mode == "lazy" for task in complete),
            "accuracy_rows": sum(task.artifacts["accuracy"].row_count for task in complete),
            "contribution_rows": sum(
                task.artifacts["contribution"].row_count for task in complete
            ),
            "time_rows": sum(task.artifacts["time"].row_count for task in complete),
        }
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    print("=== Forecast Shapley merge validation ===")
    print(f"Campaigns: {summary['campaign_count']}")
    print(
        f"Tasks: {summary['task_count']} "
        f"(complete={summary['complete_task_count']}, skipped={summary['skipped_task_count']})"
    )
    print(
        f"Issues: errors={summary['error_count']}, warnings={summary['warning_count']}, "
        f"duplicate_groups={summary['duplicate_logical_group_count']}"
    )
    for dataset, values in summary["datasets"].items():
        print(
            f"{dataset}: tasks={values['tasks']} complete={values['complete_tasks']} "
            f"accuracy={values['accuracy_rows']} contribution={values['contribution_rows']} "
            f"time={values['time_rows']}"
        )


def _write_dict_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    staging_root: Path,
    results_root: Path,
    tasks: Sequence[TaskRecord],
    global_issues: Sequence[Tuple[str, str, str]],
    summary: Mapping[str, Any],
) -> None:
    complete_tasks = sorted(
        (task for task in tasks if task.complete),
        key=lambda task: (task.dataset, task.campaign, task.task_label),
    )
    for dataset in DATASETS:
        dataset_tasks = [task for task in complete_tasks if task.dataset == dataset]
        if not dataset_tasks:
            continue
        dataset_root = staging_root / dataset
        inventory_rows = [_inventory_row(task, results_root) for task in dataset_tasks]
        _write_dict_rows(dataset_root / "tasks.csv", inventory_rows)

        for kind, spec in ARTIFACT_SPECS.items():
            output_path = dataset_root / str(spec["output_dir"]) / str(spec["output_file"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = _union_headers(tasks, dataset, kind)
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for task in dataset_tasks:
                    metadata = _source_metadata(task, kind)
                    for source_row in _iter_artifact_rows(task.artifacts[kind]):
                        writer.writerow({**metadata, **source_row})

    audit_root = staging_root / "audit"
    all_inventory = [_inventory_row(task, results_root) for task in tasks]
    _write_dict_rows(audit_root / "task_inventory.csv", all_inventory)
    issues = _issue_rows(tasks, global_issues)
    if issues:
        _write_dict_rows(audit_root / "issues.csv", issues)
    else:
        _write_dict_rows(
            audit_root / "issues.csv",
            [{"Severity": "", "SourceTaskUID": "", "Code": "", "Message": ""}],
        )
    with (staging_root / "merge_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (staging_root / "README.txt").write_text(
        "Forecast/Shapley merged export\n"
        "===============================\n\n"
        "Each dataset contains one merged CSV under accuracy/, contribution/, and time/.\n"
        "tasks.csv stores one row per merged task, including tuning and provenance fields.\n"
        "audit/task_inventory.csv also lists incomplete tasks; audit/issues.csv explains them.\n\n"
        "Use SourceTaskUID as the task-level join key.  Accuracy rows are unique by\n"
        "SourceTaskUID+Round; contribution rows by SourceTaskUID+ClientID.\n\n"
        "Important: for fixed lazy runs, the original AttackProbability column is the CLI\n"
        "default (often 1.0), not the paired experimental probability.  Use the added\n"
        "ConfiguredProbability column for the intended 0.50/0.66/0.82/0.90 pairing.\n",
        encoding="utf-8",
    )


def _archive_directory(output_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"archive already exists: {archive_path}")
    temporary = archive_path.with_name(f".{archive_path.name}.tmp.{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for path in sorted(output_root.rglob("*"), key=lambda item: str(item)):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=(Path(output_root.name) / path.relative_to(output_root)).as_posix(),
                    )
        os.replace(temporary, archive_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_runs_root(results_root: Path, configured: Optional[Path]) -> Optional[Path]:
    if configured is not None:
        return configured.resolve() if configured.is_dir() else None
    candidate = results_root.parent / ".shapley_runs"
    return candidate.resolve() if candidate.is_dir() else None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge isolated forecast/Shapley task CSVs without modifying Results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Results directory containing forecast_*_shapley_* campaigns",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Optional .shapley_runs directory (auto-detected beside Results)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="New export directory; defaults beside Results with a UTC timestamp",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--settle-seconds",
        type=int,
        default=0,
        help="Reject source files modified within this many seconds",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Inspect and print counts without writing output",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also create OUTPUT_ROOT.zip for one-file download",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 for validation errors, skipped tasks, or duplicates",
    )
    parser.add_argument(
        "--allow-partial-matrix",
        action="store_true",
        help="Do not report missing pure/lazy campaigns for selected datasets",
    )
    args = parser.parse_args(argv)
    if args.settle_seconds < 0:
        parser.error("--settle-seconds must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        print(f"ERROR: Results root does not exist: {results_root}", file=sys.stderr)
        return 1
    if args.runs_root is not None and not args.runs_root.expanduser().is_dir():
        print(
            f"ERROR: explicitly configured runs root does not exist: {args.runs_root}",
            file=sys.stderr,
        )
        return 1
    runs_root = _resolve_runs_root(results_root, args.runs_root)
    tasks, global_issues = discover_tasks(
        results_root,
        runs_root,
        args.datasets,
        args.settle_seconds,
        not args.allow_partial_matrix,
    )
    summary = summarize(tasks, global_issues)
    print_summary(summary)
    if not tasks:
        print("ERROR: no matching forecast/Shapley tasks were found", file=sys.stderr)
        return 1
    if args.validate_only:
        return 2 if args.strict and (
            summary["error_count"]
            or summary["skipped_task_count"]
            or summary["duplicate_logical_group_count"]
        ) else 0

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (results_root.parent / f"merged_forecast_shapley_{timestamp}").resolve()
    )
    if output_root.exists():
        print(f"ERROR: output already exists: {output_root}", file=sys.stderr)
        return 1
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp.",
            dir=str(output_root.parent),
        )
    )
    try:
        _write_outputs(staging_root, results_root, tasks, global_issues, summary)
        os.replace(staging_root, output_root)
    except Exception as exc:
        shutil.rmtree(staging_root, ignore_errors=True)
        print(f"ERROR: export failed: {exc}", file=sys.stderr)
        return 1

    archive_path: Optional[Path] = None
    if args.archive:
        archive_path = output_root.with_suffix(".zip")
        try:
            _archive_directory(output_root, archive_path)
        except Exception as exc:
            print(f"ERROR: archive failed: {exc}", file=sys.stderr)
            return 1

    print(f"Merged export: {output_root}")
    if archive_path is not None:
        print(f"Download this file: {archive_path}")
    if args.strict and (
        summary["error_count"]
        or summary["skipped_task_count"]
        or summary["duplicate_logical_group_count"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

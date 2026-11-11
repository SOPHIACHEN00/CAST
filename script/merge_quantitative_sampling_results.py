#!/usr/bin/env python3
"""Merge sampled-quantitative task artifacts into category-level CSV files.

The launcher deliberately keeps each task's raw output private while it is
running.  This merger is the only writer of the public result directory:

    accuracy/accuracy.csv
    contribution/contribution.csv
    time/time.csv
    diagnostics/diagnostics.csv
    audit/tasks.csv

Successful tasks are validated before their rows are published.  A partial
campaign still produces an auditable public directory, but exits with status
4 so a scheduler can distinguish it from a complete campaign (status 0).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping


ACCURACY_FIELDS = [
    "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
    "UseAttack", "DetectingType", "DetectingMethod", "UseHessian", "UseLazy",
    "LazySchedule", "AttackProbability", "LazyHonestGap", "LazyHonestPhase",
    "LazyFakePolicy", "Round", "WithAttackerAccuracy", "WithoutAttackerAccuracy",
]
CONTRIBUTION_FIELDS = [
    "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
    "UseAttack", "DetectingType", "DetectingMethod", "UseHessian", "UseLazy",
    "LazySchedule", "AttackProbability", "LazyHonestGap", "LazyHonestPhase",
    "LazyFakePolicy", "ClientID", "WithAttacker", "WithoutAttacker",
    "WithAttackerMinMax", "WithoutAttackerMinMax",
]
TIME_FIELDS = [
    "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
    "UseAttack", "DetectingType", "DetectingMethod", "UseHessian", "UseLazy",
    "LazySchedule", "AttackProbability", "LazyHonestGap", "LazyHonestPhase",
    "LazyFakePolicy", "WithAttackerSec", "WithoutAttackerSec",
]
DIAGNOSTIC_FIELDS = [
    "Dataset", "AttackMethod", "AttackRegime", "Alpha", "ClientNum", "AttackerID",
    "TrialID", "DetectingMethod", "World", "ResolvedSeed", "RequestedSamplingBudget",
    "EffectiveSamplingBudget", "SamplingBudgetUnit", "Tolerance", "LeastCoreEpsilon",
    "ElapsedSeconds",
]
PLAN_FIELDS = [
    "TaskID", "Dataset", "TrialID", "Alpha", "ClientNum", "AttackerID",
    "AttackRegime", "AttackMethod", "LazyGap", "PlannedStatus", "Reason",
]
MANIFEST_REQUIRED_FIELDS = [
    "TaskID", "Dataset", "TrialID", "Alpha", "ClientNum", "AttackerID",
    "AttackRegime", "AttackMethod", "LazyGap", "GPU", "CPUThreadsPerWorker",
    "SamplingMethod", "CommandArgv", "ResultsRoot",
]
AUDIT_FIELDS = [
    "TaskID", "CampaignComplete", "Dataset", "TrialID", "Alpha", "ClientNum",
    "AttackerID", "AttackRegime", "AttackMethod", "LazyGap", "SamplingMethod",
    "GPU", "CPUThreadsPerWorker", "Status", "Reason", "CommandArgv", "ResultsRoot",
]

HESSIAN_ATTACKS = {
    "arima", "arima_adaptive", "moirai", "moirai_online",
    "arima_mnist_imdb_layerwise", "arima_adaptive_mnist_imdb_layerwise",
    "moirai_mnist_imdb_layerwise", "moirai_online_mnist_imdb_layerwise",
}


class MergeError(ValueError):
    """Raised when a campaign artifact violates the output contract."""


def read_table(path: Path, *, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise MergeError(f"missing artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise MergeError(f"empty header: {path}")
        rows = list(reader)
    return fields, rows


def read_exact(path: Path, expected_fields: list[str], *, delimiter: str = ",") -> list[dict[str, str]]:
    fields, rows = read_table(path, delimiter=delimiter)
    if fields != expected_fields:
        raise MergeError(
            f"malformed header in {path}: expected {expected_fields}, got {fields}"
        )
    return rows


def require_fields(fields: Iterable[str], required: Iterable[str], path: Path) -> None:
    available = set(fields)
    missing = [field for field in required if field not in available]
    if missing:
        raise MergeError(f"missing fields in {path}: {missing}")


def close_number(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left) == str(right)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_finite_fields(
    row: Mapping[str, str],
    fields: Iterable[str],
    *,
    path: Path,
    allow_na: bool = False,
) -> None:
    for field in fields:
        value = str(row.get(field, "")).strip()
        if allow_na and value.upper() in {"", "N/A", "NA", "NONE"}:
            continue
        try:
            number = float(value)
        except ValueError as error:
            raise MergeError(f"non-numeric {field} in {path}: {value}") from error
        if not math.isfinite(number):
            raise MergeError(f"non-finite {field} in {path}: {value}")


def expected_hessian(attack_method: str) -> bool:
    return attack_method in HESSIAN_ATTACKS


def validate_core_row(
    row: Mapping[str, str],
    plan: Mapping[str, str],
    sampling_method: str,
    *,
    path: Path,
) -> None:
    expected_pairs = {
        "Dataset": plan["Dataset"],
        "AttackMethod": plan["AttackMethod"],
        "ClientNum": plan["ClientNum"],
        "AttackerID": plan["AttackerID"],
        "TrialID": plan["TrialID"],
        "DetectingMethod": sampling_method,
        "DetectingType": "quantitative",
    }
    for field, expected in expected_pairs.items():
        if field not in row:
            raise MergeError(f"missing {field} in {path}")
        if field == "ClientNum" or field == "TrialID" or field == "AttackerID":
            if str(row[field]) != str(expected):
                raise MergeError(f"{field} mismatch in {path}: {row[field]} != {expected}")
        elif str(row[field]) != str(expected):
            raise MergeError(f"{field} mismatch in {path}: {row[field]} != {expected}")
    if not close_number(row.get("Alpha"), plan["Alpha"]):
        raise MergeError(f"Alpha mismatch in {path}: {row.get('Alpha')} != {plan['Alpha']}")
    if not is_true(row.get("UseAttack")):
        raise MergeError(f"UseAttack is not enabled in {path}")
    if is_true(row.get("UseLazy")) != (plan["AttackRegime"] == "lazy"):
        raise MergeError(f"UseLazy mismatch in {path}")
    if plan["AttackRegime"] == "lazy":
        for field, expected in {
            "LazySchedule": "fixed",
            "LazyHonestGap": plan["LazyGap"],
            "LazyHonestPhase": "0",
            "LazyFakePolicy": "attack",
        }.items():
            if str(row.get(field, "")) != expected:
                raise MergeError(
                    f"{field} mismatch in {path}: {row.get(field)} != {expected}"
                )
    expected_hessian_value = expected_hessian(plan["AttackMethod"])
    if "UseHessian" in row and is_true(row["UseHessian"]) != expected_hessian_value:
        raise MergeError(f"UseHessian mismatch in {path}")


def validate_diagnostic_row(
    row: Mapping[str, str],
    plan: Mapping[str, str],
    sampling_method: str,
    *,
    path: Path,
) -> None:
    world = str(row.get("World", ""))
    expected_client_num = (
        plan["ClientNum"]
        if world == "with_attacker"
        else str(int(plan["ClientNum"]) - 1)
    )
    expected = {
        "Dataset": plan["Dataset"],
        "AttackMethod": plan["AttackMethod"],
        "AttackRegime": plan["AttackRegime"],
        "ClientNum": expected_client_num,
        "AttackerID": plan["AttackerID"],
        "TrialID": plan["TrialID"],
        "DetectingMethod": sampling_method,
    }
    for field, value in expected.items():
        if str(row.get(field, "")) != str(value):
            raise MergeError(f"diagnostic {field} mismatch in {path}")
    if not close_number(row.get("Alpha"), plan["Alpha"]):
        raise MergeError(f"diagnostic Alpha mismatch in {path}")
    if world not in {"with_attacker", "without_attacker"}:
        raise MergeError(f"invalid diagnostic world in {path}: {row.get('World')}")


def validate_manifest(
    manifest: Mapping[str, str],
    plan: Mapping[str, str],
    sampling_method: str,
    *,
    task_dir: Path,
) -> None:
    for field in MANIFEST_REQUIRED_FIELDS:
        if field not in manifest:
            raise MergeError(f"missing manifest field {field} for {plan['TaskID']}")
    for field in (
        "TaskID", "Dataset", "TrialID", "Alpha", "ClientNum", "AttackerID",
        "AttackRegime", "AttackMethod", "LazyGap",
    ):
        left = manifest[field]
        right = plan[field]
        if field == "Alpha":
            if not close_number(left, right):
                raise MergeError(f"manifest {field} mismatch for {plan['TaskID']}")
        elif left != right:
            raise MergeError(f"manifest {field} mismatch for {plan['TaskID']}")
    if manifest["SamplingMethod"] != sampling_method:
        raise MergeError(f"manifest SamplingMethod mismatch for {plan['TaskID']}")
    if not manifest["CommandArgv"].strip():
        raise MergeError(f"empty CommandArgv for {plan['TaskID']}")
    if not manifest["ResultsRoot"].strip():
        raise MergeError(f"empty ResultsRoot for {plan['TaskID']}")
    # The path is written by the launcher.  Resolve only for a sanity check;
    # do not require a particular slash style because runs may be merged from
    # Windows or a POSIX worker.
    results_root = Path(manifest["ResultsRoot"])
    if results_root.name != "results":
        raise MergeError(f"manifest ResultsRoot is not the task results directory for {plan['TaskID']}")
    if task_dir.name not in results_root.parts and task_dir.name not in str(results_root).replace("\\", "/"):
        raise MergeError(f"manifest ResultsRoot is not private to {plan['TaskID']}")


def public_row(task_id: str, complete: str, row: Mapping[str, str], fields: list[str]) -> dict[str, str]:
    return {"TaskID": task_id, "CampaignComplete": complete, **{field: row[field] for field in fields}}


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def read_plan(campaign_root: Path) -> list[dict[str, str]]:
    fields, rows = read_table(campaign_root / "planned.tsv", delimiter="\t")
    if fields != PLAN_FIELDS:
        raise MergeError(f"malformed plan header: expected {PLAN_FIELDS}, got {fields}")
    if not rows:
        raise MergeError("empty campaign plan")
    task_ids = [row["TaskID"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise MergeError("duplicate TaskID in campaign plan")
    return rows


def read_status(task_dir: Path) -> tuple[str, str]:
    path = task_dir / "status.tsv"
    if not path.is_file():
        return "missing", "status_missing"
    pieces = path.read_text(encoding="utf-8").strip().split("\t")
    if not pieces or not pieces[0]:
        return "failed", "status_malformed"
    return pieces[0], pieces[2] if len(pieces) > 2 else ""


def collect_task(
    campaign_root: Path,
    plan: dict[str, str],
    sampling_method: str,
    epochs: int,
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    task_id = plan["TaskID"]
    task_dir = campaign_root / "tasks" / task_id
    status, reason = read_status(task_dir)
    manifest: dict[str, str] = {}
    manifest_path = task_dir / "manifest.tsv"
    if manifest_path.is_file():
        fields, manifest_rows = read_table(manifest_path, delimiter="\t")
        require_fields(fields, MANIFEST_REQUIRED_FIELDS, manifest_path)
        if len(manifest_rows) != 1:
            raise MergeError(f"manifest row count is not one for {task_id}")
        manifest = manifest_rows[0]
    if plan["PlannedStatus"] != "planned":
        return {
            "TaskID": task_id, "Status": plan["PlannedStatus"], "Reason": plan["Reason"],
            "Manifest": manifest,
        }, [], [], [], []
    if status != "success":
        return {
            "TaskID": task_id, "Status": status, "Reason": reason,
            "Manifest": manifest,
        }, [], [], [], []
    try:
        if not manifest:
            raise MergeError(f"manifest missing for {task_id}")
        validate_manifest(manifest, plan, sampling_method, task_dir=task_dir)
        results_root = task_dir / "results"
        accuracy_path = results_root / f"Accuracy_quantitative_{sampling_method}" / f"global_accuracy_log_trial{plan['TrialID']}.csv"
        contribution_root = results_root / f"Detecting_quantitative_{sampling_method}"
        contribution_path = contribution_root / f"combined_contributions_trial{plan['TrialID']}.csv"
        time_path = results_root / f"Time_quantitative_{sampling_method}" / f"attacking_time_log_trial{plan['TrialID']}.csv"
        diagnostics_path = contribution_root / f"quantitative_diagnostics_trial{plan['TrialID']}.csv"
        accuracy = read_exact(accuracy_path, ACCURACY_FIELDS)
        contribution = read_exact(contribution_path, CONTRIBUTION_FIELDS)
        timing = read_exact(time_path, TIME_FIELDS)
        diagnostics = read_exact(diagnostics_path, DIAGNOSTIC_FIELDS)
        if len(accuracy) != epochs:
            raise MergeError(f"accuracy row count for {task_id}: {len(accuracy)} != {epochs}")
        if len(contribution) != int(plan["ClientNum"]):
            raise MergeError(
                f"contribution row count for {task_id}: {len(contribution)} != {plan['ClientNum']}"
            )
        if len(timing) != 1:
            raise MergeError(f"time row count for {task_id}: {len(timing)} != 1")
        if not diagnostics:
            raise MergeError(f"diagnostic output is empty for {task_id}")
        diagnostic_worlds = {row["World"] for row in diagnostics}
        if diagnostic_worlds != {"with_attacker", "without_attacker"}:
            raise MergeError(
                f"diagnostic worlds for {task_id} are {sorted(diagnostic_worlds)}, "
                "expected with_attacker and without_attacker"
            )
        for row in accuracy + contribution + timing:
            validate_core_row(row, plan, sampling_method, path=accuracy_path)
        for row in accuracy:
            validate_finite_fields(
                row,
                ("WithAttackerAccuracy", "WithoutAttackerAccuracy"),
                path=accuracy_path,
            )
        for row in contribution:
            validate_finite_fields(
                row,
                ("WithAttacker", "WithoutAttacker", "WithAttackerMinMax", "WithoutAttackerMinMax"),
                path=contribution_path,
                allow_na=True,
            )
        for row in timing:
            validate_finite_fields(
                row,
                ("WithAttackerSec", "WithoutAttackerSec"),
                path=time_path,
            )
        for row in diagnostics:
            validate_diagnostic_row(row, plan, sampling_method, path=diagnostics_path)
            validate_finite_fields(
                row,
                ("ResolvedSeed", "EffectiveSamplingBudget", "ElapsedSeconds"),
                path=diagnostics_path,
            )
            validate_finite_fields(
                row,
                ("Tolerance", "LeastCoreEpsilon"),
                path=diagnostics_path,
                allow_na=True,
            )
        rounds = {int(row["Round"]) for row in accuracy}
        if rounds != set(range(epochs)):
            raise MergeError(f"accuracy rounds are not 0..{epochs - 1} for {task_id}")
        client_ids = {int(row["ClientID"]) for row in contribution}
        if client_ids != set(range(int(plan["ClientNum"]))):
            raise MergeError(f"contribution client IDs are invalid for {task_id}")
        return {
            "TaskID": task_id, "Status": "success", "Reason": "", "Manifest": manifest,
        }, accuracy, contribution, timing, diagnostics
    except Exception as error:  # noqa: BLE001 - convert artifact errors into audit status
        return {
            "TaskID": task_id, "Status": "failed", "Reason": f"protocol_invalid:{error}",
            "Manifest": manifest,
        }, [], [], [], []


def merge(campaign_root: Path, public_root: Path, dataset: str, sampling_method: str, epochs: int) -> int:
    if epochs < 1:
        raise MergeError("epochs must be positive")
    if sampling_method not in {"sampling_shapley", "sampling_least_core"}:
        raise MergeError(f"unsupported sampling method: {sampling_method}")
    if public_root.exists():
        raise MergeError(f"public result already exists: {public_root}")
    plans = read_plan(campaign_root)
    if any(plan["Dataset"] != dataset for plan in plans):
        raise MergeError("campaign dataset does not match merger argument")

    statuses: dict[str, dict[str, str]] = {}
    accuracy_rows: list[dict[str, str]] = []
    contribution_rows: list[dict[str, str]] = []
    time_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, str]] = []
    for plan in plans:
        result, accuracy, contribution, timing, diagnostics = collect_task(
            campaign_root, plan, sampling_method, epochs
        )
        statuses[plan["TaskID"]] = result
        for row in accuracy:
            accuracy_rows.append(public_row(plan["TaskID"], "0", row, ACCURACY_FIELDS))
        for row in contribution:
            contribution_rows.append(public_row(plan["TaskID"], "0", row, CONTRIBUTION_FIELDS))
        for row in timing:
            time_rows.append(public_row(plan["TaskID"], "0", row, TIME_FIELDS))
        for row in diagnostics:
            diagnostic_rows.append(public_row(plan["TaskID"], "0", row, DIAGNOSTIC_FIELDS))

    planned = [plan for plan in plans if plan["PlannedStatus"] == "planned"]
    complete = all(statuses[plan["TaskID"]]["Status"] == "success" for plan in planned)
    complete_flag = "1" if complete else "0"
    for rows in (accuracy_rows, contribution_rows, time_rows, diagnostic_rows):
        for row in rows:
            row["CampaignComplete"] = complete_flag

    audit_rows: list[dict[str, str]] = []
    for plan in plans:
        result = statuses[plan["TaskID"]]
        manifest = result.get("Manifest", {})
        audit_rows.append({
            "TaskID": plan["TaskID"],
            "CampaignComplete": complete_flag,
            "Dataset": plan["Dataset"],
            "TrialID": plan["TrialID"],
            "Alpha": plan["Alpha"],
            "ClientNum": plan["ClientNum"],
            "AttackerID": plan["AttackerID"],
            "AttackRegime": plan["AttackRegime"],
            "AttackMethod": plan["AttackMethod"],
            "LazyGap": plan["LazyGap"],
            "SamplingMethod": sampling_method,
            "GPU": manifest.get("GPU", "N/A"),
            "CPUThreadsPerWorker": manifest.get("CPUThreadsPerWorker", "N/A"),
            "Status": result["Status"],
            "Reason": result["Reason"],
            "CommandArgv": manifest.get("CommandArgv", "N/A"),
            "ResultsRoot": manifest.get("ResultsRoot", "N/A"),
        })

    accuracy_rows.sort(key=lambda row: (row["TaskID"], int(row["Round"])))
    contribution_rows.sort(key=lambda row: (row["TaskID"], int(row["ClientID"])))
    time_rows.sort(key=lambda row: row["TaskID"])
    diagnostic_rows.sort(key=lambda row: (row["TaskID"], row["World"]))
    audit_rows.sort(key=lambda row: row["TaskID"])

    stage = public_root.parent / f".{public_root.name}.staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        write_csv(stage / "accuracy" / "accuracy.csv", ["TaskID", "CampaignComplete"] + ACCURACY_FIELDS, accuracy_rows)
        write_csv(stage / "contribution" / "contribution.csv", ["TaskID", "CampaignComplete"] + CONTRIBUTION_FIELDS, contribution_rows)
        write_csv(stage / "time" / "time.csv", ["TaskID", "CampaignComplete"] + TIME_FIELDS, time_rows)
        write_csv(stage / "diagnostics" / "diagnostics.csv", ["TaskID", "CampaignComplete"] + DIAGNOSTIC_FIELDS, diagnostic_rows)
        write_csv(stage / "audit" / "tasks.csv", AUDIT_FIELDS, audit_rows)
        public_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, public_root)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    print(
        f"MERGED|complete={complete_flag}|accuracy={len(accuracy_rows)}|"
        f"contribution={len(contribution_rows)}|time={len(time_rows)}|"
        f"diagnostics={len(diagnostic_rows)}|audit={len(audit_rows)}"
    )
    return 0 if complete else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("public_root", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("sampling_method", choices=["sampling_shapley", "sampling_least_core"])
    parser.add_argument("epochs", type=int)
    args = parser.parse_args()
    try:
        return merge(args.campaign_root, args.public_root, args.dataset, args.sampling_method, args.epochs)
    except Exception as error:  # noqa: BLE001 - concise CLI failure contract
        print(f"MERGE_ERROR: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

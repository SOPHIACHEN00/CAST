"""Create one compact, task-level PASS summary CSV per dataset.

The original PASS export is intentionally detailed: one row per task, submitted
round, and client.  This script keeps one row per TaskID so accuracy, detection,
coverage, and timing can be analysed without treating repeated client/round rows
as independent experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def number(value):
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', 'yes'}:
            return 1
        if normalized in {'false', 'no'}:
            return 0
    value = number(value)
    return None if value is None else int(value)


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    return statistics.pstdev(values) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def json_object(value):
    if not value:
        return {}
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def task_summary(task_id, rows):
    first = rows[0]
    client_count = integer(first["ClientNum"])
    expected_rows = 50 * client_count
    if len(rows) != expected_rows:
        raise ValueError(f"{task_id}: expected {expected_rows} rows, found {len(rows)}")

    # Accuracy is repeated for each client.  Take one value per submitted round.
    accuracy_by_round = {}
    for row in rows:
        submitted = integer(row["SubmittedRound"])
        if submitted is not None and submitted not in accuracy_by_round:
            accuracy_by_round[submitted] = (
                number(row["WithAttackerAccuracy"]),
                number(row["WithoutAttackerAccuracy"]),
            )
    with_accuracy = [pair[0] for pair in accuracy_by_round.values() if pair[0] is not None]
    without_accuracy = [pair[1] for pair in accuracy_by_round.values() if pair[1] is not None]
    accuracy_delta = [a - b for a, b in accuracy_by_round.values() if a is not None and b is not None]

    valid_rows = [row for row in rows if row["DetectorScore"] != ""]
    attacker_rows = [row for row in valid_rows if row["IsDesignatedAttacker"] == "1"]
    nonattacker_rows = [row for row in valid_rows if row["IsDesignatedAttacker"] != "1"]
    active_attacker = [row for row in attacker_rows if row["GroundTruthActiveAttack"] == "1"]
    honest_attacker = [row for row in attacker_rows if row["GroundTruthActiveAttack"] != "1"]

    def predicted(row):
        return integer(row["RoundPredictedSuspicious"]) == 1

    attacker_tp = sum(predicted(row) for row in active_attacker)
    attacker_fn = len(active_attacker) - attacker_tp
    attacker_honest_alarms = sum(predicted(row) for row in honest_attacker)
    nonattacker_alarms = sum(predicted(row) for row in nonattacker_rows)
    all_alarms = sum(predicted(row) for row in valid_rows)
    below_threshold = sum(
        number(row["DetectorScore"]) < number(row["DetectorThreshold"])
        for row in valid_rows
        if number(row["DetectorScore"]) is not None and number(row["DetectorThreshold"]) is not None
    )

    active_scores = [number(row["DetectorScore"]) for row in active_attacker]
    active_scores = [value for value in active_scores if value is not None]
    honest_scores = [number(row["DetectorScore"]) for row in honest_attacker]
    honest_scores = [value for value in honest_scores if value is not None]
    ratios = []
    for row in attacker_rows:
        score = number(row["DetectorScore"])
        threshold = number(row["DetectorThreshold"])
        if score is not None and threshold not in (None, 0):
            ratios.append(score / threshold)

    report_sources = defaultdict(int)
    report_values = []
    for row in rows:
        for report in json_object(row["IncomingReportsJSON"]).get("reports", []):
            report_sources[report.get("source", "unknown")] += 1
            value = number(report.get("report"))
            if value is not None:
                report_values.append(value)
    # The current export stores IncomingReportsJSON as a JSON list, while older
    # exports may wrap it in {"reports": [...]}; support both forms.
    if not report_sources:
        for row in rows:
            value = row["IncomingReportsJSON"]
            if not value:
                continue
            try:
                reports = json.loads(value)
            except json.JSONDecodeError:
                reports = []
            if isinstance(reports, list):
                for report in reports:
                    if not isinstance(report, dict):
                        continue
                    report_sources[report.get("source", "unknown")] += 1
                    report_value = number(report.get("report"))
                    if report_value is not None:
                        report_values.append(report_value)

    evaluation = json_object(first["TaskEvaluationJSON"])
    designated = evaluation.get("designated_identity", {})
    active_eval = evaluation.get("active_submission", {})

    result = {
        "TaskID": task_id,
        "Dataset": first["Dataset"],
        "AttackMethod": first["AttackMethod"],
        "AttackRegime": first["AttackRegime"],
        "Alpha": first["Alpha"],
        "Distribution": first["Distribution"],
        "ClientNum": client_count,
        "AttackerID": integer(first["AttackerID"]),
        "TrialID": integer(first["TrialID"]),
        "UseHessian": integer(first["UseHessian"]),
        "UseLazy": integer(first["UseLazy"]),
        "LazySchedule": first["LazySchedule"],
        "LazyFakePolicy": first["LazyFakePolicy"],
        "LambdaH": first["LambdaH"],
        "LazyHonestGap": first["LazyHonestGap"],
        "PassAlpha": number(first["PassAlpha"]),
        "PassBeta": number(first["PassBeta"]),
        "PassC0": number(first["PassC0"]),
        "ValidSubmittedRounds": len({integer(row["SubmittedRound"]) for row in valid_rows}),
        "ValidDecisionRows": len(valid_rows),
        "UnauditedRows": sum(row["DetectionSkipped"] == "1" for row in rows),
        "TaskCoverage": number(first["TaskCoverage"]),
        "ControlCoverage": number(first["ControlCoverage"]),
        "WithAttackerAccuracyMean": mean(with_accuracy),
        "WithAttackerAccuracyStd": stdev(with_accuracy),
        "WithoutAttackerAccuracyMean": mean(without_accuracy),
        "WithoutAttackerAccuracyStd": stdev(without_accuracy),
        "AccuracyDeltaMean": mean(accuracy_delta),
        "AccuracyDeltaStd": stdev(accuracy_delta),
        "ActiveAttackerValidRounds": len(active_attacker),
        "ActiveAttackerTP": attacker_tp,
        "ActiveAttackerFN": attacker_fn,
        "AttackerTPR": ratio(attacker_tp, len(active_attacker)),
        "AttackerFNR": ratio(attacker_fn, len(active_attacker)),
        "HonestAttackerValidRounds": len(honest_attacker),
        "HonestAttackerAlarms": attacker_honest_alarms,
        "LazyHonestAlarmRate": ratio(attacker_honest_alarms, len(honest_attacker)),
        "NonAttackerValidRows": len(nonattacker_rows),
        "NonAttackerAlarms": nonattacker_alarms,
        "NonAttackerFPR": ratio(nonattacker_alarms, len(nonattacker_rows)),
        "AllValidRows": len(valid_rows),
        "AllAlarms": all_alarms,
        "ScoreBelowThresholdRows": below_threshold,
        "ActiveAttackerScoreMean": mean(active_scores),
        "ActiveAttackerScoreStd": stdev(active_scores),
        "HonestAttackerScoreMean": mean(honest_scores),
        "HonestAttackerScoreStd": stdev(honest_scores),
        "AttackerScoreThresholdRatioMean": mean(ratios),
        "ReportRandomUniformCount": report_sources.get("random_uniform", 0),
        "ReportRealLocalDataCount": report_sources.get("real_local_data", 0),
        "ReportNotAuditedCount": report_sources.get("not_audited", 0),
        "ReportMean": mean(report_values),
        "EvaluationRecall": number(designated.get("recall")),
        "EvaluationFPR": number(designated.get("fpr")),
        "EvaluationF1": number(designated.get("f1")),
        "EvaluationInvalid": integer(designated.get("invalid")),
        "EvaluationActiveTP": integer(active_eval.get("tp")),
        "EvaluationActiveFP": integer(active_eval.get("fp")),
        "TaskAttackGenerationSec": number(first["TaskAttackGenerationSec"]),
        "TaskDetectorWithAttackerSec": number(first["TaskDetectorWithAttackerSec"]),
        "TaskDetectorWithoutAttackerSec": number(first["TaskDetectorWithoutAttackerSec"]),
        "TaskFLWithAttackerSec": number(first["TaskFLWithAttackerSec"]),
        "TaskFLWithoutAttackerSec": number(first["TaskFLWithoutAttackerSec"]),
        "TaskWallSec": number(first["TaskWallSec"]),
    }
    return result


FIELDNAMES = list(task_summary.__annotations__) if task_summary.__annotations__ else []


def sort_key(row):
    try:
        alpha = float(row["Alpha"])
    except (TypeError, ValueError):
        alpha = 0
    gap = row["LazyHonestGap"]
    try:
        gap_value = int(gap)
    except (TypeError, ValueError):
        gap_value = -1
    return (row["AttackRegime"], row["AttackMethod"], alpha, int(row["ClientNum"]), gap_value)


def process(input_path: Path, output_path: Path):
    grouped = defaultdict(list)
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["TaskID"]].append(row)
    if not grouped:
        raise ValueError(f"No rows found in {input_path}")
    summaries = [task_summary(task_id, rows) for task_id, rows in grouped.items()]
    fields = list(summaries[0])
    summaries.sort(key=sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"{input_path.name}: {len(summaries)} tasks -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=("adult", "tictactoe"))
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    for dataset in args.datasets:
        process(args.input_dir / f"{dataset}_pass.csv", output_dir / f"{dataset}_pass_task_summary.csv")


if __name__ == "__main__":
    main()

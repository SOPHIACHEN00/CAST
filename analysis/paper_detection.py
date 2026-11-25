"""Summarize passive detector decisions from existing CAST client-round CSVs.

Pass the historical Organized_Tidy_Passive_Detection_Results directory, or supply
individual round_client_detection_trial*.csv / merged detection CSVs. Each input
must contain complete tasks. The panels use trial 0, designated client 1, all
50 submitted rounds, four partitions, and N=4,6,8. The lazy schedule has gap 6.
Recall uses evaluated active submissions; FPR uses evaluated never-attacker
submissions. Both percentages are averaged equally across the 12 settings.
"""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd


ALPHAS = {
    "tictactoe": (10, 100, 1000, 20000),
    "adult": (1000, 100000, 1000000, 20000000),
    "dota2": (10000, 1000000, 50000000, 200000000),
    "mnist": (10, 1000, 100000, 10000000),
    "imdb": (10, 100, 1000, 200000),
}
# Recorded comparison groups in the passive detector exports.
COMPARISONS = {
    "tictactoe": (("moirai_online", "advanced_scaled_delta"), ("moirai_online", "synthetic_replacement")),
    "adult": (("moirai_online", "adam_local_deviation"), ("arima_adaptive", "synthetic_replacement")),
    "dota2": (("arima_adaptive", "dkfr"), ("moirai_online", "synthetic_replacement")),
    "mnist": (("moirai_online_mnist_imdb_layerwise", "dkfr"), ("moirai_online_mnist_imdb_layerwise", "synthetic_replacement")),
    "imdb": (("moirai_online_mnist_imdb_layerwise", "dkfr"), ("arima_adaptive_mnist_imdb_layerwise", "synthetic_replacement")),
}
DETECTORS = {
    "dagmm": ("std_dagmm", "delta_dagmm"),
    "patches": ("wef", "s2_wef", "log_based_free_rider_detection"),
    "pass": ("pass",),
}
DETAIL_NAMES = {"std_dagmm": "STD-DAGMM", "delta_dagmm": "Delta-DAGMM", "wef": "WEF", "s2_wef": "S2-WEF"}
CELL = ["Dataset", "AttackRegime", "AttackMethod", "DetectingMethod"]
SETTING = CELL + ["Alpha", "ClientNum"]
COUNTS = ["TP", "FP", "ActiveTotal", "ActiveEvaluated", "HonestTotal", "HonestEvaluated"]
FIELDS = set(SETTING + [
    "TaskID", "TrialID", "AttackerID", "Round", "ClientID", "IsDesignatedAttacker",
    "GroundTruthActiveAttack", "RoundPredictedSuspicious", "DetectionSkipped", "DecisionMode",
    "QualitativeAggregationMode", "LazySchedule", "LazyHonestGap", "LazyHonestPhase",
])


def expected_settings(panel):
    result = set()
    for dataset, comparisons in COMPARISONS.items():
        for regime, methods in zip(("pure", "lazy"), comparisons):
            if panel == "patches" and (dataset not in {"adult", "mnist", "imdb"} or regime != "lazy"):
                continue
            for combination in product(methods, DETECTORS[panel], ALPHAS[dataset], (4, 6, 8)):
                result.add((dataset, regime, *combination))
    return result


def archive_paths(root, settings):
    paths = set()
    for dataset, _, _, detector, _, _ in settings:
        if detector == "pass":
            path = root / "PASS_Wang" / f"{dataset}_pass.csv"
        elif detector == "log_based_free_rider_detection":
            path = root / "log-based_nguyen" / f"{dataset}_log_based.csv"
        else:
            path = root / "Passive_detections_4methods" / "details" / "detection" / f"{DETAIL_NAMES[detector]}.csv"
        paths.add(path)
    return sorted(paths)


def binary(values):
    flags = values.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    )
    if flags.isna().any():
        raise ValueError(f"{values.name} contains a missing or nonbinary label")
    return flags.astype(bool)


def active_report_sources(rows, active):
    detector = rows["DetectingMethod"].iloc[0]
    selected = rows.loc[active]
    if detector in {"wef", "s2_wef"}:
        field = "WEFPayloadSource"
    elif detector == "log_based_free_rider_detection":
        field = "ReportSource"
    else:
        return {}
    sources = (selected[field].str.strip() if field in selected
               else pd.Series("", index=selected.index, dtype=str))
    if field == "ReportSource" and "DetectorDetailsJSON" in selected:
        missing = sources.eq("") & selected["DetectorDetailsJSON"].ne("")
        sources = sources.copy()
        sources.loc[missing] = selected.loc[missing, "DetectorDetailsJSON"].map(
            lambda details: json.loads(details).get("report_source") or ""
        )
    return dict(Counter(sources.replace("", "unrecorded")))


def evaluate_task(rows, client_count):
    if rows["TaskID"].nunique() != 1 or rows["DecisionMode"].nunique() != 1 or not rows["DecisionMode"].iloc[0]:
        raise ValueError("Expected one task and one recorded decision rule per setting")
    if (len(rows) != 50 * client_count or rows.duplicated(["Round", "ClientID"]).any()
            or set(rows["Round"]) != set(range(50)) or set(rows["ClientID"]) != set(range(client_count))):
        raise ValueError("Expected one row per submitted round and client")
    designated = binary(rows["IsDesignatedAttacker"])
    active = binary(rows["GroundTruthActiveAttack"])
    if not designated.eq(rows["ClientID"].eq(rows["AttackerID"])).all() or (active & ~designated).any():
        raise ValueError("Designated and active labels disagree with client identities")
    valid = ~binary(rows["DetectionSkipped"])
    predictions = rows["RoundPredictedSuspicious"].str.strip()
    if not valid.eq(predictions.ne("")).all():
        raise ValueError("DetectionSkipped disagrees with the recorded decision")
    if "DecisionValid" in rows:
        known = rows["DecisionValid"].str.strip().ne("")
        if not valid[known].eq(binary(rows.loc[known, "DecisionValid"])).all():
            raise ValueError("DecisionValid disagrees with DetectionSkipped")
    alarm = binary(predictions[valid]).reindex(rows.index, fill_value=False)
    honest = ~designated
    return {
        "DecisionMode": rows["DecisionMode"].iloc[0],
        "ReportSources": active_report_sources(rows, active),
        "TP": int((active & alarm).sum()), "FP": int((honest & alarm).sum()),
        "ActiveTotal": int(active.sum()), "ActiveEvaluated": int((active & valid).sum()),
        "HonestTotal": int(honest.sum()), "HonestEvaluated": int((honest & valid).sum()),
    }


def summarize(paths, panel, *, output=None):
    expected = expected_settings(panel)
    if len(paths) == 1 and paths[0].is_dir():
        paths = archive_paths(paths[0], expected)
    if output is not None and any(path.resolve() == output.resolve() for path in paths):
        raise ValueError("Choose an output path distinct from the client-round inputs")
    settings = {}
    columns = FIELDS | {"DecisionValid"}
    if panel == "patches":
        columns |= {"WEFPayloadSource", "ReportSource", "DetectorDetailsJSON"}
    for path in paths:
        rows = pd.read_csv(path, usecols=lambda c: c in columns,
                           dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if missing := FIELDS.difference(rows.columns):
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows["Dataset"] = rows["Dataset"].str.lower()
        for name in ("Alpha", "ClientNum", "TrialID", "AttackerID", "Round", "ClientID"):
            rows[name] = pd.to_numeric(rows[name], errors="raise")
        lazy_gap = pd.to_numeric(rows["LazyHonestGap"], errors="coerce")
        scope = (rows["TrialID"].eq(0) & rows["AttackerID"].eq(1)
                 & rows["QualitativeAggregationMode"].eq("passive")
                 & (rows["AttackRegime"].eq("pure") | (rows["LazySchedule"].eq("fixed")
                    & lazy_gap.eq(6) & pd.to_numeric(rows["LazyHonestPhase"], errors="coerce").eq(0))))
        for key, task in rows.loc[scope].groupby(SETTING, sort=False):
            if key not in expected:
                continue
            if key in settings:
                raise ValueError(f"Duplicate setting in {path}: {key}")
            try:
                settings[key] = {**dict(zip(SETTING, key)), **evaluate_task(task, int(key[-1]))}
            except ValueError as exc:
                raise ValueError(f"{path}: {key}: {exc}") from exc
    missing = expected.difference(settings)
    if missing:
        raise ValueError(f"Missing {len(missing)} of {len(expected)} expected settings, including {min(missing)}")
    scores = pd.DataFrame(settings.values())
    summary = []
    for key, tasks in scores.groupby(CELL, sort=True):
        if tasks["DecisionMode"].nunique() != 1:
            raise ValueError(f"Inconsistent decision rules across settings for {key}")
        row = dict(zip(CELL, key))
        row.update(DecisionMode=tasks["DecisionMode"].iloc[0], Settings=len(tasks))
        for name in COUNTS:
            row[name] = int(tasks[name].sum())
        if panel == "patches":
            sources = Counter()
            for recorded in tasks["ReportSources"]:
                sources.update(recorded)
            row["ActiveReportSources"] = json.dumps(dict(sources), sort_keys=True)
        for name, numerator, denominator in (("Recall", "TP", "ActiveEvaluated"), ("FPR", "FP", "HonestEvaluated")):
            rates = tasks[numerator].div(tasks[denominator].where(tasks[denominator].ne(0)))
            row[f"{name}Settings"] = int(rates.count())
            row[f"{name}Pct"] = 100 * rates.mean() if rates.notna().all() else None
        summary.append(row)
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Export root or complete client-round CSVs")
    parser.add_argument("--panel", choices=DETECTORS, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = summarize(args.inputs, args.panel, output=args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

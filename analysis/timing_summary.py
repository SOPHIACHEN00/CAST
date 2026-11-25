"""Reduce recorded timings from the no-attack round probe or CAST experiment CSVs."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def nonnegative_numbers(values, column):
    numbers = pd.to_numeric(values, errors="raise")
    if not (np.isfinite(numbers) & numbers.ge(0)).all():
        raise ValueError(f"{column} must contain finite, nonnegative numbers")
    return numbers


def summarize_honest(paths):
    keys = ["Dataset", "TrialID", "RunType", "ClientNum", "LocalEpochs"]
    records = []
    for path in paths:
        rounds = pd.read_csv(path, usecols=keys + ["RoundIndex", "GlobalTrainingSec"])
        if rounds.empty or not rounds.RunType.eq("no_attack").all() or rounds[keys].isna().any().any():
            raise ValueError(f"Expected labeled no-attack timing rows in {path}")
        for identity, run in rounds.groupby(keys):
            indices = pd.to_numeric(run.RoundIndex, errors="raise")
            if sorted(indices) != list(range(len(run))):
                raise ValueError(f"Expected unique, consecutive round indices starting at zero in {path}")
            times = nonnegative_numbers(run.GlobalTrainingSec, "GlobalTrainingSec")
            records.append({**dict(zip(keys, identity)), "ObservedFLRounds": len(run),
                            "GlobalTrainingTotalSec": times.sum(), "MeanSecPerObservedFLRound": times.mean()})
    result = pd.DataFrame(records)
    if result.duplicated(keys).any():
        raise ValueError("The same no-attack run appears in more than one input file")
    return result


def summarize_generation(path, fl_rounds):
    """Report a run total and only those averages whose denominators are known."""
    frame = pd.read_csv(path)
    columns = [name for name in ("AttackGenerationWithAttackerTotalSec", "WithAttackerSec") if name in frame]
    if len(columns) != 1:
        raise ValueError("Expected exactly one generation timing column (AttackGenerationWithAttackerTotalSec or WithAttackerSec)")
    source = frame[columns[0]]
    # No-attack controls carry N/A in the generation timing column.
    if "UseAttack" in frame:
        selected = frame.UseAttack.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
    else:
        selected = source.notna()
    metadata = [name for name in ("TaskID", "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID",
                                   "TrialID", "UseHessian", "UseLazy", "LazySchedule", "LazyHonestGap",
                                   "LazyHonestPhase", "LazyFakePolicy", "AttackProbability", "ContributionMethod",
                                   "DetectingMethod") if name in frame]
    result = frame.loc[selected, metadata].copy()
    total = nonnegative_numbers(source.loc[selected], columns[0])
    result["GenerationTotalSec"] = total
    result["FLRoundsPerRun"] = fl_rounds
    result["GenerationSecPerFLRound"] = total / fl_rounds if fl_rounds is not None else np.nan
    if "FakeRoundsTimed" in frame:
        counts = nonnegative_numbers(frame.loc[selected, "FakeRoundsTimed"], "FakeRoundsTimed")
        if counts.mod(1).ne(0).any() or (fl_rounds is not None and counts.gt(fl_rounds).any()):
            raise ValueError("FakeRoundsTimed must be a whole count within the FL run")
        result["FakeRoundsTimed"] = counts
        result["GenerationSecPerTimedFakeRound"] = total / counts.where(counts.gt(0))
    else:
        result["FakeRoundsTimed"] = np.nan
        result["GenerationSecPerTimedFakeRound"] = np.nan
    return result


def archive_display_seconds(value):
    """Numeric value printed by the timing archive's table_common.fmt."""
    for threshold, decimals in [(100, 0), (10, 1), (1, 2), (0.1, 3), (0.01, 4)]:
        if value >= threshold:
            return float(f"{value:.{decimals}f}")
    return float(f"{value:.5f}")


def summarize_grid(path, fl_rounds):
    """Rebuild the archived timing grid's equal-setting-weight medians."""
    blocks = ["Dataset", "Mode", "AttackMethod"]
    settings = blocks + ["Alpha", "ClientNum", "Gap"]
    tasks = pd.read_csv(path, usecols=settings + ["TaskKey", "TimeSec"])
    if tasks.empty or tasks[settings + ["TaskKey"]].isna().any().any() or tasks.TaskKey.duplicated().any():
        raise ValueError("Expected identified timing settings and one row per unique TaskKey")
    tasks["TimeSec"] = nonnegative_numbers(tasks.TimeSec, "TimeSec")
    by_setting = tasks.groupby(settings).TimeSec.agg(SettingMean="mean", Tasks="size").reset_index()
    result = by_setting.groupby(blocks, as_index=False).agg(
        Settings=("SettingMean", "size"), Tasks=("Tasks", "sum"),
        MedianSettingMeanTotalSec=("SettingMean", "median"),
    )
    result["FLRoundsPerRun"] = fl_rounds
    result["MedianSettingMeanSecPerFLRound"] = (
        result.MedianSettingMeanTotalSec / fl_rounds if fl_rounds is not None else np.nan
    )
    # The archived display rounded cumulative seconds before the manuscript's
    # per-round comparison. Keep that second calculation alongside full precision.
    result["ArchiveDisplayTotalSec"] = result.MedianSettingMeanTotalSec.map(archive_display_seconds)
    result["ArchiveDisplaySecPerFLRound"] = (
        result.ArchiveDisplayTotalSec / fl_rounds if fl_rounds is not None else np.nan
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    honest = modes.add_parser("honest", help="Summarize recorded no-attack GlobalTrainingSec for each run")
    honest.add_argument("inputs", type=Path, nargs="+")
    generation = modes.add_parser("generation", help="Summarize recorded per-run generation totals")
    grid = modes.add_parser("grid", help="Summarize the historical full_grid_selected_time_tasks.csv task table")
    for command in (generation, grid):
        command.add_argument("input", type=Path)
        command.add_argument("--fl-rounds", type=int, help="Verified FL rounds per run, if common to every input row")
    for command in (honest, generation, grid):
        command.add_argument("-o", "--output", type=Path, help="Write a CSV (default: stdout)")
    args = parser.parse_args()
    inputs = args.inputs if args.mode == "honest" else [args.input]
    if args.output is not None and args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("Choose an output path distinct from the inputs")
    if getattr(args, "fl_rounds", None) is not None and args.fl_rounds <= 0:
        parser.error("--fl-rounds must be positive")
    if args.mode == "honest":
        summary = summarize_honest(inputs)
    elif args.mode == "generation":
        summary = summarize_generation(args.input, args.fl_rounds)
    else:
        summary = summarize_grid(args.input, args.fl_rounds)
    summary.to_csv(args.output or sys.stdout, index=False)


if __name__ == "__main__":
    main()

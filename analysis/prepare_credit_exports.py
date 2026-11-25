"""Assemble the two gap-6 credit plot tables from archived per-dataset summaries."""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


DATASETS = ("tictactoe", "adult", "dota2", "mnist", "imdb")
LEGACY_PURE = ("random", "fedavg")
EXTERNAL_PURE = ("dkfr", "adam_update_prediction", "advanced_scaled_delta")
KEYS = ["Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackMode"]
Q1 = "Q1_AttackerRatio"
Q2 = "Q2_AbsImpact"


def read_table(path, metric):
    frame = pd.read_csv(path)
    required = KEYS + [metric]
    if metric == Q2:
        required += ["ClientID", "AttackerID", "TrialID", "WithAttacker", "WithoutAttacker"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    frame["Dataset"] = frame.Dataset.str.lower()
    frame["AttackMethod"] = frame.AttackMethod.str.strip()
    frame["Alpha"] = pd.to_numeric(frame.Alpha, errors="raise").astype(float)
    counts = pd.to_numeric(frame.ClientNum, errors="raise")
    if not (counts.gt(0) & counts.mod(1).eq(0)).all():
        raise ValueError(f"Invalid client count in {path}")
    frame["ClientNum"] = counts.astype(int)
    return frame


def pure_rows(frame, methods, datasets):
    return frame.loc[frame.Dataset.isin(datasets) & frame.AttackMode.eq("pure_fake") & frame.AttackMethod.isin(methods)]


def setting_keys(frame):
    return set(frame[KEYS].itertuples(index=False, name=None))


def assemble_tables(archive_root, selected_dir):
    selection = pd.read_csv(selected_dir / "forecast_source_selection_manifest.csv",
                            dtype={"Methods": str, "Alphas": str, "ClientNums": str}).set_index("Dataset", verify_integrity=True)
    if set(selection.index) != set(DATASETS):
        raise ValueError("Expected one selection-manifest row for each of the five figure datasets")
    external = archive_root / "Pure_Attack" / "RQ_Results"
    legacy_q1, legacy_q2, external_q2, forecast_q1, forecast_q2 = [], [], [], [], []
    expected = set()
    for dataset in DATASETS:
        spec = selection.loc[dataset]
        if (spec.LazySchedule, spec.LazyHonestGap, spec.LazyAttackMode) != ("fixed", 6, "1real_5fake"):
            raise ValueError(f"{dataset}: expected the fixed gap-6 selection manifest")
        methods = [item.strip() for item in spec.Methods.split(",")]
        alphas = [float(item) for item in spec.Alphas.split(",")]
        counts = [int(item) for item in spec.ClientNums.split(",")]
        old_dir = archive_root / dataset
        legacy_q1.append(pure_rows(read_table(old_dir / f"{dataset}RQ1.csv", Q1), LEGACY_PURE, [dataset]))
        legacy_q2.append(pure_rows(read_table(old_dir / f"{dataset}RQ2_box.csv", Q2), LEGACY_PURE, [dataset]))
        external_q2.append(pure_rows(read_table(external / dataset / f"{dataset}RQ2_box.csv", Q2), EXTERNAL_PURE, [dataset]))
        forecast_dir = selected_dir / dataset
        one = read_table(forecast_dir / "forecast_RQ1.csv", Q1)
        two = read_table(forecast_dir / "forecast_RQ2_box.csv", Q2)
        grid = set(product([dataset], methods, alphas, counts, ["pure_fake", spec.LazyAttackMode]))
        if (len(one), len(two)) != (spec.RQ1GroupRows, spec.RQ2BoxRows) or setting_keys(one) != grid or setting_keys(two) != grid:
            raise ValueError(f"{dataset}: forecast summary rows do not match the selection manifest")
        forecast_q1.append(one)
        forecast_q2.append(two)
        expected.update(grid)
        expected.update(product([dataset], LEGACY_PURE + EXTERNAL_PURE, alphas, counts, ["pure_fake"]))

    external_q1 = pure_rows(read_table(external / "combined" / "pure_attack_RQ1.csv", Q1), EXTERNAL_PURE, DATASETS)
    rq1 = pd.concat([*legacy_q1, external_q1, *forecast_q1], ignore_index=True)
    rq2 = pd.concat([*legacy_q2, *external_q2, *forecast_q2], ignore_index=True)
    for frame, metric in [(rq1, Q1), (rq2, Q2)]:
        if setting_keys(frame) != expected or not np.isfinite(frame[metric]).all():
            raise ValueError(f"Incomplete setting coverage or missing scores for {metric}")
    if rq1.duplicated(KEYS).any():
        raise ValueError("Duplicate RQ1 setting")
    runs = KEYS + ["AttackerID", "TrialID"]
    if (rq2.duplicated(runs + ["ClientID"]).any() or rq2.ClientID.eq(rq2.AttackerID).any()
            or rq2.groupby(runs, dropna=False).ClientID.transform("size").ne(rq2.ClientNum - 1).any()
            or not np.allclose(rq2[Q2], (rq2.WithAttacker - rq2.WithoutAttacker).abs(), rtol=1e-8, atol=1e-12)):
        raise ValueError("Incomplete or inconsistent matched honest-client observations in RQ2")
    return rq1, rq2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True, help="Directory containing dataset exports and Pure_Attack/RQ_Results")
    parser.add_argument("--selected-dir", type=Path, required=True, help="Directory containing the forecast selection manifest and per-dataset forecast summaries")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rq1, rq2 = assemble_tables(args.archive_root, args.selected_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ("combined_RQ1_plot_data.csv", "combined_RQ2_box_plot_data.csv")
    with TemporaryDirectory(dir=args.output_dir) as staging:
        for name, frame in zip(names, (rq1, rq2)):
            frame.to_csv(Path(staging) / name, index=False)
        for name in names:
            (Path(staging) / name).replace(args.output_dir / name)


if __name__ == "__main__":
    main()

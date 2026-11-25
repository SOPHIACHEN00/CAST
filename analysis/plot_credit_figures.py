"""Render the four credit figures from the corresponding archived analysis exports."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PARTITIONS = {
    "tictactoe": (10, 100, 1000, 20000),
    "adult": (1000, 100000, 1000000, 20000000),
    "dota2": (10000, 1000000, 50000000, 200000000),
    "mnist": (10, 1000, 100000, 10000000),
    "imdb": (10, 100, 1000, 200000),
}
TITLES = {"tictactoe": "Tic-Tac-Toe", "adult": "Adult", "dota2": "Dota2", "mnist": "MNIST", "imdb": "IMDB"}
MODALITIES = ("adult", "mnist", "imdb")
MODES = {"pure_fake": "DFR", "1real_5fake": "SFR"}
FORECASTS = ("moirai", "moirai_online", "arima", "arima_adaptive")
# These are method identifiers in the archived figure exports.
BASELINES = ("random", "fedavg", "dkfr", "adam_update_prediction", "advanced_scaled_delta")
LEGEND = {
    "random": ("Rand", "#d62728"), "fedavg": ("Replay", "#ff7f0e"),
    "dkfr": ("DKFR", "#222222"), "adam_update_prediction": ("Adam", "#008b8b"),
    "advanced_scaled_delta": ("ReplayS", "#c2185b"), "synthetic_replacement": ("Syn (separate reference)", "#777777"),
    "moirai": ("CastM-Off", "#2ca02c"), "moirai_online": ("CastM-On", "#1f77b4"),
    "arima": ("CastA-Off", "#9467bd"), "arima_adaptive": ("CastA-On", "#8c564b"),
}


def read_export(path, required):
    frame = pd.read_csv(path)
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def observations(block, column, levels, metric, *, multiple=False):
    groups = []
    for value in levels:
        values = block.loc[block[column].eq(value), metric].to_numpy(dtype=float)
        if not len(values) or (len(values) != 1 and not multiple) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite {metric} at {column}={value}, got {len(values)} rows")
        groups.append(values)
    return groups


def curve(axis, block, column, levels, metric, color):
    values = observations(block, column, levels, metric)
    axis.plot(range(len(levels)), [group[0] for group in values], marker="o", ms=3, lw=1.1, color=color)


def setup_axis(axis, dataset, xlabel, labels):
    axis.set_title(TITLES[dataset])
    axis.set_xlabel(xlabel)
    axis.set_xticks(range(len(labels)), labels)
    axis.grid(axis="y", alpha=0.2)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)


def legend(fig, entries):
    handles = [Line2D([], [], color=color, label=label, lw=2) for label, color in entries]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=False)


def compare_online_offline(source):
    """Pair displayed conditions once, including the shared crossing of the two sweeps."""
    keys = ["Dataset", "AttackMode", "ClientNum", "Alpha"]
    frame = read_export(source, keys + ["AttackMethod", "Q1_AttackerRatio"])
    selected = []
    for dataset, alphas in PARTITIONS.items():
        on_client_axis = frame.Alpha.eq(alphas[0]) & frame.ClientNum.isin((4, 6, 8))
        on_partition_axis = frame.ClientNum.eq(6) & frame.Alpha.isin(alphas)
        suffix = "_mnist_imdb_layerwise" if dataset in ("mnist", "imdb") else ""
        methods = {method + suffix: method for method in FORECASTS}
        rows = frame.loc[frame.Dataset.eq(dataset) & frame.AttackMode.isin(MODES)
                         & frame.AttackMethod.isin(methods) & (on_client_axis | on_partition_axis)].copy()
        rows["Method"] = rows.AttackMethod.map(methods)
        selected.append(rows)
    paired = pd.concat(selected).pivot(index=keys, columns="Method", values="Q1_AttackerRatio")
    expected = len(PARTITIONS) * len(MODES) * 6
    if len(paired) != expected or set(paired.columns) != set(FORECASTS) or not np.isfinite(paired).all().all():
        raise ValueError("Incomplete matched-condition grid for the online/offline comparison")
    paired["MoiraiOnMinusOff"] = paired.moirai_online - paired.moirai
    paired["ArimaOnMinusOff"] = paired.arima_adaptive - paired.arima
    return paired.reset_index()


def client_and_partition_panels(number, source, reference):
    boxes = number == 2
    metric = "Q2_AbsImpact" if boxes else "Q1_AttackerRatio"
    keys = ["Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackMode"]
    frame = read_export(source, keys + [metric] + (["ClientID", "AttackerID", "TrialID", "WithAttacker", "WithoutAttacker"] if boxes else []))
    modes = ("1real_5fake",) if boxes else tuple(MODES)
    datasets = MODALITIES if boxes else tuple(PARTITIONS)
    frame = frame[frame.Dataset.isin(datasets) & frame.AttackMode.isin(modes)].copy()
    if boxes:
        frame = frame[frame.AttackMethod.str.removesuffix("_mnist_imdb_layerwise").isin(FORECASTS)].copy()
        populations = frame.groupby(keys, dropna=False)["ClientID"].transform("size")
        if (not (frame.TrialID.eq(0) & frame.AttackerID.eq(1)).all()
                or not (frame.ClientID.ge(0) & frame.ClientID.lt(frame.ClientNum) & frame.ClientID.mod(1).eq(0)).all()
                or frame.ClientID.eq(frame.AttackerID).any() or frame.duplicated(keys + ["ClientID"]).any()
                or populations.ne(frame.ClientNum - 1).any()
                or not np.allclose(frame[metric], (frame.WithAttacker - frame.WithoutAttacker).abs(), rtol=1e-8, atol=1e-12)):
            raise ValueError("Expected all matched honest clients of trial 0, designated client 1, with absolute raw-score differences")
    else:
        syn = read_export(reference, keys + [metric])
        if not syn.AttackMethod.eq("synthetic_replacement").all() or not syn.AttackMode.eq("pure_fake").all():
            raise ValueError("Expected the observed non-lazy synthetic reference export")
    fig, axes = plt.subplots(2 * len(modes), len(datasets), figsize=(3.1 * len(datasets), 2.6 * len(modes) * 2), squeeze=False)
    for row, mode in enumerate(modes):
        for col, dataset in enumerate(datasets):
            alphas = PARTITIONS[dataset]
            for offset, (x, levels) in enumerate((("ClientNum", (4, 6, 8)), ("Alpha", alphas))):
                axis = axes[row * 2 + offset, col]
                block = frame[frame.Dataset.eq(dataset) & frame.AttackMode.eq(mode)]
                fixed, value = ("Alpha", alphas[0]) if x == "ClientNum" else ("ClientNum", 6)
                block = block[block[fixed].eq(value)]
                methods = BASELINES + FORECASTS if mode == "pure_fake" else FORECASTS
                for index, method in enumerate(methods):
                    historical_id = method + "_mnist_imdb_layerwise" if method in FORECASTS and dataset in ("imdb", "mnist") else method
                    records = block[block.AttackMethod.eq(historical_id)]
                    color = LEGEND[method][1]
                    if boxes:
                        axis.boxplot(observations(records, x, levels, metric, multiple=True), whis=1.5,
                                     positions=np.arange(len(levels)) + (index - 1.5) * 0.2, widths=0.17,
                                     patch_artist=True, boxprops={"facecolor": color}, medianprops={"color": "black"},
                                     flierprops={"markersize": 2}, manage_ticks=False)
                    else:
                        curve(axis, records, x, levels, metric, color)
                if not boxes and mode == "1real_5fake":
                    curve(axis, syn[syn.Dataset.eq(dataset) & syn[fixed].eq(value)], x, levels, metric, LEGEND["synthetic_replacement"][1])
                labels = (format(v, ".0e").replace("e+0", "e") if v >= 100000 else str(v) for v in alphas[:-1])
                setup_axis(axis, dataset, "Clients" if x == "ClientNum" else "Partition α", levels if x == "ClientNum" else [*labels, "UF"])
                if col == 0:
                    axis.set_ylabel(MODES[mode] + (" |credit change|" if boxes else " signed credit share"))
    legend(fig, [LEGEND[key] for key in (FORECASTS if boxes else BASELINES + ("synthetic_replacement",) + FORECASTS)])
    return fig


def counts_or_lambda_panels(number, source):
    counts = number == 3
    metric = "Q1_AttackerMeanRatio" if counts else "Q1_AttackerRatio"
    x, levels = ("AttackerCount", (1, 2, 3)) if counts else ("LambdaH", (0.5, 0.7, 0.9, 1.0))
    required = ["Method"] if counts else ["AttackMethod", "ClientNum", "Alpha"]
    frame = read_export(source, ["Dataset", "AttackMode", metric, x] + required)
    if not counts:
        frame = frame[frame.ClientNum.eq(6) & frame.AttackMode.eq("pure_fake")].copy()
        frame["Method"] = frame.AttackMethod.str.removesuffix("_mnist_imdb_layerwise")
    rows = tuple(MODES) if counts else ("moirai_online", "arima_adaptive")
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.3))
    partition_colors = ("#2467a6", "#bc6c25", "#279078", "#705098")
    for row, key in enumerate(rows):
        for col, dataset in enumerate(MODALITIES):
            axis = axes[row, col]
            block = frame[frame.Dataset.eq(dataset)]
            if counts:
                block = block[block.AttackMode.eq(key)]
                for method in FORECASTS:
                    curve(axis, block[block.Method.eq(method)], x, levels, metric, LEGEND[method][1])
            else:
                block = block[block.Method.eq(key)]
                for alpha, color in zip(PARTITIONS[dataset], partition_colors):
                    curve(axis, block[block.Alpha.eq(alpha)], x, levels, metric, color)
            setup_axis(axis, dataset, "Free-riders" if counts else "λ", levels)
            if col == 0:
                axis.set_ylabel(MODES[key] + " share per free-rider" if counts else LEGEND[key][0] + " signed share")
    entries = [LEGEND[k] for k in FORECASTS] if counts else zip(("α₁", "α₂", "α₃", "UF"), partition_colors)
    legend(fig, entries)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--synthetic-reference", type=Path, help="Separate synthetic_baseline_RQ1.csv for Figure 1")
    parser.add_argument("--paired-csv", type=Path, help="Export the unique online/offline pairs for Figure 1")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output PDF or PNG")
    args = parser.parse_args()
    if (args.figure == 1) != (args.synthetic_reference is not None):
        parser.error("--synthetic-reference is required for Figure 1 and only applies to that figure")
    if args.paired_csv and args.figure != 1:
        parser.error("--paired-csv applies to Figure 1")
    paths = [path.resolve() for path in (args.input_csv, args.synthetic_reference, args.output, args.paired_csv) if path]
    if len(set(paths)) != len(paths):
        parser.error("Input and output files must have distinct paths")
    fig = (client_and_partition_panels(args.figure, args.input_csv, args.synthetic_reference) if args.figure <= 2
           else counts_or_lambda_panels(args.figure, args.input_csv))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(args.output, bbox_inches="tight", dpi=220)
    plt.close(fig)
    if args.paired_csv:
        compare_online_offline(args.input_csv).to_csv(args.paired_csv, index=False)


if __name__ == "__main__":
    main()

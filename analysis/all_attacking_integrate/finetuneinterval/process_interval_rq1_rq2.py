"""Derive RQ1/RQ2 tables from the pure Shapley interval campaign CSVs.

The established Q1/Q2 metric functions under ``Analysis Code`` are reused.
Unlike their older CSV entry points, this adapter retains the tuned parameter
and interval in every group key so different settings cannot be pooled.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SOURCES = {dataset: HERE / f"{dataset}.csv" for dataset in ("adult", "mnist", "imdb")}
ALPHAS = {
    "adult": (1000, 100000, 1000000, 20000000),
    "mnist": (10, 1000, 100000, 10000000),
    "imdb": (10, 100, 1000, 200000),
}
METHODS = {
    "adult": {"moirai_online", "arima_adaptive"},
    "mnist": {"moirai_online_mnist_imdb_layerwise", "arima_adaptive_mnist_imdb_layerwise"},
    "imdb": {"moirai_online_mnist_imdb_layerwise", "arima_adaptive_mnist_imdb_layerwise"},
}
INTERVALS = {"moirai_online": (2, 4, 8), "arima_adaptive": (8, 15, 25)}
TUNED_PARAMETERS = {
    "moirai_online": "moirai_adapt_interval",
    "arima_adaptive": "arima_retrain_interval",
}
CLIENTS = (2, 4, 6, 8)
GROUP_KEYS = [
    "Dataset", "AttackMethod", "PartitionSetting", "Alpha", "ClientNum",
    "LambdaH", "TunedParameter", "IntervalValue",
    "ContributionMethod", "UseHessian", "UseLazy", "AttackMode", "ShapleyRound",
]
OUTPUTS = {
    "rq1_detail": HERE / "interval_RQ1_attacker_ratio_detail.csv",
    "rq1_group": HERE / "interval_RQ1_attacker_ratio_group_mean.csv",
    "rq2_detail": HERE / "interval_RQ2_honest_impact_detail.csv",
    "rq2_group": HERE / "interval_RQ2_honest_impact_group_mean.csv",
    "rq2_honest": HERE / "interval_RQ2_honest_client_boxplot.csv",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
        f"{label}: {actual} != {expected}",
    )


def load_processor(relative_path: str) -> ModuleType:
    path = PROJECT_ROOT / "Analysis Code" / relative_path
    require(path.is_file(), f"Missing established processor: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_method(raw: str) -> str:
    return raw.removesuffix("_mnist_imdb_layerwise")


def read_sources() -> pd.DataFrame:
    frames = []
    for dataset, path in SOURCES.items():
        require(path.is_file(), f"Missing interval source: {path}")
        frame = pd.read_csv(path, dtype={"CampaignID": str, "TaskID": str})
        require(len(frame) == 42, f"{path.name}: expected 42 tasks, found {len(frame)}")
        require(frame["Dataset"].eq(dataset).all(), f"{path.name}: dataset mismatch")
        require(frame["AttackMethod"].isin(METHODS[dataset]).all(), f"{path.name}: unexpected method")
        expected_partition = frame["Alpha"].astype(str).where(
            frame["Alpha"].ne(ALPHAS[dataset][-1]), "UF"
        )
        require(frame["PartitionSetting"].astype(str).eq(expected_partition).all(),
                f"{path.name}: partition/alpha mismatch")
        frame["SourceFile"] = path.name
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    require(len(source) == 126, "Expected 126 source tasks")
    require(source["TaskID"].is_unique, "TaskID must be unique across all source files")
    require(source["CampaignID"].nunique() == 1, "Multiple campaigns in source files")
    require(source["UseHessian"].astype(str).str.lower().eq("true").all(), "Expected Hessian runs")
    require(source["UseLazy"].astype(str).str.lower().eq("false").all(), "Expected pure attacks only")
    require(source["ShapleyRound"].eq(49).all(), "Unexpected Shapley round")
    require(source["LambdaH"].eq(0.9).all(), "LambdaH must be fixed at 0.9")
    for dataset, methods in METHODS.items():
        subset = source[source["Dataset"].eq(dataset)]
        require(set(subset["AttackMethod"]) == methods, f"{dataset}: missing method")
        expected_settings = {(ALPHAS[dataset][0], n) for n in CLIENTS}
        expected_settings |= {(alpha, 6) for alpha in ALPHAS[dataset]}
        for method in methods:
            canonical = canonical_method(method)
            require(canonical in INTERVALS, f"Unknown method: {method}")
            for interval in INTERVALS[canonical]:
                block = subset[
                    subset["AttackMethod"].eq(method)
                    & subset["IntervalValue"].eq(interval)
                ]
                settings = set(zip(block["Alpha"], block["ClientNum"]))
                require(
                    len(block) == 7 and settings == expected_settings
                    and block["TunedParameter"].eq(TUNED_PARAMETERS[canonical]).all(),
                    f"{dataset}/{method}/interval={interval}: incomplete or invalid grid",
                )
    return source


def derive_details(
    source: pd.DataFrame, q1_processor: ModuleType, q2_processor: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q1_records: list[dict[str, object]] = []
    q2_records: list[dict[str, object]] = []
    honest_records: list[dict[str, object]] = []
    for _, row in source.iterrows():
        task = str(row["TaskID"])
        n = int(row["ClientNum"])
        attacker_id = int(row["AttackerID"])
        require(0 <= attacker_id < n, f"{task}: attacker index out of range")
        with_scores = json.loads(row["WithAttackerContributionsJSON"])
        without_scores = json.loads(row["WithoutAttackerContributionsJSON"])
        require(
            isinstance(with_scores, list) and isinstance(without_scores, list)
            and len(with_scores) == len(without_scores) == n,
            f"{task}: contribution array length mismatch",
        )
        require(without_scores[attacker_id] is None, f"{task}: attacker has a no-attacker score")
        require(
            all(isinstance(value, (int, float)) and math.isfinite(value) for value in with_scores),
            f"{task}: invalid with-attacker score",
        )
        require(
            all(isinstance(value, (int, float)) and math.isfinite(value)
                for i, value in enumerate(without_scores) if i != attacker_id),
            f"{task}: invalid honest no-attacker score",
        )
        contributions = pd.DataFrame({
            "ClientID": range(n), "AttackerID": attacker_id,
            "WithAttacker": with_scores, "WithoutAttacker": without_scores,
        })
        ratio = q1_processor.compute_attacker_ratio(contributions)
        require(ratio is not None and math.isfinite(float(ratio)), f"{task}: invalid RQ1 ratio")
        total = math.fsum(with_scores)
        close(with_scores[attacker_id], float(row["AttackerContribution"]), f"{task}/attacker")
        close(total, float(row["TotalContribution"]), f"{task}/total")
        close(float(ratio), float(row["Q1AttackerRatio"]), f"{task}/cached RQ1")

        honest = contributions[contributions["ClientID"].ne(attacker_id)]
        q2_metrics = q2_processor.compute_q2_metrics(honest)
        require(q2_metrics is not None, f"{task}: missing RQ2 metrics")
        require(q2_metrics["Q2_HonestClientCount"] == n - 1, f"{task}: incomplete honest clients")
        independent_impact = math.fsum(
            abs(with_scores[i] - without_scores[i]) for i in range(n) if i != attacker_id
        ) / (n - 1)
        close(q2_metrics["Q2_AvgAbsImpact"], independent_impact, f"{task}/RQ2")

        meta: dict[str, object] = {
            "CampaignID": row["CampaignID"], "SourceFile": row["SourceFile"],
            "TaskID": task, "Dataset": row["Dataset"], "AttackMethod": row["AttackMethod"],
            "PartitionSetting": str(row["PartitionSetting"]), "Alpha": int(row["Alpha"]),
            "ClientNum": n, "AttackerID": attacker_id, "TrialID": int(row["TrialID"]),
            "LambdaH": float(row["LambdaH"]),
            "TunedParameter": row["TunedParameter"],
            "IntervalValue": int(row["IntervalValue"]),
            "ContributionMethod": "shapley",
            "UseHessian": True, "UseLazy": False, "AttackMode": "pure_fake",
            "ShapleyRound": int(row["ShapleyRound"]),
        }
        q1_records.append({
            **meta, "AttackerContribution": with_scores[attacker_id],
            "TotalContribution": total, "Q1_AttackerRatio": float(ratio),
        })
        q2_records.append({**meta, **q2_metrics})
        for client_id in range(n):
            if client_id == attacker_id:
                continue
            delta = with_scores[client_id] - without_scores[client_id]
            honest_records.append({
                **meta, "ClientID": client_id,
                "WithAttacker": with_scores[client_id],
                "WithoutAttacker": without_scores[client_id],
                "Q2_Delta": delta, "Q2_AbsImpact": abs(delta),
                "Q2_HonestLoss": -delta,
            })
    q1_detail = pd.DataFrame(q1_records)
    q2_detail = pd.DataFrame(q2_records)
    q2_honest = pd.DataFrame(honest_records)
    require(len(q2_honest) == 558, "Missing honest-client observations")
    require(not q2_honest.duplicated(["TaskID", "ClientID"]).any(),
            "Duplicate honest-client observation")
    observed_means = q2_honest.groupby("TaskID")["Q2_AbsImpact"].mean()
    expected_means = q2_detail.set_index("TaskID")["Q2_AvgAbsImpact"]
    require(observed_means.index.equals(expected_means.sort_index().index),
            "Honest-client and task-level Q2 settings differ")
    require(all(math.isclose(observed_means[task], expected_means[task],
                             rel_tol=1e-10, abs_tol=1e-12)
                for task in observed_means.index),
            "Honest-client Q2 values disagree with established summary")
    return q1_detail, q2_detail, q2_honest


def derive_groups(
    q1_detail: pd.DataFrame, q2_detail: pd.DataFrame, metric_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q1_group = q1_detail.groupby(GROUP_KEYS, dropna=False, as_index=False).agg(
        Q1_AttackerRatio=("Q1_AttackerRatio", "mean"), TaskCount=("TaskID", "size"),
    )
    q2_group = q2_detail.groupby(GROUP_KEYS, dropna=False, as_index=False).agg(
        **{metric: (metric, "mean") for metric in metric_names},
        TaskCount=("TaskID", "size"),
    )
    require(len(q1_group) == len(q2_group) == len(q1_detail) == len(q2_detail) == 126,
            "Missing or pooled task in derived RQ tables")
    require(q1_group["TaskCount"].eq(1).all() and q2_group["TaskCount"].eq(1).all(),
            "Current campaign should contain one trial per setting")
    return q1_group, q2_group


def main() -> None:
    q1_processor = load_processor("Q1/compute_q1_summary.py")
    q2_processor = load_processor("Q2/compute_q2_honest_impact_summary.py")
    q2_group_processor = load_processor("Q2/q2_honest_impact_group_mean.py")
    source = read_sources()
    q1_detail, q2_detail, q2_honest = derive_details(source, q1_processor, q2_processor)
    metric_names = [
        name for name in q2_group_processor.DEFAULT_METRICS if name in q2_detail.columns
    ]
    q1_group, q2_group = derive_groups(q1_detail, q2_detail, metric_names)
    tables = {
        "rq1_detail": q1_detail, "rq1_group": q1_group,
        "rq2_detail": q2_detail, "rq2_group": q2_group,
        "rq2_honest": q2_honest,
    }
    for name, path in OUTPUTS.items():
        tables[name].to_csv(path, index=False, float_format="%.15g")
        print(f"{path.name}: {len(tables[name])} rows")


if __name__ == "__main__":
    main()

"""Read the legacy and current per-client contribution exports used for RQ1/RQ2."""

import numpy as np
import pandas as pd


RUN_COLUMNS = [
    "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
    "ContributionMethod", "UseHessian", "UseLazy", "LazyHonestGap",
    "LazyHonestPhase", "LazyFakePolicy", "AttackMode",
]

# Raw runner CSVs carry the schedule fields. Merged exports can also identify
# the campaign, task and sampling seed. A task is one observation for a mean.
OPTIONAL_SETTINGS = [
    "DetectingType", "LazySchedule", "AttackProbability", "SamplingSeed",
    "PartitionSetting", "LambdaH", "TunedParameter", "IntervalValue",
    "ShapleyRound", "FLAggregate",
]
OPTIONAL_RUN_IDS = ["Seed", "CampaignID", "TaskID"]


def parse_bool(value):
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def infer_attack_mode(use_lazy, lazy_honest_gap, schedule="fixed"):
    if not parse_bool(use_lazy):
        return "pure_fake"
    if str(schedule).strip().lower() == "probabilistic":
        return "probabilistic"
    return f"1real_{int(float(lazy_honest_gap)) - 1}fake"


def read_contribution_runs(input_csv_path, *, require_controls=False):
    frame = pd.read_csv(input_csv_path)
    method_columns = [name for name in ("ContributionMethod", "DetectingMethod") if name in frame]
    if not method_columns:
        raise ValueError("Expected a ContributionMethod or DetectingMethod column.")

    methods = frame[method_columns].astype("string").apply(lambda values: values.str.strip().str.lower())
    methods = methods.replace("", pd.NA)
    if len(method_columns) == 2:
        both = methods.dropna()
        if both["ContributionMethod"].ne(both["DetectingMethod"]).any():
            raise ValueError("ContributionMethod and DetectingMethod disagree in the same row.")
        frame["ContributionMethod"] = methods["ContributionMethod"].fillna(methods["DetectingMethod"])
    else:
        frame["ContributionMethod"] = methods.iloc[:, 0]

    frame = frame.loc[frame["UseAttack"].map(parse_bool)].copy()
    if "DetectingType" in frame:
        qualitative = frame["DetectingType"].astype("string").str.strip().str.lower().eq("qualitative")
        frame = frame.loc[~qualitative.fillna(False)].copy()
    if frame["ContributionMethod"].isna().any():
        raise ValueError("Missing contribution method for an experiment row.")
    for name in ("ClientNum", "ClientID", "AttackerID"):
        values = pd.to_numeric(frame[name], errors="raise")
        if not (np.isfinite(values) & values.ge(0) & values.mod(1).eq(0)).all():
            raise ValueError(f"{name} must be a nonnegative integer in each contribution row.")
        frame[name] = values.astype(int)
    if (frame["ClientID"].ge(frame["ClientNum"]) | frame["AttackerID"].ge(frame["ClientNum"])).any():
        raise ValueError("ClientID and AttackerID must be within the recorded client population.")
    schedules = frame.get("LazySchedule", pd.Series("fixed", index=frame.index))
    frame["AttackMode"] = [
        infer_attack_mode(lazy, gap, schedule)
        for lazy, gap, schedule in zip(frame["UseLazy"], frame["LazyHonestGap"], schedules)
    ]
    keys = RUN_COLUMNS + [name for name in OPTIONAL_SETTINGS + OPTIONAL_RUN_IDS if name in frame]
    if frame.duplicated(keys + ["ClientID"]).any():
        raise ValueError("Repeated client within the same experiment. Supply distinct task or campaign IDs.")
    observed = frame.groupby(keys, dropna=False)["ClientID"].transform("size")
    if observed.ne(frame["ClientNum"]).any():
        raise ValueError("Incomplete experiment: expected one contribution for each of the ClientNum clients.")
    frame["WithAttacker"] = pd.to_numeric(frame["WithAttacker"], errors="raise")
    if not np.isfinite(frame["WithAttacker"]).all():
        raise ValueError("WithAttacker must be finite for every client in a quantitative experiment.")
    if require_controls:
        honest = frame["ClientID"].ne(frame["AttackerID"])
        controls = pd.to_numeric(frame.loc[honest, "WithoutAttacker"], errors="raise")
        if not np.isfinite(controls).all():
            raise ValueError("WithoutAttacker must be finite for every matched honest client.")
    return frame, keys


def mean_group_columns(frame):
    keys = ["Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackMode"]
    settings = ["ContributionMethod", "UseHessian", "UseLazy", "LazyHonestGap", "LazyHonestPhase", "LazyFakePolicy"]
    return keys + [name for name in settings + OPTIONAL_SETTINGS if name in frame]

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


path = Path(__file__).resolve().parents[1] / "analysis" / "counterfactual_gap.py"
spec = importlib.util.spec_from_file_location("counterfactual_gap", path)
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(folder, *, lazy=True, method="example", values=(-0.1, 0.3, 0.2)):
    awards = folder / "combined_contributions_trial0.csv"
    rows = [
        dict(Dataset="adult", AttackMethod=method, Alpha=1000, ClientNum=3, AttackerID=0, TrialID=0,
             DetectingType="quantitative", DetectingMethod="shapley", UseAttack=True, UseHessian=False,
             UseLazy=lazy, LazySchedule="fixed", LazyHonestGap=6, LazyHonestPhase=0, LazyFakePolicy="attack",
             ClientID=i, WithAttacker=score, WithoutAttacker=0.95, WithAttackerMinMax=max(score, 0))
        for i, score in enumerate(values)
    ]
    pd.DataFrame(rows).to_csv(awards, index=False)
    stamp = {name: hashlib.sha256(name.encode()).hexdigest() for name in gap.DIGESTS}
    common = dict(RunID="award", AttackerID=0, Seed=42, UtilityUnit="accuracy_fraction", **stamp)
    audit = folder / "access.json"
    audit.write_text(json.dumps({"private_fit_count": 9 if lazy else 0}), encoding="utf-8")
    manifest = dict(**common, ContributionsCSV=awards.name, ContributionsSHA256=digest(awards),
                    DataAccess="intermittent_private_training" if lazy else "no_private_data",
                    PrivateFitCount=9 if lazy else 0, AccessEvidence=audit.name, AccessEvidenceSHA256=digest(audit))
    run_index = folder / "runs.csv"
    pd.DataFrame([manifest]).to_csv(run_index, index=False)
    ctrl = folder / "controls.csv"
    pd.DataFrame([dict(**common, World=w, WorldRunID="run-" + w, FinalUtility=v)
                  for w, v in (("honest_only_intermittent", 0.8), ("excluded", 0.85))]).to_csv(ctrl, index=False)
    return awards, run_index, ctrl


def test_signed_raw_gap_and_shared_denominator(tmp_path):
    _, run, ctrl = inputs(tmp_path)
    raw = gap.calculate(run, ctrl).iloc[0]
    assert raw.AwardedRawSV == pytest.approx(-0.1)
    assert raw.TotalSignedSV == pytest.approx(0.4)
    assert raw.TrueContributionRaw == pytest.approx(-0.05)
    assert raw.GapRaw == pytest.approx(-0.05)
    assert "GapShare" not in raw
    normalized = gap.calculate(run, ctrl, signed_share=True).iloc[0]
    assert normalized.GapShare == pytest.approx(-0.125)
    assert normalized.GapShare == pytest.approx(normalized.AwardedShare - normalized.TrueContributionShare)


@pytest.mark.parametrize("field", ["Seed", "PartitionSHA256", "InitialStateSHA256", "TrainingProtocolSHA256",
                                   "ScheduleSHA256", "UtilityProtocolSHA256", "AttackerID"])
def test_world_must_match_awarded_design(tmp_path, field):
    _, run, ctrl = inputs(tmp_path)
    table = pd.read_csv(ctrl, dtype=str)
    table.loc[0, field] = "2" if field in {"Seed", "AttackerID"} else "0" * 64
    table.to_csv(ctrl, index=False)
    with pytest.raises(ValueError, match=field):
        gap.calculate(run, ctrl)


def test_requires_two_distinct_observed_worlds(tmp_path):
    _, run, ctrl = inputs(tmp_path)
    with pytest.raises(ValueError, match="both recorded worlds"):
        gap.calculate(run)
    table = pd.read_csv(ctrl)
    table.loc[0, "WorldRunID"] = table.loc[1, "WorldRunID"]
    table.to_csv(ctrl, index=False)
    with pytest.raises(ValueError, match="distinct"):
        gap.calculate(run, ctrl)


def test_data_free_zero_requires_evidence_and_keeps_raw_gap_on_zero_total(tmp_path):
    awards, run, _ = inputs(tmp_path, lazy=False, values=(-0.25, 0, 0.25))
    record = gap.calculate(run, signed_share=True).iloc[0]
    assert record.TrueContributionRaw == 0 and record.GapRaw == -0.25
    assert pd.isna(record.GapShare) and record.ShareStatus == "zero_denominator"
    awards.write_bytes(awards.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="ContributionsSHA256"):
        gap.calculate(run)
    _, run, _ = inputs(tmp_path, lazy=False, method="synthetic_replacement")
    with pytest.raises(ValueError, match="data-free"):
        gap.calculate(run)


def test_multi_run_export_needs_unambiguous_selector(tmp_path):
    awards, run, _ = inputs(tmp_path, lazy=False)
    scores = pd.read_csv(awards)
    pd.concat([scores.assign(TrialID=0), scores.assign(TrialID=1)]).to_csv(awards, index=False)
    index = pd.read_csv(run, dtype=str)
    index["ContributionsSHA256"] = digest(awards)
    index.to_csv(run, index=False)
    with pytest.raises(ValueError, match="one matching"):
        gap.calculate(run)
    index["SettingJSON"] = '{"TrialID":1}'
    index.to_csv(run, index=False)
    assert gap.calculate(run).iloc[0].TrialID == 1
    (tmp_path / "access.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="AccessEvidenceSHA256"):
        gap.calculate(run)

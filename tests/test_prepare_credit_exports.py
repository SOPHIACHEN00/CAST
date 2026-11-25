import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "prepare_credit_exports.py"
spec = importlib.util.spec_from_file_location("credit_exports", SCRIPT)
exports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exports)


def write_tables(directory, q1_name, q2_name, methods_and_modes):
    directory.mkdir(parents=True, exist_ok=True)
    q1, q2 = [], []
    for method, mode in methods_and_modes:
        setting = dict(Dataset="adult", AttackMethod=method, Alpha=1, ClientNum=2, AttackMode=mode)
        q1.append({**setting, "Q1_AttackerRatio": 0.25})
        q2.append({**setting, "TrialID": 0, "AttackerID": 0, "ClientID": 1,
                   "WithAttacker": 0.75, "WithoutAttacker": 1.0, "Q2_AbsImpact": 0.25})
    if q1_name:
        pd.DataFrame(q1).to_csv(directory / q1_name, index=False)
    if q2_name:
        pd.DataFrame(q2).to_csv(directory / q2_name, index=False)


def test_source_selection_and_manifest_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(exports, "DATASETS", ("adult",))
    archive, derived = tmp_path / "archive", tmp_path / "selected"
    other = [("arima", "pure_fake"), ("random", "1real_5fake")]
    old = [(method, "pure_fake") for method in exports.LEGACY_PURE] + other
    write_tables(archive / "adult", "adultRQ1.csv", "adultRQ2_box.csv", old)
    external = archive / "Pure_Attack" / "RQ_Results"
    alternate = [(method, "pure_fake") for method in exports.EXTERNAL_PURE] + other
    write_tables(external / "combined", "pure_attack_RQ1.csv", None, alternate)
    write_tables(external / "adult", None, "adultRQ2_box.csv", alternate)
    write_tables(derived / "adult", "forecast_RQ1.csv", "forecast_RQ2_box.csv",
                 [("arima", "pure_fake"), ("arima", "1real_5fake")])
    manifest = pd.DataFrame([dict(Dataset="adult", LazySchedule="fixed", LazyHonestGap=6,
                                  LazyAttackMode="1real_5fake", Methods="arima", Alphas="1",
                                  ClientNums="2", RQ1GroupRows=2, RQ2BoxRows=2)])
    path = derived / "forecast_source_selection_manifest.csv"
    manifest.to_csv(path, index=False)
    one, two = exports.assemble_tables(archive, derived)
    assert len(one) == len(two) == 7
    assert list(one.AttackMethod) == [*exports.LEGACY_PURE, *exports.EXTERNAL_PURE, "arima", "arima"]
    assert list(two.AttackMode) == ["pure_fake"] * 6 + ["1real_5fake"]
    manifest["RQ2BoxRows"] = 1
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match="selection manifest"):
        exports.assemble_tables(archive, derived)

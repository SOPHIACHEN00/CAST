import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


path = Path(__file__).resolve().parents[1] / "analysis" / "paper_detection.py"
spec = importlib.util.spec_from_file_location("paper_detection", path)
detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection)


def round_rows(n, alpha, hit):
    rows = []
    for round_id in range(50):
        for client in range(n):
            designated = client == 1
            active = designated and round_id % 6 != 0
            skipped = round_id == 49
            alarm = (designated and (hit or not active)) or client == 0
            rows.append({
                "Dataset": "adult", "AttackRegime": "lazy", "AttackMethod": "synthetic_replacement",
                "DetectingMethod": "pass", "Alpha": alpha, "ClientNum": n, "TrialID": 0, "AttackerID": 1,
                "TaskID": f"task-{n}", "QualitativeAggregationMode": "passive", "LazySchedule": "fixed",
                "LazyHonestGap": 6, "LazyHonestPhase": 0, "Round": round_id, "ClientID": client,
                "IsDesignatedAttacker": int(designated), "GroundTruthActiveAttack": int(active),
                "DetectionSkipped": int(skipped), "RoundPredictedSuspicious": "" if skipped else str(int(alarm)),
                "DecisionMode": "audit", "DecisionValid": int(not skipped),
            })
    return pd.DataFrame(rows)


def test_evaluated_rounds_and_equal_setting_weights(tmp_path, monkeypatch):
    cell = ("adult", "lazy", "synthetic_replacement", "pass")
    monkeypatch.setattr(detection, "expected_settings", lambda panel: {cell + (1000, 4), cell + (1000, 8)})
    csv = tmp_path / "detection.csv"
    pd.concat([round_rows(4, 1000, True), round_rows(8, 1000, False)]).to_csv(csv, index=False)
    row = detection.summarize([csv], "pass").iloc[0]
    assert row.Settings == row.RecallSettings == row.FPRSettings == 2
    assert (row.TP, row.ActiveEvaluated, row.ActiveTotal) == (40, 80, 82)
    assert (row.FP, row.HonestEvaluated, row.HonestTotal) == (98, 490, 500)
    assert row.RecallPct == 50
    assert row.FPRPct == pytest.approx(100 * (1/3 + 1/7) / 2)
    with pytest.raises(ValueError, match="Duplicate setting"):
        detection.summarize([csv, csv], "pass")
    frame = pd.read_csv(csv)
    frame.loc[1, "ClientID"] = frame.loc[0, "ClientID"]
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="one row per submitted round and client"):
        detection.summarize([csv], "pass")


def test_missing_support_and_zero_valid_decisions(tmp_path, monkeypatch):
    csv = tmp_path / "detection.csv"
    rows = round_rows(4, 1000, True)
    rows.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Missing 239 of 240"):
        detection.summarize([csv], "pass")
    monkeypatch.setattr(detection, "expected_settings", lambda panel: {
        ("adult", "lazy", "synthetic_replacement", "pass", 1000, 4)
    })
    rows["DetectionSkipped"] = 1
    rows["RoundPredictedSuspicious"] = ""
    rows["DecisionValid"] = 0
    rows.to_csv(csv, index=False)
    result = detection.summarize([csv], "pass").iloc[0]
    assert result.RecallSettings == result.FPRSettings == 0
    assert pd.isna(result.RecallPct) and pd.isna(result.FPRPct)
    rows.loc[0, "RoundPredictedSuspicious"] = "0"
    rows.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="DetectionSkipped disagrees"):
        detection.summarize([csv], "pass")


def test_active_reporting_sources_from_merged_and_runner_exports(tmp_path, monkeypatch):
    paths = []
    settings = set()
    for method in ("wef", "s2_wef", "log_based_free_rider_detection"):
        frame = round_rows(4, 1000, False)
        frame["DetectingMethod"] = method
        settings.add(("adult", "lazy", "synthetic_replacement", method, 1000, 4))
        if method == "log_based_free_rider_detection":
            frame["ReportSource"] = "reused_last_honest"
            frame["DetectorDetailsJSON"] = '{"report_source": "reused_last_honest"}'
            # The generic runner persists source in JSON; the campaign also exports a CSV column.
            frame.loc[frame["Round"].eq(1), "ReportSource"] = ""
            frame.loc[frame["Round"].eq(2), ["ReportSource", "DetectorDetailsJSON"]] = ""
        else:
            frame["WEFPayloadSource"] = "observed_local_training"
            frame.loc[frame["Round"].eq(1), "WEFPayloadSource"] = "counterfeit_from_submitted_update"
        path = tmp_path / f"{method}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    monkeypatch.setattr(detection, "expected_settings", lambda panel: settings)
    result = detection.summarize(paths, "patches").set_index("DetectingMethod")
    for method in ("wef", "s2_wef"):
        assert json.loads(result.loc[method, "ActiveReportSources"]) == {
            "observed_local_training": 40, "counterfeit_from_submitted_update": 1,
        }
    assert json.loads(result.loc["log_based_free_rider_detection", "ActiveReportSources"]) == {
        "reused_last_honest": 40, "unrecorded": 1,
    }

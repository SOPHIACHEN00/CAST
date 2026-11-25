from io import StringIO
from pathlib import Path
import runpy
import subprocess
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "timing_summary.py"
timing = runpy.run_path(str(SCRIPT))


def test_honest_probe_mean_includes_every_observed_round(tmp_path):
    frame = pd.DataFrame({"Dataset": "example", "TrialID": 0, "RunType": "no_attack", "ClientNum": 4,
                          "LocalEpochs": 1, "RoundIndex": [0, 1, 2], "GlobalTrainingSec": [6, 1, 2]})
    path = tmp_path / "rounds.csv"
    frame.to_csv(path, index=False)
    run = subprocess.run([sys.executable, str(SCRIPT), "honest", str(path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    row = pd.read_csv(StringIO(run.stdout)).iloc[0]
    assert (row.ObservedFLRounds, row.GlobalTrainingTotalSec, row.MeanSecPerObservedFLRound) == (3, 9, 3)
    with pytest.raises(ValueError, match="same no-attack run"):
        timing["summarize_honest"]([path, path])
    frame.loc[1, "RoundIndex"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="round indices"):
        timing["summarize_honest"]([path])


def test_task_total_has_distinct_denominators_and_legacy_counts_stay_missing(tmp_path):
    path = tmp_path / "time.csv"
    pd.DataFrame({"AttackGenerationWithAttackerTotalSec": [8, 0], "FakeRoundsTimed": [2, 0]}).to_csv(path, index=False)
    rows = timing["summarize_generation"](path, 4)
    assert list(rows.GenerationTotalSec) == [8, 0]
    assert list(rows.GenerationSecPerFLRound) == [2, 0]
    assert rows.GenerationSecPerTimedFakeRound.iloc[0] == 4
    assert pd.isna(rows.GenerationSecPerTimedFakeRound.iloc[1])
    pd.DataFrame({"UseAttack": [True, False], "WithAttackerSec": [8, "N/A"]}).to_csv(path, index=False)
    rows = timing["summarize_generation"](path, None)
    assert list(rows.GenerationTotalSec) == [8]
    assert rows[["GenerationSecPerFLRound", "GenerationSecPerTimedFakeRound"]].isna().all().all()


def test_grid_takes_the_median_of_setting_means(tmp_path):
    frame = pd.DataFrame({"Dataset": "example", "Mode": "Pure", "AttackMethod": "recorded", "ClientNum": 4,
                          "Gap": 0, "Alpha": [1, 1, 1, 2], "TaskKey": ["a", "b", "c", "d"], "TimeSec": [0, 0, 6, 8]})
    path = tmp_path / "tasks.csv"
    frame.to_csv(path, index=False)
    row = timing["summarize_grid"](path, 2).iloc[0]
    assert (row.Settings, row.Tasks, row.MedianSettingMeanTotalSec, row.MedianSettingMeanSecPerFLRound) == (2, 4, 5, 2.5)
    assert pd.isna(timing["summarize_grid"](path, None).MedianSettingMeanSecPerFLRound.iloc[0])
    frame.TimeSec = 42.26844
    frame.to_csv(path, index=False)
    row = timing["summarize_grid"](path, 50).iloc[0]
    assert row.MedianSettingMeanSecPerFLRound == pytest.approx(42.26844 / 50)
    assert row.ArchiveDisplaySecPerFLRound == pytest.approx(0.846)
    frame.loc[1, "TaskKey"] = "a"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique TaskKey"):
        timing["summarize_grid"](path, 2)

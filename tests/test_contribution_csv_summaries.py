import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis" / "Analysis Code"
Q1 = "Q1/compute_q1_summary.py"
Q2 = "Q2/compute_q2_honest_impact_summary.py"
HONEST_CLIENTS = "Q2/compute_q2_honest_client_boxplot.py"


def fixture_rows(attacker_score=0.25, honest_baseline=0.5, **metadata):
    setting = {
        "Dataset": "example", "AttackMethod": "recorded", "Alpha": 1.0,
        "ClientNum": 2, "AttackerID": 0, "TrialID": 0, "UseAttack": True,
        "UseHessian": False, "UseLazy": True, "LazyHonestGap": 3,
        "LazyHonestPhase": 0, "LazyFakePolicy": "attack", **metadata,
    }
    return [
        {**setting, "ClientID": 0, "WithAttacker": attacker_score, "WithoutAttacker": "N/A"},
        {**setting, "ClientID": 1, "WithAttacker": 1 - attacker_score, "WithoutAttacker": honest_baseline},
    ]


def run_script(script, source, output):
    return subprocess.run(
        [sys.executable, str(ANALYSIS / script), str(source), "-o", str(output)],
        cwd=source.parent, capture_output=True, text=True,
    )


@pytest.mark.parametrize("method_column", ["ContributionMethod", "DetectingMethod"])
def test_old_and_new_runner_headers_give_the_same_metrics(tmp_path, method_column):
    source = tmp_path / "combined.csv"
    pd.DataFrame(fixture_rows(**{method_column: "shapley"})).to_csv(source, index=False)
    for script, metric in [(Q1, "Q1_AttackerRatio"), (Q2, "Q2_AvgAbsImpact"), (HONEST_CLIENTS, "Q2_AbsImpact")]:
        output = tmp_path / f"{metric}.csv"
        run = run_script(script, source, output)
        assert run.returncode == 0, run.stderr
        summary = pd.read_csv(output)
        assert list(summary["ContributionMethod"]) == ["shapley"]
        assert list(summary["AttackMode"]) == ["1real_2fake"]
        assert list(summary[metric]) == pytest.approx([0.25])
        if script == HONEST_CLIENTS:
            assert list(summary["ClientID"]) == [1]


def test_configurations_stay_separate_through_group_means(tmp_path):
    configurations = [
        ("fixed", 1.0, "shapley", 0.2),
        ("probabilistic", 0.5, "shapley", 0.6),
        ("probabilistic", 0.8, "shapley", 0.8),
        ("probabilistic", 0.8, "sampling_shapley", 0.9),
    ]
    rows = []
    for schedule, probability, method, score in configurations:
        rows.extend(fixture_rows(score, 1.0, LazySchedule=schedule, AttackProbability=probability,
                                 DetectingType="quantitative", DetectingMethod=method))
    source = tmp_path / "combined.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    for script, merger, metric in [
        (Q1, "Q1/q1_avg_steal_summary.py", "Q1_AttackerRatio"),
        (Q2, "Q2/q2_honest_impact_group_mean.py", "Q2_AvgAbsImpact"),
    ]:
        detail = tmp_path / f"{metric}.csv"
        mean = tmp_path / f"{metric}_mean.csv"
        for entry, input_path, output in [(script, source, detail), (merger, detail, mean)]:
            run = run_script(entry, input_path, output)
            assert run.returncode == 0, run.stderr
            result = pd.read_csv(output)
            assert len(result) == len(configurations)
            assert set(result["AttackMode"]) == {"1real_2fake", "probabilistic"}
            index = result.set_index(["LazySchedule", "AttackProbability", "ContributionMethod"])
            for schedule, probability, method, score in configurations:
                assert index.at[(schedule, probability, method), metric] == pytest.approx(score)


def test_headers_can_coexist_but_conflicting_methods_are_rejected(tmp_path):
    rows = fixture_rows()
    rows[0]["ContributionMethod"] = "shapley"
    rows[1]["DetectingMethod"] = "shapley"
    source = tmp_path / "combined.csv"
    output = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    run = run_script(Q1, source, output)
    assert run.returncode == 0, run.stderr
    assert list(pd.read_csv(output)["Q1_AttackerRatio"]) == pytest.approx([0.25])

    rows[0]["DetectingMethod"] = "sampling_least_core"
    pd.DataFrame(rows).to_csv(source, index=False)
    run = run_script(Q2, source, tmp_path / "conflict.csv")
    assert run.returncode != 0
    assert "ContributionMethod and DetectingMethod disagree" in run.stderr
    assert not (tmp_path / "conflict.csv").exists()


@pytest.mark.parametrize("script,problem,expected", [
    (Q1, "missing_client", "Incomplete experiment"),
    (Q1, "missing_score", "WithAttacker must be finite"),
    (Q2, "duplicate", "Repeated client"),
    (Q2, "missing_control", "WithoutAttacker must be finite"),
    (HONEST_CLIENTS, "missing_control", "WithoutAttacker must be finite"),
])
def test_incomplete_runs_fail_before_writing(tmp_path, script, problem, expected):
    rows = fixture_rows(DetectingMethod="shapley")
    if problem == "missing_client":
        rows.pop()
    elif problem == "missing_score":
        rows[1]["WithAttacker"] = "N/A"
    elif problem == "duplicate":
        rows.append(rows[1].copy())
    else:
        rows[1]["WithoutAttacker"] = "N/A"
    source = tmp_path / "combined.csv"
    output = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    run = run_script(script, source, output)
    assert run.returncode != 0
    assert expected in run.stderr
    assert not output.exists()

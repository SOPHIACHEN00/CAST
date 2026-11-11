"""Integration tests for the read-only Forecast/Shapley result merger."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = REPO_ROOT / "script" / "merge_forecast_shapley_results.py"

COMMON_HEADERS = [
    "Dataset",
    "AttackMethod",
    "Alpha",
    "ClientNum",
    "AttackerID",
    "TrialID",
    "UseAttack",
    "DetectingType",
    "DetectingMethod",
    "UseHessian",
    "UseLazy",
    "LazySchedule",
    "AttackProbability",
    "LazyHonestGap",
    "LazyHonestPhase",
    "LazyFakePolicy",
]

ARTIFACT_LAYOUT = {
    "accuracy": (
        "Accuracy_quantitative_shapley",
        "global_accuracy_log_trial0.csv",
        COMMON_HEADERS + ["Round", "WithAttackerAccuracy", "WithoutAttackerAccuracy"],
    ),
    "contribution": (
        "Detecting_quantitative_shapley",
        "combined_contributions_trial0.csv",
        COMMON_HEADERS
        + [
            "ClientID",
            "WithAttacker",
            "WithoutAttacker",
            "WithAttackerMinMax",
            "WithoutAttackerMinMax",
        ],
    ),
    "time": (
        "Time_quantitative_shapley",
        "attacking_time_log_trial0.csv",
        COMMON_HEADERS + ["WithAttackerSec", "WithoutAttackerSec"],
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class MergeForecastShapleyResultsTest(unittest.TestCase):
    maxDiff = None

    def _create_task(
        self,
        *,
        results_root: Path,
        runs_root: Path,
        campaign: str,
        task_label: str,
        task_id: str,
        variant: str,
        attack_method: str,
        use_lazy: bool,
        lazy_schedule: str,
        probability: str,
        include_time: bool = True,
        launcher_status: str = "success protocol_pass exit_code=0",
    ) -> List[Path]:
        task_root = results_root / campaign / task_label
        task_root.mkdir(parents=True)

        cli_args = {
            "dataset": "tictactoe",
            "alpha": 10,
            "client_num": 2,
            "attacker_id": 0,
            "attack_method": attack_method,
            "trial_id": 0,
            "num_epoch": 50,
            "num_local_epochs": 1,
            "random_round": 10,
            "use_attack": True,
            "use_hessian": True,
            "lambda_h": 0.9,
            "use_lazy": use_lazy,
            "lazy_schedule": lazy_schedule,
            # This is intentionally 1.0 even for the paired fixed run.  The
            # launcher metadata below is the source of ConfiguredProbability.
            "attack_probability": 1.0,
            "lazy_honest_gap": 2,
            "lazy_honest_phase": "honest_first",
            "lazy_fake_policy": "predict",
        }
        provenance_path = task_root / "manifests" / "run_manifest_trial0.json"
        provenance_path.parent.mkdir(parents=True)
        provenance_path.write_text(
            json.dumps(
                {
                    "created_at_utc": "2026-08-18T00:00:00Z",
                    "entry_sha256": "entry-sha",
                    "attack_module_sha256": "attack-sha",
                    "cli_args": cli_args,
                    "attack_parameters": {"lambda_h": 0.9},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        launcher_task_root = runs_root / campaign / "tasks" / task_label
        launcher_task_root.mkdir(parents=True)
        (launcher_task_root / "manifest.txt").write_text(
            "\n".join(
                [
                    f"task_id={task_id}",
                    f"variant={variant}",
                    f"probability={probability}",
                    "command_argv:",
                    "python FLCE_updated.py",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (launcher_task_root / "status.txt").write_text(
            launcher_status + "\n", encoding="utf-8"
        )

        common = {
            "Dataset": "tictactoe",
            "AttackMethod": attack_method,
            "Alpha": "10",
            "ClientNum": "2",
            "AttackerID": "0",
            "TrialID": "0",
            "UseAttack": "True",
            "DetectingType": "quantitative",
            "DetectingMethod": "shapley",
            "UseHessian": "True",
            "UseLazy": str(use_lazy),
            "LazySchedule": lazy_schedule,
            "AttackProbability": "1.0",
            "LazyHonestGap": "2",
            "LazyHonestPhase": "honest_first",
            "LazyFakePolicy": "predict",
        }

        accuracy_rows = [
            {
                **common,
                "Round": round_id,
                "WithAttackerAccuracy": f"{0.70 + round_id / 1000:.3f}",
                "WithoutAttackerAccuracy": f"{0.69 + round_id / 1000:.3f}",
            }
            for round_id in range(50)
        ]
        contribution_rows = [
            {
                **common,
                "ClientID": client_id,
                "WithAttacker": f"{0.001 + client_id / 1000:.4f}",
                "WithoutAttacker": f"{0.0008 + client_id / 1000:.4f}",
                "WithAttackerMinMax": f"{0.1 + client_id / 10:.2f}",
                "WithoutAttackerMinMax": f"{0.08 + client_id / 10:.2f}",
            }
            for client_id in range(2)
        ]
        time_rows = [
            {**common, "WithAttackerSec": "12.5", "WithoutAttackerSec": "10.1"}
        ]

        created = [provenance_path]
        for kind, rows in (
            ("accuracy", accuracy_rows),
            ("contribution", contribution_rows),
            ("time", time_rows),
        ):
            if kind == "time" and not include_time:
                continue
            directory, filename, headers = ARTIFACT_LAYOUT[kind]
            csv_path = task_root / directory / filename
            _write_csv(csv_path, headers, rows)
            created.append(csv_path)
        return created

    def test_validate_merge_archive_audit_and_source_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            results_root = temp_root / "Results"
            runs_root = temp_root / ".shapley_runs"
            output_root = temp_root / "merged_export"
            results_root.mkdir()

            pure_campaign = (
                "forecast_tictactoe_pure_shapley_20260818-000001_pid100"
            )
            lazy_campaign = (
                "forecast_tictactoe_lazy_shapley_20260818-000002_pid200"
            )
            pure_task = (
                "task_000001_tictactoe_pure_arima_offline_"
                "a10_n2_pure_gna_pna_trial0"
            )
            lazy_task = (
                "task_000002_tictactoe_lazy_moirai_online_"
                "a10_n2_fixed_g2_p0p50_trial0"
            )
            incomplete_task = (
                "task_000003_tictactoe_lazy_arima_adaptive_"
                "a10_n2_probabilistic_g2_p0p50_trial0"
            )

            source_paths: List[Path] = []
            source_paths.extend(
                self._create_task(
                    results_root=results_root,
                    runs_root=runs_root,
                    campaign=pure_campaign,
                    task_label=pure_task,
                    task_id="000001",
                    variant="arima_offline",
                    attack_method="arima",
                    use_lazy=False,
                    lazy_schedule="pure",
                    probability="na",
                )
            )
            source_paths.extend(
                self._create_task(
                    results_root=results_root,
                    runs_root=runs_root,
                    campaign=lazy_campaign,
                    task_label=lazy_task,
                    task_id="000002",
                    variant="moirai_online",
                    attack_method="moirai_online",
                    use_lazy=True,
                    lazy_schedule="fixed",
                    probability="0.50",
                )
            )
            source_paths.extend(
                self._create_task(
                    results_root=results_root,
                    runs_root=runs_root,
                    campaign=lazy_campaign,
                    task_label=incomplete_task,
                    task_id="000003",
                    variant="arima_adaptive",
                    attack_method="arima_adaptive",
                    use_lazy=True,
                    lazy_schedule="probabilistic",
                    probability="0.50",
                    include_time=False,
                )
            )
            (runs_root / pure_campaign / "tasks.tsv").write_text(
                "planned-pure-task\n", encoding="utf-8"
            )
            (runs_root / lazy_campaign / "tasks.tsv").write_text(
                "planned-lazy-task-1\nplanned-lazy-task-2\n", encoding="utf-8"
            )
            original_bytes = {path: path.read_bytes() for path in source_paths}
            original_hashes = {path: _sha256(path) for path in source_paths}

            validate = subprocess.run(
                [
                    sys.executable,
                    str(MERGE_SCRIPT),
                    "--results-root",
                    str(results_root),
                    "--runs-root",
                    str(runs_root),
                    "--datasets",
                    "tictactoe",
                    "--validate-only",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
            self.assertIn("Tasks: 3 (complete=2, skipped=1)", validate.stdout)
            self.assertIn(
                "tictactoe: tasks=3 complete=2 accuracy=100 contribution=4 time=2",
                validate.stdout,
            )
            self.assertFalse(output_root.exists())

            merge = subprocess.run(
                [
                    sys.executable,
                    str(MERGE_SCRIPT),
                    "--results-root",
                    str(results_root),
                    "--runs-root",
                    str(runs_root),
                    "--output-root",
                    str(output_root),
                    "--datasets",
                    "tictactoe",
                    "--archive",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(merge.returncode, 0, merge.stdout + merge.stderr)

            accuracy_rows = _read_csv(
                output_root / "tictactoe" / "accuracy" / "accuracy_all_tasks.csv"
            )
            contribution_rows = _read_csv(
                output_root
                / "tictactoe"
                / "contribution"
                / "contribution_all_tasks.csv"
            )
            time_rows = _read_csv(
                output_root / "tictactoe" / "time" / "time_all_tasks.csv"
            )
            self.assertEqual(len(accuracy_rows), 100)
            self.assertEqual(len(contribution_rows), 4)
            self.assertEqual(len(time_rows), 2)

            lazy_accuracy = [
                row for row in accuracy_rows if row["SourceMode"] == "lazy"
            ]
            pure_accuracy = [
                row for row in accuracy_rows if row["SourceMode"] == "pure"
            ]
            self.assertEqual(len(lazy_accuracy), 50)
            self.assertTrue(
                all(row["ConfiguredProbability"] == "0.50" for row in lazy_accuracy)
            )
            self.assertTrue(
                all(row["AttackProbability"] == "1.0" for row in lazy_accuracy)
            )
            self.assertTrue(
                all(row["ConfiguredProbability"] == "" for row in pure_accuracy)
            )

            expected_uids = {
                f"{pure_campaign}/{pure_task}",
                f"{lazy_campaign}/{lazy_task}",
            }
            self.assertEqual(
                {row["SourceTaskUID"] for row in accuracy_rows}, expected_uids
            )
            self.assertEqual(
                {row["SourceTaskUID"] for row in contribution_rows}, expected_uids
            )
            self.assertEqual({row["SourceTaskUID"] for row in time_rows}, expected_uids)

            task_rows = _read_csv(output_root / "tictactoe" / "tasks.csv")
            self.assertEqual(len(task_rows), 2)
            self.assertEqual({row["SourceTaskUID"] for row in task_rows}, expected_uids)

            inventory_rows = _read_csv(output_root / "audit" / "task_inventory.csv")
            self.assertEqual(len(inventory_rows), 3)
            incomplete_inventory = next(
                row
                for row in inventory_rows
                if row["SourceTaskUID"] == f"{lazy_campaign}/{incomplete_task}"
            )
            self.assertEqual(incomplete_inventory["MergeStatus"], "skipped_incomplete")
            self.assertEqual(incomplete_inventory["TimeRows"], "0")

            issue_rows = _read_csv(output_root / "audit" / "issues.csv")
            self.assertTrue(
                any(
                    row["SourceTaskUID"]
                    == f"{lazy_campaign}/{incomplete_task}"
                    and row["Code"] == "time_file_count"
                    for row in issue_rows
                )
            )
            summary = json.loads(
                (output_root / "merge_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["task_count"], 3)
            self.assertEqual(summary["complete_task_count"], 2)
            self.assertEqual(summary["skipped_task_count"], 1)

            archive_path = output_root.with_suffix(".zip")
            self.assertTrue(archive_path.is_file())
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            archive_prefix = output_root.name
            self.assertIn(
                f"{archive_prefix}/tictactoe/accuracy/accuracy_all_tasks.csv", names
            )
            self.assertIn(
                f"{archive_prefix}/tictactoe/contribution/contribution_all_tasks.csv",
                names,
            )
            self.assertIn(f"{archive_prefix}/tictactoe/tasks.csv", names)
            self.assertIn(f"{archive_prefix}/audit/task_inventory.csv", names)

            for source_path in source_paths:
                self.assertEqual(source_path.read_bytes(), original_bytes[source_path])
                self.assertEqual(_sha256(source_path), original_hashes[source_path])

    def test_protocol_failed_task_is_audited_but_not_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            results_root = temp_root / "Results"
            runs_root = temp_root / ".shapley_runs"
            output_root = temp_root / "merged_export"
            results_root.mkdir()
            campaign = "forecast_tictactoe_pure_shapley_20260818-000003_pid300"
            task_label = (
                "task_000001_tictactoe_pure_arima_offline_"
                "a10_n2_pure_gna_pna_trial0"
            )
            self._create_task(
                results_root=results_root,
                runs_root=runs_root,
                campaign=campaign,
                task_label=task_label,
                task_id="000001",
                variant="arima_offline",
                attack_method="arima",
                use_lazy=False,
                lazy_schedule="pure",
                probability="na",
                launcher_status="failed protocol_invalid exit_code=3",
            )
            (runs_root / campaign / "tasks.tsv").write_text(
                "planned-task\n", encoding="utf-8"
            )

            merge = subprocess.run(
                [
                    sys.executable,
                    str(MERGE_SCRIPT),
                    "--results-root",
                    str(results_root),
                    "--runs-root",
                    str(runs_root),
                    "--output-root",
                    str(output_root),
                    "--datasets",
                    "tictactoe",
                    "--allow-partial-matrix",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(merge.returncode, 0, merge.stdout + merge.stderr)
            summary = json.loads(
                (output_root / "merge_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["complete_task_count"], 0)
            self.assertEqual(summary["skipped_task_count"], 1)
            issues = _read_csv(output_root / "audit" / "issues.csv")
            self.assertTrue(
                any(row["Code"] == "launcher_status_not_success" for row in issues)
            )
            self.assertFalse((output_root / "tictactoe" / "tasks.csv").exists())


if __name__ == "__main__":
    unittest.main()

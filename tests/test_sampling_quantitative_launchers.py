import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"


DATASETS = ("tictactoe", "adult", "dota2", "mnist", "imdb")
METHODS = ("sampling_shapley", "sampling_least_core")


class SamplingLauncherContractTests(unittest.TestCase):
    def test_ten_dataset_method_wrappers_exist(self):
        for dataset in DATASETS:
            for method in METHODS:
                path = SCRIPT_DIR / f"run_{dataset}_{method}.sh"
                self.assertTrue(path.is_file(), path)
                source = path.read_text(encoding="utf-8")
                self.assertIn(f"SAMPLING_DATASET={dataset}", source)
                self.assertIn(f"SAMPLING_METHOD={method}", source)

    def test_every_launcher_is_standalone_and_contains_the_full_campaign_logic(self):
        for dataset in DATASETS:
            for method in METHODS:
                path = SCRIPT_DIR / f"run_{dataset}_{method}.sh"
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("quantitative_sampling_launcher_common.inc", source)
                self.assertNotIn("source \"$(cd", source)
                self.assertIn("LAZY_GAPS=(3 6 10)", source)
                self.assertIn("for attacker_id", source)
                self.assertIn("--lazy_schedule fixed", source)
                self.assertIn("--use_last_round_sv", source)
                self.assertIn("main \"$@\"", source)
                self.assertIn("DEFAULT_WORKERS=10", source)
                self.assertIn("DEFAULT_THREADS=15", source)

    def test_every_launcher_invokes_the_category_result_merger(self):
        for dataset in DATASETS:
            for method in METHODS:
                path = SCRIPT_DIR / f"run_{dataset}_{method}.sh"
                source = path.read_text(encoding="utf-8")
                self.assertIn('MERGER_SCRIPT="${SCRIPT_DIR}/merge_quantitative_sampling_results.py"', source)
                self.assertIn("run_merger", source)


class QuantitativeSamplingMergerTests(unittest.TestCase):
    def test_merger_consolidates_task_artifacts_into_category_csvs(self):
        merger = SCRIPT_DIR / "merge_quantitative_sampling_results.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            campaign = temp / "campaign"
            public = temp / "public"
            task_id = "q000001_tictactoe_t0_a10_n2_att0_pure_advanced_scaled_delta"
            task = campaign / "tasks" / task_id
            result = task / "results"
            result.mkdir(parents=True)

            plan_header = [
                "TaskID", "Dataset", "TrialID", "Alpha", "ClientNum", "AttackerID",
                "AttackRegime", "AttackMethod", "LazyGap", "PlannedStatus", "Reason",
            ]
            plan_row = [
                task_id, "tictactoe", "0", "10", "2", "0", "pure",
                "advanced_scaled_delta", "N/A", "planned", "",
            ]
            campaign.mkdir(parents=True, exist_ok=True)
            with (campaign / "planned.tsv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(plan_header)
                writer.writerow(plan_row)

            manifest_header = [
                "TaskID", "Dataset", "TrialID", "Alpha", "ClientNum", "AttackerID",
                "AttackRegime", "AttackMethod", "LazyGap", "GPU",
                "CPUThreadsPerWorker", "SamplingMethod", "CommandArgv", "ResultsRoot",
            ]
            with (task / "manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(manifest_header)
                writer.writerow([
                    task_id, "tictactoe", "0", "10", "2", "0", "pure",
                    "advanced_scaled_delta", "N/A", "0", "2", "sampling_shapley",
                    "python FLCE_updated.py", str(result),
                ])
            (task / "status.tsv").write_text("success\t0\t\n", encoding="utf-8")

            accuracy_header = [
                "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
                "UseAttack", "DetectingType", "DetectingMethod", "UseHessian", "UseLazy",
                "LazySchedule", "AttackProbability", "LazyHonestGap", "LazyHonestPhase",
                "LazyFakePolicy", "Round", "WithAttackerAccuracy", "WithoutAttackerAccuracy",
            ]
            accuracy_dir = result / "Accuracy_quantitative_sampling_shapley"
            accuracy_dir.mkdir()
            with (accuracy_dir / "global_accuracy_log_trial0.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(accuracy_header)
                writer.writerow([
                    "tictactoe", "advanced_scaled_delta", "10", "2", "0", "0", "1",
                    "quantitative", "sampling_shapley", "0", "0", "fixed", "1.0", "3", "0",
                    "attack", "0", "0.5", "0.5",
                ])

            contribution_header = [
                "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
                "UseAttack", "DetectingType", "DetectingMethod", "UseHessian", "UseLazy",
                "LazySchedule", "AttackProbability", "LazyHonestGap", "LazyHonestPhase",
                "LazyFakePolicy", "ClientID", "WithAttacker", "WithoutAttacker",
                "WithAttackerMinMax", "WithoutAttackerMinMax",
            ]
            contribution_dir = result / "Detecting_quantitative_sampling_shapley"
            contribution_dir.mkdir()
            with (contribution_dir / "combined_contributions_trial0.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(contribution_header)
                writer.writerow([
                    "tictactoe", "advanced_scaled_delta", "10", "2", "0", "0", "1",
                    "quantitative", "sampling_shapley", "0", "0", "fixed", "1.0", "3", "0",
                    "attack", "0", "0.1", "N/A", "0.0", "N/A",
                ])
                writer.writerow([
                    "tictactoe", "advanced_scaled_delta", "10", "2", "0", "0", "1",
                    "quantitative", "sampling_shapley", "0", "0", "fixed", "1.0", "3", "0",
                    "attack", "1", "0.2", "N/A", "1.0", "N/A",
                ])

            time_header = [
                "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
                "UseAttack", "DetectingType", "DetectingMethod", "UseHessian", "UseLazy",
                "LazySchedule", "AttackProbability", "LazyHonestGap", "LazyHonestPhase",
                "LazyFakePolicy", "WithAttackerSec", "WithoutAttackerSec",
            ]
            time_dir = result / "Time_quantitative_sampling_shapley"
            time_dir.mkdir()
            with (time_dir / "attacking_time_log_trial0.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(time_header)
                writer.writerow([
                    "tictactoe", "advanced_scaled_delta", "10", "2", "0", "0", "1",
                    "quantitative", "sampling_shapley", "0", "0", "fixed", "1.0", "3", "0",
                    "attack", "1.0", "1.0",
                ])

            diagnostic_header = [
                "Dataset", "AttackMethod", "AttackRegime", "Alpha", "ClientNum", "AttackerID",
                "TrialID", "DetectingMethod", "World", "ResolvedSeed", "RequestedSamplingBudget",
                "EffectiveSamplingBudget", "SamplingBudgetUnit", "Tolerance", "LeastCoreEpsilon",
                "ElapsedSeconds",
            ]
            diagnostic_dir = contribution_dir
            with (diagnostic_dir / "quantitative_diagnostics_trial0.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(diagnostic_header)
                writer.writerow([
                    "tictactoe", "advanced_scaled_delta", "pure", "10", "2", "0", "0",
                    "sampling_shapley", "with_attacker", "42", "auto", "1", "permutations",
                    "1e-6", "N/A", "0.1",
                ])
                writer.writerow([
                    "tictactoe", "advanced_scaled_delta", "pure", "10", "1", "0", "0",
                    "sampling_shapley", "without_attacker", "42", "auto", "1", "permutations",
                    "1e-6", "N/A", "0.1",
                ])

            completed = subprocess.run(
                [sys.executable, str(merger), str(campaign), str(public), "tictactoe", "sampling_shapley", "1"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            audit = ""
            audit_path = public / "audit" / "tasks.csv"
            if audit_path.is_file():
                audit = audit_path.read_text(encoding="utf-8")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr + audit)
            self.assertEqual(
                sorted(path.relative_to(public).as_posix() for path in public.rglob("*.csv")),
                [
                    "accuracy/accuracy.csv",
                    "audit/tasks.csv",
                    "contribution/contribution.csv",
                    "diagnostics/diagnostics.csv",
                    "time/time.csv",
                ],
            )


if __name__ == "__main__":
    unittest.main()

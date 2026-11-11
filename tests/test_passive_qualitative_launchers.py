import collections
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"

PROFILES = {
    "tictactoe": {
        "file": "run_tictactoe_passive_qualitative.sh",
        "alphas": {"10", "100", "1000", "20000"},
        "gpu": "1", "workers": "15", "threads": "15",
        "pure": {"moirai_online", "advanced_scaled_delta"},
        "lazy": {"moirai_online", "synthetic_replacement"},
        "arima": ("0.005", "0.016"), "moirai": ("0.05", "0.025"),
    },
    "adult": {
        "file": "run_adult_passive_qualitative.sh",
        "alphas": {"1000", "100000", "1000000", "20000000"},
        "gpu": "2", "workers": "15", "threads": "15",
        "pure": {"moirai_online", "adam_local_deviation"},
        "lazy": {"arima_adaptive", "synthetic_replacement"},
        "arima": ("0.005", "0.08"), "moirai": ("0.003", "0.12"),
    },
    "dota2": {
        "file": "run_dota2_passive_qualitative.sh",
        "alphas": {"10000", "1000000", "50000000", "200000000"},
        "gpu": "3", "workers": "15", "threads": "15",
        "pure": {"arima_adaptive", "dkfr"},
        "lazy": {"synthetic_replacement", "moirai_online"},
        "arima": ("0.005", "0.18"), "moirai": ("0.001", "0.25"),
    },
    "mnist": {
        "file": "run_mnist_passive_qualitative.sh",
        "alphas": {"10", "1000", "100000", "10000000"},
        "gpu": "0", "workers": "12", "threads": "10",
        "pure": {"moirai_online_mnist_imdb_layerwise", "dkfr"},
        "lazy": {"synthetic_replacement", "moirai_online_mnist_imdb_layerwise"},
        "arima": ("0.05", "0.05"), "moirai": ("0.003", "0.06"),
    },
    "imdb": {
        "file": "run_imdb_passive_qualitative.sh",
        "alphas": {"10", "100", "1000", "200000"},
        "gpu": "1", "workers": "12", "threads": "10",
        "pure": {"dkfr", "moirai_online_mnist_imdb_layerwise"},
        "lazy": {"synthetic_replacement", "arima_adaptive_mnist_imdb_layerwise"},
        "arima": ("0.10", "0.015"), "moirai": ("0.01", "0.025"),
    },
}

DETECTORS = {"std_dagmm", "delta_dagmm", "wef", "s2_wef"}
HESSIAN_METHODS = {
    "arima_adaptive", "arima_adaptive_mnist_imdb_layerwise",
    "moirai_online", "moirai_online_mnist_imdb_layerwise",
}
LAYERWISE_METHODS = {
    "arima_adaptive_mnist_imdb_layerwise",
    "moirai_online_mnist_imdb_layerwise",
}

DKFR = {
    "dota2": ("32", "2", "2", "5e-4", "5e-4", "2", "0.1", "1", "1", "128", "1", "0", "0", "0", "0", "5", "0", "3"),
    "mnist": ("32", "2", "2", "1e-5", "5e-6", "12", "0", "0.05", "1", "128", "0", "1", "5", "8", "0.15", "0.2", "0.1", "2"),
    "imdb": ("32", "2", "2", "5e-5", "5e-5", "4", "0", "0.1", "1", "128", "0", "1", "2", "4", "0", "0.5", "0.1", "2"),
}
DKFR_OPTIONS = (
    "--dkfr_synth_batch_size", "--dkfr_generator_steps", "--dkfr_kd_steps",
    "--dkfr_lr_g", "--dkfr_lr_student", "--dkfr_temperature",
    "--dkfr_alpha_feat", "--dkfr_alpha_kl", "--dkfr_alpha_cls",
    "--dkfr_hidden_dim", "--dkfr_lambda_kl_g", "--dkfr_kl_cap",
    "--dkfr_student_kl_cap", "--dkfr_student_loss_cap",
    "--dkfr_teacher_conf_min", "--dkfr_grad_clip",
    "--dkfr_max_update_norm", "--dkfr_guard_reset_patience",
)


def option_value(argv: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1:
        raise AssertionError(f"expected one {option}, found {len(positions)}")
    if positions[0] + 1 >= len(argv):
        raise AssertionError(f"{option} has no value")
    return argv[positions[0] + 1]


class PassiveQualitativeLauncherContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = [os.environ.get("BASH_EXE"), os.environ.get("BASH"), shutil.which("bash")]
        git = shutil.which("git")
        if git:
            candidates.append(str(pathlib.Path(git).resolve().parent.parent / "bin" / "bash.exe"))
        cls.bash = None
        for value in candidates:
            if not value or not pathlib.Path(value).is_file():
                continue
            probe = subprocess.run(
                [value, "--version"], text=True, capture_output=True, check=False
            )
            if probe.returncode == 0 and "GNU bash" in probe.stdout + probe.stderr:
                cls.bash = value
                break
        if cls.bash is None:
            raise unittest.SkipTest("GNU Bash is unavailable")

    def run_launcher(self, filename: str, overrides=None):
        environment = os.environ.copy()
        for name in (
            "ATTACKER_ID", "CAMPAIGN_ID", "CONFIRM_QUALITATIVE_RUN",
            "CPU_THREADS_PER_WORKER", "DRY_RUN", "GPU_IDS", "MERGE_ONLY",
            "NUM_EPOCH", "PROJECT_ROOT", "TRIAL_IDS", "WORKERS_PER_GPU",
        ):
            environment.pop(name, None)
        environment.update({"DRY_RUN": "1", "CAMPAIGN_ID": "contract"})
        environment.update(overrides or {})
        return subprocess.run(
            [self.bash, "-lc", 'bash "$1"', "launcher-test", str(SCRIPT_DIR / filename)],
            cwd=ROOT, env=environment,
            text=True, encoding="utf-8", errors="replace", capture_output=True,
            check=False,
        )

    def test_exhaustive_dry_run_contract(self):
        total_tasks = total_na = 0
        core_blocks = []
        for dataset, profile in PROFILES.items():
            with self.subTest(dataset=dataset):
                result = self.run_launcher(profile["file"])
                self.assertEqual(result.returncode, 0, result.stderr)
                task_lines = [line for line in result.stdout.splitlines() if line.startswith("TASK|")]
                na_lines = [line for line in result.stdout.splitlines() if line.startswith("N/A|")]
                command_lines = [line for line in result.stdout.splitlines() if line.startswith("command:")]
                self.assertEqual(len(task_lines), 224)
                self.assertEqual(len(na_lines), 32)
                self.assertEqual(len(command_lines), 224)
                total_tasks += len(task_lines)
                total_na += len(na_lines)

                text = (SCRIPT_DIR / profile["file"]).read_text(encoding="utf-8")
                self.assertNotRegex(text, r"(?m)^\s*(?:source|\.)\s+")
                self.assertNotIn("paper_online", text)
                self.assertNotIn("candidate", text.lower())
                self.assertNotIn("sha256", text.lower())
                self.assertIn(
                    f"CONFIG|{dataset}|gpu={profile['gpu']}|workers={profile['workers']}|threads={profile['threads']}",
                    result.stdout,
                )
                self.assertIn(
                    "ensure_prerequisites; acquire_gpu_locks; probe_gpus; prepare_campaign",
                    text,
                )
                core_blocks.append(text.split("# BEGIN IDENTICAL CORE\n", 1)[1])

                records = []
                for task_line, command_line in zip(task_lines, command_lines):
                    fields = task_line.split("|")
                    self.assertEqual(len(fields), 10)
                    _, task_id, observed_dataset, trial, alpha, clients, regime, attack, detector, state = fields
                    self.assertEqual(state, "planned")
                    self.assertEqual(observed_dataset, dataset)
                    self.assertEqual(trial, "0")
                    self.assertIn(alpha, profile["alphas"])
                    self.assertIn(clients, {"2", "4", "6", "8"})
                    self.assertIn(attack, profile[regime])
                    self.assertIn(detector, DETECTORS)
                    self.assertFalse(clients == "2" and detector in {"wef", "s2_wef"})

                    argv = re.findall(r"<([^>]*)>", command_line)
                    for option, wanted in (
                        ("--dataset", dataset), ("--alpha", alpha),
                        ("--client_num", clients), ("--attacker_id", "1"),
                        ("--attack_method", attack), ("--detecting_method", detector),
                        ("--detecting_type", "qualitative"),
                        ("--qualitative_aggregation_mode", "passive"),
                        ("--trial_id", "0"), ("--num_epoch", "50"),
                        ("--num_local_epochs", "1"),
                    ):
                        self.assertEqual(option_value(argv, option), wanted)
                    self.assertEqual(argv.count("--use_attack"), 1)
                    self.assertEqual(
                        option_value(argv, "--random_round"),
                        "2" if attack == "advanced_scaled_delta" else "10",
                    )
                    self.assertNotIn("--attack_probability", argv)

                    if regime == "lazy":
                        self.assertEqual(argv.count("--use_lazy"), 1)
                        self.assertEqual(option_value(argv, "--lazy_schedule"), "fixed")
                        self.assertEqual(option_value(argv, "--lazy_honest_gap"), "6")
                        self.assertEqual(option_value(argv, "--lazy_honest_phase"), "0")
                        self.assertEqual(option_value(argv, "--lazy_fake_policy"), "attack")
                    else:
                        for option in ("--use_lazy", "--lazy_schedule", "--lazy_honest_gap", "--lazy_honest_phase", "--lazy_fake_policy"):
                            self.assertNotIn(option, argv)

                    if attack in HESSIAN_METHODS:
                        self.assertEqual(argv.count("--use_hessian"), 1)
                        self.assertEqual(option_value(argv, "--lambda_h"), "0.9")
                    else:
                        self.assertNotIn("--use_hessian", argv)
                        self.assertNotIn("--lambda_h", argv)

                    if attack.startswith("arima_adaptive"):
                        self.assertEqual(option_value(argv, "--arima_var_window"), "3")
                        self.assertEqual(option_value(argv, "--arima_var_threshold"), profile["arima"][0])
                        self.assertEqual(option_value(argv, "--arima_err_threshold"), profile["arima"][1])
                        self.assertEqual(option_value(argv, "--arima_min_rounds_after_start"), "4")
                        self.assertEqual(option_value(argv, "--arima_retrain_interval"), "15")
                        self.assertEqual(option_value(argv, "--arima_err_ema_beta"), "0.9")
                        self.assertIn("--arima_trim_history", argv)
                    else:
                        self.assertNotIn("--arima_var_window", argv)

                    if attack.startswith("moirai_online"):
                        expected_lr, expected_error = profile["moirai"]
                        expected = {
                            "--moirai_tune_lr": expected_lr, "--moirai_tune_steps": "1",
                            "--moirai_tune_start_round": "10", "--moirai_adapt_interval": "4",
                            "--moirai_residual_decay": "0.95", "--moirai_residual_clip": "5",
                            "--moirai_err_ema_beta": "0.9", "--moirai_err_threshold": expected_error,
                            "--moirai_min_rounds_after_start": "4",
                        }
                        self.assertIn("--moirai_online_tuning", argv)
                        self.assertIn("--moirai_trim_history", argv)
                        for option, wanted in expected.items():
                            self.assertEqual(option_value(argv, option), wanted)
                    else:
                        self.assertNotIn("--moirai_tune_lr", argv)

                    if attack in LAYERWISE_METHODS:
                        layerwise = {
                            "--layerwise_selection_metric": "normalized_rms",
                            "--layerwise_selection_fraction": "0.5",
                            "--layerwise_pca_threshold": "512",
                            "--layerwise_pca_components": "8",
                        }
                        for option, wanted in layerwise.items():
                            self.assertEqual(option_value(argv, option), wanted)
                    else:
                        self.assertNotIn("--layerwise_selection_metric", argv)

                    self.assert_baseline_profile(dataset, attack, argv)
                    records.append((alpha, clients, regime, attack, detector, task_id))

                for line in na_lines:
                    fields = line.split("|")
                    self.assertEqual(len(fields), 11)
                    _, _, na_dataset, trial, alpha, clients, regime, attack, detector, status, reason = fields
                    self.assertEqual((na_dataset, trial, clients, status, reason),
                                     (dataset, "0", "2", "N/A", "wef_family_requires_at_least_3_clients"))
                    self.assertIn(alpha, profile["alphas"])
                    self.assertIn(attack, profile[regime])
                    self.assertIn(detector, {"wef", "s2_wef"})

                expected_counts = collections.Counter()
                for regime in ("pure", "lazy"):
                    for attack in profile[regime]:
                        for detector in DETECTORS:
                            expected_counts[(regime, attack, detector)] = 12 if detector in {"wef", "s2_wef"} else 16
                self.assertEqual(collections.Counter((r[2], r[3], r[4]) for r in records), expected_counts)
                self.assertEqual(len({r[5] for r in records}), 224)

        self.assertEqual(total_tasks, 1120)
        self.assertEqual(total_na, 160)
        self.assertTrue(all(block == core_blocks[0] for block in core_blocks[1:]))

    def assert_baseline_profile(self, dataset, attack, argv):
        if attack == "advanced_scaled_delta":
            expected = {
                "--random_round": "2", "--scaled_delta_c_const": "1.0",
                "--scaled_delta_noise_fraction": "1.0",
                "--scaled_delta_max_noise_ratio": "5.0", "--scaled_delta_max_scale": "10.0",
            }
            for option, wanted in expected.items():
                self.assertEqual(option_value(argv, option), wanted)
            self.assertNotIn("--scaled_delta_lambda_est", argv)
        if attack == "adam_local_deviation":
            expected = {
                "--wang_adam_lr": "8e-3", "--wang_adam_beta1": "0.9",
                "--wang_adam_beta2": "0.999", "--wang_adam_eps": "1e-8",
                "--adam_public_fraction": "0.05", "--adam_public_seed": "20260726",
            }
            for option, wanted in expected.items():
                self.assertEqual(option_value(argv, option), wanted)
        if attack == "dkfr":
            self.assertEqual(tuple(option_value(argv, option) for option in DKFR_OPTIONS), DKFR[dataset])
            if dataset == "mnist":
                self.assertEqual(option_value(argv, "--dkfr_mnist_tv"), "0.1")
                self.assertEqual(option_value(argv, "--dkfr_mnist_pixel_reg"), "0.2")
                self.assertEqual(option_value(argv, "--dkfr_mnist_logit_reg"), "1e-4")
            else:
                self.assertNotIn("--dkfr_mnist_tv", argv)
        if attack == "synthetic_replacement":
            expected = {
                "--synthetic_mixup_alpha": "0.4", "--synthetic_noise_std": "0.02",
                "--synthetic_tabular_noise_std": "0.05",
                "--synthetic_tabular_flip_prob": "0.02", "--synthetic_seed": "0",
            }
            for option, wanted in expected.items():
                self.assertEqual(option_value(argv, option), wanted)

    def test_validation_and_write_guards(self):
        filename = PROFILES["tictactoe"]["file"]
        for overrides in (
            {"GPU_IDS": "1 1"}, {"GPU_IDS": "x"}, {"WORKERS_PER_GPU": "0"},
            {"CPU_THREADS_PER_WORKER": "0"}, {"ATTACKER_ID": "0"},
            {"NUM_EPOCH": "49"}, {"TRIAL_IDS": "00"}, {"TRIAL_IDS": "0 0"},
            {"DRY_RUN": "2"}, {"MERGE_ONLY": "2"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_launcher(filename, overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ERROR:", result.stderr)

        with tempfile.TemporaryDirectory() as directory:
            project = pathlib.Path(directory) / "project"
            project.mkdir()
            result = self.run_launcher(filename, {"PROJECT_ROOT": str(project)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((project / ".qualitative_runs").exists())
            self.assertFalse((project / "Results").exists())

            result = self.run_launcher(filename, {
                "PROJECT_ROOT": str(project), "DRY_RUN": "0",
                "CONFIRM_QUALITATIVE_RUN": "0",
            })
            self.assertEqual(result.returncode, 2)
            self.assertFalse((project / ".qualitative_runs").exists())
            self.assertFalse((project / "Results").exists())

    def test_merger_rejects_empty_plan_wrong_raw_task_id_and_nonbinary_flags(self):
        for dataset, profile in PROFILES.items():
            with self.subTest(dataset=dataset):
                text = (SCRIPT_DIR / profile["file"]).read_text(encoding="utf-8")
                core = text.split("# BEGIN IDENTICAL CORE", 1)[1].split("# END IDENTICAL CORE", 1)[0]
                self.assertIn('if not plans: raise ValueError("empty campaign plan")', core)
                self.assertIn('if not trials: raise ValueError("empty campaign trials")', core)
                self.assertIn('expected_raw_task_id="{}_{}_{}_alpha{}_clients{}_attacker1_trial{}".format(', core)
                self.assertEqual(core.count('if raw_task_ids!={expected_raw_task_id}:'), 2)
                self.assertEqual(core.count('str(float(p["Alpha"]))'), 2)
                self.assertIn('if binary(row.get("UseHessian"),"UseHessian") != expected_hessian(p):', core)
                self.assertIn('if binary(row.get("UseLazy"),"UseLazy") != lazy:', core)

    def test_raw_task_id_uses_flce_float_alpha_normalization(self):
        self.assertEqual(str(float("10")), "10.0")
        self.assertEqual(str(float("200000000")), "200000000.0")
        scheduler = (ROOT / "tests" / "test_passive_qualitative_scheduler.sh").read_text(encoding="utf-8")
        self.assertIn("printf -v raw_alpha '%.1f'", scheduler)
        self.assertIn("_alpha${raw_alpha}_clients", scheduler)

if __name__ == "__main__":
    unittest.main()

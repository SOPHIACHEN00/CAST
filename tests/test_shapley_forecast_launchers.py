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

LAUNCHERS = {
    "run_tictactoe_forecast_pure_shapley.sh": ("tictactoe", "pure"),
    "run_tictactoe_forecast_lazy_shapley.sh": ("tictactoe", "lazy"),
    "run_adult_forecast_pure_shapley.sh": ("adult", "pure"),
    "run_adult_forecast_lazy_shapley.sh": ("adult", "lazy"),
    "run_dota2_forecast_pure_shapley.sh": ("dota2", "pure"),
    "run_dota2_forecast_lazy_shapley.sh": ("dota2", "lazy"),
}

ALPHAS = {
    "tictactoe": {"10", "100", "1000", "20000"},
    "adult": {"1000", "100000", "1000000", "20000000"},
    "dota2": {"10000", "1000000", "50000000", "200000000"},
}

GPU_BY_DATASET = {"tictactoe": "1", "adult": "2", "dota2": "3"}

METHODS = {
    "arima_offline": "arima",
    "arima_adaptive": "arima_adaptive",
    "moirai_offline": "moirai",
    "moirai_online": "moirai_online",
}

ARIMA_ERR = {"tictactoe": "0.016", "adult": "0.08", "dota2": "0.18"}
MOIRAI_LR = {"tictactoe": "0.05", "adult": "0.003", "dota2": "0.001"}
MOIRAI_ERR = {"tictactoe": "0.025", "adult": "0.12", "dota2": "0.25"}

PAIRS = {"2": "0.50", "3": "0.66", "6": "0.82", "10": "0.90"}


def option_value(argv, option):
    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1:
        raise AssertionError(f"expected one {option}, found {len(positions)}")
    position = positions[0]
    if position + 1 >= len(argv):
        raise AssertionError(f"{option} has no value")
    return argv[position + 1]


class ShapleyForecastLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = []
        for configured_name in ("BASH_EXE", "BASH"):
            configured_path = os.environ.get(configured_name)
            if configured_path:
                candidates.append(pathlib.Path(configured_path))

        discovered_bash = shutil.which("bash")
        if discovered_bash:
            candidates.append(pathlib.Path(discovered_bash))

        # On Windows, shutil.which("bash") may resolve to an unusable WSL
        # shim.  Git for Windows ships the Bash needed by these launchers.
        discovered_git = shutil.which("git")
        if discovered_git:
            git_root = pathlib.Path(discovered_git).resolve().parent.parent
            candidates.append(git_root / "bin" / "bash.exe")

        cls.bash = None
        seen_candidates = set()
        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in seen_candidates or not candidate.is_file():
                continue
            seen_candidates.add(candidate_key)
            probe = subprocess.run(
                [candidate_key, "--version"],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0 and "GNU bash" in (probe.stdout + probe.stderr):
                cls.bash = candidate_key
                break

        if cls.bash is None:
            raise unittest.SkipTest("GNU Bash is unavailable")

    def run_launcher(self, filename, print_limit, extra_env=None):
        environment = os.environ.copy()
        environment.update(
            {
                "DRY_RUN": "1",
                "DRY_RUN_PRINT_LIMIT": str(print_limit),
                "CAMPAIGN_ID": f"contract_{filename.removesuffix('.sh')}",
            }
        )
        if extra_env:
            environment.update(extra_env)
        return subprocess.run(
            [self.bash, str(SCRIPT_DIR / filename)],
            cwd=ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )

    def test_bash_syntax(self):
        for path in (SCRIPT_DIR / name for name in LAUNCHERS):
            with self.subTest(path=path.name):
                result = subprocess.run(
                    [self.bash, "-n", str(path)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_deployment_attests_launchers_and_contract_tests(self):
        deploy_text = (SCRIPT_DIR / "deploy_and_smoke_moirai_arima.sh").read_text(
            encoding="utf-8"
        )
        trusted_files = [
            "tests/test_shapley_forecast_launchers.py",
            "tests/test_shapley_forecast_scheduler.sh",
            *(f"script/{name}" for name in LAUNCHERS),
        ]
        for relative_path in trusted_files:
            with self.subTest(relative_path=relative_path):
                self.assertGreaterEqual(
                    deploy_text.count(relative_path),
                    3,
                    "file must appear in required overlay, deployment hash, and smoke hash",
                )
        self.assertIn("tests.test_shapley_forecast_launchers", deploy_text)
        self.assertIn("sh tests/test_shapley_forecast_scheduler.sh", deploy_text)

    def test_launchers_are_self_contained_with_visible_configuration(self):
        for filename, (expected_dataset, expected_mode) in LAUNCHERS.items():
            with self.subTest(filename=filename):
                launcher_text = (SCRIPT_DIR / filename).read_text(encoding="utf-8")
                header_text = "\n".join(launcher_text.splitlines()[:60])
                expected_alphas = " ".join(
                    sorted(ALPHAS[expected_dataset], key=int)
                )

                self.assertTrue(
                    launcher_text.startswith("#!/usr/bin/env bash\n"),
                    "self-contained launchers must explicitly require Bash",
                )
                self.assertRegex(
                    header_text,
                    rf'(?m)^DATASET="{re.escape(expected_dataset)}"$',
                )
                self.assertRegex(
                    header_text,
                    rf'(?m)^MODE="{re.escape(expected_mode)}"$',
                )
                self.assertRegex(
                    header_text,
                    rf"(?m)^ALPHAS=\({re.escape(expected_alphas)}\)$",
                )
                expected_gpu = GPU_BY_DATASET[expected_dataset]
                self.assertRegex(
                    header_text,
                    rf'(?m)^GPU_IDS_REQUESTED="\$\{{GPU_IDS:-{expected_gpu}\}}"$',
                )
                self.assertRegex(
                    header_text,
                    rf"(?m)^GPU_IDS=\({expected_gpu}\)$",
                )
                self.assertRegex(
                    header_text,
                    r'(?m)^WORKERS_PER_GPU="\$\{WORKERS_PER_GPU:-15\}"$',
                )
                self.assertRegex(
                    header_text,
                    r'(?m)^CPU_THREADS_PER_WORKER="\$\{CPU_THREADS_PER_WORKER:-15\}"$',
                )
                self.assertNotIn(
                    "forecast_shapley_launcher_common.inc",
                    launcher_text,
                    "self-contained launchers must not source the former common file",
                )

    def test_all_six_dry_run_contracts(self):
        all_launcher_tasks = 0
        for filename, (expected_dataset, expected_mode) in LAUNCHERS.items():
            expected_count = 64 if expected_mode == "pure" else 512
            with self.subTest(filename=filename):
                result = self.run_launcher(filename, expected_count)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()
                task_lines = [line for line in lines if line.startswith("TASK|")]
                command_lines = [line for line in lines if line.startswith("command:")]
                self.assertEqual(len(task_lines), expected_count)
                self.assertEqual(len(command_lines), expected_count)
                all_launcher_tasks += len(task_lines)
                self.assertIn(f"total_tasks          : {expected_count} ", result.stdout)
                self.assertIn(f"enumerated {expected_count} tasks", result.stdout)

                records = []
                results_roots = set()
                for task_line, command_line in zip(task_lines, command_lines):
                    fields = task_line.split("|")
                    self.assertEqual(len(fields), 12)
                    (
                        _, task_number, dataset, mode, alpha, clients, variant,
                        method, schedule, gap, probability, trial,
                    ) = fields
                    self.assertEqual(dataset, expected_dataset)
                    self.assertEqual(mode, expected_mode)
                    self.assertIn(alpha, ALPHAS[dataset])
                    self.assertIn(clients, {"2", "4", "6", "8"})
                    self.assertEqual(METHODS[variant], method)
                    self.assertEqual(trial, "0")
                    self.assertEqual(int(task_number), len(records) + 1)

                    argv = re.findall(r"<([^>]*)>", command_line)
                    self.assertGreater(len(argv), 1)
                    self.assertEqual(option_value(argv, "--dataset"), dataset)
                    self.assertEqual(option_value(argv, "--alpha"), alpha)
                    self.assertEqual(option_value(argv, "--client_num"), clients)
                    self.assertEqual(option_value(argv, "--attacker_id"), "1")
                    self.assertEqual(option_value(argv, "--attack_method"), method)
                    self.assertEqual(option_value(argv, "--contribution_method"), "shapley")
                    self.assertEqual(option_value(argv, "--detecting_type"), "quantitative")
                    self.assertEqual(option_value(argv, "--num_epoch"), "50")
                    self.assertEqual(option_value(argv, "--num_local_epochs"), "1")
                    self.assertEqual(option_value(argv, "--random_round"), "10")
                    self.assertEqual(option_value(argv, "--lambda_h"), "0.9")
                    self.assertEqual(argv.count("--use_last_round_sv"), 1)
                    self.assertEqual(argv.count("--use_attack"), 1)
                    self.assertEqual(argv.count("--use_hessian"), 1)
                    self.assertFalse(any("layerwise" in argument for argument in argv))
                    self.assertFalse(any("eds" in argument or "dagmm" in argument for argument in argv))

                    results_root = option_value(argv, "--results_root")
                    self.assertNotIn(results_root, results_roots)
                    results_roots.add(results_root)

                    if mode == "pure":
                        self.assertEqual((schedule, gap, probability), ("pure", "na", "na"))
                        self.assertNotIn("--use_lazy", argv)
                        self.assertNotIn("--lazy_schedule", argv)
                        self.assertNotIn("--attack_probability", argv)
                    else:
                        self.assertEqual(argv.count("--use_lazy"), 1)
                        self.assertEqual(option_value(argv, "--lazy_schedule"), schedule)
                        self.assertEqual(option_value(argv, "--lazy_honest_gap"), gap)
                        self.assertEqual(option_value(argv, "--lazy_honest_phase"), "0")
                        self.assertEqual(option_value(argv, "--lazy_fake_policy"), "attack")
                        self.assertEqual(PAIRS[gap], probability)
                        if schedule == "fixed":
                            self.assertNotIn("--attack_probability", argv)
                        else:
                            self.assertEqual(schedule, "probabilistic")
                            self.assertEqual(option_value(argv, "--attack_probability"), probability)

                    if variant.startswith("arima_"):
                        self.assertEqual(option_value(argv, "--arima_var_window"), "3")
                        self.assertEqual(option_value(argv, "--arima_var_threshold"), "0.005")
                    else:
                        self.assertNotIn("--arima_var_window", argv)

                    if variant == "arima_adaptive":
                        self.assertEqual(option_value(argv, "--arima_err_threshold"), ARIMA_ERR[dataset])
                        self.assertEqual(option_value(argv, "--arima_min_rounds_after_start"), "4")
                        self.assertEqual(option_value(argv, "--arima_retrain_interval"), "15")
                        self.assertEqual(option_value(argv, "--arima_err_ema_beta"), "0.9")
                        self.assertEqual(argv.count("--arima_trim_history"), 1)
                    else:
                        self.assertNotIn("--arima_err_threshold", argv)
                        self.assertNotIn("--arima_min_rounds_after_start", argv)
                        self.assertNotIn("--arima_retrain_interval", argv)
                        self.assertNotIn("--arima_err_ema_beta", argv)
                        self.assertNotIn("--arima_trim_history", argv)

                    if variant == "moirai_online":
                        self.assertEqual(argv.count("--moirai_online_tuning"), 1)
                        self.assertEqual(option_value(argv, "--moirai_tune_lr"), MOIRAI_LR[dataset])
                        self.assertEqual(option_value(argv, "--moirai_tune_steps"), "1")
                        self.assertEqual(option_value(argv, "--moirai_tune_start_round"), "10")
                        self.assertEqual(option_value(argv, "--moirai_adapt_interval"), "4")
                        self.assertEqual(option_value(argv, "--moirai_residual_decay"), "0.95")
                        self.assertEqual(option_value(argv, "--moirai_residual_clip"), "5.0")
                        self.assertEqual(option_value(argv, "--moirai_err_ema_beta"), "0.9")
                        self.assertEqual(option_value(argv, "--moirai_err_threshold"), MOIRAI_ERR[dataset])
                        self.assertEqual(option_value(argv, "--moirai_min_rounds_after_start"), "4")
                        self.assertEqual(argv.count("--moirai_trim_history"), 1)
                    elif variant != "moirai_online":
                        self.assertNotIn("--moirai_tune_lr", argv)

                    records.append((alpha, clients, variant, schedule, gap, probability))

                by_method = collections.Counter(record[2] for record in records)
                expected_per_method = 16 if expected_mode == "pure" else 128
                self.assertEqual(by_method, collections.Counter({key: expected_per_method for key in METHODS}))
                self.assertEqual({record[0] for record in records}, ALPHAS[expected_dataset])
                self.assertEqual({record[1] for record in records}, {"2", "4", "6", "8"})
                expected_per_alpha = 16 if expected_mode == "pure" else 128
                expected_per_client = 16 if expected_mode == "pure" else 128
                self.assertEqual(
                    collections.Counter(record[0] for record in records),
                    collections.Counter(
                        {alpha: expected_per_alpha for alpha in ALPHAS[expected_dataset]}
                    ),
                )
                self.assertEqual(
                    collections.Counter(record[1] for record in records),
                    collections.Counter(
                        {client: expected_per_client for client in {"2", "4", "6", "8"}}
                    ),
                )
                if expected_mode == "lazy":
                    arms = collections.Counter((record[3], record[4], record[5]) for record in records)
                    expected_arms = {
                        (schedule, gap, probability): 64
                        for gap, probability in PAIRS.items()
                        for schedule in ("fixed", "probabilistic")
                    }
                    self.assertEqual(arms, collections.Counter(expected_arms))
        self.assertEqual(all_launcher_tasks, 1728)

    def test_protocol_and_gpu_drift_are_rejected(self):
        cases = [
            {"CONTRIBUTION_METHOD": "eds"},
            {"NUM_EPOCH": "49"},
            {"USE_HESSIAN": "0"},
            {"GPU_IDS": "0 1 2"},
            {"WORKERS_PER_GPU": "0"},
            {"WORKERS_PER_GPU": "not-an-integer"},
            {"CPU_THREADS_PER_WORKER": "0"},
            {"CPU_THREADS_PER_WORKER": "not-an-integer"},
            {"ATTACKER_ID": "0"},
            {"TRIAL_IDS": "00"},
            {"TRIAL_IDS": "0 00"},
        ]
        for override in cases:
            with self.subTest(override=override):
                result = self.run_launcher(
                    "run_tictactoe_forecast_pure_shapley.sh", 0, override
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ERROR:", result.stderr)

    def test_dry_run_creates_no_campaign_or_results(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            project_root = pathlib.Path(temporary_root) / "project"
            project_root.mkdir()
            result = self.run_launcher(
                "run_adult_forecast_lazy_shapley.sh",
                0,
                {"PROJECT_ROOT": str(project_root)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((project_root / ".shapley_runs").exists())
            self.assertFalse((project_root / "Results").exists())

    def test_real_run_is_confirmation_locked_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            project_root = pathlib.Path(temporary_root) / "project"
            project_root.mkdir()
            result = self.run_launcher(
                "run_dota2_forecast_pure_shapley.sh",
                0,
                {
                    "PROJECT_ROOT": str(project_root),
                    "DRY_RUN": "0",
                    "CONFIRM_SHAPLEY_RUN": "0",
                },
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Run is locked", result.stderr)
            self.assertFalse((project_root / ".shapley_runs").exists())
            self.assertFalse((project_root / "Results").exists())


if __name__ == "__main__":
    unittest.main()

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from module.method.detecting_registry import resolve_detecting_method
from module.method.quantitative_detecting.runner_support import build_sampling_call_kwargs


ROOT = Path(__file__).resolve().parents[1]


class ThresholdModel(torch.nn.Module):
    num_epoch = 1
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.threshold = torch.nn.Parameter(torch.zeros(1))

    def score(self, x, y, value_functions):
        accuracy = ((x < self.threshold) == y).float().mean().item()
        return [accuracy for _ in value_functions]


class LeaveOneOutTests(unittest.TestCase):
    def test_signed_held_out_marginals_use_selected_server_rule(self):
        x = torch.tensor([0.15, 0.30, 0.45, 0.55, 0.65, 0.75])
        labels = torch.ones_like(x, dtype=torch.bool)
        loader = SimpleNamespace(
            X_train_parts=[x[:2], x[:1], x[:1]],
            y_train_parts=[labels[:2], labels[:1], labels[:1]],
            X_test=x,
            y_test=labels,
        )
        spec = resolve_detecting_method("leave_one_out", requested_type="quantitative")
        self.assertEqual(spec.name, "loo")
        options = build_sampling_call_kwargs(
            spec.name, num_parts=3, experiment_seed=42, trial_id=0,
            shapley_num_permutations=None, shapley_seed=None, shapley_tolerance=0,
            least_core_num_coalitions=None, least_core_seed=None,
        )
        expectations = {"fedavg": [2/6, 0, -2/6], "weighted_median": [3/6, -2/6, -2/6]}
        deltas = {i: [torch.tensor([u])] for i, u in enumerate([0.8, 0.4, 0.0])}
        for method, expected in expectations.items():
            with self.subTest(method=method):
                estimator = spec.implementation(
                    loader, ThresholdModel(), {}, ["accuracy"], use_last_round_sv=True,
                    last_round_state=ThresholdModel().state_dict(),
                    last_round_weights=[0.5, 0.25, 0.25],
                    last_round_client_grads=deltas,
                    aggregation_config={"method": method},
                )
                credits, elapsed = estimator.get_contributions(**options)
                np.testing.assert_allclose(credits, [expected], atol=1e-7)
                self.assertGreaterEqual(elapsed, 0)

    def test_cli_registers_loo_as_quantitative(self):
        args = [sys.executable, str(ROOT / "FLCE_updated.py"), "--detecting_method", "loo",
                "--detecting_type", "quantitative", "--check_detecting_setup"]
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertIn("type=quantitative, method=loo", result.stdout)

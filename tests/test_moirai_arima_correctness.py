"""Regression tests for corrected Moirai/ARIMA attack state machines.

These tests intentionally stub optional forecasting dependencies.  They verify
round alignment and document-faithful adaptive-refit semantics without
downloading Moirai weights or requiring pmdarima/scikit-learn in the desktop
test environment.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
_MISSING_MODULE = object()


class _StubMoiraiModule:
    @classmethod
    def from_pretrained(cls, _source):
        return cls()


class _StubMoiraiForecast:
    def __init__(self, **_kwargs):
        pass

    def create_predictor(self, batch_size=1):
        raise AssertionError(f"Unexpected real predictor construction, batch_size={batch_size}")


class _NumpyIncrementalPCA:
    """Small SVD-backed stand-in with the IncrementalPCA methods under test."""

    def __init__(self, n_components):
        self.n_components = int(n_components)
        self.mean_ = None
        self.components_ = None

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        _, _, vh = np.linalg.svd(values - self.mean_, full_matrices=False)
        self.components_ = vh[: self.n_components]
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return (values - self.mean_) @ self.components_.T

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return values @ self.components_ + self.mean_


def _install_optional_dependency_stubs():
    stub_names = (
        "huggingface_hub",
        "pmdarima",
        "sklearn",
        "sklearn.decomposition",
        "uni2ts",
        "uni2ts.model",
        "uni2ts.model.moirai",
        "uni2ts.model.moirai.module",
    )
    previous_modules = {
        name: sys.modules.get(name, _MISSING_MODULE) for name in stub_names
    }

    huggingface = types.ModuleType("huggingface_hub")
    huggingface.hf_hub_download = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Network/model download is forbidden in focused unit tests")
    )
    sys.modules["huggingface_hub"] = huggingface

    pmdarima = types.ModuleType("pmdarima")
    pmdarima.auto_arima = lambda *_args, **_kwargs: None
    sys.modules["pmdarima"] = pmdarima

    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []
    decomposition = types.ModuleType("sklearn.decomposition")
    decomposition.IncrementalPCA = _NumpyIncrementalPCA
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.decomposition"] = decomposition

    uni2ts = types.ModuleType("uni2ts")
    uni2ts.__path__ = []
    uni2ts_model = types.ModuleType("uni2ts.model")
    uni2ts_model.__path__ = []
    moirai = types.ModuleType("uni2ts.model.moirai")
    moirai.__path__ = []
    moirai.MoiraiForecast = _StubMoiraiForecast
    moirai.MoiraiModule = _StubMoiraiModule
    moirai_module = types.ModuleType("uni2ts.model.moirai.module")
    moirai_module.decode_distr_output = lambda value: value
    sys.modules["uni2ts"] = uni2ts
    sys.modules["uni2ts.model"] = uni2ts_model
    sys.modules["uni2ts.model.moirai"] = moirai
    sys.modules["uni2ts.model.moirai.module"] = moirai_module
    return previous_modules


def _restore_optional_dependency_modules(previous_modules):
    for name, previous in previous_modules.items():
        if previous is _MISSING_MODULE:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _load_attack_modules():
    previous_modules = _install_optional_dependency_stubs()

    package_paths = {
        "module": ROOT / "module",
        "module.method": ROOT / "module" / "method",
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    loaded = []
    try:
        for short_name in ("attack_updated_nonlayerwise", "attack_updated_layerwise"):
            full_name = f"module.method.{short_name}"
            source = package_paths["module.method"] / f"{short_name}.py"
            spec = importlib.util.spec_from_file_location(full_name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            loaded.append(module)
    finally:
        _restore_optional_dependency_modules(previous_modules)
    return loaded


nonlayer, layerwise = _load_attack_modules()


class _RecordingARIMA:
    def __init__(self, series=None):
        self.series = list(np.asarray(series if series is not None else [0.0], dtype=float))
        self.predict_horizons = []
        self.updates = []
        self.update_kwargs = []

    def predict(self, n_periods=1):
        n_periods = int(n_periods)
        self.predict_horizons.append(n_periods)
        # The last element makes increasing-horizon behavior directly visible.
        return np.arange(1, n_periods + 1, dtype=np.float32)

    def update(self, value, *args, **kwargs):
        scalar = float(np.asarray(value).reshape(-1)[0])
        self.updates.append(scalar)
        self.update_kwargs.append(dict(kwargs))
        self.series.append(scalar)
        return self


class _Forecast:
    def __init__(self, values):
        self.samples = np.asarray(values, dtype=np.float32).reshape(1, 1, -1)


class _RecordingPredictor:
    def __init__(self, *values):
        self.values = [np.asarray(value, dtype=np.float32) for value in values]
        self.calls = 0

    def predict(self, _input_data):
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        yield _Forecast(self.values[index])


class MNISTMLP(torch.nn.Module):
    def __init__(self, size=2, two_modules=False):
        super().__init__()
        if two_modules:
            # Deliberately give fc1 four times as many parameters as fc2.  A
            # raw L2 norm would therefore prefer fc1 for the warm-up below,
            # while dimension-normalized RMS correctly prefers fc2.
            self.fc1 = torch.nn.Linear(4, 4)
            self.fc2 = torch.nn.Linear(4, 1)
        else:
            self.weight = torch.nn.Parameter(torch.zeros(size))


def _two_module_gradient(model, fc1_value, fc2_value):
    gradients = []
    for name, param in model.named_parameters():
        value = fc1_value if name.startswith("fc1.") else fc2_value
        gradients.append(torch.full_like(param, float(value)))
    return gradients


def _record_two_module_warmup(attack, model):
    # The per-element warm-up RMS values are fc1=2 and fc2=3.  Because fc1
    # has 20 parameters and fc2 has five, the unnormalized mean L2 norms have
    # the opposite ordering; this makes accidental size-biased selection
    # observable.
    observations = ((1.0, 2.0), (3.0, 4.0))
    for round_num, (fc1_value, fc2_value) in enumerate(observations):
        gradient = _two_module_gradient(model, fc1_value, fc2_value)
        if getattr(attack, "supports_round_aware_recording", False):
            attack.record_global_gradient(gradient, round_num=round_num)
        else:
            attack.record_global_gradient(gradient)


def _outputs_by_parameter_name(model, outputs):
    return {
        name: output
        for (name, _param), output in zip(model.named_parameters(), outputs)
    }


class LayerwiseSelectionTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def _selection_kwargs(self):
        return {
            "selection_metric": "normalized_rms",
            "selection_fraction": 0.5,
            "selection_top_n": None,
            "pca_threshold": 1000,
            "pca_components": 3,
        }

    def test_all_layerwise_constructors_expose_selection_and_pca_controls(self):
        constructors = (
            layerwise.LayerwiseOfflineARIMAAttackMNISTIMDB,
            layerwise.LayerwiseAdaptiveARIMAAttackMNISTIMDB,
            layerwise.LayerwiseOfflineMoiraiAttackMNISTIMDB,
            layerwise.LayerwiseOnlineMoiraiAttackMNISTIMDB,
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor.__name__):
                model = MNISTMLP(two_modules=True)
                attack = constructor(
                    k=2,
                    random_round=2,
                    model=model,
                    selection_metric="normalized_rms",
                    selection_fraction=0.5,
                    selection_top_n=None,
                    pca_threshold=7,
                    pca_components=3,
                    lambda_h=1.0,
                )

                self.assertEqual(attack.selection_metric, "normalized_rms")
                self.assertEqual(attack.selection_fraction, 0.5)
                self.assertIsNone(attack.selection_top_n)
                self.assertEqual(attack.pca_threshold, 7)
                self.assertEqual(attack.pca_components, 3)
                self.assertIsInstance(attack.selected_modules, tuple)
                self.assertIsInstance(attack.selection_scores, dict)
                self.assertIsInstance(attack.last_layer_sources, dict)
                self.assertIsInstance(attack.selection_telemetry, list)
                self.assertEqual(
                    [state.module_name for state in attack.layer_states],
                    ["fc1", "fc1", "fc2", "fc2"],
                )
                self.assertTrue(
                    all(isinstance(state.selected, bool) for state in attack.layer_states)
                )
                self.assertTrue(attack.layer_states[0].use_pca)
                self.assertEqual(attack.layer_states[0].compressed_dim, 3)

    def test_normalized_rms_warmup_selects_one_module_and_groups_weight_bias(self):
        model = MNISTMLP(two_modules=True)
        attack = layerwise.LayerwiseOfflineMoiraiAttackMNISTIMDB(
            k=2,
            random_round=2,
            model=model,
            history_length=2,
            lambda_h=1.0,
            **self._selection_kwargs(),
        )
        _record_two_module_warmup(attack, model)

        forecasted = []

        def _forecast(state):
            forecasted.append(state.name)
            return np.full(state.size, 9.0, dtype=np.float32)

        with patch.object(attack, "_predict_state", side_effect=_forecast):
            attack.get_fake_gradient(2, self.device, model)

        # An unnormalized norm would rank fc1 first because 2*sqrt(20) is
        # larger than 3*sqrt(5).  Normalized RMS must remove that size bias.
        self.assertGreater(2.0 * np.sqrt(20.0), 3.0 * np.sqrt(5.0))
        self.assertGreater(
            attack.selection_scores["fc2"], attack.selection_scores["fc1"]
        )
        self.assertEqual(attack.selected_modules, ("fc2",))
        self.assertEqual(
            {state.name for state in attack.layer_states if state.selected},
            {"fc2.weight", "fc2.bias"},
        )
        self.assertEqual(forecasted, ["fc2.weight", "fc2.bias"])
        self.assertEqual(len(attack.selection_telemetry), 1)
        self.assertEqual(
            attack.last_layer_sources,
            {
                "fc1.weight": "recent_mean",
                "fc1.bias": "recent_mean",
                "fc2.weight": "forecast",
                "fc2.bias": "forecast",
            },
        )

    def test_offline_arima_fits_and_forecasts_only_selected_module(self):
        model = MNISTMLP(two_modules=True)
        attack = layerwise.LayerwiseOfflineARIMAAttackMNISTIMDB(
            k=2,
            random_round=2,
            model=model,
            lambda_h=1.0,
            **self._selection_kwargs(),
        )
        _record_two_module_warmup(attack, model)

        def _fit_model(series, *_args, **_kwargs):
            return _RecordingARIMA(series)

        with patch.object(layerwise, "auto_arima", side_effect=_fit_model) as fit:
            attack.train_offline_arima_model(fit_round=2, reason="selection_test")

        selected_states = [state for state in attack.layer_states if state.selected]
        unselected_states = [state for state in attack.layer_states if not state.selected]
        self.assertEqual(fit.call_count, sum(state.size for state in selected_states))
        self.assertTrue(all(state.arima_models is not None for state in selected_states))
        self.assertTrue(all(state.arima_models is None for state in unselected_states))

        attack.arima_started = True
        attack.arima_start_round = 2
        outputs = _outputs_by_parameter_name(
            model,
            attack.get_fake_gradient(2, self.device, model),
        )

        self.assertTrue(torch.equal(outputs["fc1.weight"], torch.full_like(model.fc1.weight, 2.0)))
        self.assertTrue(torch.equal(outputs["fc1.bias"], torch.full_like(model.fc1.bias, 2.0)))
        self.assertTrue(torch.equal(outputs["fc2.weight"], torch.ones_like(model.fc2.weight)))
        self.assertTrue(torch.equal(outputs["fc2.bias"], torch.ones_like(model.fc2.bias)))
        for state in selected_states:
            for arima_model in state.arima_models["models"]:
                self.assertEqual(arima_model.predict_horizons, [1])
        self.assertEqual(
            attack.last_layer_sources,
            {
                "fc1.weight": "recent_mean",
                "fc1.bias": "recent_mean",
                "fc2.weight": "forecast",
                "fc2.bias": "forecast",
            },
        )

    def test_offline_moirai_calls_predictor_only_for_selected_module(self):
        model = MNISTMLP(two_modules=True)
        attack = layerwise.LayerwiseOfflineMoiraiAttackMNISTIMDB(
            k=2,
            random_round=2,
            model=model,
            history_length=2,
            lambda_h=1.0,
            **self._selection_kwargs(),
        )
        _record_two_module_warmup(attack, model)
        predictors = {}
        for state in attack.layer_states:
            predictor = _RecordingPredictor(
                np.full(state.compressed_dim, 7.0, dtype=np.float32)
            )
            state.moirai_model = object()
            state.moirai_predictor = predictor
            predictors[state.name] = predictor

        outputs = _outputs_by_parameter_name(
            model,
            attack.get_fake_gradient(2, self.device, model),
        )

        self.assertEqual(predictors["fc1.weight"].calls, 0)
        self.assertEqual(predictors["fc1.bias"].calls, 0)
        self.assertEqual(predictors["fc2.weight"].calls, 1)
        self.assertEqual(predictors["fc2.bias"].calls, 1)
        self.assertTrue(torch.equal(outputs["fc1.weight"], torch.full_like(model.fc1.weight, 2.0)))
        self.assertTrue(torch.equal(outputs["fc1.bias"], torch.full_like(model.fc1.bias, 2.0)))
        self.assertTrue(torch.equal(outputs["fc2.weight"], torch.full_like(model.fc2.weight, 7.0)))
        self.assertTrue(torch.equal(outputs["fc2.bias"], torch.full_like(model.fc2.bias, 7.0)))
        self.assertEqual(
            attack.last_layer_sources,
            {
                "fc1.weight": "recent_mean",
                "fc1.bias": "recent_mean",
                "fc2.weight": "forecast",
                "fc2.bias": "forecast",
            },
        )

    def test_online_moirai_scores_and_tunes_only_same_round_selected_prediction(self):
        model = MNISTMLP(two_modules=True)
        attack = layerwise.LayerwiseOnlineMoiraiAttackMNISTIMDB(
            k=2,
            random_round=2,
            model=model,
            history_length=2,
            lambda_h=1.0,
            tune_lr=0.5,
            tune_steps=1,
            tune_start_round=0,
            adapt_interval=0,
            err_threshold=float("inf"),
            min_rounds_after_start=1,
            **self._selection_kwargs(),
        )
        _record_two_module_warmup(attack, model)
        predictors = {}
        for state in attack.layer_states:
            predictor = _RecordingPredictor(
                np.full(state.compressed_dim, 6.0, dtype=np.float32)
            )
            state.moirai_model = object()
            state.moirai_predictor = predictor
            predictors[state.name] = predictor

        submitted = _outputs_by_parameter_name(
            model,
            attack.get_fake_gradient(2, self.device, model),
        )
        attack.record_global_gradient(
            _two_module_gradient(model, fc1_value=100.0, fc2_value=4.0),
            round_num=2,
        )

        self.assertTrue(torch.equal(submitted["fc1.weight"], torch.full_like(model.fc1.weight, 2.0)))
        self.assertTrue(torch.equal(submitted["fc1.bias"], torch.full_like(model.fc1.bias, 2.0)))
        self.assertTrue(torch.equal(submitted["fc2.weight"], torch.full_like(model.fc2.weight, 6.0)))
        self.assertTrue(torch.equal(submitted["fc2.bias"], torch.full_like(model.fc2.bias, 6.0)))
        self.assertEqual(predictors["fc1.weight"].calls, 0)
        self.assertEqual(predictors["fc1.bias"].calls, 0)
        self.assertEqual(predictors["fc2.weight"].calls, 1)
        self.assertEqual(predictors["fc2.bias"].calls, 1)
        for state in attack.layer_states:
            expected_residual = -1.0 if state.module_name == "fc2" else 0.0
            np.testing.assert_allclose(
                state.residual,
                np.full(state.size, expected_residual, dtype=np.float32),
            )
        self.assertAlmostEqual(attack._err_ema, 2.0)
        self.assertEqual(attack._aligned_observations, 1)
        self.assertIsNone(attack._last_pred_round)

    def test_adaptive_arima_scores_actual_hessian_fused_selected_submission(self):
        model = MNISTMLP(two_modules=True)

        def _hessian_provider(current_model, device):
            return [
                torch.full_like(
                    param,
                    99.0 if name.startswith("fc1.") else 8.0,
                    device=device,
                )
                for name, param in current_model.named_parameters()
            ]

        attack = layerwise.LayerwiseAdaptiveARIMAAttackMNISTIMDB(
            k=2,
            random_round=2,
            model=model,
            lambda_h=0.25,
            hessian_provider=_hessian_provider,
            retrain_interval=0,
            err_threshold=float("inf"),
            **self._selection_kwargs(),
        )
        _record_two_module_warmup(attack, model)
        attack.arima_started = True
        attack.arima_start_round = 2
        attack._fixed_origin_round = 2
        attack.offline_model_trained = True

        with patch.object(
            attack,
            "_predict_state",
            side_effect=lambda state, horizon: np.full(
                state.size, 4.0, dtype=np.float32
            ),
        ):
            submitted = _outputs_by_parameter_name(
                model,
                attack.get_fake_gradient(2, self.device, model),
            )

        # Unselected modules bypass Hessian fusion exactly.  Selected modules
        # submit 0.25 * 4 + 0.75 * 8 = 7, which is the value that must be
        # cached and scored when the same-round observation arrives.
        self.assertTrue(
            torch.equal(
                submitted["fc1.weight"], torch.full_like(model.fc1.weight, 2.0)
            )
        )
        self.assertTrue(
            torch.equal(
                submitted["fc1.bias"], torch.full_like(model.fc1.bias, 2.0)
            )
        )
        self.assertTrue(
            torch.equal(
                submitted["fc2.weight"], torch.full_like(model.fc2.weight, 7.0)
            )
        )
        self.assertTrue(
            torch.equal(
                submitted["fc2.bias"], torch.full_like(model.fc2.bias, 7.0)
            )
        )
        np.testing.assert_allclose(attack._last_pred_flat, 7.0)

        attack.record_global_gradient(
            _two_module_gradient(model, fc1_value=100.0, fc2_value=5.0),
            round_num=2,
        )

        self.assertAlmostEqual(attack._err_ema, 2.0)
        self.assertEqual(attack._aligned_observations, 1)
        self.assertIsNone(attack._last_pred_round)

    def test_online_moirai_tunes_from_actual_hessian_fused_submission(self):
        model = MNISTMLP(two_modules=True)

        def _hessian_provider(current_model, device):
            return [
                torch.full_like(
                    param,
                    99.0 if name.startswith("fc1.") else 8.0,
                    device=device,
                )
                for name, param in current_model.named_parameters()
            ]

        attack = layerwise.LayerwiseOnlineMoiraiAttackMNISTIMDB(
            k=2,
            random_round=2,
            model=model,
            history_length=2,
            lambda_h=0.25,
            hessian_provider=_hessian_provider,
            tune_lr=0.5,
            tune_steps=1,
            tune_start_round=0,
            adapt_interval=0,
            err_threshold=float("inf"),
            min_rounds_after_start=1,
            **self._selection_kwargs(),
        )
        _record_two_module_warmup(attack, model)
        for state in attack.layer_states:
            state.moirai_model = object()
            state.moirai_predictor = _RecordingPredictor(
                np.full(state.compressed_dim, 4.0, dtype=np.float32)
            )
            if state.module_name == "fc2":
                state.residual.fill(1.0)

        first_submission = _outputs_by_parameter_name(
            model,
            attack.get_fake_gradient(2, self.device, model),
        )

        # Selected submission is 0.25 * (4 + 1) + 0.75 * 8 = 7.25.
        # Unselected layers must remain the exact recent-two mean (2), even
        # though their supplied Hessian value is deliberately very large.
        self.assertTrue(
            torch.equal(
                first_submission["fc1.weight"],
                torch.full_like(model.fc1.weight, 2.0),
            )
        )
        self.assertTrue(
            torch.equal(
                first_submission["fc1.bias"], torch.full_like(model.fc1.bias, 2.0)
            )
        )
        self.assertTrue(
            torch.equal(
                first_submission["fc2.weight"],
                torch.full_like(model.fc2.weight, 7.25),
            )
        )
        self.assertTrue(
            torch.equal(
                first_submission["fc2.bias"], torch.full_like(model.fc2.bias, 7.25)
            )
        )
        np.testing.assert_allclose(attack._last_pred_flat, 7.25)

        attack.record_global_gradient(
            _two_module_gradient(model, fc1_value=100.0, fc2_value=5.0),
            round_num=2,
        )

        # The residual lives before fusion, so one gradient step includes the
        # fusion derivative: 1 - 0.5 * 0.25 * (7.25 - 5) = 0.71875.  Using
        # the pre-fusion value (4 + 1) would incorrectly leave it unchanged.
        self.assertAlmostEqual(attack._err_ema, 2.25)
        for state in attack.layer_states:
            expected = 0.71875 if state.module_name == "fc2" else 0.0
            np.testing.assert_allclose(state.residual, expected)

        second_submission = _outputs_by_parameter_name(
            model,
            attack.get_fake_gradient(3, self.device, model),
        )
        first_distance = torch.abs(first_submission["fc2.weight"] - 5.0)
        second_distance = torch.abs(second_submission["fc2.weight"] - 5.0)
        self.assertTrue(torch.all(second_distance < first_distance))
        self.assertTrue(
            torch.equal(
                second_submission["fc1.weight"],
                torch.full_like(model.fc1.weight, 51.5),
            )
        )
        self.assertTrue(
            torch.equal(
                second_submission["fc1.bias"],
                torch.full_like(model.fc1.bias, 51.5),
            )
        )

    def test_adaptive_arima_selection_survives_reset_and_refit(self):
        model = MNISTMLP(two_modules=True)
        attack = layerwise.LayerwiseAdaptiveARIMAAttackMNISTIMDB(
            k=2,
            random_round=2,
            model=model,
            lambda_h=1.0,
            retrain_interval=1,
            trim_history=False,
            **self._selection_kwargs(),
        )
        _record_two_module_warmup(attack, model)

        with patch.object(
            layerwise,
            "auto_arima",
            side_effect=lambda series, *_args, **_kwargs: _RecordingARIMA(series),
        ):
            attack.train_offline_arima_model(fit_round=2, reason="initial_test")
        initial_scores = dict(attack.selection_scores)
        self.assertEqual(attack.selected_modules, ("fc2",))
        self.assertEqual(len(attack.selection_telemetry), 1)

        attack._reset_arima_state()
        attack.record_global_gradient(
            _two_module_gradient(model, fc1_value=1000.0, fc2_value=0.5),
            round_num=2,
        )
        attack.record_global_gradient(
            _two_module_gradient(model, fc1_value=1000.0, fc2_value=0.5),
            round_num=3,
        )
        with patch.object(
            layerwise,
            "auto_arima",
            side_effect=lambda series, *_args, **_kwargs: _RecordingARIMA(series),
        ):
            attack.train_offline_arima_model(fit_round=4, reason="refit_test")

        self.assertEqual(attack.selected_modules, ("fc2",))
        self.assertEqual(attack.selection_scores, initial_scores)
        self.assertEqual(len(attack.selection_telemetry), 1)
        self.assertEqual(
            {state.name for state in attack.layer_states if state.selected},
            {"fc2.weight", "fc2.bias"},
        )


class NonLayerARIMATest(unittest.TestCase):
    def setUp(self):
        self.model = MNISTMLP(size=1)
        self.device = torch.device("cpu")

    def test_offline_arima_cannot_fit_before_configured_warmup(self):
        attack = nonlayer.OfflineARIMAAttack(
            k=2,
            random_round=10,
            model_params=list(self.model.parameters()),
            lambda_h=1.0,
        )
        attack.global_gradient_history = [
            np.asarray([value], dtype=np.float32) for value in range(8)
        ]
        with patch.object(attack, "check_arima_ready", return_value=True), patch.object(
            attack, "train_offline_arima_model"
        ) as train:
            attack.get_fake_gradient(8, self.device, self.model)
        self.assertFalse(attack.arima_started)
        train.assert_not_called()

    def test_nonlayer_readiness_includes_newest_observation(self):
        histories = [0.0, 0.0, 0.0, 1000.0]
        for attack_class in (
            nonlayer.OfflineARIMAAttack,
            nonlayer.AdaptiveARIMAAttack,
        ):
            with self.subTest(attack_class=attack_class.__name__):
                attack = attack_class(
                    k=2,
                    random_round=0,
                    model_params=list(self.model.parameters()),
                    var_window=2,
                    var_threshold=1.0,
                    lambda_h=1.0,
                )
                attack.global_gradient_history = [
                    np.asarray([value], dtype=np.float32)
                    for value in histories
                ]
                self.assertFalse(attack.check_arima_ready())

    def test_offline_arima_uses_increasing_fixed_origin_horizon(self):
        attack = nonlayer.OfflineARIMAAttack(
            k=2,
            random_round=0,
            model_params=list(self.model.parameters()),
            lambda_h=1.0,
        )
        recording_model = _RecordingARIMA()
        attack.global_gradient_history = [np.asarray([0.0], dtype=np.float32)]
        attack.arima_started = True
        attack.arima_start_round = 5
        attack.offline_model_trained = True
        attack.offline_model = {0: [recording_model]}

        first = attack.get_fake_gradient(5, self.device, self.model)[0].item()
        third = attack.get_fake_gradient(7, self.device, self.model)[0].item()

        self.assertEqual(recording_model.predict_horizons, [1, 3])
        self.assertEqual(first, 1.0)
        self.assertEqual(third, 3.0)

    def test_offline_arima_failed_dimension_falls_back_to_recent_mean(self):
        attack = nonlayer.OfflineARIMAAttack(
            k=2,
            random_round=0,
            model_params=list(self.model.parameters()),
            lambda_h=1.0,
        )
        attack.global_gradient_history = [
            np.asarray([2.0], dtype=np.float32),
            np.asarray([4.0], dtype=np.float32),
        ]
        attack.arima_started = True
        attack.arima_start_round = 2
        attack.offline_model_trained = True
        attack.offline_model = {0: [None]}

        predicted = attack.get_fake_gradient(2, self.device, self.model)[0].item()

        self.assertEqual(predicted, 3.0)

    def test_adaptive_arima_keeps_model_frozen_and_scores_only_same_round(self):
        attack = nonlayer.AdaptiveARIMAAttack(
            k=2,
            random_round=0,
            model_params=list(self.model.parameters()),
            lambda_h=1.0,
            retrain_interval=0,
            err_threshold=float("inf"),
        )
        recording_model = _RecordingARIMA()
        attack.arima_started = True
        attack.arima_start_round = 2
        attack.offline_model_trained = True
        attack.offline_model = {0: [recording_model]}
        attack._last_pred_round = 2
        attack._last_pred_flat = np.asarray([1.0], dtype=np.float32)

        with patch.object(
            recording_model,
            "update",
            side_effect=AssertionError("adaptive observation must not update frozen ARIMA"),
        ) as update:
            attack.record_global_gradient([torch.tensor([2.0])], round_num=3)
            self.assertIsNone(attack._err_ema)
            self.assertIsNone(attack._last_pred_round)
            self.assertIsNone(attack._last_pred_flat)

            attack._last_pred_round = 4
            attack._last_pred_flat = np.asarray([1.0], dtype=np.float32)
            attack.record_global_gradient([torch.tensor([3.0])], round_num=4)

        update.assert_not_called()
        self.assertEqual(recording_model.updates, [])
        self.assertAlmostEqual(attack._err_ema, 2.0)
        self.assertEqual(attack._aligned_observations, 1)
        self.assertEqual(attack._skipped_alignments, 1)
        self.assertIsNone(attack._last_pred_round)
        self.assertIsNone(attack._last_pred_flat)

    def test_adaptive_arima_caches_actual_hessian_fused_submission(self):
        hessian_calls = []

        def _hessian_provider(current_model, device):
            hessian_calls.append((current_model, device))
            return [torch.full_like(param, 10.0, device=device) for param in current_model.parameters()]

        attack = nonlayer.AdaptiveARIMAAttack(
            k=1,
            random_round=0,
            model_params=list(self.model.parameters()),
            lambda_h=0.25,
            hessian_provider=_hessian_provider,
            retrain_interval=0,
            min_rounds_after_start=1,
            err_threshold=float("inf"),
        )
        attack.global_gradient_history = [np.asarray([0.0], dtype=np.float32)]
        attack.arima_started = True
        attack.arima_start_round = 2
        attack._fixed_origin_round = 2
        attack.offline_model_trained = True
        attack.offline_model = {0: [_RecordingARIMA()]}

        submitted = attack.get_fake_gradient(2, self.device, self.model)[0]

        self.assertEqual(len(hessian_calls), 1)
        self.assertTrue(torch.equal(submitted, torch.tensor([7.75])))
        np.testing.assert_allclose(attack._last_pred_flat, [7.75])

        attack.record_global_gradient([submitted.clone()], round_num=2)

        self.assertEqual(attack._aligned_observations, 1)
        self.assertEqual(attack._err_ema, 0.0)

    def test_adaptive_arima_resets_fixed_origin_after_periodic_refit(self):
        attack = nonlayer.AdaptiveARIMAAttack(
            k=2,
            random_round=0,
            model_params=list(self.model.parameters()),
            lambda_h=1.0,
            retrain_interval=1,
            err_threshold=float("inf"),
        )
        old_model = _RecordingARIMA()
        replacement_model = _RecordingARIMA()
        attack.global_gradient_history = [np.asarray([0.0], dtype=np.float32)]
        attack.arima_started = True
        attack.arima_start_round = 5
        attack.offline_model_trained = True
        attack.offline_model = {0: [old_model]}
        attack._fixed_origin_round = 5
        attack._last_full_refit_observation = 0

        first = attack.get_fake_gradient(5, self.device, self.model)[0].item()
        with patch.object(nonlayer, "auto_arima", return_value=replacement_model) as fit:
            attack.record_global_gradient([torch.tensor([2.0])], round_num=5)

        fit.assert_called_once()
        self.assertEqual(first, 1.0)
        self.assertEqual(old_model.predict_horizons, [1])
        self.assertEqual(old_model.updates, [])
        self.assertEqual(attack._fixed_origin_round, 6)
        self.assertIs(attack.offline_model[0][0], replacement_model)

        attack.retrain_interval = 0
        after_refit = attack.get_fake_gradient(6, self.device, self.model)[0].item()
        attack.record_global_gradient([torch.tensor([1.0])], round_num=6)
        third_from_refit = attack.get_fake_gradient(8, self.device, self.model)[0].item()

        self.assertEqual(replacement_model.predict_horizons, [1, 3])
        self.assertEqual(replacement_model.updates, [])
        self.assertEqual(after_refit, 1.0)
        self.assertEqual(third_from_refit, 3.0)

    def test_reset_history_does_not_bypass_fresh_observation_cooldown(self):
        attack = nonlayer.AdaptiveARIMAAttack(
            k=2,
            random_round=0,
            model_params=list(self.model.parameters()),
            lambda_h=1.0,
            retrain_interval=0,
        )
        attack.global_gradient_history = [
            np.asarray([value], dtype=np.float32) for value in range(8)
        ]
        attack._reset_arima_state()

        with patch.object(attack, "check_arima_ready", return_value=True), patch.object(
            attack, "train_offline_arima_model"
        ) as train:
            attack.get_fake_gradient(20, self.device, self.model)
        self.assertFalse(attack.arima_started)
        train.assert_not_called()

        for round_num in range(20, 20 + attack.k + attack.var_window):
            attack.record_global_gradient(
                [torch.tensor([float(round_num)])], round_num=round_num
            )
        with patch.object(attack, "check_arima_ready", return_value=True), patch.object(
            attack, "train_offline_arima_model"
        ) as train:
            attack.get_fake_gradient(30, self.device, self.model)
        self.assertTrue(attack.arima_started)
        train.assert_called_once()


class LayerwiseARIMATest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_layerwise_readiness_includes_newest_observation(self):
        model = MNISTMLP(size=1)
        attack = layerwise.LayerwiseOfflineARIMAAttackMNISTIMDB(
            k=2,
            random_round=0,
            model=model,
            var_window=2,
            var_threshold=1.0,
            lambda_h=1.0,
        )
        for value in (0.0, 0.0, 0.0, 1000.0):
            attack.record_global_gradient([torch.tensor([value])])
        self.assertFalse(attack.check_arima_ready())

    def test_offline_layerwise_uses_increasing_fixed_origin_horizon(self):
        model = MNISTMLP(size=2)
        attack = layerwise.LayerwiseOfflineARIMAAttackMNISTIMDB(
            k=2,
            random_round=0,
            model=model,
            lambda_h=1.0,
        )
        recording_models = [_RecordingARIMA(), _RecordingARIMA()]
        state = attack.layer_states[0]
        state.raw_history.append(np.asarray([0.0, 0.0], dtype=np.float32))
        attack.global_gradient_history.append([state.raw_history[-1]])
        state.arima_models = {"ipca": None, "models": recording_models}
        attack.arima_started = True
        attack.arima_start_round = 5
        attack._fixed_origin_round = 5
        attack.offline_model_trained = True

        first = attack.get_fake_gradient(5, self.device, model)[0]
        third = attack.get_fake_gradient(7, self.device, model)[0]

        self.assertTrue(torch.equal(first, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(third, torch.tensor([3.0, 3.0])))
        for recording_model in recording_models:
            self.assertEqual(recording_model.predict_horizons, [1, 3])

    def test_offline_layerwise_failed_dimension_uses_recent_layer_mean(self):
        model = MNISTMLP(size=2)
        attack = layerwise.LayerwiseOfflineARIMAAttackMNISTIMDB(
            k=2,
            random_round=0,
            model=model,
            lambda_h=1.0,
        )
        state = attack.layer_states[0]
        first = np.asarray([2.0, 6.0], dtype=np.float32)
        second = np.asarray([4.0, 10.0], dtype=np.float32)
        state.raw_history.extend([first, second])
        attack.global_gradient_history.extend([[first], [second]])
        state.arima_models = {"ipca": None, "models": [None, None]}
        attack.arima_started = True
        attack.arima_start_round = 2
        attack._fixed_origin_round = 2
        attack.offline_model_trained = True

        predicted = attack.get_fake_gradient(2, self.device, model)[0]

        self.assertTrue(torch.equal(predicted, torch.tensor([3.0, 8.0])))

    def test_adaptive_layerwise_keeps_models_frozen_and_scores_exact_round(self):
        model = MNISTMLP(size=3)
        attack = layerwise.LayerwiseAdaptiveARIMAAttackMNISTIMDB(
            k=2,
            random_round=0,
            model=model,
            lambda_h=1.0,
            retrain_interval=0,
            err_threshold=float("inf"),
        )
        state = attack.layer_states[0]
        state.use_pca = True
        state.compressed_dim = 2
        old_history = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 1.0], [2.0, 1.0, 3.0]],
            dtype=np.float64,
        )
        frozen_ipca = _NumpyIncrementalPCA(n_components=2).fit(old_history)
        recording_models = [_RecordingARIMA(), _RecordingARIMA()]
        state.raw_history.extend(row.astype(np.float32) for row in old_history)
        attack.global_gradient_history.extend(
            [[row.astype(np.float32)] for row in old_history]
        )
        # This fixture manually injects an active-model prediction cache, so
        # explicitly establish the selection invariant that get_fake_gradient
        # would establish in the real call path first.
        attack._ensure_layer_selection(selection_round=3)
        state.arima_models = {
            "ipca": frozen_ipca,
            "models": recording_models,
            "fit_observations": len(old_history),
        }
        attack.arima_started = True
        attack.arima_start_round = 3
        attack.offline_model_trained = True
        attack._last_pred_round = 3
        attack._last_pred_flat = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

        target = torch.tensor([3.0, 4.0, 2.0])
        with patch.object(
            attack,
            "_fit_ipca",
            side_effect=AssertionError("online observation must not refit PCA"),
        ), patch.object(
            recording_models[0],
            "update",
            side_effect=AssertionError("adaptive observation must not update frozen ARIMA"),
        ) as first_update, patch.object(
            recording_models[1],
            "update",
            side_effect=AssertionError("adaptive observation must not update frozen ARIMA"),
        ) as second_update:
            attack.record_global_gradient([target], round_num=3)

        first_update.assert_not_called()
        second_update.assert_not_called()
        self.assertAlmostEqual(attack._err_ema, 2.0)
        for recording_model in recording_models:
            self.assertEqual(recording_model.updates, [])
        self.assertIs(state.arima_models["ipca"], frozen_ipca)
        self.assertIsNone(attack._last_pred_round)
        self.assertIsNone(attack._last_pred_flat)

    def test_adaptive_layerwise_resets_fixed_origin_after_periodic_refit(self):
        model = MNISTMLP(size=2)
        attack = layerwise.LayerwiseAdaptiveARIMAAttackMNISTIMDB(
            k=2,
            random_round=0,
            model=model,
            lambda_h=1.0,
            retrain_interval=1,
            err_threshold=float("inf"),
        )
        state = attack.layer_states[0]
        initial = np.asarray([0.0, 0.0], dtype=np.float32)
        state.raw_history.append(initial)
        attack.global_gradient_history.append([initial])
        old_models = [_RecordingARIMA(), _RecordingARIMA()]
        replacement_models = [_RecordingARIMA(), _RecordingARIMA()]
        state.arima_models = {
            "ipca": None,
            "models": old_models,
            "fit_observations": 1,
        }
        attack.arima_started = True
        attack.arima_start_round = 5
        attack.offline_model_trained = True
        attack._fixed_origin_round = 5
        attack._observations_since_full_refit = 0

        first = attack.get_fake_gradient(5, self.device, model)[0]
        with patch.object(
            layerwise,
            "auto_arima",
            side_effect=replacement_models,
        ) as fit:
            attack.record_global_gradient([torch.tensor([2.0, 4.0])], round_num=5)

        self.assertEqual(fit.call_count, 2)
        self.assertTrue(torch.equal(first, torch.tensor([1.0, 1.0])))
        self.assertEqual(attack._fixed_origin_round, 6)
        self.assertEqual(attack.arima_fit_telemetry[-1]["origin_round"], 6)
        for old_model in old_models:
            self.assertEqual(old_model.predict_horizons, [1])
            self.assertEqual(old_model.updates, [])

        attack.retrain_interval = 0
        after_refit = attack.get_fake_gradient(6, self.device, model)[0]
        attack.record_global_gradient([torch.tensor([1.0, 1.0])], round_num=6)
        third_from_refit = attack.get_fake_gradient(8, self.device, model)[0]

        self.assertTrue(torch.equal(after_refit, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(third_from_refit, torch.tensor([3.0, 3.0])))
        for replacement_model in replacement_models:
            self.assertEqual(replacement_model.predict_horizons, [1, 3])
            self.assertEqual(replacement_model.updates, [])


class MoiraiAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def _run_nonlayer_residual_updates(self, rounds, adapt_interval):
        model = MNISTMLP(size=1)
        attack = nonlayer.OnlineMoiraiAttack(
            k=1,
            random_round=0,
            history_length=1,
            lambda_h=1.0,
            tune_lr=0.5,
            tune_start_round=0,
            adapt_interval=adapt_interval,
            residual_decay=0.5,
            err_threshold=float("inf"),
        )
        attack.D = 1
        attack.param_shapes = [model.weight.shape]
        attack.param_sizes = [1]
        attack.residual = np.ones(1, dtype=np.float32)
        attack.model = object()
        attack._predictor = _RecordingPredictor([0.0])
        attack.global_gradient_history = [np.asarray([0.0], dtype=np.float32)]

        with patch("builtins.print") as print_mock:
            for round_num in rounds:
                submitted = attack.get_fake_gradient(
                    round_num, self.device, model
                )[0]
                # Make the calibration error zero so any residual change is
                # solely attributable to interval decay.
                attack.record_global_gradient(
                    [submitted.detach().clone()], round_num=round_num
                )
        telemetry = [
            " ".join(str(arg) for arg in call.args)
            for call in print_mock.call_args_list
        ]
        return attack, telemetry

    def _run_layerwise_residual_updates(self, rounds, adapt_interval):
        model = MNISTMLP(size=1)
        attack = layerwise.LayerwiseOnlineMoiraiAttackMNISTIMDB(
            k=1,
            random_round=0,
            model=model,
            history_length=1,
            lambda_h=1.0,
            tune_lr=0.5,
            tune_start_round=0,
            adapt_interval=adapt_interval,
            residual_decay=0.5,
            err_threshold=float("inf"),
            selection_fraction=1.0,
            pca_threshold=1000,
        )
        state = attack.layer_states[0]
        state.residual.fill(1.0)
        state.moirai_model = object()
        state.moirai_predictor = _RecordingPredictor([0.0])
        attack.record_global_gradient([torch.tensor([0.0])], round_num=0)

        with patch("builtins.print") as print_mock:
            for round_num in rounds:
                submitted = attack.get_fake_gradient(
                    round_num, self.device, model
                )[0]
                # Make the calibration error zero so any residual change is
                # solely attributable to interval decay.
                attack.record_global_gradient(
                    [submitted.detach().clone()], round_num=round_num
                )
        telemetry = [
            " ".join(str(arg) for arg in call.args)
            for call in print_mock.call_args_list
        ]
        return attack, telemetry

    def test_nonlayer_online_uses_submitted_forecast_for_residual_update_once(self):
        model = MNISTMLP(size=2)
        attack = nonlayer.OnlineMoiraiAttack(
            k=2,
            random_round=0,
            history_length=2,
            lambda_h=1.0,
            tune_lr=0.5,
            tune_steps=1,
            tune_start_round=0,
            adapt_interval=0,
            err_threshold=float("inf"),
        )
        predictor = _RecordingPredictor([4.0, 4.0], [99.0, 99.0])
        attack.D = 2
        attack.param_shapes = [model.weight.shape]
        attack.param_sizes = [model.weight.numel()]
        attack.residual = np.zeros(2, dtype=np.float32)
        attack.model = object()
        attack._predictor = predictor
        attack.global_gradient_history = [
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0], dtype=np.float32),
        ]

        submitted = attack.get_fake_gradient(2, self.device, model)[0]
        attack.record_global_gradient([torch.tensor([2.0, 2.0])], round_num=2)

        self.assertEqual(predictor.calls, 1)
        self.assertTrue(torch.equal(submitted, torch.tensor([4.0, 4.0])))
        np.testing.assert_allclose(attack.residual, [-1.0, -1.0])
        self.assertIsNone(attack._last_pred_round)

    def test_nonlayer_online_uses_hessian_fused_submission_and_chain_rule(self):
        model = MNISTMLP(size=2)
        hessian_calls = []

        def _hessian_provider(current_model, device):
            hessian_calls.append((current_model, device))
            return [torch.full_like(param, 8.0, device=device) for param in current_model.parameters()]

        attack = nonlayer.OnlineMoiraiAttack(
            k=2,
            random_round=0,
            history_length=2,
            lambda_h=0.25,
            hessian_provider=_hessian_provider,
            tune_lr=0.4,
            tune_steps=2,
            tune_start_round=0,
            adapt_interval=0,
            err_threshold=float("inf"),
        )
        predictor = _RecordingPredictor([4.0, 4.0])
        attack.D = 2
        attack.param_shapes = [model.weight.shape]
        attack.param_sizes = [model.weight.numel()]
        attack.residual = np.asarray([2.0, 2.0], dtype=np.float32)
        attack.model = object()
        attack._predictor = predictor
        attack.global_gradient_history = [
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 1.0], dtype=np.float32),
        ]

        submitted = attack.get_fake_gradient(2, self.device, model)[0]

        self.assertEqual(predictor.calls, 1)
        self.assertEqual(len(hessian_calls), 1)
        self.assertTrue(torch.equal(submitted, torch.tensor([7.5, 7.5])))
        np.testing.assert_allclose(attack._last_pred_flat, [7.5, 7.5])
        np.testing.assert_allclose(attack._last_base_pred_flat, [7.0, 7.0])

        attack.record_global_gradient(
            [torch.tensor([6.5, 6.5])], round_num=2
        )

        self.assertEqual(attack._aligned_observations, 1)
        self.assertEqual(attack._err_ema, 1.0)
        np.testing.assert_allclose(attack.residual, [1.8025, 1.8025])
        self.assertIsNone(attack._last_pred_round)

    def test_nonlayer_online_ignores_stale_prediction_on_honest_round(self):
        model = MNISTMLP(size=1)
        attack = nonlayer.OnlineMoiraiAttack(
            k=1,
            random_round=0,
            history_length=1,
            lambda_h=1.0,
            tune_lr=0.5,
            tune_start_round=0,
            adapt_interval=0,
            err_threshold=float("inf"),
        )
        predictor = _RecordingPredictor([4.0])
        attack.D = 1
        attack.param_shapes = [model.weight.shape]
        attack.param_sizes = [1]
        attack.residual = np.zeros(1, dtype=np.float32)
        attack.model = object()
        attack._predictor = predictor
        attack.global_gradient_history = [np.asarray([1.0], dtype=np.float32)]

        attack.get_fake_gradient(2, self.device, model)
        attack.record_global_gradient([torch.tensor([2.0])], round_num=3)

        self.assertEqual(predictor.calls, 1)
        np.testing.assert_allclose(attack.residual, [0.0])
        self.assertIsNone(attack._err_ema)
        self.assertIsNone(attack._last_pred_round)

    def test_layerwise_online_does_not_resample_or_refit_on_observation(self):
        model = MNISTMLP(size=3)
        attack = layerwise.LayerwiseOnlineMoiraiAttackMNISTIMDB(
            k=2,
            random_round=0,
            model=model,
            history_length=2,
            lambda_h=1.0,
            tune_lr=0.5,
            tune_start_round=0,
            adapt_interval=0,
            err_threshold=float("inf"),
        )
        predictor = _RecordingPredictor([4.0, 4.0, 4.0], [99.0, 99.0, 99.0])
        state = attack.layer_states[0]
        state.moirai_model = object()
        state.moirai_predictor = predictor
        attack.record_global_gradient([torch.tensor([0.0, 0.0, 0.0])], round_num=0)
        attack.record_global_gradient([torch.tensor([1.0, 1.0, 1.0])], round_num=1)

        submitted = attack.get_fake_gradient(2, self.device, model)[0]
        attack.record_global_gradient([torch.tensor([2.0, 2.0, 2.0])], round_num=2)

        self.assertEqual(predictor.calls, 1)
        self.assertTrue(torch.equal(submitted, torch.tensor([4.0, 4.0, 4.0])))
        np.testing.assert_allclose(state.residual, [-1.0, -1.0, -1.0])

    def test_layerwise_online_does_not_guess_alignment_without_round_number(self):
        model = MNISTMLP(size=1)
        attack = layerwise.LayerwiseOnlineMoiraiAttackMNISTIMDB(
            k=1,
            random_round=0,
            model=model,
            history_length=1,
            lambda_h=1.0,
            tune_lr=0.5,
            tune_start_round=0,
            adapt_interval=0,
            err_threshold=float("inf"),
        )
        predictor = _RecordingPredictor([4.0])
        state = attack.layer_states[0]
        state.moirai_model = object()
        state.moirai_predictor = predictor
        attack.record_global_gradient([torch.tensor([1.0])], round_num=0)
        attack.get_fake_gradient(1, self.device, model)

        # The inferred legacy sequence number happens to be 1, but it is not
        # authoritative enough to calibrate a cached submitted prediction.
        attack.record_global_gradient([torch.tensor([2.0])])

        np.testing.assert_allclose(state.residual, [0.0])
        self.assertIsNone(attack._err_ema)
        self.assertIsNone(attack._last_pred_round)

    def test_nonlayer_decay_counts_updates_across_fixed_gap_cadences(self):
        # Fake rounds under fixed-gap 2 are odd (1, 3, ...), while fixed-gap
        # 6 begins with consecutive fake rounds (1, 2, ...).  Both schedules
        # must decay after the same number of successful calibrations.
        for rounds in ((1, 3), (1, 2)):
            with self.subTest(rounds=rounds):
                attack, telemetry = self._run_nonlayer_residual_updates(
                    rounds, adapt_interval=2
                )
                self.assertEqual(attack._residual_update_count, 2)
                np.testing.assert_allclose(attack.residual, [0.5])
                self.assertTrue(
                    any(
                        "residual_updates=2" in line
                        and "decay_applied=1" in line
                        for line in telemetry
                    )
                )

    def test_layerwise_decay_counts_updates_across_fixed_gap_cadences(self):
        for rounds in ((1, 3), (1, 2)):
            with self.subTest(rounds=rounds):
                attack, telemetry = self._run_layerwise_residual_updates(
                    rounds, adapt_interval=2
                )
                self.assertEqual(attack._residual_update_count, 2)
                np.testing.assert_allclose(
                    attack.layer_states[0].residual, [0.5]
                )
                self.assertTrue(
                    any(
                        "residual_updates=2" in line
                        and "decay_applied=1" in line
                        for line in telemetry
                    )
                )

    def test_hard_reset_clears_residual_update_counter(self):
        nonlayer_attack, _ = self._run_nonlayer_residual_updates(
            (1,), adapt_interval=2
        )
        self.assertEqual(nonlayer_attack._residual_update_count, 1)
        nonlayer_attack._hard_reset()
        self.assertEqual(nonlayer_attack._residual_update_count, 0)
        np.testing.assert_allclose(nonlayer_attack.residual, [0.0])

        layerwise_attack, _ = self._run_layerwise_residual_updates(
            (1,), adapt_interval=2
        )
        self.assertEqual(layerwise_attack._residual_update_count, 1)
        layerwise_attack._hard_reset()
        self.assertEqual(layerwise_attack._residual_update_count, 0)
        np.testing.assert_allclose(
            layerwise_attack.layer_states[0].residual, [0.0]
        )

    def test_zero_adapt_interval_disables_decay_but_counts_updates(self):
        nonlayer_attack, nonlayer_telemetry = (
            self._run_nonlayer_residual_updates((1, 2), adapt_interval=0)
        )
        self.assertEqual(nonlayer_attack._residual_update_count, 2)
        np.testing.assert_allclose(nonlayer_attack.residual, [1.0])
        self.assertFalse(
            any("decay_applied=1" in line for line in nonlayer_telemetry)
        )

        layerwise_attack, layerwise_telemetry = (
            self._run_layerwise_residual_updates((1, 2), adapt_interval=0)
        )
        self.assertEqual(layerwise_attack._residual_update_count, 2)
        np.testing.assert_allclose(
            layerwise_attack.layer_states[0].residual, [1.0]
        )
        self.assertFalse(
            any("decay_applied=1" in line for line in layerwise_telemetry)
        )


class FLCEWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "FLCE_updated.py"
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_moirai_online_cli_parameters_exist(self):
        flags = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
        expected = {
            "--moirai_online_tuning",
            "--moirai_tune_lr",
            "--moirai_tune_steps",
            "--moirai_tune_start_round",
            "--moirai_adapt_interval",
            "--moirai_residual_decay",
            "--moirai_residual_clip",
            "--moirai_err_ema_beta",
            "--moirai_err_threshold",
            "--moirai_min_rounds_after_start",
            "--moirai_trim_history",
        }
        self.assertTrue(expected.issubset(flags), expected - flags)

    def test_arima_readiness_cli_routes_only_to_four_arima_constructors(self):
        flags = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
        expected_flags = {"--arima_var_window", "--arima_var_threshold"}
        self.assertTrue(expected_flags.issubset(flags), expected_flags - flags)

        expected_constructors = {
            "OfflineARIMAAttack",
            "AdaptiveARIMAAttack",
            "LayerwiseOfflineARIMAAttackMNISTIMDB",
            "LayerwiseAdaptiveARIMAAttackMNISTIMDB",
        }
        expanded_by = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "arima_readiness_parameters"
                for keyword in node.keywords
            ):
                expanded_by.append(node.func.id)
        self.assertEqual(set(expanded_by), expected_constructors)
        self.assertEqual(len(expanded_by), len(expected_constructors))

    def test_arima_readiness_effective_defaults_preserve_both_routes(self):
        helper = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_effective_arima_readiness_parameters"
        )
        namespace = {
            "LAYERWISE_ARIMA_METHODS": {
                "arima_mnist_imdb_layerwise",
                "arima_adaptive_mnist_imdb_layerwise",
            }
        }
        module = ast.Module(body=[helper], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(self.path), "exec"), namespace)

        parsed = argparse.Namespace(arima_var_window=3, arima_var_threshold=None)
        ordinary = namespace["_effective_arima_readiness_parameters"](
            parsed, "arima"
        )
        layerwise_values = namespace["_effective_arima_readiness_parameters"](
            parsed, "arima_mnist_imdb_layerwise"
        )
        self.assertEqual(ordinary, {"var_window": 3, "var_threshold": 5e-3})
        self.assertEqual(
            layerwise_values, {"var_window": 3, "var_threshold": 5e-2}
        )

        parsed.arima_var_threshold = 0.1
        self.assertEqual(
            namespace["_effective_arima_readiness_parameters"](
                parsed, "arima_adaptive_mnist_imdb_layerwise"
            )["var_threshold"],
            0.1,
        )

    def test_num_epoch_cli_defaults_to_50_and_accepts_override(self):
        matching_calls = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if any(
                isinstance(arg, ast.Constant) and arg.value == "--num_epoch"
                for arg in node.args
            ):
                matching_calls.append(node)

        self.assertEqual(len(matching_calls), 1)
        call = matching_calls[0]
        keyword_values = {keyword.arg: keyword.value for keyword in call.keywords}
        self.assertEqual(ast.literal_eval(keyword_values["default"]), 50)
        self.assertIsInstance(keyword_values["type"], ast.Name)
        self.assertEqual(keyword_values["type"].id, "int")

        parser = argparse.ArgumentParser()
        parser.add_argument("--num_epoch", type=int, default=50)
        self.assertEqual(parser.parse_args([]).num_epoch, 50)
        self.assertEqual(parser.parse_args(["--num_epoch", "7"]).num_epoch, 7)

    def test_num_epoch_routes_from_args_without_hardcoded_assignment(self):
        assignments = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "num_epoch" for target in node.targets):
                assignments.append(node.value)

        self.assertEqual(len(assignments), 1)
        value = assignments[0]
        self.assertIsInstance(value, ast.Attribute)
        self.assertIsInstance(value.value, ast.Name)
        self.assertEqual(value.value.id, "args")
        self.assertEqual(value.attr, "num_epoch")

        validations = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and isinstance(node.test.left.value, ast.Name)
            and node.test.left.value.id == "args"
            and node.test.left.attr == "num_epoch"
        ]
        self.assertTrue(validations, "--num_epoch must be validated before the run")

    def test_layerwise_selection_and_pca_cli_parameters_exist(self):
        flags = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
        expected = {
            "--layerwise_selection_metric",
            "--layerwise_selection_fraction",
            "--layerwise_selection_top_n",
            "--layerwise_pca_threshold",
            "--layerwise_pca_components",
        }
        self.assertTrue(expected.issubset(flags), expected - flags)

    def test_only_layerwise_constructors_receive_layerwise_parameters(self):
        expected_constructors = {
            "LayerwiseOfflineARIMAAttackMNISTIMDB",
            "LayerwiseAdaptiveARIMAAttackMNISTIMDB",
            "LayerwiseOfflineMoiraiAttackMNISTIMDB",
            "LayerwiseOnlineMoiraiAttackMNISTIMDB",
        }
        expanded_by = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "layerwise_parameters"
                for keyword in node.keywords
            ):
                expanded_by.append(node.func.id)

        self.assertEqual(set(expanded_by), expected_constructors)
        self.assertEqual(len(expanded_by), len(expected_constructors))

    def test_runner_passes_round_number_to_round_aware_attacker(self):
        matching_calls = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "record_global_gradient":
                matching_calls.append(node)
        self.assertTrue(
            any(any(keyword.arg == "round_num" for keyword in call.keywords) for call in matching_calls)
        )

    def test_layerwise_arima_override_detects_equals_style_cli(self):
        helper = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_cli_option_was_supplied"
        )
        namespace = {"sys": sys}
        module = ast.Module(body=[helper], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(self.path), "exec"), namespace)

        original_argv = sys.argv
        try:
            sys.argv = ["FLCE_updated.py", "--arima_err_threshold=0.03"]
            self.assertTrue(
                namespace["_cli_option_was_supplied"]("--arima_err_threshold")
            )
            self.assertFalse(
                namespace["_cli_option_was_supplied"](
                    "--arima_min_rounds_after_start"
                )
            )
        finally:
            sys.argv = original_argv

    def test_adaptive_arima_effective_defaults_match_actual_constructors(self):
        helpers = [
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "_cli_option_was_supplied",
                "_effective_arima_adaptive_parameters",
            }
        ]
        namespace = {"sys": sys}
        module = ast.Module(body=helpers, type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(self.path), "exec"), namespace)

        parsed = argparse.Namespace(
            arima_err_ema_beta=0.9,
            arima_err_threshold=0.005,
            arima_min_rounds_after_start=2,
            arima_retrain_interval=10,
            arima_trim_history=True,
        )
        original_argv = sys.argv
        try:
            sys.argv = ["FLCE_updated.py"]
            ordinary = namespace["_effective_arima_adaptive_parameters"](
                parsed, "tictactoe", "arima_adaptive"
            )
            mnist = namespace["_effective_arima_adaptive_parameters"](
                parsed, "mnist", "arima_adaptive_mnist_imdb_layerwise"
            )
            imdb = namespace["_effective_arima_adaptive_parameters"](
                parsed, "imdb", "arima_adaptive_mnist_imdb_layerwise"
            )
            self.assertEqual(ordinary["err_threshold"], 0.005)
            self.assertEqual(ordinary["min_rounds_after_start"], 2)
            self.assertEqual(mnist["err_threshold"], 0.035)
            self.assertEqual(mnist["min_rounds_after_start"], 4)
            self.assertEqual(imdb["err_threshold"], 0.015)
            self.assertEqual(imdb["min_rounds_after_start"], 4)

            parsed.arima_err_threshold = 0.08
            parsed.arima_min_rounds_after_start = 6
            sys.argv = [
                "FLCE_updated.py",
                "--arima_err_threshold=0.08",
                "--arima_min_rounds_after_start",
                "6",
            ]
            explicit = namespace["_effective_arima_adaptive_parameters"](
                parsed, "imdb", "arima_adaptive_mnist_imdb_layerwise"
            )
            self.assertEqual(explicit["err_threshold"], 0.08)
            self.assertEqual(explicit["min_rounds_after_start"], 6)
        finally:
            sys.argv = original_argv

    def test_online_moirai_constructors_receive_explicit_adapter_parameters(self):
        expected_keywords = {
            "tune_lr",
            "tune_steps",
            "tune_start_round",
            "adapt_interval",
            "residual_decay",
            "residual_clip",
            "err_ema_beta",
            "err_threshold",
            "min_rounds_after_start",
            "trim_history",
        }
        constructors = {
            "OnlineMoiraiAttack": False,
            "LayerwiseOnlineMoiraiAttackMNISTIMDB": False,
        }
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in constructors:
                constructors[node.func.id] = any(
                    keyword.arg is None
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "moirai_online_parameters"
                    for keyword in node.keywords
                )
        for name, expands_effective_parameters in constructors.items():
            self.assertTrue(expands_effective_parameters, name)

        helper = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_effective_moirai_online_parameters"
        )
        returned_keys = set()
        for node in ast.walk(helper):
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
                continue
            returned_keys.update(
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
        self.assertTrue(
            expected_keywords.issubset(returned_keys),
            expected_keywords - returned_keys,
        )

    def test_provenance_manifest_is_exclusive_and_hashes_support_modules(self):
        writer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_write_run_provenance"
        )
        string_constants = {
            node.value
            for node in ast.walk(writer)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("attack_modules", string_constants)
        self.assertIn("entry_sha256", string_constants)
        self.assertIn("attack_module_sha256", string_constants)
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
                and any(
                    isinstance(arg, ast.Constant) and arg.value == "x"
                    for arg in node.args
                )
                for node in ast.walk(writer)
            )
        )

    def test_forecast_fake_rounds_skip_private_local_training(self):
        assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "FORECAST_FREE_RIDER_METHODS"
                for target in node.targets
            )
        )
        methods = {
            element.value
            for element in assignment.value.elts
            if isinstance(element, ast.Constant)
        }
        expected = {
            "replay",
            "arima",
            "arima_adaptive",
            "moirai",
            "moirai_online",
            "arima_mnist_imdb_layerwise",
            "arima_adaptive_mnist_imdb_layerwise",
            "moirai_mnist_imdb_layerwise",
            "moirai_online_mnist_imdb_layerwise",
        }
        self.assertEqual(methods, expected)
        self.assertIn("elif forecast_fake_round:", self.source)
        self.assertIn("private_training={'skipped' if forecast_fake_round", self.source)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Real-package smoke checks for the updated ARIMA and Moirai attackers.

This script is intentionally independent of pytest and the FL training loop.  It
can be copied either into ``<project>/tests`` or directly into an isolated copy
of the project root.  It discovers the project from its location/current
directory, never modifies experiment data, and only writes a report when
``--json-output`` is explicitly supplied.

Examples
--------
Lightweight checks (no model download and no Moirai inference)::

    python tests/server_smoke_moirai_arima.py --component all --dataset all

Use an existing local Moirai checkpoint for one real forecast::

    python tests/server_smoke_moirai_arima.py --component moirai \
        --dataset mnist --run-moirai-forecast --device cuda:0

Allow a missing checkpoint to be downloaded from Hugging Face and exercise the
real attack forecast/state machine as well::

    python tests/server_smoke_moirai_arima.py --component moirai \
        --dataset imdb --run-moirai-forecast --run-moirai-attack-forecast \
        --allow-download --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DATASETS = ("tictactoe", "adult", "dota2", "mnist", "imdb")
NONLAYER_DATASETS = frozenset(("tictactoe", "adult", "dota2"))
LAYERWISE_DATASETS = frozenset(("mnist", "imdb"))


class SmokeSkip(RuntimeError):
    """A deliberate, non-failing skip for an opt-in expensive operation."""


@dataclass
class SmokeResult:
    status: str
    component: str
    check: str
    elapsed_s: float
    details: dict[str, Any]


class SmokeRunner:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = bool(verbose)
        self.results: list[SmokeResult] = []
        self.started = time.perf_counter()

    @staticmethod
    def _safe_details(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return {"result": value}

    def run(self, component: str, check: str, fn: Callable[[], Any]) -> None:
        started = time.perf_counter()
        status = "PASS"
        details: dict[str, Any]
        try:
            details = self._safe_details(fn())
        except SmokeSkip as exc:
            status = "SKIP"
            details = {"reason": str(exc)}
        except Exception as exc:  # each check must be reported before continuing
            status = "FAIL"
            details = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if self.verbose:
                details["traceback"] = traceback.format_exc()
        elapsed = time.perf_counter() - started
        result = SmokeResult(status, component, check, elapsed, details)
        self.results.append(result)
        print(
            f"SMOKE {status} component={component} check={check} "
            f"elapsed_s={elapsed:.3f} details={json.dumps(details, sort_keys=True, default=str)}",
            flush=True,
        )

    def finish(self) -> int:
        counts = {
            status: sum(result.status == status for result in self.results)
            for status in ("PASS", "SKIP", "FAIL")
        }
        status = "FAIL" if counts["FAIL"] else "PASS"
        elapsed = time.perf_counter() - self.started
        print(
            f"SMOKE SUMMARY status={status} pass={counts['PASS']} "
            f"skip={counts['SKIP']} fail={counts['FAIL']} elapsed_s={elapsed:.3f}",
            flush=True,
        )
        return 1 if counts["FAIL"] else 0


def discover_project_root() -> Path:
    markers = (
        Path("module/method/attack_updated_nonlayerwise.py"),
        Path("module/method/attack_updated_layerwise.py"),
        Path("module/model/model.py"),
        Path("uni2ts/src"),
    )
    candidates: list[Path] = []
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        candidates.extend((start, *start.parents))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / marker).exists() for marker in markers):
            return candidate
    raise FileNotFoundError(
        "Could not locate the FLCE attack project. Expected module/method/"
        "attack_updated_nonlayerwise.py and uni2ts/src near the script or cwd."
    )


PROJECT_ROOT = discover_project_root()
os.chdir(PROJECT_ROOT)
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "uni2ts" / "src"):
    import_text = str(import_root)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def import_runtime() -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import pmdarima
    import torch

    nonlayer = importlib.import_module("module.method.attack_updated_nonlayerwise")
    layerwise = importlib.import_module("module.method.attack_updated_layerwise")
    model_module = importlib.import_module("module.model.model")
    moirai_module = importlib.import_module("uni2ts.model.moirai")
    return {
        "np": np,
        "pd": pd,
        "pmdarima": pmdarima,
        "torch": torch,
        "nonlayer": nonlayer,
        "layerwise": layerwise,
        "model_module": model_module,
        "MoiraiForecast": moirai_module.MoiraiForecast,
    }


def resolve_device(torch: Any, requested: str) -> Any:
    device = torch.device(requested)
    if device.type == "cuda":
        require(torch.cuda.is_available(), f"CUDA requested ({device}) but torch.cuda.is_available() is false")
        if device.index is not None:
            require(
                device.index < torch.cuda.device_count(),
                f"CUDA device index {device.index} is out of range; count={torch.cuda.device_count()}",
            )
    return device


def parse_datasets(raw_values: Sequence[str] | None) -> list[str]:
    if not raw_values:
        return list(DATASETS)
    requested: list[str] = []
    for raw in raw_values:
        for item in raw.split(","):
            dataset = item.strip().lower()
            if not dataset:
                continue
            if dataset == "all":
                return list(DATASETS)
            if dataset not in DATASETS:
                raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASETS} or 'all'")
            if dataset not in requested:
                requested.append(dataset)
    require(bool(requested), "At least one dataset must be selected")
    return requested


def model_name_for_dataset(dataset: str) -> str:
    return {
        "tictactoe": "TicTacToeMLP",
        "adult": "AdultMLP",
        "dota2": "Dota2MLP",
        "mnist": "MNISTMLP",
        "imdb": "IMDBMLP",
    }[dataset]


def build_model(runtime: dict[str, Any], dataset: str, device: Any, args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {
        "seed": args.seed,
        "lr": 1e-3,
        "num_epoch": 1,
        "hidden_layer_size": args.hidden_size,
        "device": device,
        "batch_size": 4,
    }
    if dataset == "imdb":
        kwargs["input_size"] = args.imdb_input_size
    model = runtime["model_module"].return_model(model_name_for_dataset(dataset), **kwargs)
    require(model is not None, f"return_model returned None for dataset={dataset}")
    require(sum(parameter.numel() for parameter in model.parameters()) > 0, "model has no parameters")
    return model


def tensor_list_close(torch: Any, actual: Iterable[Any], expected: Iterable[Any], atol: float = 2e-4) -> None:
    actual_list = list(actual)
    expected_list = list(expected)
    require(len(actual_list) == len(expected_list), "gradient tensor counts differ")
    for index, (left, right) in enumerate(zip(actual_list, expected_list)):
        require(left.shape == right.shape, f"tensor {index} shape mismatch: {left.shape} != {right.shape}")
        require(bool(torch.isfinite(left).all()), f"tensor {index} contains NaN/Inf")
        require(
            bool(torch.allclose(left, right, atol=atol, rtol=1e-4)),
            f"tensor {index} mismatch; max_abs={float((left - right).abs().max().item()):.6e}",
        )


def synthetic_gradient(torch: Any, model: Any, round_index: int = 0, varying: bool = False) -> list[Any]:
    gradients: list[Any] = []
    parameter_count = len(list(model.parameters()))
    for index, parameter in enumerate(model.parameters()):
        # Earlier modules receive larger scores, so layerwise selection is
        # deterministic and selects fc1 in the current MNIST/IMDB MLPs.
        base = 0.01 * float(parameter_count - index)
        drift = (round_index * 1e-4 * float(index + 1)) if varying else 0.0
        gradients.append(torch.full_like(parameter, base + drift))
    return gradients


class TrackingHessian:
    def __init__(self, torch: Any, scale: float) -> None:
        self.torch = torch
        self.scale = float(scale)
        self.calls = 0

    def __call__(self, model: Any, device: Any) -> list[Any]:
        self.calls += 1
        return [self.scale * self.torch.ones_like(parameter, device=device) for parameter in model.parameters()]


def check_imports(runtime: dict[str, Any], datasets: Sequence[str], device: Any, args: argparse.Namespace) -> dict[str, Any]:
    required_nonlayer = (
        "OfflineARIMAAttack",
        "AdaptiveARIMAAttack",
        "OfflineMoiraiAttack",
        "OnlineMoiraiAttack",
        "frobenius_scaled_identity_provider",
        "_resolve_moirai_source",
        "_load_moirai_module",
    )
    required_layerwise = (
        "LayerwiseOfflineARIMAAttackMNISTIMDB",
        "LayerwiseAdaptiveARIMAAttackMNISTIMDB",
        "LayerwiseOfflineMoiraiAttackMNISTIMDB",
        "LayerwiseOnlineMoiraiAttackMNISTIMDB",
    )
    for name in required_nonlayer:
        require(hasattr(runtime["nonlayer"], name), f"missing nonlayer symbol: {name}")
    for name in required_layerwise:
        require(hasattr(runtime["layerwise"], name), f"missing layerwise symbol: {name}")

    for cls_name in required_nonlayer[:4]:
        signature = inspect.signature(getattr(runtime["nonlayer"], cls_name))
        require("hessian_provider" in signature.parameters, f"{cls_name} lacks hessian_provider")
    for cls_name in required_layerwise:
        signature = inspect.signature(getattr(runtime["layerwise"], cls_name))
        for parameter in ("hessian_provider", "selection_metric", "selection_fraction", "pca_components"):
            require(parameter in signature.parameters, f"{cls_name} lacks {parameter}")

    models = {}
    for dataset in datasets:
        model = build_model(runtime, dataset, device, args)
        models[dataset] = {
            "class": model.__class__.__name__,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        }
    torch = runtime["torch"]
    return {
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "pmdarima": runtime["pmdarima"].__version__,
        "device": str(device),
        "models": models,
        "nonlayer_file": str(Path(runtime["nonlayer"].__file__).resolve()),
        "layerwise_file": str(Path(runtime["layerwise"].__file__).resolve()),
    }


def check_real_pmdarima(runtime: dict[str, Any]) -> dict[str, Any]:
    np = runtime["np"]
    series = np.asarray(
        [0.08 * index + np.sin(index / 2.5) for index in range(24)],
        dtype=np.float64,
    )
    model = runtime["pmdarima"].auto_arima(
        series,
        start_p=0,
        start_q=0,
        max_p=2,
        max_q=2,
        d=None,
        max_d=1,
        seasonal=False,
        stationary=False,
        error_action="raise",
        suppress_warnings=True,
        stepwise=True,
        maxiter=20,
    )
    forecast = np.asarray(model.predict(n_periods=2), dtype=np.float64)
    require(forecast.shape == (2,), f"unexpected ARIMA forecast shape: {forecast.shape}")
    require(bool(np.isfinite(forecast).all()), "ARIMA forecast contains NaN/Inf")
    return {
        "order": tuple(int(value) for value in model.order),
        "observations": int(series.size),
        "forecast": forecast.tolist(),
    }


def _record_history(attacker: Any, histories: Sequence[list[Any]], round_aware: bool = False) -> None:
    for round_index, gradient in enumerate(histories):
        if round_aware:
            attacker.record_global_gradient(gradient, round_num=round_index)
        else:
            attacker.record_global_gradient(gradient)


def check_arima_attack_state(runtime: dict[str, Any], dataset: str, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    np = runtime["np"]
    torch = runtime["torch"]
    model = build_model(runtime, dataset, device, args)
    k = 2
    warmup = 4
    var_window = 2
    active_round = warmup
    histories = [synthetic_gradient(torch, model) for _ in range(warmup)]
    hessian_scale = 0.037
    lambda_h = 0.73
    methods: dict[str, Any] = {}

    if dataset in NONLAYER_DATASETS:
        classes = (
            ("offline", runtime["nonlayer"].OfflineARIMAAttack, False),
            ("adaptive", runtime["nonlayer"].AdaptiveARIMAAttack, True),
        )
        for method_name, attack_cls, round_aware in classes:
            provider = TrackingHessian(torch, hessian_scale)
            kwargs: dict[str, Any] = {
                "k": k,
                "random_round": warmup,
                "model_params": list(model.parameters()),
                "var_window": var_window,
                "var_threshold": 1.0,
                "lambda_h": lambda_h,
                "hessian_provider": provider,
            }
            if method_name == "adaptive":
                kwargs.update(
                    err_threshold=1e9,
                    min_rounds_after_start=1,
                    retrain_interval=0,
                )
            attacker = attack_cls(**kwargs)
            _record_history(attacker, histories, round_aware=round_aware)
            submitted = attacker.get_fake_gradient(active_round, device, model)
            require(provider.calls == 1, f"{method_name}: Hessian provider was not called exactly once")
            expected = [
                lambda_h * source + (1.0 - lambda_h) * hessian_scale
                for source in histories[-1]
            ]
            tensor_list_close(torch, submitted, expected)
            require("arima" in attacker.last_source, f"{method_name}: did not enter ARIMA forecast state")

            alignment_mae = None
            if method_name == "adaptive":
                cached_before_observe = torch.as_tensor(attacker._last_pred_flat, device=device)
                submitted_flat = torch.cat([tensor.reshape(-1) for tensor in submitted])
                cache_max_abs = float(
                    (cached_before_observe - submitted_flat).abs().max().item()
                )
                attacker.record_global_gradient(
                    [tensor.detach().clone() for tensor in submitted],
                    round_num=active_round,
                )
                alignment_mae = attacker._err_ema
                require(attacker._aligned_observations == 1, "adaptive: same-round observation was not aligned")
                require(
                    cache_max_abs <= 2e-6
                    and alignment_mae is not None
                    and abs(float(alignment_mae)) <= 2e-6,
                    "adaptive: cached prediction is not the actual Hessian-fused submission; "
                    f"cache_max_abs={cache_max_abs:.6e}, same_round_MAE={alignment_mae}",
                )
            methods[method_name] = {
                "source": attacker.last_source,
                "fit_stats": dict(attacker._fit_stats),
                "hessian_calls": provider.calls,
                "actual_submission_mae": alignment_mae,
            }
    else:
        classes = (
            ("offline_layerwise", runtime["layerwise"].LayerwiseOfflineARIMAAttackMNISTIMDB, False),
            ("adaptive_layerwise", runtime["layerwise"].LayerwiseAdaptiveARIMAAttackMNISTIMDB, True),
        )
        for method_name, attack_cls, round_aware in classes:
            provider = TrackingHessian(torch, hessian_scale)
            kwargs = {
                "k": k,
                "random_round": warmup,
                "model": model,
                "var_window": var_window,
                "var_threshold": 1.0,
                "lambda_h": lambda_h,
                "hessian_provider": provider,
                "selection_top_n": 1,
                "pca_threshold": args.layerwise_pca_threshold,
                "pca_components": args.layerwise_pca_components,
            }
            if method_name == "adaptive_layerwise":
                kwargs.update(
                    err_threshold=1e9,
                    min_rounds_after_start=1,
                    retrain_interval=0,
                )
            attacker = attack_cls(**kwargs)
            _record_history(attacker, histories, round_aware=round_aware)
            submitted = attacker.get_fake_gradient(active_round, device, model)
            require(provider.calls == 1, f"{method_name}: Hessian provider was not called exactly once")
            expected: list[Any] = []
            for state, source in zip(attacker.layer_states, histories[-1]):
                if state.selected:
                    expected.append(lambda_h * source + (1.0 - lambda_h) * hessian_scale)
                else:
                    expected.append(source)
            tensor_list_close(torch, submitted, expected, atol=6e-4)
            require(attacker._selection_frozen, f"{method_name}: layer selection was not frozen")
            require(len(attacker.selected_modules) == 1, f"{method_name}: expected one selected module")

            alignment_mae = None
            if method_name == "adaptive_layerwise":
                attacker.record_global_gradient(
                    [tensor.detach().clone() for tensor in submitted],
                    round_num=active_round,
                )
                alignment_mae = attacker._err_ema
                require(attacker._aligned_observations == 1, "adaptive layerwise alignment failed")
                require(
                    alignment_mae is not None and abs(float(alignment_mae)) <= 2e-6,
                    f"adaptive layerwise actual-submission MAE should be zero, got {alignment_mae}",
                )
            methods[method_name] = {
                "source": attacker.last_source,
                "selected_modules": list(attacker.selected_modules),
                "layer_sources": dict(attacker.last_layer_sources),
                "fit_stats": dict(attacker._fit_stats),
                "hessian_calls": provider.calls,
                "actual_submission_mae": alignment_mae,
            }

    return {
        "dataset": dataset,
        "model": model.__class__.__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "methods": methods,
    }


def resolve_moirai(runtime: dict[str, Any], args: argparse.Namespace) -> tuple[str, bool]:
    if args.moirai_source:
        source = str(Path(args.moirai_source).expanduser().resolve()) if Path(args.moirai_source).expanduser().exists() else args.moirai_source
    else:
        source = str(runtime["nonlayer"]._resolve_moirai_source(args.moirai_size))
    return source, Path(source).exists()


def check_moirai_resolution(runtime: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source, is_local = resolve_moirai(runtime, args)
    if args.moirai_source and not is_local and "/" not in source:
        raise FileNotFoundError(f"Explicit --moirai-source does not exist and is not a Hub repo id: {source}")
    return {
        "source": source,
        "source_kind": "local" if is_local else "hub",
        "allow_download": bool(args.allow_download),
        "forecast_requested": bool(args.run_moirai_forecast),
        "attack_forecast_requested": bool(args.run_moirai_attack_forecast),
    }


def require_moirai_load_permission(runtime: dict[str, Any], args: argparse.Namespace) -> tuple[str, bool]:
    source, is_local = resolve_moirai(runtime, args)
    if not is_local and not args.allow_download:
        raise RuntimeError(
            f"Moirai source resolves to Hub ({source}) but downloads are disabled. "
            "Provide a local checkpoint or add --allow-download."
        )
    return source, is_local


def check_real_moirai_forecast(runtime: dict[str, Any], device: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_moirai_forecast:
        raise SmokeSkip("real Moirai load/forecast is opt-in; add --run-moirai-forecast")
    source, is_local = require_moirai_load_permission(runtime, args)
    np = runtime["np"]
    pd = runtime["pd"]
    module = runtime["nonlayer"]._load_moirai_module(source)
    forecast_model = runtime["MoiraiForecast"](
        module=module,
        prediction_length=1,
        context_length=args.moirai_context_length,
        patch_size="auto",
        num_samples=1,
        target_dim=2,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = forecast_model.create_predictor(batch_size=1, device=str(device))
    steps = max(args.moirai_context_length + 2, 8)
    target = np.stack(
        (
            np.linspace(0.0, 0.7, steps, dtype=np.float32),
            np.cos(np.arange(steps, dtype=np.float32) / 3.0),
        ),
        axis=0,
    )
    forecast = next(
        predictor.predict(
            [{"start": pd.Period("2000-01-01", freq="D"), "target": target}]
        )
    )
    samples = np.asarray(forecast.samples)
    require(samples.size >= 2, f"Moirai returned too few values: shape={samples.shape}")
    require(bool(np.isfinite(samples).all()), "Moirai forecast contains NaN/Inf")
    return {
        "source": source,
        "source_kind": "local" if is_local else "hub",
        "samples_shape": list(samples.shape),
        "sample_preview": samples.reshape(-1)[:4].astype(float).tolist(),
        "device": str(device),
    }


def check_moirai_warmup_state(runtime: dict[str, Any], dataset: str, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    torch = runtime["torch"]
    model = build_model(runtime, dataset, device, args)
    history_length = max(3, args.moirai_context_length)
    gradient = synthetic_gradient(torch, model)
    methods: dict[str, Any] = {}
    if dataset in NONLAYER_DATASETS:
        classes = (
            ("offline", runtime["nonlayer"].OfflineMoiraiAttack, False),
            ("online", runtime["nonlayer"].OnlineMoiraiAttack, True),
        )
        for method_name, attack_cls, round_aware in classes:
            provider = TrackingHessian(torch, 0.037)
            kwargs: dict[str, Any] = {
                "k": 2,
                "random_round": history_length,
                "history_length": history_length,
                "size": args.moirai_size,
                "lambda_h": 0.73,
                "hessian_provider": provider,
            }
            if method_name == "online":
                kwargs.update(err_threshold=1e9, min_rounds_after_start=1)
            attacker = attack_cls(**kwargs)
            if round_aware:
                attacker.record_global_gradient(gradient, round_num=0)
            else:
                attacker.record_global_gradient(gradient)
            # Both current nonlayer get_fake_gradient implementations call
            # _lazy_init before checking the warm-up branch. Calling them here
            # would load/download Moirai despite this being the default cheap
            # check. Exercise the real history and averaging helpers instead;
            # the opt-in attack-forecast check below covers get_fake_gradient.
            submitted = attacker._avg_fallback(device, model)
            tensor_list_close(torch, submitted, gradient)
            require(provider.calls == 0, "Hessian must not run on the warm-up mean path")
            require(attacker.D is None and attacker._predictor is None, "default check unexpectedly loaded Moirai")
            methods[method_name] = {
                "source": "history_recorded_avg_helper",
                "hessian_calls": provider.calls,
                "weights_loaded": False,
            }
    else:
        classes = (
            ("offline_layerwise", runtime["layerwise"].LayerwiseOfflineMoiraiAttackMNISTIMDB, False),
            ("online_layerwise", runtime["layerwise"].LayerwiseOnlineMoiraiAttackMNISTIMDB, True),
        )
        for method_name, attack_cls, round_aware in classes:
            provider = TrackingHessian(torch, 0.037)
            kwargs = {
                "k": 2,
                "random_round": history_length,
                "model": model,
                "history_length": history_length,
                "size": args.moirai_size,
                "lambda_h": 0.73,
                "hessian_provider": provider,
                "selection_top_n": 1,
                "pca_threshold": args.layerwise_pca_threshold,
                "pca_components": args.layerwise_pca_components,
            }
            if method_name == "online_layerwise":
                kwargs.update(err_threshold=1e9, min_rounds_after_start=1)
            attacker = attack_cls(**kwargs)
            if round_aware:
                attacker.record_global_gradient(gradient, round_num=0)
            else:
                attacker.record_global_gradient(gradient)
            submitted = attacker.get_fake_gradient(0, device, model)
            tensor_list_close(torch, submitted, gradient)
            require(provider.calls == 0, "Hessian must not run on the warm-up mean path")
            require(not attacker._selection_frozen, "layer selection must not freeze before warm-up ends")
            methods[method_name] = {"source": attacker.last_source, "hessian_calls": provider.calls}
    return {"dataset": dataset, "methods": methods}


def check_moirai_attack_forecast(runtime: dict[str, Any], dataset: str, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_moirai_attack_forecast:
        raise SmokeSkip("real attack-level Moirai forecast is opt-in; add --run-moirai-attack-forecast")
    source, is_local = require_moirai_load_permission(runtime, args)
    if args.moirai_source:
        # Attack APIs accept a size token rather than an arbitrary source.  An
        # explicit source is therefore valid for the direct forecast check but
        # cannot be injected here without replacing production loading logic.
        resolved_from_size = str(runtime["nonlayer"]._resolve_moirai_source(args.moirai_size))
        require(
            str(Path(resolved_from_size).resolve()) == str(Path(source).resolve()) or resolved_from_size == source,
            "--moirai-source differs from the production source selected by --moirai-size; "
            "attack-level smoke must exercise the production resolver",
        )

    np = runtime["np"]
    torch = runtime["torch"]
    model = build_model(runtime, dataset, device, args)
    history_length = max(3, args.moirai_context_length)
    histories = [
        synthetic_gradient(torch, model, round_index=index, varying=True)
        for index in range(history_length)
    ]
    active_round = history_length
    hessian_scale = 0.037
    lambda_h = 0.73
    methods: dict[str, Any] = {}

    if dataset in NONLAYER_DATASETS:
        classes = (
            ("offline", runtime["nonlayer"].OfflineMoiraiAttack, False),
            ("online", runtime["nonlayer"].OnlineMoiraiAttack, True),
        )
        for method_name, attack_cls, round_aware in classes:
            provider = TrackingHessian(torch, hessian_scale)
            kwargs: dict[str, Any] = {
                "k": 2,
                "random_round": history_length,
                "history_length": history_length,
                "size": args.moirai_size,
                "lambda_h": lambda_h,
                "hessian_provider": provider,
            }
            if method_name == "online":
                kwargs.update(
                    err_threshold=1e9,
                    min_rounds_after_start=1,
                    tune_lr=1e-3,
                    tune_steps=1,
                )
            attacker = attack_cls(**kwargs)
            _record_history(attacker, histories, round_aware=round_aware)

            fused_inputs: dict[str, Any] = {}
            if method_name == "online":
                original_hessian_fuse = attacker._hessian_fuse

                def capture_hessian_input(
                    predicted_gradient: Sequence[Any],
                    current_model: Any,
                    current_device: Any,
                ) -> list[Any]:
                    fused_inputs["predicted_flat"] = torch.cat(
                        [tensor.reshape(-1) for tensor in predicted_gradient]
                    ).detach().clone()
                    fused_inputs["residual"] = np.asarray(
                        attacker.residual, dtype=np.float32
                    ).copy()
                    return original_hessian_fuse(
                        predicted_gradient, current_model, current_device
                    )

                attacker._hessian_fuse = capture_hessian_input

            submitted = attacker.get_fake_gradient(active_round, device, model)
            require(provider.calls == 1, f"{method_name}: Hessian provider call count={provider.calls}")
            for index, tensor in enumerate(submitted):
                require(bool(torch.isfinite(tensor).all()), f"{method_name}: tensor {index} contains NaN/Inf")
            require("moirai" in attacker.last_source, f"{method_name}: did not enter Moirai forecast state")

            alignment_mae = None
            if method_name == "online":
                submitted_flat = torch.cat([tensor.reshape(-1) for tensor in submitted])
                predicted_flat = fused_inputs["predicted_flat"]
                residual_at_submit = fused_inputs["residual"]
                expected = lambda_h * predicted_flat + (1.0 - lambda_h) * hessian_scale
                require(
                    bool(torch.allclose(submitted_flat, expected, atol=2e-5, rtol=1e-4)),
                    "online Moirai returned submission does not match Hessian fusion formula",
                )
                cached_submission = torch.as_tensor(attacker._last_pred_flat, device=device)
                cache_max_abs = float(
                    (cached_submission - submitted_flat).abs().max().item()
                )
                expected_fused_base = (
                    submitted_flat.detach().cpu().numpy()
                    - lambda_h * residual_at_submit
                )
                require(
                    bool(
                        np.allclose(
                            attacker._last_base_pred_flat,
                            expected_fused_base,
                            atol=2e-5,
                            rtol=1e-4,
                        )
                    ),
                    "online Moirai fused-base cache is inconsistent with the submitted residual",
                )
                attacker.record_global_gradient(
                    [tensor.detach().clone() for tensor in submitted],
                    round_num=active_round,
                )
                alignment_mae = attacker._err_ema
                require(attacker._aligned_observations == 1, "online Moirai same-round alignment failed")
                require(
                    cache_max_abs <= 2e-6
                    and alignment_mae is not None
                    and abs(float(alignment_mae)) <= 2e-6,
                    "online Moirai: cached prediction is not the actual Hessian-fused submission; "
                    f"cache_max_abs={cache_max_abs:.6e}, same_round_MAE={alignment_mae}",
                )
            methods[method_name] = {
                "source": attacker.last_source,
                "hessian_calls": provider.calls,
                "actual_submission_mae": alignment_mae,
            }
    else:
        classes = (
            ("offline_layerwise", runtime["layerwise"].LayerwiseOfflineMoiraiAttackMNISTIMDB, False),
            ("online_layerwise", runtime["layerwise"].LayerwiseOnlineMoiraiAttackMNISTIMDB, True),
        )
        for method_name, attack_cls, round_aware in classes:
            provider = TrackingHessian(torch, hessian_scale)
            kwargs = {
                "k": 2,
                "random_round": history_length,
                "model": model,
                "history_length": history_length,
                "size": args.moirai_size,
                "lambda_h": lambda_h,
                "hessian_provider": provider,
                "selection_top_n": 1,
                "pca_threshold": args.layerwise_pca_threshold,
                "pca_components": args.layerwise_pca_components,
            }
            if method_name == "online_layerwise":
                kwargs.update(err_threshold=1e9, min_rounds_after_start=1)
            attacker = attack_cls(**kwargs)
            _record_history(attacker, histories, round_aware=round_aware)
            submitted = attacker.get_fake_gradient(active_round, device, model)
            require(provider.calls == 1, f"{method_name}: Hessian provider call count={provider.calls}")
            require(attacker._selection_frozen, f"{method_name}: selection not frozen")
            require(len(attacker.selected_modules) == 1, f"{method_name}: expected one selected module")
            for index, tensor in enumerate(submitted):
                require(bool(torch.isfinite(tensor).all()), f"{method_name}: tensor {index} contains NaN/Inf")
            for state, tensor, history_index in zip(
                attacker.layer_states,
                submitted,
                range(len(attacker.layer_states)),
            ):
                if not state.selected:
                    recent_mean = sum(
                        history[history_index] for history in histories[-2:]
                    ) / 2.0
                    require(
                        bool(torch.allclose(tensor, recent_mean, atol=2e-5, rtol=1e-4)),
                        f"{method_name}: unselected layer {state.name} was changed by Hessian/forecast",
                    )

            selected_pca_states = [
                state
                for state in attacker.layer_states
                if state.selected and state.use_pca
            ]
            require(
                selected_pca_states,
                f"{method_name}: synthetic layerwise gate did not select a PCA-eligible tensor",
            )
            effective_pca_dims: dict[str, int] = {}
            for state in selected_pca_states:
                expected_dim = min(
                    args.layerwise_pca_components,
                    history_length - 1,
                    state.size - 1,
                )
                require(
                    state.moirai_ipca is not None,
                    f"{method_name}: selected tensor {state.name} has no frozen PCA basis",
                )
                require(
                    int(state.moirai_ipca.n_components_) == expected_dim,
                    f"{method_name}: tensor {state.name} frozen PCA rank="
                    f"{state.moirai_ipca.n_components_}, expected {expected_dim}",
                )
                require(
                    state.moirai_effective_dim == expected_dim,
                    f"{method_name}: tensor {state.name} target_dim="
                    f"{state.moirai_effective_dim}, expected {expected_dim}",
                )
                effective_pca_dims[state.name] = int(state.moirai_effective_dim)

            alignment_mae = None
            if method_name == "online_layerwise":
                cached_submission = torch.as_tensor(attacker._last_pred_flat, device=device)
                selected_submission = torch.cat(
                    [
                        tensor.reshape(-1)
                        for state, tensor in zip(attacker.layer_states, submitted)
                        if state.selected
                    ]
                )
                require(
                    bool(torch.allclose(cached_submission, selected_submission, atol=2e-5, rtol=1e-4)),
                    "online layerwise cache is not the actual selected-layer submission",
                )
                attacker.record_global_gradient(
                    [tensor.detach().clone() for tensor in submitted],
                    round_num=active_round,
                )
                alignment_mae = attacker._err_ema
                require(attacker._aligned_observations == 1, "online layerwise same-round alignment failed")
                require(
                    alignment_mae is not None and abs(float(alignment_mae)) <= 2e-6,
                    f"online layerwise actual-submission MAE should be zero, got {alignment_mae}",
                )
            methods[method_name] = {
                "source": attacker.last_source,
                "selected_modules": list(attacker.selected_modules),
                "layer_sources": dict(attacker.last_layer_sources),
                "effective_pca_dims": effective_pca_dims,
                "hessian_calls": provider.calls,
                "actual_submission_mae": alignment_mae,
            }

    return {
        "dataset": dataset,
        "model": model.__class__.__name__,
        "production_source": source,
        "source_kind": "local" if is_local else "hub",
        "methods": methods,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-package smoke validation for updated ARIMA/Moirai attacks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--component", choices=("imports", "arima", "moirai", "all"), default="imports")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name, comma-separated names, or 'all'; option may be repeated",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=1, help="Small hidden size for state-machine smoke")
    parser.add_argument("--imdb-input-size", type=int, default=768, help="Synthetic IMDB embedding width")
    parser.add_argument("--layerwise-pca-threshold", type=int, default=512)
    parser.add_argument("--layerwise-pca-components", type=int, default=3)
    parser.add_argument("--moirai-size", default="small", help="Production Moirai size token")
    parser.add_argument("--moirai-source", default=None, help="Optional local path/Hub id for direct forecast only")
    parser.add_argument("--moirai-context-length", type=int, default=5)
    parser.add_argument(
        "--run-moirai-forecast",
        action="store_true",
        help="Load real weights and execute one minimal direct Moirai forecast",
    )
    parser.add_argument(
        "--run-moirai-attack-forecast",
        action="store_true",
        help="Execute real offline/online attacker forecasts; potentially slow and memory-heavy",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Moirai loading from Hugging Face when no local checkpoint is present",
    )
    parser.add_argument("--json-output", type=Path, default=None, help="Optional new JSON report path")
    parser.add_argument("--verbose", action="store_true", help="Include tracebacks in failed result records")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    require(args.hidden_size >= 1, "--hidden-size must be >= 1")
    require(args.imdb_input_size >= 1, "--imdb-input-size must be >= 1")
    require(args.layerwise_pca_threshold >= 1, "--layerwise-pca-threshold must be >= 1")
    require(args.layerwise_pca_components >= 2, "--layerwise-pca-components must be >= 2")
    require(args.moirai_context_length >= 3, "--moirai-context-length must be >= 3")
    if args.run_moirai_attack_forecast:
        require(
            args.component in ("moirai", "all"),
            "--run-moirai-attack-forecast requires --component moirai or all",
        )
    if args.run_moirai_forecast:
        require(args.component in ("moirai", "all"), "--run-moirai-forecast requires --component moirai or all")


def write_json_report(path: Path, project_root: Path, results: Sequence[SmokeResult]) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {resolved}. Choose a new isolated path."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_root": str(project_root),
        "generated_at_unix": time.time(),
        "results": [asdict(result) for result in results],
    }
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"SMOKE REPORT path={resolved}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        datasets = parse_datasets(args.dataset)
    except Exception as exc:
        parser.error(str(exc))

    runner = SmokeRunner(verbose=args.verbose)
    runtime_holder: dict[str, Any] = {}

    def load_runtime() -> dict[str, Any]:
        if not runtime_holder:
            runtime_holder.update(import_runtime())
        return runtime_holder

    runner.run(
        "imports",
        "updated_modules_models_and_packages",
        lambda: check_imports(
            load_runtime(),
            datasets,
            resolve_device(load_runtime()["torch"], args.device),
            args,
        ),
    )
    if runner.results[-1].status == "FAIL":
        exit_code = runner.finish()
        if args.json_output is not None:
            write_json_report(args.json_output, PROJECT_ROOT, runner.results)
        return exit_code

    runtime = load_runtime()
    device = resolve_device(runtime["torch"], args.device)

    if args.component in ("arima", "all"):
        runner.run("arima", "real_pmdarima_fit_predict", lambda: check_real_pmdarima(runtime))
        for dataset in datasets:
            runner.run(
                "arima",
                f"{dataset}_attack_state_hessian",
                lambda dataset=dataset: check_arima_attack_state(runtime, dataset, device, args),
            )

    if args.component in ("moirai", "all"):
        runner.run("moirai", "source_resolution", lambda: check_moirai_resolution(runtime, args))
        for dataset in datasets:
            runner.run(
                "moirai",
                f"{dataset}_construct_and_warmup",
                lambda dataset=dataset: check_moirai_warmup_state(runtime, dataset, device, args),
            )
        runner.run(
            "moirai",
            "real_minimal_forecast",
            lambda: check_real_moirai_forecast(runtime, device, args),
        )
        for dataset in datasets:
            runner.run(
                "moirai",
                f"{dataset}_real_attack_forecast_hessian",
                lambda dataset=dataset: check_moirai_attack_forecast(runtime, dataset, device, args),
            )

    exit_code = runner.finish()
    if args.json_output is not None:
        try:
            write_json_report(args.json_output, PROJECT_ROOT, runner.results)
        except Exception as exc:
            print(
                f"SMOKE FAIL component=report check=json_output elapsed_s=0.000 "
                f"details={json.dumps({'error_type': type(exc).__name__, 'error': str(exc)}, sort_keys=True)}",
                flush=True,
            )
            return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

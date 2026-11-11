import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA

try:
    from pmdarima import auto_arima
except ImportError:  # Optional: required only by ARIMA attacks.
    auto_arima = None

from module.method.attack_updated_nonlayerwise import (
    _avg_history_np,
    _constant_series_model,
    _is_constant_series,
    _load_moirai_module,
    _log_fallback,
    _resolve_moirai_source,
)
try:
    from uni2ts.model.moirai import MoiraiForecast
except ImportError:  # Optional: required only by Moirai attacks.
    MoiraiForecast = None


DEFAULT_LAYERWISE_PCA_THRESHOLD = 512
DEFAULT_LAYERWISE_PCA_COMPONENTS = 8


class _LayerState:
    def __init__(
        self,
        name,
        shape,
        module_name,
        pca_threshold,
        pca_components,
    ):
        self.name = str(name)
        self.module_name = str(module_name)
        self.shape = tuple(shape)
        self.size = int(np.prod(self.shape))
        self.use_pca = self.size > int(pca_threshold)
        self.compressed_dim = self._resolve_compressed_dim(pca_components)
        self.selected = False

        self.raw_history = []

        self.arima_models = None
        self.moirai_model = None
        self.moirai_predictor = None
        self.moirai_ipca = None
        self.moirai_effective_dim = None

        self.residual = np.zeros((self.size,), dtype=np.float32)

    def _resolve_compressed_dim(self, pca_components):
        if not self.use_pca:
            return self.size
        upper = max(2, min(self.size - 1, int(pca_components)))
        return upper


class LayerwiseAttackBaseMNISTIMDB:
    def __init__(
        self,
        k,
        random_round,
        model,
        history_length=5,
        size="small",
        patch_size="auto",
        lambda_h=0.9,
        hessian_provider=None,
        selection_metric="normalized_rms",
        selection_fraction=0.5,
        selection_top_n=None,
        pca_threshold=DEFAULT_LAYERWISE_PCA_THRESHOLD,
        pca_components=DEFAULT_LAYERWISE_PCA_COMPONENTS,
    ):
        self.k = int(k)
        self.random_round = int(random_round)
        self.history_length = int(history_length)
        self.size = size
        self.patch_size = patch_size
        self.lambda_h = float(lambda_h)
        self.hessian_provider = hessian_provider
        if selection_metric not in {"normalized_rms", "mean_norm"}:
            raise ValueError(
                "selection_metric must be 'normalized_rms' or 'mean_norm', "
                f"got {selection_metric!r}"
            )
        self.selection_metric = str(selection_metric)
        self.selection_fraction = float(selection_fraction)
        if not 0.0 < self.selection_fraction <= 1.0:
            raise ValueError(
                "selection_fraction must be in (0, 1], "
                f"got {selection_fraction!r}"
            )
        if selection_top_n is not None:
            if isinstance(selection_top_n, bool):
                raise ValueError("selection_top_n must be a positive integer or None")
            resolved_top_n = int(selection_top_n)
            if resolved_top_n <= 0 or resolved_top_n != float(selection_top_n):
                raise ValueError("selection_top_n must be a positive integer or None")
            selection_top_n = resolved_top_n
        self.selection_top_n = selection_top_n
        self.pca_threshold = int(pca_threshold)
        self.pca_components = int(pca_components)
        if self.pca_threshold < 1:
            raise ValueError(
                f"pca_threshold must be >= 1, got {pca_threshold!r}"
            )
        if self.pca_components < 2:
            raise ValueError(
                f"pca_components must be >= 2, got {pca_components!r}"
            )

        self.layer_states = self._build_layer_states(model)
        if not self.layer_states:
            raise ValueError("Layerwise attacker requires at least one model parameter")
        self.selected_modules = tuple()
        self.selection_scores = {}
        self.selection_telemetry = []
        self.selection_round = None
        self._selection_frozen = False
        self.global_gradient_history = []
        self.last_source = "uninitialized"
        self.last_layer_sources = {
            state.name: "uninitialized" for state in self.layer_states
        }
        # Keep an absolute observation counter separate from the retained
        # history length. Adaptive attacks may trim history after a reset.
        self._recorded_observation_count = 0

    @staticmethod
    def _module_name_from_parameter(parameter_name):
        name = str(parameter_name)
        if "." in name:
            prefix, suffix = name.rsplit(".", 1)
            if suffix in {"weight", "bias"}:
                return prefix
        return name

    def _build_layer_states(self, model):
        model_name = model.__class__.__name__
        supported_models = {"MNISTMLP", "IMDBMLP", "TicTacToeMLP", "AdultMLP", "Dota2MLP"}
        if model_name not in supported_models:
            raise ValueError(
                f"Layerwise attackers only support {sorted(supported_models)}, got {model_name}"
            )
        return [
            _LayerState(
                name=name,
                shape=param.shape,
                module_name=self._module_name_from_parameter(name),
                pca_threshold=self.pca_threshold,
                pca_components=self.pca_components,
            )
            for name, param in model.named_parameters()
        ]

    def _split_by_layer(self, global_gradient):
        if len(global_gradient) != len(self.layer_states):
            raise ValueError(
                f"Expected {len(self.layer_states)} tensors, got {len(global_gradient)}"
            )
        return [
            np.asarray(grad.detach().reshape(-1).cpu().numpy(), dtype=np.float32)
            for grad in global_gradient
        ]

    def _record_raw_layers(self, global_gradient):
        layer_vectors = self._split_by_layer(global_gradient)
        for state, vec in zip(self.layer_states, layer_vectors):
            state.raw_history.append(vec.copy())
        self.global_gradient_history.append(layer_vectors)
        return layer_vectors

    def _layer_avg(self, state):
        return _avg_history_np(state.raw_history, self.k).astype(np.float32)

    def _module_states(self):
        grouped = {}
        for state in self.layer_states:
            grouped.setdefault(state.module_name, []).append(state)
        return grouped

    def _ensure_layer_selection(self, selection_round=None):
        """Select module groups once, using only causal warm-up observations."""
        if self._selection_frozen:
            return

        grouped = self._module_states()
        available = min(
            (len(state.raw_history) for state in self.layer_states),
            default=0,
        )
        if available <= 0:
            raise RuntimeError("Cannot select layer modules without gradient history")
        observations = (
            min(available, self.random_round)
            if self.random_round > 0
            else available
        )
        if observations <= 0:
            raise RuntimeError("Layer selection requires at least one observation")

        scores = {}
        module_sizes = {}
        for module_name, states in grouped.items():
            total_elements = int(sum(state.size for state in states))
            per_round_scores = []
            for observation_idx in range(observations):
                squared_norm = 0.0
                for state in states:
                    vec = np.asarray(
                        state.raw_history[observation_idx], dtype=np.float64
                    )
                    squared_norm += float(np.dot(vec, vec))
                module_norm = float(np.sqrt(max(0.0, squared_norm)))
                if self.selection_metric == "normalized_rms":
                    module_norm /= float(np.sqrt(max(1, total_elements)))
                per_round_scores.append(module_norm)
            score = float(np.mean(per_round_scores))
            if not np.isfinite(score):
                score = float("-inf")
            scores[module_name] = score
            module_sizes[module_name] = total_elements

        module_names = list(grouped)
        if self.selection_top_n is None:
            selected_count = int(np.ceil(self.selection_fraction * len(module_names)))
        else:
            selected_count = int(self.selection_top_n)
        selected_count = max(1, min(len(module_names), selected_count))
        module_order = {name: idx for idx, name in enumerate(module_names)}
        ranked = sorted(
            module_names,
            key=lambda name: (-scores[name], module_order[name]),
        )
        selected = tuple(ranked[:selected_count])
        selected_set = set(selected)
        for state in self.layer_states:
            state.selected = state.module_name in selected_set

        self.selected_modules = selected
        self.selection_scores = dict(scores)
        self.selection_round = (
            None if selection_round is None else int(selection_round)
        )
        telemetry = {
            "selection_round": self.selection_round,
            "history_observations": observations,
            "metric": self.selection_metric,
            "fraction": self.selection_fraction,
            "top_n": self.selection_top_n,
            "selected_modules": list(selected),
            "scores": dict(scores),
            "module_sizes": module_sizes,
        }
        self.selection_telemetry.append(telemetry)
        self._selection_frozen = True
        score_text = ", ".join(
            f"{name}={scores[name]:.6e}" for name in ranked
        )
        print(
            "[LAYERWISE][SELECTION] "
            f"round={self.selection_round} metric={self.selection_metric} "
            f"observations={observations} selected={list(selected)} "
            f"scores=[{score_text}]"
        )

    def _set_all_layer_sources(self, source):
        self.last_layer_sources = {
            state.name: str(source) for state in self.layer_states
        }

    def _avg_fallback(self, device, model):
        fake_gradient = []
        sources = {}
        for state, param in zip(self.layer_states, model.parameters()):
            if state.raw_history:
                avg_vec = self._layer_avg(state)
                fake_gradient.append(
                    torch.as_tensor(avg_vec, device=device).view_as(param).to(dtype=param.dtype)
                )
                sources[state.name] = "recent_mean"
            else:
                fake_gradient.append(torch.randn_like(param, device=device))
                sources[state.name] = "random"
        self.last_layer_sources = sources
        return fake_gradient

    def _flatten_raw_layers(self, layer_vecs):
        if not layer_vecs:
            return np.array([], dtype=np.float32)
        return np.concatenate([np.asarray(v, dtype=np.float32) for v in layer_vecs], axis=0)

    def _flatten_selected_layers(self, layer_vecs):
        selected_vecs = [
            vec
            for state, vec in zip(self.layer_states, layer_vecs)
            if state.selected
        ]
        return self._flatten_raw_layers(selected_vecs)

    @staticmethod
    def _tensor_layers_to_raw(layer_tensors):
        return [
            np.asarray(
                tensor.detach().reshape(-1).cpu().numpy(),
                dtype=np.float32,
            ).copy()
            for tensor in layer_tensors
        ]

    def _hessian_fuse(self, predicted_gradient, model, device):
        provider = self.hessian_provider or (
            lambda m, d: [torch.zeros_like(pp, device=d) for pp in m.parameters()]
        )
        h_term = provider(model, device)
        return [
            self.lambda_h * pg + (1.0 - self.lambda_h) * hh
            if state.selected
            else pg
            for state, pg, hh in zip(self.layer_states, predicted_gradient, h_term)
        ]

    def _fit_ipca(self, state, raw_history=None):
        source_history = state.raw_history if raw_history is None else raw_history
        history = np.stack(source_history, axis=0).astype(np.float64)
        if not state.use_pca:
            return None, history
        n_samples = history.shape[0]
        n_components = min(state.compressed_dim, n_samples - 1, state.size - 1)
        if n_components < 2:
            raise RuntimeError(
                f"Layer '{state.name}' does not have enough history for PCA yet: samples={n_samples}"
            )
        ipca = IncrementalPCA(n_components=n_components)
        ipca.fit(history)
        projected = ipca.transform(history)
        return ipca, projected

    def _project_latest_history(self, state, length):
        if len(state.raw_history) < length:
            raise RuntimeError(
                f"Layer '{state.name}' needs >= {length} history points, got {len(state.raw_history)}"
            )
        if not state.use_pca:
            raw = np.stack(state.raw_history[-length:], axis=0).astype(np.float32)
            return None, raw
        ipca, _ = self._fit_ipca(state)
        raw = np.stack(state.raw_history[-length:], axis=0).astype(np.float64)
        projected = ipca.transform(raw).astype(np.float32)
        return ipca, projected

    def _reconstruct(self, state, latent_pred, ipca):
        pred = np.asarray(latent_pred, dtype=np.float32)
        if ipca is None:
            return pred.copy()
        return ipca.inverse_transform(pred.reshape(1, -1))[0].astype(np.float32)

    def _format_moirai_target(self, series):
        arr = np.asarray(series, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D history array, got shape={arr.shape}")
        # GluonTS expects a 1D target for univariate series and 2D for multivariate.
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr.T

    def _current_round_count(self):
        return len(self.global_gradient_history)

    def _resolve_record_round(self, round_num):
        if round_num is None:
            resolved_round = int(self._recorded_observation_count)
        else:
            resolved_round = int(round_num)
        self._recorded_observation_count = max(
            int(self._recorded_observation_count),
            resolved_round + 1,
        )
        return resolved_round


class LayerwiseOfflineARIMAAttackMNISTIMDB(LayerwiseAttackBaseMNISTIMDB):
    def __init__(
        self,
        k,
        random_round,
        model,
        var_window=3,
        # case-study previous value: var_threshold=2e-2
        var_threshold=5e-2,
        lambda_h=0.9,
        hessian_provider=None,
        selection_metric="normalized_rms",
        selection_fraction=0.5,
        selection_top_n=None,
        pca_threshold=DEFAULT_LAYERWISE_PCA_THRESHOLD,
        pca_components=DEFAULT_LAYERWISE_PCA_COMPONENTS,
    ):
        super().__init__(
            k=k,
            random_round=random_round,
            model=model,
            lambda_h=lambda_h,
            hessian_provider=hessian_provider,
            selection_metric=selection_metric,
            selection_fraction=selection_fraction,
            selection_top_n=selection_top_n,
            pca_threshold=pca_threshold,
            pca_components=pca_components,
        )
        self.var_window = int(var_window)
        self.var_threshold = float(var_threshold)
        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self._fixed_origin_round = None
        self._fit_generation = 0
        self._fit_stats = {"success": 0, "constant": 0, "failed": 0}
        self.arima_fit_telemetry = []
        self._arima_mode_label = "OFFLINE"

    def record_global_gradient(self, global_gradient):
        self._record_raw_layers(global_gradient)

    def check_arima_ready(self):
        needed = self.k + self.var_window
        if self._current_round_count() < needed:
            return False
        if (
            getattr(self, "_cooldown_active", False)
            and int(getattr(self, "_observations_since_reset", 0)) < needed
        ):
            return False
        diffs = []
        readiness_states = (
            [state for state in self.layer_states if state.selected]
            if self._selection_frozen
            else self.layer_states
        )
        for state in readiness_states:
            hist = np.stack(state.raw_history[-needed:], axis=0)
            norms = np.linalg.norm(hist, axis=1)
            windows = []
            # ``needed = k + var_window`` gives ``var_window + 1`` sliding
            # windows and exactly ``var_window`` adjacent variance changes.
            # Include all of them so readiness reacts to the newest gradient.
            for idx in range(self.var_window + 1):
                segment = norms[idx:idx + self.k]
                windows.append(float(np.var(segment)))
            if len(windows) > 1:
                diffs.extend(np.abs(np.diff(windows)).tolist())
        if not diffs:
            return False
        var_diff = np.asarray(diffs, dtype=np.float32)
        print(f"[ARIMA][LAYERWISE][{self._arima_mode_label}] var diff: {var_diff}")
        # case-study previous readiness: return bool(np.all(var_diff < self.var_threshold))
        return bool(np.mean(var_diff < self.var_threshold) >= 0.8)

    def train_offline_arima_model(self, fit_round=None, history_limit=None, reason="initial"):
        self._ensure_layer_selection(selection_round=fit_round)
        print(
            f"[ARIMA][LAYERWISE][{self._arima_mode_label}] "
            "Starting layerwise auto_arima training..."
        )
        successful_dims = 0
        constant_dims = 0
        failed_dims = 0
        fit_observations = None
        for state in self.layer_states:
            if not state.selected:
                state.arima_models = None
                continue
            source_history = state.raw_history
            if history_limit is not None:
                source_history = source_history[-max(1, int(history_limit)):]
            fit_observations = len(source_history)
            try:
                ipca, series_mat = self._fit_ipca(state, raw_history=source_history)
            except Exception as exc:
                # Preserve the raw layer shape so prediction remains well-formed
                # even when a layer-level PCA fit cannot be constructed.
                ipca = None
                model_list = [None] * state.size
                failed_dims += state.size
                state.arima_models = {
                    "ipca": ipca,
                    "models": model_list,
                    "fit_stats": {"success": 0, "constant": 0, "failed": state.size},
                    "fit_observations": fit_observations,
                }
                print(
                    f"[ARIMA][LAYERWISE][{self._arima_mode_label}][FIT] layer={state.name} "
                    f"PCA failed -> {state.size} failed dims: {exc}"
                )
                continue

            model_list = []
            layer_success = 0
            layer_constant = 0
            layer_failed = 0
            for dim_idx in range(series_mat.shape[1]):
                series = np.asarray(series_mat[:, dim_idx], dtype=np.float32)
                if _is_constant_series(series):
                    constant_model = _constant_series_model(series)
                    model_list.append(
                        (constant_model[0], constant_model[1], int(series.size))
                    )
                    constant_dims += 1
                    layer_constant += 1
                    continue
                try:
                    model_list.append(
                        auto_arima(
                            series,
                            start_p=0,
                            start_q=0,
                            max_p=2,
                            max_q=2,
                            d=None,
                            max_d=1,
                            seasonal=False,
                            stationary=False,
                            error_action="ignore",
                            suppress_warnings=True,
                            stepwise=True,
                            maxiter=20,
                        )
                    )
                    successful_dims += 1
                    layer_success += 1
                except Exception:
                    model_list.append(None)
                    failed_dims += 1
                    layer_failed += 1
            state.arima_models = {
                "ipca": ipca,
                "models": model_list,
                "fit_stats": {
                    "success": layer_success,
                    "constant": layer_constant,
                    "failed": layer_failed,
                },
                "fit_observations": fit_observations,
            }
        self.offline_model_trained = True
        self._fixed_origin_round = (
            int(fit_round) if fit_round is not None else self._current_round_count()
        )
        self._fit_generation += 1
        self._fit_stats = {
            "success": successful_dims,
            "constant": constant_dims,
            "failed": failed_dims,
        }
        telemetry = {
            "reason": str(reason),
            "generation": self._fit_generation,
            "origin_round": self._fixed_origin_round,
            "observations": int(fit_observations or 0),
            **self._fit_stats,
        }
        self.arima_fit_telemetry.append(telemetry)
        print(
            f"[ARIMA][LAYERWISE][{self._arima_mode_label}][FIT] "
            f"reason={reason} generation={self._fit_generation} "
            f"origin_round={self._fixed_origin_round} observations={fit_observations or 0} "
            f"success={successful_dims} constant={constant_dims} failed={failed_dims}"
        )

    def _predict_state(self, state, horizon=1):
        bundle = state.arima_models or {}
        model_list = bundle.get("models", [])
        ipca = bundle.get("ipca")
        latent_pred = np.zeros((len(model_list),), dtype=np.float32)
        fallback_latent = np.zeros_like(latent_pred)
        if state.raw_history and len(model_list) > 0:
            try:
                recent_raw = np.stack(
                    state.raw_history[-min(len(state.raw_history), max(1, self.k)):],
                    axis=0,
                ).astype(np.float64)
                recent_latent = (
                    recent_raw if ipca is None else ipca.transform(recent_raw)
                )
                recent_mean = np.asarray(
                    np.mean(recent_latent, axis=0), dtype=np.float32
                ).reshape(-1)
                fallback_latent[: min(len(fallback_latent), len(recent_mean))] = (
                    recent_mean[: len(fallback_latent)]
                )
            except Exception:
                # The zero-initialized fallback remains shape-safe; fit
                # telemetry still exposes the layer-level failure.
                pass
        for idx, arima_model in enumerate(model_list):
            if isinstance(arima_model, tuple) and len(arima_model) >= 2 and arima_model[0] == "constant":
                latent_pred[idx] = float(arima_model[1])
            elif arima_model is not None:
                try:
                    forecast = arima_model.predict(n_periods=max(1, int(horizon)))
                    latent_pred[idx] = float(forecast[-1])
                except Exception:
                    latent_pred[idx] = fallback_latent[idx]
            else:
                latent_pred[idx] = fallback_latent[idx]
        return self._reconstruct(state, latent_pred, ipca)

    def get_fake_gradient(self, round_num, device, model):
        if self._current_round_count() == 0:
            self.last_source = "random_no_history"
            self._set_all_layer_sources("random")
            _log_fallback("ARIMA][LAYERWISE][OFFLINE", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        # Offline means a single frozen fit. Do not inspect readiness or fit
        # before the configured warmup boundary.
        if int(round_num) < self.random_round:
            self.last_source = "avg_warmup"
            _log_fallback(
                "ARIMA][LAYERWISE][OFFLINE",
                round_num,
                "avg",
                "warmup",
                extra=f"used={min(self._current_round_count(), self.k)}",
            )
            return self._avg_fallback(device, model)

        self._ensure_layer_selection(selection_round=round_num)

        if (not self.arima_started) and self.check_arima_ready():
            self.arima_started = True
            self.arima_start_round = int(round_num)
            self.train_offline_arima_model(fit_round=round_num, reason="initial")
            print(f"[ARIMA][LAYERWISE][OFFLINE] round={round_num} SWITCHED TO AUTO-ARIMA")

        if self.arima_started and self.offline_model_trained:
            origin_round = (
                self._fixed_origin_round
                if self._fixed_origin_round is not None
                else self.arima_start_round
            )
            horizon = max(1, int(round_num) - int(origin_round) + 1)
            predicted_gradient = []
            layer_sources = {}
            for state, param in zip(self.layer_states, model.parameters()):
                if state.selected:
                    pred = self._predict_state(state, horizon=horizon)
                    layer_sources[state.name] = "forecast"
                else:
                    pred = self._layer_avg(state)
                    layer_sources[state.name] = "recent_mean"
                predicted_gradient.append(
                    torch.as_tensor(pred, device=device).view_as(param).to(dtype=param.dtype)
                )
            print(
                "[ARIMA][LAYERWISE][OFFLINE][FORECAST] "
                f"round={round_num} origin_round={origin_round} horizon={horizon}"
            )
            self.last_source = "arima_layerwise_fixed_origin"
            self.last_layer_sources = layer_sources
            return self._hessian_fuse(predicted_gradient, model, device)

        self.last_source = "avg_not_ready"
        _log_fallback(
            "ARIMA][LAYERWISE][OFFLINE",
            round_num,
            "avg",
            "warmup_or_not_ready",
            extra=f"used={min(self._current_round_count(), self.k)}",
        )
        return self._avg_fallback(device, model)


class LayerwiseAdaptiveARIMAAttackMNISTIMDB(LayerwiseOfflineARIMAAttackMNISTIMDB):
    supports_round_aware_recording = True

    def __init__(
        self,
        k,
        random_round,
        model,
        var_window=3,
        # case-study previous value: var_threshold=2e-2
        var_threshold=5e-2,
        lambda_h=0.9,
        hessian_provider=None,
        err_ema_beta=0.9,
        err_threshold=5e-3,
        min_rounds_after_start=2,
        retrain_interval=10,
        trim_history=True,
        selection_metric="normalized_rms",
        selection_fraction=0.5,
        selection_top_n=None,
        pca_threshold=DEFAULT_LAYERWISE_PCA_THRESHOLD,
        pca_components=DEFAULT_LAYERWISE_PCA_COMPONENTS,
    ):
        super().__init__(
            k=k,
            random_round=random_round,
            model=model,
            var_window=var_window,
            var_threshold=var_threshold,
            lambda_h=lambda_h,
            hessian_provider=hessian_provider,
            selection_metric=selection_metric,
            selection_fraction=selection_fraction,
            selection_top_n=selection_top_n,
            pca_threshold=pca_threshold,
            pca_components=pca_components,
        )
        self.err_ema_beta = float(err_ema_beta)
        self.err_threshold = float(err_threshold)
        self.min_rounds_after_start = int(min_rounds_after_start)
        self.retrain_interval = int(retrain_interval)
        self.trim_history = bool(trim_history)

        self._err_ema = None
        self._since_start = 0
        self._last_pred_flat = None
        self._last_pred_round = None
        self._cooldown_active = False
        self._observations_since_reset = None
        self._observations_since_full_refit = 0
        self._aligned_observations = 0
        self._skipped_alignments = 0
        self._arima_mode_label = "ADAPT"

    def train_offline_arima_model(
        self,
        fit_round=None,
        history_limit=None,
        reason="adaptive",
    ):
        super().train_offline_arima_model(
            fit_round=fit_round,
            history_limit=history_limit,
            reason=reason,
        )
        self._observations_since_full_refit = 0

    def _clear_prediction_cache(self):
        self._last_pred_flat = None
        self._last_pred_round = None

    def _reset_arima_state(self):
        self.arima_started = False
        self.arima_start_round = None
        self.offline_model_trained = False
        self._fixed_origin_round = None
        for state in self.layer_states:
            state.arima_models = None
        self._clear_prediction_cache()
        self._err_ema = None
        self._since_start = 0
        self._cooldown_active = True
        self._observations_since_reset = 0
        self._observations_since_full_refit = 0
        if self.trim_history:
            keep = max(self.k + self.var_window, self.k)
            if len(self.global_gradient_history) > keep:
                self.global_gradient_history = self.global_gradient_history[-keep:]
            for state in self.layer_states:
                if len(state.raw_history) > keep:
                    state.raw_history = state.raw_history[-keep:]
        print("[ARIMA][LAYERWISE][ADAPT] RESET completed.")

    def record_global_gradient(self, global_gradient, round_num=None):
        round_was_explicit = round_num is not None
        observed_round = self._resolve_record_round(round_num)
        raw_layers = self._record_raw_layers(global_gradient)
        flat = (
            self._flatten_selected_layers(raw_layers)
            if self._selection_frozen
            else np.array([], dtype=np.float32)
        )

        # Retained pre-reset history is useful as a fallback, but it never
        # contributes to the fresh-observation cooldown counter.
        if self._observations_since_reset is not None:
            self._observations_since_reset += 1

        predicted_round = self._last_pred_round
        pending_prediction = (
            self._last_pred_flat is not None and predicted_round is not None
        )
        exact_alignment = bool(
            round_was_explicit
            and pending_prediction
            and int(predicted_round) == int(observed_round)
        )
        mae = None
        should_reset = False
        if exact_alignment:
            self._aligned_observations += 1
            self._since_start += 1
            mae = float(np.mean(np.abs(flat - self._last_pred_flat)))
            if self._err_ema is None:
                self._err_ema = mae
            else:
                beta = self.err_ema_beta
                self._err_ema = beta * self._err_ema + (1.0 - beta) * mae
            should_reset = bool(
                self._since_start >= self.min_rounds_after_start
                and self._err_ema > self.err_threshold
            )
        elif pending_prediction:
            self._skipped_alignments += 1

        # One submitted prediction can be scored at most once. A mismatched
        # observation also invalidates it, preventing lazy honest rounds from
        # consuming a stale cache later.
        self._clear_prediction_cache()

        if should_reset:
            print(
                "[ARIMA][LAYERWISE][ADAPT][OBSERVE] "
                f"round={observed_round} pred_round={predicted_round} "
                f"mae={mae:.6e} ema={self._err_ema:.6e} reset=1"
            )
            self._reset_arima_state()
            return

        active_model = self.arima_started and self.offline_model_trained
        periodic_refit = False
        if active_model:
            # Document-faithful adaptive behavior: PCA and ARIMA states remain
            # frozen between scheduled full refits.
            self._observations_since_full_refit += 1
            periodic_refit = bool(
                self.retrain_interval > 0
                and self._observations_since_full_refit >= self.retrain_interval
            )
            if periodic_refit:
                history_limit = (
                    self._observations_since_reset
                    if self._observations_since_reset is not None
                    else None
                )
                self.train_offline_arima_model(
                    fit_round=observed_round + 1,
                    history_limit=history_limit,
                    reason=f"periodic@observed_round={observed_round}",
                )

        if pending_prediction or active_model or self._cooldown_active:
            alignment = "exact" if exact_alignment else (
                "mismatch" if pending_prediction else "none"
            )
            print(
                "[ARIMA][LAYERWISE][ADAPT][OBSERVE] "
                f"round={observed_round} pred_round={predicted_round} "
                f"alignment={alignment} "
                f"mae={'n/a' if mae is None else f'{mae:.6e}'} "
                f"ema={'n/a' if self._err_ema is None else f'{self._err_ema:.6e}'} "
                f"periodic_refit={int(periodic_refit)} "
                f"fresh_since_reset={self._observations_since_reset}"
            )

    def get_fake_gradient(self, round_num, device, model):
        if self._current_round_count() == 0:
            self.last_source = "random_no_history"
            self._set_all_layer_sources("random")
            _log_fallback("ARIMA][LAYERWISE][ADAPT", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if int(round_num) < self.random_round:
            self.last_source = "avg_warmup"
            _log_fallback(
                "ARIMA][LAYERWISE][ADAPT",
                round_num,
                "avg",
                "warmup",
                extra=f"used={min(self._current_round_count(), self.k)}",
            )
            return self._avg_fallback(device, model)

        self._ensure_layer_selection(selection_round=round_num)

        needed_fresh = self.k + self.var_window
        if (
            not self.arima_started
            and self._cooldown_active
            and int(self._observations_since_reset or 0) < needed_fresh
        ):
            self.last_source = "avg_cooldown"
            _log_fallback(
                "ARIMA][LAYERWISE][ADAPT",
                round_num,
                "avg",
                "reset_cooldown",
                extra=(
                    f"used={min(self._current_round_count(), self.k)} "
                    f"fresh={int(self._observations_since_reset or 0)}/{needed_fresh}"
                ),
            )
            return self._avg_fallback(device, model)

        if (not self.arima_started) and self.check_arima_ready():
            history_limit = (
                self._observations_since_reset
                if self._cooldown_active
                else None
            )
            self.train_offline_arima_model(
                fit_round=round_num,
                history_limit=history_limit,
                reason="reset_recovery" if self._cooldown_active else "initial",
            )
            self.arima_started = True
            self.arima_start_round = int(round_num)
            self._cooldown_active = False
            print(f"[ARIMA][LAYERWISE][ADAPT] round={round_num} SWITCHED TO ADAPTIVE")

        if self.arima_started and self.offline_model_trained:
            origin_round = (
                self._fixed_origin_round
                if self._fixed_origin_round is not None
                else self.arima_start_round
            )
            horizon = max(1, int(round_num) - int(origin_round) + 1)
            predicted_gradient = []
            layer_sources = {}
            for state, param in zip(self.layer_states, model.parameters()):
                if state.selected:
                    pred = self._predict_state(state, horizon=horizon)
                    layer_sources[state.name] = "forecast"
                else:
                    pred = self._layer_avg(state)
                    layer_sources[state.name] = "recent_mean"
                predicted_gradient.append(
                    torch.as_tensor(pred, device=device).view_as(param).to(dtype=param.dtype)
                )

            submitted_gradient = self._hessian_fuse(predicted_gradient, model, device)
            submitted_raw = self._tensor_layers_to_raw(submitted_gradient)
            self._last_pred_round = int(round_num)
            self._last_pred_flat = self._flatten_selected_layers(submitted_raw)
            print(
                "[ARIMA][LAYERWISE][ADAPT][FORECAST] "
                f"round={round_num} origin_round={origin_round} horizon={horizon}"
            )
            self.last_source = "arima_layerwise_adaptive"
            self.last_layer_sources = layer_sources
            return submitted_gradient

        self.last_source = "avg_cooldown" if self._cooldown_active else "avg_not_ready"
        _log_fallback(
            "ARIMA][LAYERWISE][ADAPT",
            round_num,
            "avg",
            "reset_cooldown" if self._cooldown_active else "not_ready",
            extra=f"used={min(self._current_round_count(), self.k)}",
        )
        return self._avg_fallback(device, model)


class LayerwiseOfflineMoiraiAttackMNISTIMDB(LayerwiseAttackBaseMNISTIMDB):
    def __init__(
        self,
        k,
        random_round,
        model,
        history_length=5,
        size="small",
        patch_size="auto",
        lambda_h=0.9,
        hessian_provider=None,
        selection_metric="normalized_rms",
        selection_fraction=0.5,
        selection_top_n=None,
        pca_threshold=DEFAULT_LAYERWISE_PCA_THRESHOLD,
        pca_components=DEFAULT_LAYERWISE_PCA_COMPONENTS,
    ):
        super().__init__(
            k=k,
            random_round=random_round,
            model=model,
            history_length=history_length,
            size=size,
            patch_size=patch_size,
            lambda_h=lambda_h,
            hessian_provider=hessian_provider,
            selection_metric=selection_metric,
            selection_fraction=selection_fraction,
            selection_top_n=selection_top_n,
            pca_threshold=pca_threshold,
            pca_components=pca_components,
        )

    def record_global_gradient(self, global_gradient):
        self._record_raw_layers(global_gradient)

    def _ensure_predictor(self, state):
        if not state.selected:
            raise RuntimeError(
                f"Moirai predictor requested for unselected layer '{state.name}'"
            )
        if state.moirai_predictor is not None:
            return
        self._ensure_moirai_projection(state)
        state.moirai_model = MoiraiForecast(
            module=_load_moirai_module(_resolve_moirai_source(self.size)),
            prediction_length=1,
            context_length=self.history_length,
            patch_size=self.patch_size,
            num_samples=1,
            target_dim=state.moirai_effective_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        state.moirai_predictor = state.moirai_model.create_predictor(batch_size=1)

    def _ensure_moirai_projection(self, state):
        """Freeze one causal PCA basis before creating the Moirai predictor."""
        if state.moirai_effective_dim is not None:
            return
        if state.use_pca:
            ipca, projected = self._fit_ipca(state)
            state.moirai_ipca = ipca
            state.moirai_effective_dim = int(projected.shape[1])
        else:
            state.moirai_ipca = None
            state.moirai_effective_dim = int(state.size)

    def _project_frozen_moirai_history(self, state):
        if len(state.raw_history) < self.history_length:
            raise RuntimeError(
                f"Layer '{state.name}' needs >= {self.history_length} history points, "
                f"got {len(state.raw_history)}"
            )
        self._ensure_moirai_projection(state)
        raw = np.stack(
            state.raw_history[-self.history_length:], axis=0
        ).astype(np.float64)
        if state.moirai_ipca is None:
            return raw.astype(np.float32)
        return state.moirai_ipca.transform(raw).astype(np.float32)

    def _predict_state(self, state):
        series = self._project_frozen_moirai_history(state)
        self._ensure_predictor(state)
        input_data = [{
            "start": pd.Period("2000-01-01", freq="D"),
            "target": self._format_moirai_target(series),
        }]
        forecast = next(state.moirai_predictor.predict(input_data))
        latent_pred = np.asarray(forecast.samples[0, 0], dtype=np.float32)
        return self._reconstruct(state, latent_pred, state.moirai_ipca)

    def get_fake_gradient(self, round_num, device, model):
        if self._current_round_count() == 0:
            self.last_source = "random_no_history"
            self._set_all_layer_sources("random")
            _log_fallback("Moirai][LAYERWISE][OFFLINE", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if self._current_round_count() < self.history_length or round_num < self.random_round:
            self.last_source = "avg_warmup_or_short_history"
            _log_fallback(
                "Moirai][LAYERWISE][OFFLINE",
                round_num,
                "avg",
                "warmup_or_short_history",
                extra=f"used={min(self._current_round_count(), self.k)}",
            )
            return self._avg_fallback(device, model)

        self._ensure_layer_selection(selection_round=round_num)

        print(f"[Moirai][LAYERWISE][OFFLINE] round={round_num} USING MOIRAI PREDICTION")
        predicted_gradient = []
        layer_sources = {}
        for state, param in zip(self.layer_states, model.parameters()):
            if state.selected:
                pred = self._predict_state(state)
                layer_sources[state.name] = "forecast"
            else:
                pred = self._layer_avg(state)
                layer_sources[state.name] = "recent_mean"
            predicted_gradient.append(
                torch.as_tensor(pred, device=device).view_as(param).to(dtype=param.dtype)
            )
        self.last_source = "moirai_layerwise_offline"
        self.last_layer_sources = layer_sources
        return self._hessian_fuse(predicted_gradient, model, device)


class LayerwiseOnlineMoiraiAttackMNISTIMDB(LayerwiseOfflineMoiraiAttackMNISTIMDB):
    supports_round_aware_recording = True

    def __init__(
        self,
        k,
        random_round,
        model,
        history_length=5,
        size="small",
        patch_size="auto",
        lambda_h=0.9,
        hessian_provider=None,
        online_tuning=True,
        tune_lr=1e-3,
        tune_steps=1,
        tune_start_round=None,
        residual_clip=5.0,
        adapt_interval=10,
        residual_decay=0.9,
        err_ema_beta=0.9,
        err_threshold=5e-3,
        min_rounds_after_start=2,
        trim_history=True,
        selection_metric="normalized_rms",
        selection_fraction=0.5,
        selection_top_n=None,
        pca_threshold=DEFAULT_LAYERWISE_PCA_THRESHOLD,
        pca_components=DEFAULT_LAYERWISE_PCA_COMPONENTS,
    ):
        super().__init__(
            k=k,
            random_round=random_round,
            model=model,
            history_length=history_length,
            size=size,
            patch_size=patch_size,
            lambda_h=lambda_h,
            hessian_provider=hessian_provider,
            selection_metric=selection_metric,
            selection_fraction=selection_fraction,
            selection_top_n=selection_top_n,
            pca_threshold=pca_threshold,
            pca_components=pca_components,
        )
        self.online_tuning = bool(online_tuning)
        self.tune_lr = float(tune_lr)
        self.tune_steps = int(max(1, tune_steps))
        self.tune_start_round = self.random_round if tune_start_round is None else int(tune_start_round)
        self.residual_clip = residual_clip
        self.adapt_interval = int(adapt_interval)
        self.residual_decay = float(residual_decay)
        self.trim_history = bool(trim_history)

        self.err_ema_beta = float(err_ema_beta)
        self.err_threshold = float(err_threshold)
        self.min_rounds_after_start = int(min_rounds_after_start)
        self._err_ema = None
        self._since_start = 0
        self._last_pred_flat = None
        self._last_pred_round = None
        self._last_base_preds = None
        self._aligned_observations = 0
        self._skipped_alignments = 0
        # This is an observation-level counter: one successful aligned
        # calibration counts once even when several selected layers are tuned.
        # It deliberately does not use the global round number, which is tied
        # to the chosen lazy-attack cadence.
        self._residual_update_count = 0

    def _soft_adjust(self):
        for state in self.layer_states:
            if state.selected:
                state.residual *= self.residual_decay

    def _clear_submission_cache(self):
        self._last_pred_flat = None
        self._last_pred_round = None
        self._last_base_preds = None

    def _hard_reset(self):
        for state in self.layer_states:
            if state.selected:
                state.residual.fill(0.0)
        self._err_ema = None
        self._since_start = 0
        self._residual_update_count = 0
        self._clear_submission_cache()
        if self.trim_history:
            keep = max(self.k + self.history_length, self.k)
            if len(self.global_gradient_history) > keep:
                self.global_gradient_history = self.global_gradient_history[-keep:]
            for state in self.layer_states:
                if len(state.raw_history) > keep:
                    state.raw_history = state.raw_history[-keep:]
        print(
            "[Moirai][LAYERWISE][OnlineTune] "
            "RESET completed residual_updates=0."
        )

    def record_global_gradient(self, global_gradient, round_num=None):
        round_was_explicit = round_num is not None
        observed_round = self._resolve_record_round(round_num)
        raw_layers = self._record_raw_layers(global_gradient)
        flat = (
            self._flatten_selected_layers(raw_layers)
            if self._selection_frozen
            else np.array([], dtype=np.float32)
        )
        predicted_round = self._last_pred_round
        pending_prediction = (
            self._last_pred_flat is not None
            and predicted_round is not None
            and self._last_base_preds is not None
        )
        exact_alignment = bool(
            round_was_explicit
            and pending_prediction
            and int(predicted_round) == int(observed_round)
        )

        if not exact_alignment:
            if pending_prediction:
                self._skipped_alignments += 1
                print(
                    "[Moirai][LAYERWISE][OnlineTune][OBSERVE] "
                    f"round={observed_round} pred_round={predicted_round} "
                    "alignment=mismatch update=skipped"
                )
            # Honest rounds only append the observation. Any stale submitted
            # forecast is consumed without error scoring or residual tuning.
            self._clear_submission_cache()
            return

        self._aligned_observations += 1
        self._since_start += 1
        mae = float(np.mean(np.abs(flat - self._last_pred_flat)))
        if self._err_ema is None:
            self._err_ema = mae
        else:
            beta = self.err_ema_beta
            self._err_ema = beta * self._err_ema + (1.0 - beta) * mae

        if (
            self._since_start >= self.min_rounds_after_start
            and self._err_ema > self.err_threshold
        ):
            print(
                "[Moirai][LAYERWISE][OnlineTune][OBSERVE] "
                f"round={observed_round} pred_round={predicted_round} "
                f"mae={mae:.6e} ema={self._err_ema:.6e} reset=1"
            )
            self._hard_reset()
            return

        cached_base_preds = self._last_base_preds
        tuned_layers = 0
        failed_layers = 0
        decay_applied = False
        if self.online_tuning and observed_round >= self.tune_start_round:
            for state, target_raw in zip(self.layer_states, raw_layers):
                if not state.selected:
                    continue
                try:
                    submitted_base = cached_base_preds[state.name]
                    submitted_base = np.asarray(submitted_base, dtype=np.float32)
                    target_raw = np.asarray(target_raw, dtype=np.float32)
                    for _ in range(self.tune_steps):
                        submitted_pred = (
                            submitted_base + self.lambda_h * state.residual
                        )
                        err = submitted_pred - target_raw
                        state.residual -= self.tune_lr * self.lambda_h * err
                        if self.residual_clip is not None:
                            np.clip(
                                state.residual,
                                -float(self.residual_clip),
                                float(self.residual_clip),
                                out=state.residual,
                            )
                    tuned_layers += 1
                except Exception as exc:
                    failed_layers += 1
                    print(
                        f"[Moirai][LAYERWISE][OnlineTune] skip cached residual "
                        f"update for layer {state.name}: {exc}"
                    )

        residual_updated = tuned_layers > 0
        if residual_updated:
            self._residual_update_count += 1
            if (
                self.adapt_interval > 0
                and self._residual_update_count % self.adapt_interval == 0
            ):
                # Decay only after a real residual calibration and measure the
                # interval in successful updates, independent of lazy cadence.
                self._soft_adjust()
                decay_applied = True

        residual_norm = float(
            np.sqrt(
                sum(
                    float(np.dot(state.residual, state.residual))
                    for state in self.layer_states
                    if state.selected
                )
            )
        )

        print(
            "[Moirai][LAYERWISE][OnlineTune][OBSERVE] "
            f"round={observed_round} pred_round={predicted_round} alignment=exact "
            f"mae={mae:.6e} ema={self._err_ema:.6e} "
            f"tuned_layers={tuned_layers} failed_layers={failed_layers} "
            f"residual_updated={int(residual_updated)} "
            f"residual_updates={self._residual_update_count} "
            f"decay_applied={int(decay_applied)} "
            f"residual_norm={residual_norm:.6e} "
            f"aligned_total={self._aligned_observations} "
            f"skipped_total={self._skipped_alignments}"
        )
        self._clear_submission_cache()

    def get_fake_gradient(self, round_num, device, model):
        if self._current_round_count() == 0:
            self._clear_submission_cache()
            self.last_source = "random_no_history"
            self._set_all_layer_sources("random")
            _log_fallback("Moirai][LAYERWISE][OnlineTune", round_num, "random", "no_history")
            return [torch.randn_like(p, device=device) for p in model.parameters()]

        if self._current_round_count() < self.history_length or round_num < self.random_round:
            self._clear_submission_cache()
            self.last_source = "avg_warmup_or_short_history"
            _log_fallback(
                "Moirai][LAYERWISE][OnlineTune",
                round_num,
                "avg",
                "warmup_or_short_history",
                extra=f"used={min(self._current_round_count(), self.k)}",
            )
            return self._avg_fallback(device, model)

        self._ensure_layer_selection(selection_round=round_num)

        print(f"[Moirai][LAYERWISE][OnlineTune] round={round_num} USING MOIRAI PREDICTION")
        predicted_gradient = []
        layer_sources = {}
        for state, param in zip(self.layer_states, model.parameters()):
            if not state.selected:
                pred = self._layer_avg(state)
                layer_sources[state.name] = "recent_mean"
                predicted_gradient.append(
                    torch.as_tensor(pred, device=device).view_as(param).to(dtype=param.dtype)
                )
                continue
            try:
                base_pred = np.asarray(self._predict_state(state), dtype=np.float32)
                pred = (base_pred + state.residual).astype(np.float32)
                layer_sources[state.name] = "forecast"
                predicted_gradient.append(
                    torch.as_tensor(pred, device=device).view_as(param).to(dtype=param.dtype)
                )
            except Exception as exc:
                self._clear_submission_cache()
                raise RuntimeError(
                    f"Moirai layer forecast failed for '{state.name}'"
                ) from exc

        submitted_gradient = self._hessian_fuse(predicted_gradient, model, device)
        submitted_raw = self._tensor_layers_to_raw(submitted_gradient)
        self._last_pred_round = int(round_num)
        self._last_pred_flat = self._flatten_selected_layers(submitted_raw)
        self._last_base_preds = {
            state.name: submitted.copy() - self.lambda_h * state.residual
            for state, submitted in zip(self.layer_states, submitted_raw)
            if state.selected
        }
        self.last_source = "moirai_layerwise_online_residual"
        self.last_layer_sources = layer_sources
        return submitted_gradient



import typing
setattr(__builtins__, 'Union', typing.Union)
setattr(__builtins__, 'Tuple', typing.Tuple)

import sys
sys.path.append("./uni2ts/src")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import torch
import os
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy.stats
import config
from script.datapre import *
from module.model.model import return_model
from module.method.detecting_registry import resolve_detecting_method
from module.method.fl_aggregation import AggregationConfig, ServerAggregator, add_aggregation_arguments
from module.data_manager.client_validation import reserve_client_validation
from module.method.qualitative_detecting.pass_detector import (
    PASSAuditObserver, PASSConfig, attack_resource_notes,
    resolve_forged_auditors, write_pass_artifacts,
)
from module.method.qualitative_detecting.log_based_free_rider_detection import (
    LogBasedFreeRiderConfig, LogBasedFreeRiderObserver,
    write_log_based_free_rider_artifacts,
)
from module.method.quantitative_detecting.runner_support import (
    build_quantitative_diagnostic_row,
    build_sampling_call_kwargs,
    publish_realized_fake_rounds,
    sample_probabilistic_fake_rounds,
    write_quantitative_diagnostics,
)
# from module.method.sv_v2 import ShapleyValue

# Historical import kept only as a comment; this isolated runner imports the
# attack_updated modules below.
from module.method.attack_updated_nonlayerwise import (
    ReplayAttack, ARIMAAttack, RandomAttack, MoiraiAttack,
    OfflineARIMAAttack, AdaptiveARIMAAttack, OfflineMoiraiAttack, OnlineMoiraiAttack,
    ScaledDeltaAttack, AdvancedScaledDeltaAttack, WangLocalDeviationAttack,
    FraboniDisguisedAttack,
    DKFRLiteAttack, SyntheticReplacementAttack,
    frobenius_scaled_identity_provider,   # 便宜版曲率
    # hutchinson_hvp_provider,            # 要真HVP再解开 & 提供 loss_fn/batch
    # zeros_hessian_provider,             # 显式禁用时可用
)
from module.method.attack_updated_layerwise import (
    LayerwiseAdaptiveARIMAAttackMNISTIMDB,
    LayerwiseOfflineARIMAAttackMNISTIMDB,
    LayerwiseOfflineMoiraiAttackMNISTIMDB,
    LayerwiseOnlineMoiraiAttackMNISTIMDB,
)


import copy
import time

import argparse


MOIRAI_ONLINE_METHODS = {
    "moirai_online",
    "moirai_online_mnist_imdb_layerwise",
}

ARIMA_METHODS = {
    "arima",
    "arima_adaptive",
    "arima_mnist_imdb_layerwise",
    "arima_adaptive_mnist_imdb_layerwise",
}

ADAPTIVE_ARIMA_METHODS = {
    "arima_adaptive",
    "arima_adaptive_mnist_imdb_layerwise",
}

LAYERWISE_ARIMA_METHODS = {
    "arima_mnist_imdb_layerwise",
    "arima_adaptive_mnist_imdb_layerwise",
}


def _cli_option_was_supplied(option_name):
    """Recognize both ``--name value`` and ``--name=value`` spellings."""
    return any(
        token == option_name or token.startswith(option_name + "=")
        for token in sys.argv[1:]
    )

# History/forecast attacks are strict free-riders on fake rounds: they may
# observe the broadcast aggregate but must not train on the attacker's private
# shard before discarding that local update.
FORECAST_FREE_RIDER_METHODS = {
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

ATTACK_IMPLEMENTATIONS = {
    "random": RandomAttack,
    "replay": ReplayAttack,
    "fraboni_disguised": FraboniDisguisedAttack,
    "scaled_delta": ScaledDeltaAttack,
    "advanced_scaled_delta": AdvancedScaledDeltaAttack,
    "adam_local_deviation": WangLocalDeviationAttack,
    "dkfr": DKFRLiteAttack,
    "synthetic_replacement": SyntheticReplacementAttack,
    "arima": OfflineARIMAAttack,
    "arima_adaptive": AdaptiveARIMAAttack,
    "moirai": OfflineMoiraiAttack,
    "moirai_online": OnlineMoiraiAttack,
    "arima_mnist_imdb_layerwise": LayerwiseOfflineARIMAAttackMNISTIMDB,
    "arima_adaptive_mnist_imdb_layerwise": LayerwiseAdaptiveARIMAAttackMNISTIMDB,
    "moirai_mnist_imdb_layerwise": LayerwiseOfflineMoiraiAttackMNISTIMDB,
    "moirai_online_mnist_imdb_layerwise": LayerwiseOnlineMoiraiAttackMNISTIMDB,
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_moirai_online_parameters(parsed_args, dataset_name, method_name):
    is_layerwise = method_name == "moirai_online_mnist_imdb_layerwise"
    if parsed_args.moirai_err_threshold is None:
        err_threshold = (
            0.045 if dataset_name == "mnist" else 0.015
        ) if is_layerwise else 5e-3
    else:
        err_threshold = parsed_args.moirai_err_threshold

    if parsed_args.moirai_min_rounds_after_start is None:
        min_rounds_after_start = 4 if is_layerwise else 2
    else:
        min_rounds_after_start = parsed_args.moirai_min_rounds_after_start

    return {
        "online_tuning": parsed_args.moirai_online_tuning,
        "tune_lr": parsed_args.moirai_tune_lr,
        "tune_steps": parsed_args.moirai_tune_steps,
        "tune_start_round": parsed_args.moirai_tune_start_round,
        "adapt_interval": parsed_args.moirai_adapt_interval,
        "residual_decay": parsed_args.moirai_residual_decay,
        "residual_clip": parsed_args.moirai_residual_clip,
        "err_ema_beta": parsed_args.moirai_err_ema_beta,
        "err_threshold": err_threshold,
        "min_rounds_after_start": min_rounds_after_start,
        "trim_history": parsed_args.moirai_trim_history,
    }


def _effective_arima_readiness_parameters(parsed_args, method_name):
    """Resolve constructor defaults without conflating ordinary/layerwise ARIMA."""
    is_layerwise = method_name in LAYERWISE_ARIMA_METHODS
    if parsed_args.arima_var_threshold is None:
        # Preserve the constructor defaults that were previously obtained by
        # omitting these arguments at the two routes.
        var_threshold = 5e-2 if is_layerwise else 5e-3
    else:
        var_threshold = parsed_args.arima_var_threshold
    return {
        "var_window": parsed_args.arima_var_window,
        "var_threshold": var_threshold,
    }


def _effective_arima_adaptive_parameters(parsed_args, dataset_name, method_name):
    """Resolve the actual adaptive constructor values for routing/provenance."""
    is_layerwise = method_name == "arima_adaptive_mnist_imdb_layerwise"
    if is_layerwise and not _cli_option_was_supplied("--arima_err_threshold"):
        err_threshold = 0.035 if dataset_name == "mnist" else 0.015
    else:
        err_threshold = parsed_args.arima_err_threshold

    if is_layerwise and not _cli_option_was_supplied(
        "--arima_min_rounds_after_start"
    ):
        min_rounds_after_start = 4
    else:
        min_rounds_after_start = parsed_args.arima_min_rounds_after_start

    return {
        "err_ema_beta": parsed_args.arima_err_ema_beta,
        "err_threshold": err_threshold,
        "min_rounds_after_start": min_rounds_after_start,
        "retrain_interval": parsed_args.arima_retrain_interval,
        "trim_history": parsed_args.arima_trim_history,
    }


def _collect_attack_parameters(parsed_args, dataset_name, method_name):
    exact_names = {
        "alpha",
        "attacker_id",
        "attacker_ids",
        "seed",
        "attack_method",
        "attack_probability",
        "client_num",
        "dataset",
        "lambda_h",
        "random_round",
        "trial_id",
        "use_attack",
        "use_hessian",
        "use_lazy",
    }
    attack_prefixes = (
        "adam_",
        "arima_",
        "dkfr_",
        "fraboni_",
        "lazy_",
        "moirai_",
        "random_",
        "scaled_delta_",
        "synthetic_",
        "wang_",
    )
    parameters = {
        name: value
        for name, value in vars(parsed_args).items()
        if name in exact_names or name.startswith(attack_prefixes)
    }
    if method_name in MOIRAI_ONLINE_METHODS:
        parameters["effective_moirai_online_adapter"] = (
            _effective_moirai_online_parameters(
                parsed_args,
                dataset_name,
                method_name,
            )
        )
    if method_name in ARIMA_METHODS:
        parameters["effective_arima_readiness"] = (
            _effective_arima_readiness_parameters(parsed_args, method_name)
        )
    if method_name in ADAPTIVE_ARIMA_METHODS:
        parameters["effective_arima_adaptive"] = (
            _effective_arima_adaptive_parameters(
                parsed_args,
                dataset_name,
                method_name,
            )
        )
    return parameters


def _write_run_provenance(parsed_args, dataset_name, method_name, detecting_name):
    entry_invoked_path = Path(os.path.abspath(sys.argv[0]))
    entry_executed_path = Path(__file__).resolve()
    attack_class = ATTACK_IMPLEMENTATIONS[method_name]
    attack_module = sys.modules[attack_class.__module__]
    attack_module_path = Path(attack_module.__file__).resolve()
    attack_module_names = {
        attack_class.__module__,
        # Layerwise attacks import shared Moirai/ARIMA helpers from the
        # non-layerwise module, so both files are required for provenance.
        OfflineARIMAAttack.__module__,
    }
    attack_modules = []
    for module_name in sorted(attack_module_names):
        module_obj = sys.modules[module_name]
        module_path = Path(module_obj.__file__).resolve()
        attack_modules.append(
            {
                "module": module_name,
                "path": str(module_path),
                "sha256": _sha256_file(module_path),
            }
        )
    results_path = Path(parsed_args.results_root).resolve()
    manifest_dir = results_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    safe_dataset = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in dataset_name
    )
    safe_method = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in method_name
    )
    base_name = (
        f"run_manifest_{safe_dataset}_{safe_method}_trial{parsed_args.trial_id}_"
        f"{timestamp}_pid{os.getpid()}"
    )
    attack_parameters = _collect_attack_parameters(
        parsed_args,
        dataset_name,
        method_name,
    )
    from module.method.fl_aggregation import AggregationConfig
    aggregation_path = entry_executed_path.parent / 'module' / 'method' / 'fl_aggregation.py'
    payload = {
        "schema_version": 1,
        "created_at_utc": created_at.isoformat(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd().resolve()),
        "argv": list(sys.argv),
        "invoked_entry_path": str(entry_invoked_path),
        "executed_entry_path": str(entry_executed_path),
        "entry_sha256": _sha256_file(entry_executed_path),
        "attack_class": f"{attack_class.__module__}.{attack_class.__name__}",
        "attack_module_path": str(attack_module_path),
        "attack_module_sha256": _sha256_file(attack_module_path),
        "attack_modules": attack_modules,
        "results_root": str(results_path),
        "detecting_method": detecting_name,
        "fl_aggregation": AggregationConfig.from_args(parsed_args).to_dict(),
        "aggregation_module_sha256": _sha256_file(aggregation_path),
        "attack_parameters": attack_parameters,
        "cli_args": dict(vars(parsed_args)),
    }

    counter = 0
    while True:
        suffix = "" if counter == 0 else f"_{counter}"
        manifest_path = manifest_dir / f"{base_name}{suffix}.json"
        try:
            payload["manifest_path"] = str(manifest_path.resolve())
            with manifest_path.open("x", encoding="utf-8") as file_obj:
                json.dump(payload, file_obj, indent=2, sort_keys=True, default=str)
                file_obj.write("\n")
            break
        except FileExistsError:
            counter += 1

    return payload, manifest_path

parser = argparse.ArgumentParser()
parser.add_argument('--attacker_id', type=int, default=1)
parser.add_argument('--rq1_strict_data_free', action='store_true',
                    help='Opt-in RQ1 advanced_scaled_delta/DKFR: no attacker data training or empirical data statistics.')
parser.add_argument('--rq1_only', action='store_true',
                    help='Single-attacker RQ1: six clients, pure attacks, exact last-round Shapley; write JSON and skip removal control/legacy CSVs.')
parser.add_argument('--control_excluded_ids', type=int, nargs='+', default=None,
                    help='RQ2 control only: partition 8 clients, remove these original IDs, train honest remainder.')
parser.add_argument('--attacker_ids', type=int, nargs='+', default=None,
                    help='Opt into the isolated multi-free-rider experiment (client IDs, including k=1).')
parser.add_argument('--seed', type=int, default=42,
                    help='Data/model root seed; legacy default remains 42.')
parser.add_argument('--alpha', type=float, default=10)
parser.add_argument('--dataset', type=str, default="tictactoe")
parser.add_argument(
    '--attack_method',
    type=str,
    default="moirai",
    choices=[
        "random",
        "replay",
        "fraboni_disguised",
        "scaled_delta",
        "advanced_scaled_delta",
        "adam_local_deviation",
        "dkfr",
        "synthetic_replacement",
        "arima",
        "arima_adaptive",
        "moirai",
        "moirai_online",
        "arima_mnist_imdb_layerwise",
        "arima_adaptive_mnist_imdb_layerwise",
        "moirai_mnist_imdb_layerwise",
        "moirai_online_mnist_imdb_layerwise",
    ]
)
parser.add_argument('--client_num', type=int, default=4)
parser.add_argument('--num_epoch', type=int, default=50,
                    help='Total number of federated learning rounds; defaults to the full 50-round experiment.')
parser.add_argument('--num_local_epochs', type=int, default=1,
                    help='Local optimization epochs per FL round; WEF records one update per local epoch.')
add_aggregation_arguments(parser)
parser.add_argument(
    '--detecting_method', '--contribution_method',
    dest='detecting_method',
    type=str,
    default='shapley',
    choices=[
        'shapley', 'sampling_shapley', 'sampling_least_core',
        'eds', 'std_dagmm', 'delta_dagmm', 'wef', 's2_wef', 'pass',
        'log_based_free_rider_detection',
    ],
    help=(
        'Detecting method. shapley, sampling_shapley, and '
        'sampling_least_core are quantitative; eds, std_dagmm, '
        'delta_dagmm, wef, s2_wef, pass, and '
        'log_based_free_rider_detection are qualitative. '
        '--contribution_method is retained as a backward-compatible alias.'
    ),
)
parser.add_argument(
    '--detecting_type',
    type=str,
    default=None,
    choices=['quantitative', 'qualitative'],
    help='Optional category check for --detecting_method.',
)
parser.add_argument(
    '--qualitative_aggregation_mode',
    type=str,
    default='paper_online',
    choices=['passive', 'paper_online'],
    help=(
        'WEF/S2-WEF aggregation behavior. passive records detections while '
        'using the selected sample-weighted FL rule; paper_online retains the existing '
        'online isolation/broadcast behavior.'
    ),
)
# PASS parameters describe the approved passive audit experiment.
parser.add_argument('--pass_alpha', type=float, default=0.95,
                    help='PASS literal recurrence history coefficient, in [0,1].')
parser.add_argument('--pass_beta', type=float, default=1.75,
                    help='PASS threshold coefficient: contribution < 1/(beta*N).')
parser.add_argument('--pass_c0', type=float, default=None,
                    help='PASS initial contribution; omitted uses 1/N for each run.')
parser.add_argument('--pass_report_low', type=float, default=-1.0,
                    help='Pure forged divergence uniform lower bound (experiment assumption).')
parser.add_argument('--pass_report_high', type=float, default=1.0,
                    help='Pure forged divergence uniform upper bound (experiment assumption).')
parser.add_argument('--pass_seed', type=int, default=42,
                    help='Independent audit RNG seed; effective seed adds trial_id*100003.')
parser.add_argument('--pass_audit_batch_size', type=int, default=256,
                    help='PASS forward-only local audit batch size.')
parser.add_argument('--pass_max_audit_samples', type=int, default=0,
                    help='0 audits the full original local shard; positive caps a fixed random subset.')
# Log-based free-rider detector: paper-unspecified clustering conventions are
# configurable for the separate GPU pilot and must be frozen before evaluation.
parser.add_argument('--logfr_val_fraction', type=float, default=0.15,
                    help='Fraction held out within each already-partitioned client training shard.')
parser.add_argument('--logfr_val_seed', type=int, default=42,
                    help='Fixed per-client validation split root seed.')
parser.add_argument('--logfr_report_seed', type=int, default=42,
                    help='Independent pure-fake reported-accuracy RNG seed.')
parser.add_argument('--logfr_linkage', choices=['single', 'average', 'complete'], default='average',
                    help='Pilot parameter: hierarchical linkage for submitted full-model cosine distance.')
parser.add_argument('--logfr_cut_ratio', type=float, default=0.7,
                    help='Pilot parameter: dendrogram cut as a fraction of the largest merge distance.')
parser.add_argument('--logfr_group_order', choices=['client_id', 'medoid_first'], default='client_id',
                    help='Pilot parameter: within-cluster server audit order.')
parser.add_argument('--logfr_complaint_policy',
                    choices=['silent_fake', 'truthful_fake', 'always_audit'], default='silent_fake',
                    help='Client complaint simulation; fake-round behavior is not specified by the paper.')
parser.add_argument('--logfr_shadow_full_audit', action='store_true',
                    help='Pilot only: score otherwise skipped submissions for diagnostics without changing decisions.')
parser.add_argument(
    '--check_detecting_setup',
    action='store_true',
    help='Validate the detecting category/import and exit before loading data.',
)
parser.add_argument(
    '--sampling_shapley_num_permutations',
    type=int,
    default=None,
    help='TMC-Shapley permutation count; omitted selects the upstream automatic budget.',
)
parser.add_argument(
    '--sampling_shapley_seed',
    type=int,
    default=None,
    help='TMC-Shapley RNG seed; omitted derives a deterministic seed from trial_id.',
)
parser.add_argument(
    '--sampling_shapley_tolerance',
    type=float,
    default=1e-6,
    help='Absolute TMC prefix-to-grand utility threshold for early truncation.',
)
parser.add_argument(
    '--sampling_least_core_num_coalitions',
    type=int,
    default=None,
    help='Number of unique Least Core coalition constraints; omitted uses min(2^n, n^2).',
)
parser.add_argument(
    '--sampling_least_core_seed',
    type=int,
    default=None,
    help='Sampled Least Core RNG seed; omitted derives a deterministic seed from trial_id.',
)

# Lin et al. STD-DAGMM options
parser.add_argument('--std_dagmm_epochs', type=int, default=100,
                    help='Full-batch optimization epochs for each round-local STD-DAGMM model.')
parser.add_argument('--std_dagmm_lr', type=float, default=1e-3,
                    help='Adam learning rate for STD-DAGMM.')
parser.add_argument('--std_dagmm_latent_dim', type=int, default=1,
                    help='Autoencoder latent dimension for STD-DAGMM.')
parser.add_argument('--std_dagmm_components', type=int, default=2,
                    help='Number of Gaussian mixture components in STD-DAGMM.')
parser.add_argument('--std_dagmm_lambda_energy', type=float, default=0.1,
                    help='Sample-energy loss weight in STD-DAGMM.')
parser.add_argument('--std_dagmm_lambda_cov_diag', type=float, default=0.005,
                    help='Covariance diagonal penalty weight in STD-DAGMM.')
parser.add_argument('--std_dagmm_covariance_eps', type=float, default=1e-4,
                    help='Positive covariance regularizer for stable GMM energy evaluation.')
parser.add_argument('--std_dagmm_min_cluster_separation', type=float, default=0.5,
                    help='Archived v1 compatibility option; canonical STD-DAGMM uses the paper median rule.')
parser.add_argument('--std_dagmm_energy_mad_scale', type=float, default=3.0,
                    help='Archived v1 compatibility option; canonical STD-DAGMM uses the paper median rule.')
parser.add_argument('--std_dagmm_normalize_input', action=argparse.BooleanOptionalAction, default=False,
                    help='Optional RMS variant; disabled by default for the paper-faithful STD-DAGMM path.')
parser.add_argument('--std_dagmm_seed', type=int, default=42,
                    help='Base random seed for round-local STD-DAGMM initialization.')

# Huang et al. Delta-DAGMM options
parser.add_argument('--delta_dagmm_epochs', type=int, default=100,
                    help='Full-batch optimization epochs for each round-local Delta-DAGMM model.')
parser.add_argument('--delta_dagmm_lr', type=float, default=1e-3,
                    help='Adam learning rate for Delta-DAGMM.')
parser.add_argument('--delta_dagmm_latent_dim', type=int, default=1,
                    help='Autoencoder latent dimension for Delta-DAGMM.')
parser.add_argument('--delta_dagmm_components', type=int, default=2,
                    help='Number of Gaussian mixture components in Delta-DAGMM.')
parser.add_argument('--delta_dagmm_lambda_energy', type=float, default=0.1,
                    help='Sample-energy loss weight in Delta-DAGMM.')
parser.add_argument('--delta_dagmm_lambda_cov_diag', type=float, default=0.005,
                    help='Covariance diagonal penalty weight in Delta-DAGMM.')
parser.add_argument('--delta_dagmm_covariance_eps', type=float, default=1e-4,
                    help='Positive covariance regularizer for stable GMM energy evaluation.')
parser.add_argument('--delta_dagmm_linear_scale', type=float, default=1.0,
                    help='Paper Eq. (5) affine multiplier k for the submitted model increment.')
parser.add_argument('--delta_dagmm_linear_offset', type=float, default=0.0,
                    help='Paper Eq. (5) affine offset b for the submitted model increment.')
parser.add_argument('--delta_dagmm_output_layer_only', action=argparse.BooleanOptionalAction, default=True,
                    help='Use only the final matrix-shaped parameter update, as in the paper.')
parser.add_argument('--delta_dagmm_normalize_input', action=argparse.BooleanOptionalAction, default=False,
                    help='Optional RMS variant; disabled by default for the paper-faithful Delta-DAGMM path.')
parser.add_argument('--delta_dagmm_threshold_multiplier', type=float, default=1.08,
                    help='Paper per-round threshold multiplier applied to mean sample energy.')
parser.add_argument('--delta_dagmm_signed_energy_safe_threshold', action=argparse.BooleanOptionalAction, default=False,
                    help='Optional signed-energy safeguard; disabled by default for the literal paper threshold.')
parser.add_argument('--delta_dagmm_seed', type=int, default=42,
                    help='Base random seed for Delta-DAGMM initialization.')

# Chen et al. WEF options
parser.add_argument('--wef_epsilon', type=float, default=0.05,
                    help='Paper reputation-threshold epsilon in max(Dev) - epsilon.')
parser.add_argument('--wef_weight_from_end', type=int, default=2,
                    help='Matrix-shaped parameter index from the end; 2 selects the penultimate layer.')

# Nakamura et al. S2-WEF options
parser.add_argument('--s2_wef_weight_from_end', type=int, default=2,
                    help='Matrix-shaped parameter index from the end; 2 selects the penultimate layer.')
parser.add_argument('--s2_wef_gamma_multiplier', type=float, default=1.5,
                    help='Paper Eq. (9) multiplier on the median gamma score.')
parser.add_argument('--s2_wef_deviation_epsilon', type=float, default=0.05,
                    help='Paper Eq. (10) epsilon in max(Dev) - epsilon.')
parser.add_argument('--s2_wef_silhouette_threshold', type=float, default=0.30,
                    help='Minimum silhouette coefficient for retaining two clusters.')
parser.add_argument('--s2_wef_merge_jump_threshold', type=float, default=0.90,
                    help='Minimum final dendrogram merge-distance jump ratio.')
parser.add_argument(
    '--use_last_round_sv',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Use last-round server-gradient Shapley utility (default: True).'
)
parser.add_argument('--use_attack', action='store_true', help='Enable attack or not')
parser.add_argument('--trial_id', type=int, default=0, help='Repetition ID')
parser.add_argument('--random_round', type=int, default=10,
                    help='Warmup/fallback rounds before prediction-style attackers start; default keeps the old WARMUP=10 behavior.')

# Layerwise-only prediction controls. These options are intentionally passed
# only to the four Layerwise* attackers below.
parser.add_argument(
    '--layerwise_selection_metric',
    choices=['normalized_rms', 'mean_norm'],
    default='normalized_rms',
    help='Warm-up importance metric used to select forecasted layers.',
)
parser.add_argument(
    '--layerwise_selection_fraction',
    type=float,
    default=0.5,
    help='Fraction of layer groups selected for forecasting when top-N is unset.',
)
parser.add_argument(
    '--layerwise_selection_top_n',
    type=int,
    default=None,
    help='Optional exact maximum number of layer groups selected for forecasting.',
)
parser.add_argument(
    '--layerwise_pca_threshold',
    type=int,
    default=512,
    help='Apply PCA when a selected parameter tensor exceeds this dimension.',
)
parser.add_argument(
    '--layerwise_pca_components',
    type=int,
    default=8,
    help='Requested latent dimension for PCA-compressed selected tensors.',
)

# hessian optional
parser.add_argument('--use_hessian', action='store_true', help='Enable Hessian fusion')
parser.add_argument('--lambda_h', type=float, default=0.9, help='Weight for f_P in Eq.(3)')

# ARIMA readiness plus adaptive reset/retraining options
parser.add_argument('--arima_var_window', type=int, default=3,
                    help=('Number of adjacent variance differences for offline/adaptive '
                          'ARIMA; evaluates var_window+1 sliding windows from '
                          'k+var_window observations. Must be at least 2.'))
parser.add_argument('--arima_var_threshold', type=float, default=None,
                    help=('ARIMA readiness variance-difference threshold. Defaults preserve '
                          'the existing routes: 5e-3 ordinary and 5e-2 layerwise.'))
parser.add_argument('--arima_err_threshold', type=float, default=5e-3,
                    help='Adaptive ARIMA reset threshold on EMA prediction error.')
parser.add_argument('--arima_min_rounds_after_start', type=int, default=2,
                    help='Minimum observed rounds after ARIMA starts before reset is allowed.')
parser.add_argument('--arima_retrain_interval', type=int, default=10,
                    help='Periodic retrain interval for adaptive ARIMA; <=0 disables periodic retraining.')
parser.add_argument('--arima_err_ema_beta', type=float, default=0.9,
                    help='EMA smoothing factor for adaptive ARIMA prediction error.')
parser.add_argument('--arima_trim_history', action=argparse.BooleanOptionalAction, default=True,
                    help='Trim gradient history after adaptive ARIMA reset.')

# Online Moirai residual-adapter options. These are intentionally separate
# from adaptive ARIMA controls because the two algorithms maintain different
# online state.
parser.add_argument('--moirai_online_tuning', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Enable the online Moirai residual adapter (enabled by default).')
parser.add_argument('--moirai_tune_lr', type=float, default=1e-3,
                    help=('Residual-adapter learning rate for online Moirai. '
                          'The compatibility default is 1e-3; set 0.1 explicitly '
                          'to test stronger online adaptation.'))
parser.add_argument('--moirai_tune_steps', type=int, default=1,
                    help='Residual-adapter update steps per observed global gradient.')
parser.add_argument('--moirai_tune_start_round', type=int, default=None,
                    help='First round for residual adaptation; defaults to --random_round.')
parser.add_argument('--moirai_adapt_interval', type=int, default=10,
                    help=('Successful aligned residual updates between decay adjustments; '
                          '<=0 disables decay.'))
parser.add_argument('--moirai_residual_decay', type=float, default=0.9,
                    help='Multiplicative residual decay used at each adaptation interval.')
parser.add_argument('--moirai_residual_clip', type=float, default=5.0,
                    help='Absolute elementwise clipping bound for the online residual.')
parser.add_argument('--moirai_err_ema_beta', type=float, default=0.9,
                    help='EMA smoothing factor for online Moirai prediction error.')
parser.add_argument('--moirai_err_threshold', type=float, default=None,
                    help=('Online Moirai reset threshold. Defaults preserve the previous '
                          'route: 5e-3 non-layerwise; 0.045 for MNIST layerwise and '
                          '0.015 for other layerwise datasets.'))
parser.add_argument('--moirai_min_rounds_after_start', type=int, default=None,
                    help=('Minimum observed rounds before an online Moirai reset. '
                          'Defaults preserve the previous route: 2 non-layerwise, '
                          '4 layerwise.'))
parser.add_argument('--moirai_trim_history', action=argparse.BooleanOptionalAction, default=True,
                    help='Trim online Moirai gradient history after a hard reset.')

# scaled-delta / advanced scaled-delta options
parser.add_argument('--scaled_delta_c_const', type=float, default=1.0,
                    help='C constant for advanced scaled-delta cosine estimate.')
parser.add_argument('--scaled_delta_lambda_est', type=float, default=None,
                    help='Optional fixed lambda for advanced scaled-delta; if omitted, estimate from update-norm decay.')
parser.add_argument('--scaled_delta_noise_fraction', type=float, default=1.0,
                    help='Fraction of flattened dimensions eligible for advanced scaled-delta noise.')
parser.add_argument('--scaled_delta_max_noise_ratio', type=float, default=5.0,
                    help='Cap for ||phi|| / ||scaled_delta|| in advanced scaled-delta.')
parser.add_argument('--scaled_delta_max_scale', type=float, default=10.0,
                    help='Cap for scaled-delta norm-decay multiplier.')

# Lin 2019 random attack options
parser.add_argument(
    '--random_radius', '--random_R',
    dest='random_radius',
    type=float,
    default=None,
    help='Lin random-attack radius R. Required when --attack_method random is used.',
)

# Fraboni 2021 disguised free-rider options
parser.add_argument(
    '--fraboni_noise_std',
    type=float,
    default=None,
    help=(
        'Positive bootstrap/fixed Gaussian sigma for fraboni_disguised. '
        'Provide an externally calibrated value; history_rms uses it only before history exists.'
    ),
)
parser.add_argument(
    '--fraboni_calibration_mode',
    choices=['fixed', 'history_rms'],
    default='history_rms',
    help=(
        'fixed uses --fraboni_noise_std every round. history_rms uses the '
        'recent global-update RMS per layer after round 0; it is a documented project adaptation.'
    ),
)
parser.add_argument('--fraboni_calibration_window', type=int, default=5,
                    help='Number of prior global updates used by history_rms calibration.')
parser.add_argument('--fraboni_min_noise_std', type=float, default=1e-12,
                    help='Lower bound for a history-calibrated Fraboni sigma.')

# Wang 2022 public-warm-start local-deviation Adam adaptation
parser.add_argument('--wang_adam_lr', type=float, default=8e-3,
                    help='Adam learning rate for the local/global-deviation update.')
parser.add_argument('--wang_adam_beta1', type=float, default=0.9,
                    help='Adam beta1 for the local/global-deviation update.')
parser.add_argument('--wang_adam_beta2', type=float, default=0.999,
                    help='Adam beta2 for the local/global-deviation update.')
parser.add_argument('--wang_adam_eps', type=float, default=1e-8,
                    help='Adam numerical epsilon for the local/global-deviation update.')
parser.add_argument('--adam_public_fraction', type=float, default=0.05,
                    help='Fraction of the pre-partition training pool reserved as the simulated public pool.')
parser.add_argument('--adam_public_size', type=int, default=None,
                    help='Optional exact size of the simulated public pool; overrides --adam_public_fraction.')
parser.add_argument('--adam_public_seed', type=int, default=20260726,
                    help='Seed for stratified extraction of the simulated public pool.')

# DKFR-lite options
parser.add_argument('--dkfr_synth_batch_size', type=int, default=64,
                    help='Synthetic batch size for DKFR-lite generator/KD steps.')
parser.add_argument('--dkfr_generator_steps', type=int, default=5,
                    help='Number of Eq.(1)-lite generator optimization steps per fake round.')
parser.add_argument('--dkfr_kd_steps', type=int, default=5,
                    help='Number of Eq.(4)-lite student optimization steps per fake round.')
parser.add_argument('--dkfr_lr_g', type=float, default=1e-3,
                    help='Generator learning rate for DKFR-lite.')
parser.add_argument('--dkfr_lr_student', type=float, default=1e-3,
                    help='Student/local model learning rate for DKFR-lite.')
parser.add_argument('--dkfr_temperature', type=float, default=2.0,
                    help='KD temperature for DKFR-lite.')
parser.add_argument('--dkfr_alpha_feat', type=float, default=0.1,
                    help='Hidden-feature cosine loss weight for DKFR-lite.')
parser.add_argument('--dkfr_alpha_kl', type=float, default=1.0,
                    help='Student KL weight for DKFR-lite.')
parser.add_argument('--dkfr_alpha_cls', type=float, default=1.0,
                    help='Student supervised synthetic-label CE/BCE weight for DKFR-lite.')
parser.add_argument('--dkfr_hidden_dim', type=int, default=128,
                    help='Hidden size of the DKFR-lite generator MLP.')
parser.add_argument('--dkfr_lambda_kl_g', type=float, default=1.0,
                    help='Weight for adversarial teacher-student KL in DKFR-lite generator loss.')
parser.add_argument('--dkfr_kl_cap', type=float, default=0.0,
                    help='Cap generator KL term before applying adversarial sign; <=0 disables capping.')
parser.add_argument('--dkfr_student_kl_cap', type=float, default=0.0,
                    help='Smooth cap for student KD loss; <=0 disables capping.')
parser.add_argument('--dkfr_student_loss_cap', type=float, default=0.0,
                    help='Smooth cap for total student DKFR loss; <=0 disables capping.')
parser.add_argument('--dkfr_teacher_conf_min', type=float, default=0.0,
                    help='Minimum teacher confidence for multiclass synthetic KD; <=0 keeps all samples.')
parser.add_argument('--dkfr_guard_reset_patience', type=int, default=3,
                    help='Reset DKFR generator after this many consecutive guarded rounds; <=0 disables reset.')
parser.add_argument('--dkfr_mnist_tv', type=float, default=0.0,
                    help='Total-variation image prior weight for MNIST DKFR generator.')
parser.add_argument('--dkfr_mnist_pixel_reg', type=float, default=0.0,
                    help='MNIST pixel mean/std regularization weight for DKFR generator.')
parser.add_argument('--dkfr_mnist_logit_reg', type=float, default=0.0,
                    help='MNIST-only student logit L2 regularization weight for synthetic DKFR training.')
parser.add_argument('--dkfr_grad_clip', type=float, default=5.0,
                    help='Clip DKFR-lite generator/student gradient norm; <=0 disables clipping.')
parser.add_argument('--dkfr_max_update_norm', type=float, default=0.0,
                    help='Clip DKFR-lite fake update norm before upload; <=0 disables clipping.')

# Lee-style synthetic local-data replacement options
parser.add_argument('--synthetic_mixup_alpha', type=float, default=0.4,
                    help='Beta concentration for same-label MixUp on MNIST/IMDB synthetic replacement.')
parser.add_argument('--synthetic_noise_std', type=float, default=0.02,
                    help='Feature-scaled Gaussian noise for MNIST/IMDB synthetic replacement.')
parser.add_argument('--synthetic_tabular_noise_std', type=float, default=0.05,
                    help='Feature-scaled Gaussian noise for non-binary tabular columns.')
parser.add_argument('--synthetic_tabular_flip_prob', type=float, default=0.02,
                    help='Bit-flip probability for binary-like tabular/relational columns.')
parser.add_argument('--synthetic_seed', type=int, default=0,
                    help='Seed offset for fixed per-client synthetic replacement shards.')

# lazy mode
parser.add_argument('--use_lazy', action='store_true', help='Enable Q3 lazy mode: sparse honest rounds, fake otherwise')
parser.add_argument('--lazy_schedule', type=str, default='fixed', choices=['fixed', 'probabilistic'],
                    help='Lazy scheduling rule. fixed uses lazy_honest_gap; probabilistic randomly places a fixed fake-round budget set by --attack_probability.')
parser.add_argument('--attack_probability', type=float, default=1.0,
                    help='Target fake-round ratio when --lazy_schedule probabilistic is used; exactly round(p * total_rounds) fake rounds are sampled.')
parser.add_argument('--lazy_honest_gap', type=int, default=3, help='Honest cadence in lazy mode, e.g. 3 means 1 honest / 2 fake')
parser.add_argument('--lazy_honest_phase', type=int, default=0, help='Honest phase offset; default 0 makes round 0 honest')
parser.add_argument('--lazy_fake_policy', type=str, default='attack', choices=['attack', 'zero'], help='Submission policy on non-honest rounds in lazy mode')


# dump clean global-grad history for Moirai fine-tuning 
parser.add_argument('--dump_grad_history', action='store_true',
                    help='Dump flattened global gradients each round (clean run, no attacker)')
parser.add_argument('--dump_dir', type=str, default='data/grad_hist',
                    help='Directory to save gradient CSVs, e.g., data/grad_hist')
parser.add_argument('--results_root', type=str, default='Results_updated',
                    help='Root directory for isolated updated-run csv outputs.')


args = parser.parse_args()
task_wall_started_at = time.perf_counter()
try:
    fl_aggregation_config = AggregationConfig.from_args(args)
except ValueError as exc:
    parser.error(str(exc))
args.fl_aggregate = fl_aggregation_config.method
# Existing FedAvg paths stay intact. New rules/settings cannot overwrite them.
aggregation_subdirectory = fl_aggregation_config.output_subdirectory()
if aggregation_subdirectory:
    args.results_root = os.path.join(args.results_root, aggregation_subdirectory)
    args.dump_dir = os.path.join(args.dump_dir, aggregation_subdirectory)
results_root = args.results_root

try:
    detecting_spec = resolve_detecting_method(
        method_name=args.detecting_method,
        requested_type=args.detecting_type,
    )
except ValueError as exc:
    parser.error(str(exc))

detecting_method = detecting_spec.name
detecting_type = detecting_spec.detecting_type
if fl_aggregation_config.method != "fedavg":
    if detecting_type == "quantitative" and not args.use_last_round_sv:
        parser.error('FedAdam/weighted Median contribution evaluation requires --use_last_round_sv.')
    if detecting_method in {"wef", "s2_wef"} and args.qualitative_aggregation_mode == "paper_online":
        parser.error('FedAdam/weighted Median with WEF/S2-WEF requires --qualitative_aggregation_mode passive.')
if detecting_type == "quantitative":
    ContributionCalculator = detecting_spec.implementation
else:
    QualitativeDetector = detecting_spec.implementation

if detecting_method == "pass":
    try:
        PASSConfig.from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    # PASS is an observer: retain FedAvg even when the WEF default is paper_online.
    args.qualitative_aggregation_mode = "passive"
elif detecting_method == "log_based_free_rider_detection":
    try:
        LogBasedFreeRiderConfig.from_args(args, seed=42, trial_id=args.trial_id)
    except ValueError as exc:
        parser.error(str(exc))
    args.qualitative_aggregation_mode = "passive"

if args.check_detecting_setup:
    print(
        f"Detecting setup OK: type={detecting_type}, method={detecting_method}, "
        f"implementation={detecting_spec.implementation.__module__}."
        f"{detecting_spec.implementation.__name__}"
    )
    raise SystemExit(0)


if args.control_excluded_ids is not None:
    from module.method.freerider_control import validate_control
    validate_control(args)

if args.rq1_strict_data_free and not args.rq1_only:
    parser.error('--rq1_strict_data_free requires --rq1_only')
if args.rq1_only:
    from module.method.rq1_single_attacker import validate_config as validate_rq1_config, configure_worker_threads as configure_rq1_threads
    try:
        validate_rq1_config(args)
        configure_rq1_threads()
    except ValueError as exc:
        parser.error(str(exc))

if args.attacker_ids is not None:
    from module.method.multi_freerider import validate_config

    if any(token.split('=')[0] == '--attacker_id' for token in sys.argv[1:]):
        parser.error('Use --attacker_ids alone; do not combine it with --attacker_id.')
    try:
        validate_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.attacker_id = None
attacker_id = args.attacker_id if args.attacker_ids is None else None
alpha = args.alpha
dataset = args.dataset.lower()
if dataset == "adult2":
    dataset = "adult"
attack_method = args.attack_method.lower()
num_parts = args.client_num
use_attack = args.use_attack
trial_id = args.trial_id

if args.num_epoch < 1:
    raise ValueError(f"--num_epoch must be at least 1, got {args.num_epoch}")

if detecting_method == "sampling_shapley":
    if (
        args.sampling_shapley_num_permutations is not None
        and args.sampling_shapley_num_permutations < 1
    ):
        raise ValueError("--sampling_shapley_num_permutations must be at least 1")
    if (
        not np.isfinite(args.sampling_shapley_tolerance)
        or args.sampling_shapley_tolerance < 0.0
    ):
        raise ValueError(
            "--sampling_shapley_tolerance must be a finite non-negative number"
        )
elif detecting_method == "sampling_least_core":
    evaluated_world_sizes = [num_parts]
    if use_attack:
        evaluated_world_sizes.append(num_parts - 1)
    if min(evaluated_world_sizes) < 1:
        raise ValueError(
            "sampling_least_core requires at least one client in every evaluated world"
        )
    if args.sampling_least_core_num_coalitions is not None:
        if args.sampling_least_core_num_coalitions < 2:
            raise ValueError(
                "--sampling_least_core_num_coalitions must be at least 2"
            )
        smallest_world_limit = min(2 ** size for size in evaluated_world_sizes)
        if args.sampling_least_core_num_coalitions > smallest_world_limit:
            raise ValueError(
                "--sampling_least_core_num_coalitions may be at most "
                f"{smallest_world_limit} across the evaluated client worlds"
            )

if not 0.0 <= args.attack_probability <= 1.0:
    raise ValueError(f"--attack_probability must be in [0, 1], got {args.attack_probability}")

if not 0.0 < args.layerwise_selection_fraction <= 1.0:
    raise ValueError(
        "--layerwise_selection_fraction must be in (0, 1], got "
        f"{args.layerwise_selection_fraction}"
    )
if args.layerwise_selection_top_n is not None and args.layerwise_selection_top_n < 1:
    raise ValueError("--layerwise_selection_top_n must be at least 1 when supplied")
if args.layerwise_pca_threshold < 1:
    raise ValueError("--layerwise_pca_threshold must be at least 1")
if args.layerwise_pca_components < 2:
    raise ValueError("--layerwise_pca_components must be at least 2")

if use_attack and attack_method == "random":
    if args.random_radius is None or args.random_radius <= 0.0:
        raise ValueError(
            "--attack_method random requires a positive --random_radius / --random_R; "
            "Lin et al. leaves R to attacker-side scale calibration."
        )

if use_attack and attack_method == "fraboni_disguised":
    if args.fraboni_noise_std is None or args.fraboni_noise_std <= 0.0:
        raise ValueError(
            "--attack_method fraboni_disguised requires a positive --fraboni_noise_std"
        )
    if args.fraboni_calibration_window < 1:
        raise ValueError("--fraboni_calibration_window must be at least 1")
    if args.fraboni_min_noise_std <= 0.0:
        raise ValueError("--fraboni_min_noise_std must be positive")

if use_attack and attack_method == "adam_local_deviation":
    if args.adam_public_size is not None and args.adam_public_size < 1:
        raise ValueError("--adam_public_size must be at least 1 when supplied")
    if args.adam_public_size is None and not 0.0 < args.adam_public_fraction < 1.0:
        raise ValueError(
            "--adam_public_fraction must lie strictly between 0 and 1 when "
            "--adam_public_size is not supplied"
        )
    if args.use_lazy:
        raise ValueError(
            "adam_local_deviation is data-free after its public warm start and cannot be "
            "combined with --use_lazy: a lazy 'honest' round would require the attacker's "
            "private shard. Use the non-lazy Wang path or a separately declared public-retrain variant."
        )

if use_attack and attack_method in ARIMA_METHODS:
    if args.arima_var_window < 2:
        raise ValueError("--arima_var_window must be at least 2")
    if (
        args.arima_var_threshold is not None
        and (
            not np.isfinite(args.arima_var_threshold)
            or args.arima_var_threshold <= 0.0
        )
    ):
        raise ValueError("--arima_var_threshold must be positive when supplied")

if use_attack and attack_method in ADAPTIVE_ARIMA_METHODS:
    if (
        not np.isfinite(args.arima_err_ema_beta)
        or not 0.0 <= args.arima_err_ema_beta <= 1.0
    ):
        raise ValueError("--arima_err_ema_beta must be in [0, 1]")
    if not np.isfinite(args.arima_err_threshold) or args.arima_err_threshold <= 0.0:
        raise ValueError("--arima_err_threshold must be positive")
    if args.arima_min_rounds_after_start < 0:
        raise ValueError("--arima_min_rounds_after_start must be non-negative")

if use_attack and attack_method in MOIRAI_ONLINE_METHODS:
    if not np.isfinite(args.moirai_tune_lr) or args.moirai_tune_lr < 0.0:
        raise ValueError("--moirai_tune_lr must be non-negative")
    if args.moirai_tune_steps < 1:
        raise ValueError("--moirai_tune_steps must be at least 1")
    if args.moirai_tune_start_round is not None and args.moirai_tune_start_round < 0:
        raise ValueError("--moirai_tune_start_round must be non-negative when supplied")
    if (
        not np.isfinite(args.moirai_residual_decay)
        or not 0.0 <= args.moirai_residual_decay <= 1.0
    ):
        raise ValueError("--moirai_residual_decay must be in [0, 1]")
    if not np.isfinite(args.moirai_residual_clip) or args.moirai_residual_clip <= 0.0:
        raise ValueError("--moirai_residual_clip must be positive")
    if (
        not np.isfinite(args.moirai_err_ema_beta)
        or not 0.0 <= args.moirai_err_ema_beta <= 1.0
    ):
        raise ValueError("--moirai_err_ema_beta must be in [0, 1]")
    if (
        args.moirai_err_threshold is not None
        and (
            not np.isfinite(args.moirai_err_threshold)
            or args.moirai_err_threshold <= 0.0
        )
    ):
        raise ValueError("--moirai_err_threshold must be positive when supplied")
    if (
        args.moirai_min_rounds_after_start is not None
        and args.moirai_min_rounds_after_start < 0
    ):
        raise ValueError(
            "--moirai_min_rounds_after_start must be non-negative when supplied"
        )

need_history = (detecting_type == "qualitative")
need_gradients = (
    detecting_type == "quantitative"
    or attack_method in [
        "replay",
        "scaled_delta",
        "advanced_scaled_delta",
        "adam_local_deviation",
        "dkfr",
        "arima",
        "arima_adaptive",
        "moirai",
        "moirai_online",
        "arima_mnist_imdb_layerwise",
        "arima_adaptive_mnist_imdb_layerwise",
        "moirai_mnist_imdb_layerwise",
        "moirai_online_mnist_imdb_layerwise",
    ]
)

# datasets=("tictactoe" "adult" "dota2")
# attack_methods=("random" "replay" "arima" "moirai")
# client_nums=(2 4 6 8)
# declare -A alpha_map
# alpha_map[tictactoe]="10 100 1000 20000"
# alpha_map[adult]="1000 100000 1000000 20000000"
# alpha_map[dota2]="10000 1000000 50000000 200000000"
# alpha_map[mnist] = [10, 1000, 100000, 10000000] 
# alpha_map[imdb] = [10, 100, 1000, 200000] 

uniform_thresholds = {
    "tictactoe": 20000,
    "adult": 20000000,
    "dota2": 200000000,
    "mnist": 10000000,
    "imdb": 200000,
}

distribution = "uniform" if alpha == uniform_thresholds[dataset] else "quantity skew"

# model_name_map = {
#     "tictactoe": "TicTacToeLR",
#     "adult": "AdultLR",
#     "dota2": "Dota2LR",
#     "mnist": "MNISTLR"
# }
model_name_map = {
    "tictactoe": "TicTacToeMLP",
    "adult": "AdultMLP",
    "dota2": "Dota2MLP",
    "mnist": "MNISTMLP",
    "imdb": "IMDBMLP",
}
model_name = model_name_map[dataset]



# ======== Configuration ========
seed = args.seed
num_epoch = args.num_epoch
num_local_epochs = args.num_local_epochs
if num_local_epochs < 1:
    raise ValueError("--num_local_epochs must be at least 1")

lr = 0.008
hidden_layer_size = 16
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attack_time_log = []



# Fixed window for SV evaluation (start from checkpoint at T-R_FOR_SV)
R_FOR_SV = 20
provenance_payload, provenance_manifest_path = _write_run_provenance(
    args,
    dataset,
    attack_method,
    detecting_method,
)
print(
    "=== Provenance: "
    + json.dumps(
        {
            "executed_entry_path": provenance_payload["executed_entry_path"],
            "entry_sha256": provenance_payload["entry_sha256"],
            "attack_module_path": provenance_payload["attack_module_path"],
            "attack_module_sha256": provenance_payload["attack_module_sha256"],
            "attack_modules": provenance_payload["attack_modules"],
            "results_root": provenance_payload["results_root"],
            "manifest_path": str(provenance_manifest_path.resolve()),
        },
        sort_keys=True,
    )
    + " ===",
    flush=True,
)
print(
    "=== Attack parameters: "
    + json.dumps(
        provenance_payload["attack_parameters"],
        sort_keys=True,
        default=str,
    )
    + " ===",
    flush=True,
)
# ======== Nohup Logs ========
print(f"=== Running {dataset}, attack={attack_method}, N={num_parts}, alpha={alpha}, "
      f"use_attack={use_attack}, attacker_id={attacker_id}, trial_id={trial_id} ===", flush=True)
print(f"=== FL aggregation: {fl_aggregation_config.to_dict()} (sample-size weighted) ===", flush=True)
print(
    f"=== Detecting config: type={detecting_type}, method={detecting_method} ===",
    flush=True,
)
print(
    f"=== Lazy config: use_lazy={args.use_lazy}, schedule={args.lazy_schedule}, "
    f"attack_probability={args.attack_probability}, honest_gap={args.lazy_honest_gap}, "
    f"honest_phase={args.lazy_honest_phase}, fake_policy={args.lazy_fake_policy} ===",
    flush=True,
)


if device.type == 'cuda':
    print(f"detect available GPU: {torch.cuda.get_device_name(0)}, FL training is running in pure GPU mode.")
else:
    print("Warning: No available GPU detected. The training process will revert to CPU mode.")

# ======== Attack Configuration ========
K = 5              # look back rounds
WARMUP = args.random_round        # warmup rounds / random rounds
HISTORY = K        # Moirai's history_length = k 



def _select_stratified_public_indices(labels, public_size, seed):
    """Return a deterministic approximately label-stratified index set."""
    labels_np = labels.detach().cpu().reshape(-1).numpy()
    total = int(labels_np.shape[0])
    if public_size < 1 or public_size >= total:
        raise ValueError(
            f"public_size must be in [1, {total - 1}], got {public_size}"
        )

    rng = np.random.default_rng(int(seed))
    _, inverse, counts = np.unique(labels_np, return_inverse=True, return_counts=True)
    raw_allocations = counts.astype(np.float64) * float(public_size) / float(total)
    allocations = np.floor(raw_allocations).astype(np.int64)
    remainder = int(public_size - int(allocations.sum()))
    fractional_order = np.argsort(-(raw_allocations - allocations), kind="stable")
    for class_position in fractional_order:
        if remainder <= 0:
            break
        if allocations[class_position] < counts[class_position]:
            allocations[class_position] += 1
            remainder -= 1
    if remainder != 0:
        raise RuntimeError("Could not allocate the requested simulated public-pool size")

    selected = []
    for class_position, allocation in enumerate(allocations):
        if allocation == 0:
            continue
        class_indices = np.flatnonzero(inverse == class_position)
        selected.append(rng.choice(class_indices, size=int(allocation), replace=False))

    public_indices = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(public_indices)
    return torch.as_tensor(public_indices, dtype=torch.long)


def _reserve_simulated_public_pool(loader, public_fraction, public_size, public_seed):
    """Remove a disjoint in-domain public pool before client partitioning.

    This is a controlled simulated-public-data protocol, not a claim that the
    samples are an external public dataset.  ``loader.X_test/y_test`` are never
    read or modified here.
    """
    total = int(len(loader.y_train))
    requested_size = (
        int(public_size)
        if public_size is not None
        else int(round(float(public_fraction) * total))
    )
    if requested_size < 1 or requested_size >= total:
        raise ValueError(
            "The simulated public pool must leave at least one training sample; "
            f"requested={requested_size}, available={total}"
        )

    public_indices = _select_stratified_public_indices(
        loader.y_train,
        requested_size,
        public_seed,
    )
    train_mask = torch.ones(total, dtype=torch.bool)
    train_mask[public_indices] = False

    public_X = loader.X_train[public_indices].detach().clone()
    public_y = loader.y_train[public_indices].detach().clone()
    loader.X_train = loader.X_train[train_mask]
    loader.y_train = loader.y_train[train_mask]

    if len(loader.y_train) + len(public_y) != total:
        raise RuntimeError("Simulated public-pool extraction changed the sample count")
    if public_X.shape[0] != public_y.shape[0]:
        raise RuntimeError("Simulated public-pool feature/label count mismatch")

    print(
        "[ADAM_PUBLIC_POOL] "
        f"source=in_domain_held_out size={len(public_y)} "
        f"remaining_train={len(loader.y_train)} seed={public_seed}",
        flush=True,
    )
    return public_X, public_y


def _partition_loader_after_public_pool(loader, seed, distribution, alpha, num_parts):
    """Mirror ``script.datapre.load_and_partition`` after pool extraction."""
    if distribution == "uniform":
        loader.uniform_split(num_parts)
    elif distribution == "quantity skew":
        alpha_list = np.full(int(num_parts), float(alpha), dtype=np.float64)
        ratios = scipy.stats.dirichlet.rvs(alpha_list, random_state=seed).reshape(-1)
        loader.ratio_split(ratios)
    elif distribution == "label skew":
        alpha_list = np.full(int(num_parts), float(alpha), dtype=np.float64)
        loader.non_iid_split(num_parts, alpha_list, random_state=seed)
    else:
        raise ValueError(f"Unsupported client distribution: {distribution}")

    empty_clients = [
        client_id
        for client_id, client_labels in enumerate(loader.y_train_parts)
        if len(client_labels) == 0
    ]
    if empty_clients:
        raise ValueError(
            "The simulated-public-pool split produced empty client shards: "
            f"{empty_clients}. Reduce --adam_public_fraction/--adam_public_size or adjust alpha."
        )


def _load_data_with_simulated_public_pool(seed, dataset, distribution, alpha, num_parts):
    """Read, reserve the public pool, then partition remaining client data."""
    loader = str_to_dataset(dataset)
    loader.read(
        test_ratio=config.test_ratio,
        shuffle_seed=seed,
        nrows=config.num_rows_of_dataset[dataset],
    )
    public_pool = _reserve_simulated_public_pool(
        loader,
        public_fraction=args.adam_public_fraction,
        public_size=args.adam_public_size,
        public_seed=args.adam_public_seed,
    )
    _partition_loader_after_public_pool(
        loader,
        seed=seed,
        distribution=distribution,
        alpha=alpha,
        num_parts=num_parts,
    )
    return loader, public_pool


# Load Data & Model
if use_attack and attack_method == "adam_local_deviation":
    loader, adam_public_pool = _load_data_with_simulated_public_pool(
        seed=seed,
        dataset=dataset,
        distribution=distribution,
        alpha=alpha,
        num_parts=num_parts,
    )
else:
    loader = get_data(seed=seed, dataset=dataset, distribution=distribution,
                      alpha=alpha, num_parts=num_parts)
    adam_public_pool = None

logfr_validation_metadata = None
if detecting_method == "log_based_free_rider_detection":
    logfr_validation_metadata = reserve_client_validation(
        loader, fraction=args.logfr_val_fraction, seed=args.logfr_val_seed,
    )

model_kwargs = dict(
    seed=seed,
    num_epoch=num_epoch,
    lr=lr,
    batch_size=batch_size,
    hidden_layer_size=hidden_layer_size,
    device=device,
    dataset=dataset,
)
if dataset == "imdb":
    model_kwargs["input_size"] = int(loader.X_train.shape[1])
model = return_model(model_name, **model_kwargs)

model.init_args = {
    "seed": seed,
    "num_epoch": num_epoch,
    "lr": lr,
    "device": device,
    "hidden_layer_size": hidden_layer_size,
    "batch_size": batch_size
}
if dataset == "imdb":
    model.init_args["input_size"] = int(loader.X_train.shape[1])



# VLDB-aligned: save common initialization for SV
init_state = copy.deepcopy(model.state_dict())
def prepare_loader_without_attacker(loader, attacker_id):
    ids = [i for i in range(len(loader.X_train_parts)) if i != attacker_id]
    new_loader = copy.deepcopy(loader)
    new_loader.X_train_parts = [loader.X_train_parts[i] for i in ids]
    new_loader.y_train_parts = [loader.y_train_parts[i] for i in ids]
    if hasattr(loader, "X_val_parts"):
        new_loader.X_val_parts = [loader.X_val_parts[i] for i in ids]
        new_loader.y_val_parts = [loader.y_val_parts[i] for i in ids]
        new_loader.X_train = torch.cat(new_loader.X_train_parts, dim=0)
        new_loader.y_train = torch.cat(new_loader.y_train_parts, dim=0)
    return new_loader, ids


def _assert_all_grads_on_device(grads_by_client, device, tag=""):
    # grads_by_client: List[List[Tensor]]
    for ci, grad_list in enumerate(grads_by_client):
        for pi, g in enumerate(grad_list):
            if not isinstance(g, torch.Tensor):
                raise RuntimeError(f"[{tag}] grad is not a Tensor: client={ci} param={pi} type={type(g)}")
            if g.device != device:
                raise RuntimeError(
                    f"[{tag}] grad device mismatch: client={ci} param={pi} g.device={g.device} expected={device}"
                )


def _assert_submitted_update_contract(gradient, model, device, tag=""):
    """Ensure every attack update remains safe for the unchanged detectors."""
    parameters = list(model.parameters())
    if not isinstance(gradient, (list, tuple)):
        raise RuntimeError(f"[{tag}] submitted update must be a list/tuple, got {type(gradient)}")
    if len(gradient) != len(parameters):
        raise RuntimeError(
            f"[{tag}] submitted-update length mismatch {len(gradient)} != {len(parameters)}"
        )
    for parameter_index, (update, parameter) in enumerate(zip(gradient, parameters)):
        if not isinstance(update, torch.Tensor):
            raise RuntimeError(
                f"[{tag}] submitted update {parameter_index} is not a tensor: {type(update)}"
            )
        if update.shape != parameter.shape:
            raise RuntimeError(
                f"[{tag}] submitted update {parameter_index} shape {tuple(update.shape)} "
                f"does not match parameter shape {tuple(parameter.shape)}"
            )
        if update.dtype != parameter.dtype:
            raise RuntimeError(
                f"[{tag}] submitted update {parameter_index} dtype {update.dtype} "
                f"does not match parameter dtype {parameter.dtype}"
            )
        if update.device != device:
            raise RuntimeError(
                f"[{tag}] submitted update {parameter_index} device {update.device} "
                f"does not match expected device {device}"
            )
        if not torch.isfinite(update).all():
            raise FloatingPointError(f"[{tag}] submitted update {parameter_index} is non-finite")

def _assert_model_params_on_device(model, device, tag=""):
    for pi, p in enumerate(model.parameters()):
        if p.device != device:
            raise RuntimeError(
                f"[{tag}] param device mismatch: param={pi} p.device={p.device} expected={device}"
            )

def _assert_weights_on_device(weights, device, tag=""):
    # weights can be Tensor on GPU or Python floats; both ok
    if isinstance(weights, torch.Tensor) and weights.device != device:
        raise RuntimeError(f"[{tag}] weights.device={weights.device} expected={device}")


def _should_use_online_qualitative_aggregation(
    collect_history: bool,
    resolved_detecting_type: str,
    resolved_detecting_method: str,
    aggregation_mode: str,
) -> bool:
    """Return whether WEF-family detections may alter this FL trajectory."""
    return bool(
        collect_history
        and resolved_detecting_type == "qualitative"
        and resolved_detecting_method in {"wef", "s2_wef"}
        and aggregation_mode == "paper_online"
    )


def _aggregate_round_updates(
    submitted_updates: typing.Sequence[typing.Sequence[torch.Tensor]],
    weights: typing.Sequence[typing.Any],
    device: torch.device,
    current_global_parameters: typing.Sequence[torch.Tensor],
    *,
    use_online_qualitative_aggregation: bool,
    detecting_method: str,
    round_client_base_parameters: typing.Sequence[typing.Sequence[torch.Tensor]],
    suspicious_clients: typing.Sequence[int],
    server_aggregator=None,
) -> typing.Tuple[
    typing.List[torch.Tensor],
    typing.Optional[typing.List[typing.List[torch.Tensor]]],
]:
    """Return the actual server step, or preserve the legacy online group rule."""
    if use_online_qualitative_aggregation:
        if server_aggregator is not None and server_aggregator.config.method != "fedavg":
            raise ValueError('New FL aggregation rules require passive WEF/S2-WEF mode')
        from module.method.qualitative_detecting.online_aggregation import (
            aggregate_online_round,
        )

        aggregation_result = aggregate_online_round(
            method=detecting_method,
            base_parameters=round_client_base_parameters,
            submitted_updates=submitted_updates,
            suspicious_clients=suspicious_clients,
            device=device,
        )
        aggregated_gradient = [
            target - previous
            for target, previous in zip(
                aggregation_result["global_parameters"],
                current_global_parameters,
            )
        ]
        return aggregated_gradient, aggregation_result["client_parameters"]

    if server_aggregator is not None:
        return server_aggregator.step(submitted_updates, weights, current_global_parameters), None

    aggregated_gradient = [
        torch.zeros_like(parameter, device=device)
        for parameter in current_global_parameters
    ]
    for client_gradient, weight in zip(submitted_updates, weights):
        for aggregate, gradient in zip(aggregated_gradient, client_gradient):
            aggregate += gradient * weight
    return aggregated_gradient, None


def _resolve_round_attack_activity(
    round_index: int,
    *,
    client_ids: typing.Sequence[int],
    use_attack: bool,
    attacker_id: typing.Optional[int],
    use_lazy: bool,
    lazy_schedule: str,
    lazy_honest_gap: int,
    lazy_honest_phase: int,
    probabilistic_fake_rounds: typing.AbstractSet[int],
    public_warm_start: bool = False,
    attacker_ids: typing.Optional[typing.Sequence[int]] = None,
) -> typing.Dict[str, typing.Any]:
    """Resolve the realized attack truth for one FL round."""
    resolved_client_ids = [int(client_id) for client_id in client_ids]
    designated_ids = (
        list(attacker_ids) if attacker_ids is not None
        else ([int(attacker_id)] if attacker_id is not None else [])
    ) if use_attack else []
    designated_attacker = bool(designated_ids)

    if not designated_attacker:
        attack_state = "benign"
        is_fake_round = False
        is_honest_round = False
        lazy_attack_selected = None
    elif public_warm_start:
        attack_state = "public_warm_start"
        is_fake_round = False
        is_honest_round = False
        lazy_attack_selected = None
    elif not use_lazy:
        attack_state = "fake"
        is_fake_round = True
        is_honest_round = False
        lazy_attack_selected = None
    elif lazy_schedule == "probabilistic":
        lazy_attack_selected = bool(round_index in probabilistic_fake_rounds)
        is_fake_round = lazy_attack_selected
        is_honest_round = not is_fake_round
        attack_state = "fake" if is_fake_round else "honest"
    else:
        lazy_attack_selected = None
        is_honest_round = bool(
            (
                (round_index - lazy_honest_phase)
                % max(1, lazy_honest_gap)
            )
            == 0
        )
        is_fake_round = not is_honest_round
        attack_state = "honest" if is_honest_round else "fake"

    active_attackers = list(designated_ids) if is_fake_round else []
    active_attacker_set = set(active_attackers)
    designated_attacker_labels = {
        client_id: int(client_id in designated_ids)
        for client_id in resolved_client_ids
    }
    ground_truth_labels = {
        client_id: int(client_id in active_attacker_set)
        for client_id in resolved_client_ids
    }
    return {
        "round_idx": int(round_index),
        "attacker_id": int(attacker_id) if designated_attacker and attacker_id is not None else None,
        "attack_state": attack_state,
        "is_fake_round": bool(is_fake_round),
        "is_honest_round": bool(is_honest_round),
        "lazy_attack_selected": lazy_attack_selected,
        "active_attackers": active_attackers,
        "ground_truth_labels": ground_truth_labels,
        "designated_attacker_labels": designated_attacker_labels,
    }


def _attach_round_attack_ground_truth(
    qualitative_result: typing.Optional[typing.Dict[str, typing.Any]],
    round_attack_activity: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    score_evaluator: typing.Optional[typing.Callable[..., typing.Dict[str, typing.Any]]] = None,
) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Attach realized round labels and refresh label-dependent score metrics."""
    if qualitative_result is None:
        return None

    activity_by_round = {
        int(activity["round_idx"]): activity
        for activity in round_attack_activity
    }
    evaluated_records = []
    for record in qualitative_result.get("round_records", []):
        round_index = int(record["round_idx"])
        if round_index not in activity_by_round:
            raise ValueError(
                f"Missing attack activity for qualitative round {round_index}"
            )
        activity = activity_by_round[round_index]
        record["attack_state"] = activity["attack_state"]
        record["is_fake_round"] = bool(activity["is_fake_round"])
        record["is_honest_round"] = bool(activity["is_honest_round"])
        record["active_attackers"] = list(activity["active_attackers"])
        record["ground_truth_labels"] = dict(activity["ground_truth_labels"])
        record["designated_attacker_labels"] = dict(
            activity["designated_attacker_labels"]
        )

        if score_evaluator is not None and "client_scores" in record:
            evaluation = score_evaluator(
                record["client_scores"],
                ground_truth_attackers=activity["active_attackers"],
            )
            record.update(
                {
                    "ground_truth_labels": evaluation["labels"],
                    "energy_ranks": evaluation["energy_ranks"],
                    "energy_percentiles": evaluation["energy_percentiles"],
                    "top_clients": evaluation["top_clients"],
                    "roc_auc": evaluation["roc_auc"],
                    "average_precision": evaluation["average_precision"],
                    "attacker_mean_rank": evaluation["attacker_mean_rank"],
                    "top1_hit": evaluation["top1_hit"],
                }
            )
            evaluated_records.append(record)

    activity_copy = [dict(activity) for activity in round_attack_activity]
    qualitative_result["round_attack_activity"] = activity_copy
    qualitative_result["active_attack_rounds"] = [
        int(activity["round_idx"])
        for activity in activity_copy
        if activity["is_fake_round"]
    ]
    qualitative_result["honest_attacker_rounds"] = [
        int(activity["round_idx"])
        for activity in activity_copy
        if activity["is_honest_round"]
    ]
    qualitative_result["ground_truth_semantics"] = "active_fake_submission"

    if score_evaluator is not None:
        def _metric_values(name: str) -> typing.List[float]:
            return [
                float(record[name])
                for record in evaluated_records
                if record.get(name) is not None
            ]

        auc_values = _metric_values("roc_auc")
        average_precision_values = _metric_values("average_precision")
        attacker_rank_values = _metric_values("attacker_mean_rank")
        top1_values = _metric_values("top1_hit")
        evaluation_summary = qualitative_result.setdefault("evaluation", {})
        evaluation_summary.update(
            {
                "mean_round_auc": (
                    float(np.mean(auc_values)) if auc_values else None
                ),
                "std_round_auc": (
                    float(np.std(auc_values)) if auc_values else None
                ),
                "mean_average_precision": (
                    float(np.mean(average_precision_values))
                    if average_precision_values
                    else None
                ),
                "mean_attacker_rank": (
                    float(np.mean(attacker_rank_values))
                    if attacker_rank_values
                    else None
                ),
                "top1_hit_rate": (
                    float(np.mean(top1_values)) if top1_values else None
                ),
                "valid_auc_rounds": len(auc_values),
                "ground_truth_attackers_by_round": {
                    int(activity["round_idx"]): list(activity["active_attackers"])
                    for activity in activity_copy
                },
                "ground_truth_semantics": "active_fake_submission",
            }
        )

    return qualitative_result


def _write_round_attack_activity_csv(
    output_path: typing.Union[str, Path],
    round_attack_activity: typing.Sequence[typing.Mapping[str, typing.Any]],
    metadata: typing.Mapping[str, typing.Any],
) -> None:
    """Persist realized round/client attack labels for qualitative evaluation."""
    metadata_fields = [
        "Dataset",
        "AttackMethod",
        "DetectingMethod",
        "QualitativeAggregationMode",
        "Alpha",
        "ClientNum",
        "AttackerID",
        "TrialID",
        "UseAttack",
        "UseLazy",
        "LazySchedule",
        "AttackProbability",
        "LazyHonestGap",
        "LazyHonestPhase",
        "LazyFakePolicy",
    ]
    activity_fields = [
        "Round",
        "ClientID",
        "IsDesignatedAttacker",
        "GroundTruthActiveAttack",
        "AttackState",
        "IsFakeRound",
        "IsHonestRound",
    ]
    missing_metadata = [field for field in metadata_fields if field not in metadata]
    if missing_metadata:
        raise ValueError(
            "Missing attack activity metadata fields: "
            + ", ".join(missing_metadata)
        )

    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metadata_fields + activity_fields,
        )
        if handle.tell() == 0:
            writer.writeheader()
        for activity in round_attack_activity:
            ground_truth_labels = activity["ground_truth_labels"]
            designated_labels = activity["designated_attacker_labels"]
            for client_id in sorted(ground_truth_labels):
                writer.writerow(
                    {
                        **{field: metadata[field] for field in metadata_fields},
                        "Round": int(activity["round_idx"]),
                        "ClientID": int(client_id),
                        "IsDesignatedAttacker": int(designated_labels[client_id]),
                        "GroundTruthActiveAttack": int(
                            ground_truth_labels[client_id]
                        ),
                        "AttackState": activity["attack_state"],
                        "IsFakeRound": int(bool(activity["is_fake_round"])),
                        "IsHonestRound": int(bool(activity["is_honest_round"])),
                    }
                )


def _build_qualitative_detection_rows(
    qualitative_result: typing.Mapping[str, typing.Any],
    metadata: typing.Mapping[str, typing.Any],
) -> typing.List[typing.Dict[str, typing.Any]]:
    """Build the uniform round-by-client qualitative detection artifact."""
    metadata_fields = [
        "TaskID",
        "CampaignComplete",
        "Dataset",
        "AttackMethod",
        "AttackRegime",
        "Alpha",
        "ClientNum",
        "AttackerID",
        "TrialID",
        "UseHessian",
        "UseLazy",
        "LazySchedule",
        "LazyHonestGap",
        "LazyHonestPhase",
        "LazyFakePolicy",
        "DetectingMethod",
        "QualitativeAggregationMode",
    ]
    missing_metadata = [field for field in metadata_fields if field not in metadata]
    if missing_metadata:
        raise ValueError(
            "Missing qualitative detection metadata fields: "
            + ", ".join(missing_metadata)
        )

    method = str(metadata["DetectingMethod"])
    is_wef_family = method in {"wef", "s2_wef"}
    use_hessian = bool(metadata["UseHessian"])
    lambda_h = metadata.get("LambdaH", "") if use_hessian else "N/A"
    final_suspicious_clients = {
        int(client_id)
        for client_id in qualitative_result.get("suspicious_clients", [])
    }
    decision_mode = qualitative_result.get("decision_mode", "")
    if method == "delta_dagmm":
        delta_config = qualitative_result.get("config", {})
        threshold_multiplier = (
            delta_config.get("threshold_multiplier", 1.08)
            if isinstance(delta_config, typing.Mapping)
            else 1.08
        )
        decision_mode = (
            "native_delta_dagmm:round_energy>"
            f"{float(threshold_multiplier):g}*round_mean_energy;"
            "client_detected_round_count>ceil(2R/3)"
        )

    def _value_for_client(
        values: typing.Any,
        client_id: int,
    ) -> typing.Any:
        if not isinstance(values, typing.Mapping):
            return ""
        return values.get(client_id, values.get(str(client_id), ""))

    def _detector_threshold(record: typing.Mapping[str, typing.Any]) -> typing.Any:
        if method in {"std_dagmm", "delta_dagmm"}:
            return record.get("energy_threshold", "")
        if method == "pass":
            return record.get("contribution_threshold", "")
        if method == "log_based_free_rider_detection":
            return record.get("a_gap", "")
        if method == "wef":
            return record.get("reputation_threshold", "")
        if method == "s2_wef":
            thresholds = {
                key: record[key]
                for key in ("gamma_threshold", "deviation_threshold")
                if record.get(key) is not None
            }
            return (
                json.dumps(thresholds, sort_keys=True, separators=(",", ":"))
                if thresholds
                else ""
            )
        return ""

    diagnostic_fields = {
        "pass": {
            "mean_divergences": "mean_incoming_divergence",
            "denominators": "contribution_denominator",
            "valid_by_client": "decision_valid",
            "invalid_reasons": "invalid_reason",
            "report_sources": "outgoing_report_source",
            "audit_sample_counts": "local_audit_sample_count",
        },
        "log_based_free_rider_detection": {
            "reported_accuracies": "reported_accuracy",
            "report_sources": "report_source",
            "complaints": "complaint",
            "copy_flags": "exact_copy",
            "audit_accuracies": "server_audit_accuracy",
            "cluster_labels": "cluster_label",
            "cosine_distances_to_global": "cosine_distance_to_global",
            "valid_by_client": "decision_valid",
            "invalid_reasons": "invalid_reason",
        },
        "std_dagmm": {
            "client_stds": "client_std",
            "energy_ranks": "energy_rank",
            "energy_percentiles": "energy_percentile",
        },
        "delta_dagmm": {"client_means": "client_mean"},
        "wef": {
            "euclidean_distances": "euclidean_distance",
            "cosine_similarities": "cosine_similarity",
            "average_frequencies": "average_frequency",
            "wef_matrix_sums": "wef_matrix_sum",
        },
        "s2_wef": {
            "gamma_scores": "gamma_score",
            "deviation_scores": "deviation_score",
            "cosine_to_simulated": "cosine_to_simulated",
            "l1_to_simulated": "l1_to_simulated",
            "robust_z_gamma": "robust_z_gamma",
            "robust_z_deviation": "robust_z_deviation",
            "cluster_labels": "cluster_label",
            "gamma_flags": "gamma_flag",
            "deviation_flags": "deviation_flag",
            "wef_matrix_sums": "wef_matrix_sum",
        },
    }

    rows = []
    for record in qualitative_result.get("round_records", []):
        ground_truth_labels = record.get("ground_truth_labels", {})
        designated_labels = record.get("designated_attacker_labels", {})
        client_scores = record.get("client_scores", {})
        round_suspicious_clients = {
            int(client_id)
            for client_id in record.get("suspicious_clients", [])
        }
        detection_skipped = bool(record.get("detection_skipped", False))
        round_has_active_attack = int(bool(record.get("active_attackers", [])))
        delta_ranks = {}
        delta_percentiles = {}
        if method == "delta_dagmm" and not detection_skipped:
            resolved_scores = {
                int(client_id): float(score)
                for client_id in ground_truth_labels
                for score in [_value_for_client(client_scores, int(client_id))]
                if score != "" and score is not None
            }
            score_values = list(resolved_scores.values())
            percentile_denominator = max(len(score_values) - 1, 1)
            for client_id, score in resolved_scores.items():
                greater = sum(value > score for value in score_values)
                equal_others = sum(value == score for value in score_values) - 1
                lower = sum(value < score for value in score_values)
                delta_ranks[client_id] = 1.0 + greater + 0.5 * equal_others
                delta_percentiles[client_id] = (
                    lower + 0.5 * equal_others
                ) / percentile_denominator

        mean_energy = record.get("mean_energy", "")
        delta_threshold_pathology = int(
            method == "delta_dagmm"
            and mean_energy != ""
            and mean_energy is not None
            and float(mean_energy) < 0.0
            and not bool(record.get("signed_energy_adjusted", False))
        )
        for client_id in sorted(ground_truth_labels):
            resolved_client_id = int(client_id)
            ground_truth_active_attack = int(ground_truth_labels[client_id])
            details = {}
            for source_field, detail_name in diagnostic_fields.get(method, {}).items():
                value = _value_for_client(record.get(source_field), resolved_client_id)
                if value != "" and value is not None:
                    details[detail_name] = value
            if method in {"std_dagmm", "delta_dagmm"}:
                for source_field in ("mean_energy", "signed_energy_adjusted"):
                    value = record.get(source_field, "")
                    if value != "" and value is not None:
                        details[source_field] = value
            if method == "delta_dagmm":
                if resolved_client_id in delta_ranks:
                    details["energy_rank"] = delta_ranks[resolved_client_id]
                    details["energy_percentile"] = delta_percentiles[
                        resolved_client_id
                    ]
                suspicious_count = _value_for_client(
                    qualitative_result.get("suspicious_counts", {}),
                    resolved_client_id,
                )
                details["client_suspicious_count"] = (
                    0 if suspicious_count == "" or suspicious_count is None
                    else suspicious_count
                )
                details["final_persistence_threshold"] = (
                    qualitative_result.get("decision_threshold")
                )
            if method == "s2_wef" and not detection_skipped:
                for source_field in (
                    "cluster_count",
                    "silhouette_score",
                    "merge_jump_ratio",
                    "tentative_suspicious_clients",
                    "gamma_vote",
                    "deviation_vote",
                    "simulated_wef_sum",
                ):
                    details[source_field] = record.get(
                        source_field,
                        [] if source_field == "tentative_suspicious_clients" else None,
                    )

            client_detection_skipped = detection_skipped or (
                method in {"pass", "log_based_free_rider_detection"}
                and not _value_for_client(record.get("valid_by_client"), resolved_client_id)
            )
            if method == "log_based_free_rider_detection":
                final_detection_skipped = not _value_for_client(
                    qualitative_result.get("final_valid_by_client"), resolved_client_id,
                )
            else:
                final_detection_skipped = client_detection_skipped or (
                    method == "pass"
                    and not _value_for_client(qualitative_result.get("final_valid_by_client"), resolved_client_id)
                )
            if method == "pass":
                for field in ("submitted_round", "audit_round", "decision_round"):
                    details[field] = record.get(field)
                details["final_audited_submission_round"] = qualitative_result.get("final_audited_submission_round")
            elif method == "log_based_free_rider_detection":
                for field in ("a_gap", "audit_triggered", "audit_reason"):
                    details[field] = record.get(field)

            row = {
                **{field: metadata[field] for field in metadata_fields},
                "LambdaH": lambda_h,
                "Round": int(record["round_idx"]),
                "ClientID": resolved_client_id,
                "IsDesignatedAttacker": int(
                    _value_for_client(designated_labels, resolved_client_id) or 0
                ),
                "GroundTruthActiveAttack": ground_truth_active_attack,
                "DesignatedAttackerRoundState": record.get("attack_state", ""),
                "RoundHasActiveAttack": round_has_active_attack,
                "RoundIsHonestForDesignatedAttacker": int(
                    bool(record.get("is_honest_round", False))
                ),
                "WEFPayloadSource": (
                    "counterfeit_from_submitted_update"
                    if is_wef_family
                    and ground_truth_active_attack
                    and bool(record.get("is_fake_round", False))
                    and metadata["AttackMethod"] != "synthetic_replacement"
                    else "observed_local_training"
                    if is_wef_family
                    else "N/A"
                ),
                "DetectionSkipped": int(client_detection_skipped),
                "DetectorScore": (
                    ""
                    if client_detection_skipped
                    else _value_for_client(client_scores, resolved_client_id)
                ),
                "DetectorThreshold": "" if client_detection_skipped else _detector_threshold(record),
                "DeltaThresholdPathology": delta_threshold_pathology,
                "RoundPredictedSuspicious": (
                    ""
                    if client_detection_skipped
                    else int(resolved_client_id in round_suspicious_clients)
                ),
                "FinalPredictedSuspicious": (
                    ""
                    if final_detection_skipped
                    else int(resolved_client_id in final_suspicious_clients)
                ),
                "DecisionMode": decision_mode,
                "DetectorDetailsJSON": json.dumps(
                    details,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
            rows.append(row)
    return rows


def _write_qualitative_detection_csv(
    output_path: typing.Union[str, Path],
    qualitative_result: typing.Mapping[str, typing.Any],
    metadata: typing.Mapping[str, typing.Any],
) -> None:
    """Write the new per-task uniform qualitative detection CSV once."""
    fieldnames = [
        "TaskID", "CampaignComplete", "Dataset", "AttackMethod", "AttackRegime",
        "Alpha", "ClientNum", "AttackerID", "TrialID", "UseHessian", "LambdaH",
        "UseLazy", "LazySchedule", "LazyHonestGap", "LazyHonestPhase",
        "LazyFakePolicy", "DetectingMethod", "QualitativeAggregationMode", "Round",
        "ClientID", "IsDesignatedAttacker", "GroundTruthActiveAttack",
        "DesignatedAttackerRoundState", "RoundHasActiveAttack",
        "RoundIsHonestForDesignatedAttacker", "WEFPayloadSource", "DetectionSkipped",
        "DetectorScore", "DetectorThreshold", "DeltaThresholdPathology",
        "RoundPredictedSuspicious",
        "FinalPredictedSuspicious", "DecisionMode", "DetectorDetailsJSON",
    ]
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_build_qualitative_detection_rows(qualitative_result, metadata))


def _minmax_normalize_rows(contribs, eps=1e-12):
    if contribs is None or len(contribs) == 0:
        return contribs
    arr = np.asarray(contribs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_min = arr.min(axis=1, keepdims=True)
    row_max = arr.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, eps)
    return ((arr - row_min) / denom).tolist()


def _build_qualitative_time_row(
    metadata: typing.Mapping[str, typing.Any],
    *,
    use_attack: bool,
    attack_time_log_with_attacker: typing.Sequence[float],
    round_attack_activity_with: typing.Sequence[typing.Mapping[str, typing.Any]],
    qualitative_result_with: typing.Optional[typing.Mapping[str, typing.Any]],
    qualitative_result_without: typing.Optional[typing.Mapping[str, typing.Any]],
    fl_with_attacker_wall_sec: typing.Any,
    fl_without_attacker_wall_sec: typing.Any,
    task_wall_sec: float,
) -> typing.Dict[str, typing.Any]:
    """Build the one-row qualitative timing artifact from realized execution state."""
    detecting_method = metadata["DetectingMethod"]
    if use_attack:
        attack_generation_total_sec = sum(attack_time_log_with_attacker)
        fake_rounds_timed = sum(
            int(activity.get("is_fake_round", False))
            for activity in round_attack_activity_with
        )
        detector_with_attacker_total_sec = (
            qualitative_result_with.get("detector_time_sec", "")
            if qualitative_result_with is not None
            else ""
        )
        if detecting_method in {"wef", "s2_wef", "pass", "log_based_free_rider_detection"}:
            detector_without_attacker_total_sec = (
                qualitative_result_without.get("detector_time_sec", "")
                if qualitative_result_without is not None
                else ""
            )
            detector_without_attacker_measured = 1
            detector_without_attacker_not_measured_reason = ""
        else:
            detector_without_attacker_total_sec = ""
            detector_without_attacker_measured = 0
            detector_without_attacker_not_measured_reason = (
                "posthoc_detector_not_run_for_internal_n_minus_1_control"
            )
    else:
        attack_generation_total_sec = ""
        fake_rounds_timed = 0
        detector_with_attacker_total_sec = ""
        detector_without_attacker_total_sec = (
            qualitative_result_without.get("detector_time_sec", "")
            if qualitative_result_without is not None
            else ""
        )
        detector_without_attacker_measured = 1
        detector_without_attacker_not_measured_reason = ""

    return {
        "TaskID": metadata["TaskID"],
        "CampaignComplete": metadata["CampaignComplete"],
        "Dataset": metadata["Dataset"],
        "AttackMethod": metadata["AttackMethod"],
        "AttackRegime": metadata["AttackRegime"],
        "Alpha": metadata["Alpha"],
        "ClientNum": metadata["ClientNum"],
        "AttackerID": metadata["AttackerID"],
        "TrialID": metadata["TrialID"],
        "UseHessian": metadata["UseHessian"],
        "LambdaH": metadata["LambdaH"],
        "UseLazy": metadata["UseLazy"],
        "LazySchedule": metadata["LazySchedule"],
        "LazyHonestGap": metadata["LazyHonestGap"],
        "LazyHonestPhase": metadata["LazyHonestPhase"],
        "LazyFakePolicy": metadata["LazyFakePolicy"],
        "DetectingMethod": detecting_method,
        "QualitativeAggregationMode": metadata["QualitativeAggregationMode"],
        "AttackGenerationWithAttackerTotalSec": attack_generation_total_sec,
        "FakeRoundsTimed": fake_rounds_timed,
        "DetectorDecisionWithAttackerTotalSec": detector_with_attacker_total_sec,
        "DetectorDecisionWithoutAttackerTotalSec": (
            detector_without_attacker_total_sec
        ),
        "DetectorDecisionWithoutAttackerMeasured": (
            detector_without_attacker_measured
        ),
        "DetectorDecisionWithoutAttackerNotMeasuredReason": (
            detector_without_attacker_not_measured_reason
        ),
        "FLWithAttackerWallSec": fl_with_attacker_wall_sec,
        "FLWithoutAttackerWallSec": fl_without_attacker_wall_sec,
        "TaskWallSec": task_wall_sec,
    }


def _timed_qualitative_call(callable_, device=None):
    """Measure a qualitative detector operation, synchronizing CUDA when needed."""
    should_synchronize = bool(
        device is not None
        and torch.cuda.is_available()
        and torch.device(device).type == "cuda"
    )
    if should_synchronize:
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    result = callable_()
    if should_synchronize:
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - started_at


def _run_qualitative_detection(
    round_client_gradients,
    round_wef_matrices=None,
    round_global_gradients=None,
    ground_truth_attackers=None,
):
    common_kwargs = dict(
        round_client_gradients=round_client_gradients,
        warmup_rounds=0,
        persistence_ratio=0.5,
        min_suspicious_rounds=2,
    )
    if detecting_method == "eds":
        return QualitativeDetector.detect_free_riders_from_history(
            dynamic_weighting=False,
            **common_kwargs,
        )
    if detecting_method == "std_dagmm":
        from module.method.qualitative_detecting.std_dagmm import STDDAGMMConfig

        config = STDDAGMMConfig(
            latent_dim=args.std_dagmm_latent_dim,
            n_components=args.std_dagmm_components,
            epochs=args.std_dagmm_epochs,
            learning_rate=args.std_dagmm_lr,
            lambda_energy=args.std_dagmm_lambda_energy,
            lambda_cov_diag=args.std_dagmm_lambda_cov_diag,
            covariance_eps=args.std_dagmm_covariance_eps,
            normalize_input=args.std_dagmm_normalize_input,
            min_cluster_separation=args.std_dagmm_min_cluster_separation,
            energy_mad_scale=args.std_dagmm_energy_mad_scale,
            random_state=args.std_dagmm_seed,
        )
        return QualitativeDetector.detect_free_riders_from_history(
            config=config,
            device=device,
            ground_truth_attackers=ground_truth_attackers,
            **common_kwargs,
        )
    if detecting_method == "delta_dagmm":
        from module.method.qualitative_detecting.delta_dagmm import DeltaDAGMMConfig

        config = DeltaDAGMMConfig(
            latent_dim=args.delta_dagmm_latent_dim,
            n_components=args.delta_dagmm_components,
            epochs=args.delta_dagmm_epochs,
            learning_rate=args.delta_dagmm_lr,
            lambda_energy=args.delta_dagmm_lambda_energy,
            lambda_cov_diag=args.delta_dagmm_lambda_cov_diag,
            covariance_eps=args.delta_dagmm_covariance_eps,
            linear_scale=args.delta_dagmm_linear_scale,
            linear_offset=args.delta_dagmm_linear_offset,
            output_layer_only=args.delta_dagmm_output_layer_only,
            normalize_input=args.delta_dagmm_normalize_input,
            threshold_multiplier=args.delta_dagmm_threshold_multiplier,
            signed_energy_safe_threshold=args.delta_dagmm_signed_energy_safe_threshold,
            random_state=args.delta_dagmm_seed,
        )
        return QualitativeDetector.detect_free_riders_from_history(
            round_client_gradients=round_client_gradients,
            config=config,
            device=device,
            warmup_rounds=0,
            persistence_ratio=2.0 / 3.0,
            min_suspicious_rounds=1,
        )
    if detecting_method == "wef":
        from module.method.qualitative_detecting.wef import WEFConfig

        if round_wef_matrices is None:
            raise ValueError("WEF detection requires local-epoch WEF-Matrix history")
        config = WEFConfig(
            reputation_epsilon=args.wef_epsilon,
            weight_from_end=args.wef_weight_from_end,
        )
        return QualitativeDetector.detect_free_riders_from_history(
            round_wef_matrices=round_wef_matrices,
            config=config,
            warmup_rounds=0,
        )
    if detecting_method == "s2_wef":
        from module.method.qualitative_detecting.s2_wef import S2WEFConfig

        if round_wef_matrices is None or round_global_gradients is None:
            raise ValueError(
                "S2-WEF detection requires round-local WEF and global-gradient histories"
            )
        config = S2WEFConfig(
            weight_from_end=args.s2_wef_weight_from_end,
            gamma_multiplier=args.s2_wef_gamma_multiplier,
            deviation_epsilon=args.s2_wef_deviation_epsilon,
            silhouette_threshold=args.s2_wef_silhouette_threshold,
            merge_jump_threshold=args.s2_wef_merge_jump_threshold,
        )
        return QualitativeDetector.detect_free_riders_from_history(
            round_wef_matrices=round_wef_matrices,
            round_global_gradients=round_global_gradients,
            local_iterations=num_local_epochs,
            config=config,
        )
    raise ValueError(f"Unsupported qualitative detecting method: {detecting_method}")


def _detect_qualitative_round_online(
    round_index,
    round_wef_matrices,
    previous_global_difference=None,
    local_iterations=1,
):
    """Run one WEF-family decision before the current round is aggregated."""
    if detecting_method == "wef":
        from module.method.qualitative_detecting.wef import WEFConfig, WEFDetector

        config = WEFConfig(
            reputation_epsilon=args.wef_epsilon,
            weight_from_end=args.wef_weight_from_end,
        )
        score_result = WEFDetector.compute_round_scores(
            {client_id: matrix for client_id, matrix in enumerate(round_wef_matrices)},
            config=config,
        )
        record = {
            "round_idx": round_index,
            "client_scores": score_result["client_scores"],
            "euclidean_distances": score_result["euclidean_distances"],
            "cosine_similarities": score_result["cosine_similarities"],
            "average_frequencies": score_result["average_frequencies"],
            "wef_matrix_sums": score_result["wef_matrix_sums"],
            "suspicious_clients": score_result["suspicious_clients"],
            "reputation_threshold": score_result["reputation_threshold"],
            "weights": score_result["config"],
            "detection_skipped": False,
        }
        return record, score_result["reputation_threshold"]

    if detecting_method == "s2_wef":
        from module.method.qualitative_detecting.s2_wef import (
            S2WEFConfig,
            S2WEFDetector,
        )

        config = S2WEFConfig(
            weight_from_end=args.s2_wef_weight_from_end,
            gamma_multiplier=args.s2_wef_gamma_multiplier,
            deviation_epsilon=args.s2_wef_deviation_epsilon,
            silhouette_threshold=args.s2_wef_silhouette_threshold,
            merge_jump_threshold=args.s2_wef_merge_jump_threshold,
        )
        if previous_global_difference is None:
            record = {
                "round_idx": round_index,
                "client_scores": {},
                "gamma_scores": {},
                "deviation_scores": {},
                "suspicious_clients": [],
                "cluster_count": 1,
                "silhouette_score": None,
                "merge_jump_ratio": None,
                "gamma_vote": 0.0,
                "deviation_vote": 0.0,
                "simulated_wef_sum": 0.0,
                "detection_skipped": True,
                "config": vars(config),
                "weights": vars(config),
            }
            return record, None

        simulated = S2WEFDetector.simulate_wef_matrix(
            previous_global_difference,
            round_wef_matrices[0].shape,
            local_iterations,
            config,
        )
        record = S2WEFDetector.detect_round(
            {client_id: matrix for client_id, matrix in enumerate(round_wef_matrices)},
            simulated,
            config,
        )
        record["round_idx"] = round_index
        record["detection_skipped"] = False
        return record, None

    raise ValueError(
        "Online qualitative rounds are only supported for WEF and S2-WEF"
    )


def _fit_local_model_with_optional_wef(
    model,
    X_train,
    y_train,
    num_local_epochs,
    wef_tracker=None,
):
    """Run the unchanged local optimizer with optional per-epoch WEF sampling."""
    previous_callback = getattr(model, "_epoch_end_callback", None)
    had_callback = hasattr(model, "_epoch_end_callback")
    if wef_tracker is not None:
        model._epoch_end_callback = (
            lambda trained_model, _epoch: wef_tracker.observe_model(trained_model)
        )
    try:
        model.fit(
            X_train,
            y_train,
            incremental=True,
            num_epochs=num_local_epochs,
        )
    finally:
        if wef_tracker is not None:
            if had_callback:
                model._epoch_end_callback = previous_callback
            else:
                delattr(model, "_epoch_end_callback")



def run_federated_learning(loader, model, attacker_id, attacker, use_attack,
                         num_epoch, num_local_epochs, device, record_gradients=False,
                         collect_history=False, public_data=None,
                         realized_attack_rounds_out=None, last_round_aggregation_out=None,
                         single_attacker_audit_out=None):
    aggregation_config = AggregationConfig.from_args(args)
    server_aggregator = ServerAggregator(aggregation_config)
    if last_round_aggregation_out is not None:
        last_round_aggregation_out.clear()
    # The dedicated path supplies independent attack instances keyed by client ID.
    multi_attackers = attacker if isinstance(attacker, dict) else None
    # Snapshot used by windowed-SV: T-R_FOR_SV checkpoint
    state_t_minus_R = None  # model state right before the last R rounds
    num_parts = len(loader.X_train_parts)
    if single_attacker_audit_out is not None:
        if multi_attackers is not None or not use_attack or attacker_id is None:
            raise ValueError('The RQ1 audit requires the existing singular attacker path')
        single_attacker_audit_out.clear()
        single_attacker_audit_out.update(
            private_fit_calls_by_client={str(i): 0 for i in range(num_parts)},
            fake_call_rounds=[], fake_sources=[], round_activity=[], public_warm_start_rounds=[],
        )
    wang_public_data = None
    if use_attack and args.attack_method == "adam_local_deviation":
        if public_data is None:
            raise ValueError(
                "adam_local_deviation requires the disjoint simulated public pool "
                "prepared by FLCE_updated before client partitioning"
            )
        public_X, public_y = public_data
        if len(public_X) != len(public_y) or len(public_y) == 0:
            raise ValueError("Invalid simulated public pool: X/y must be non-empty and aligned")
        wang_public_data = (public_X, public_y)
    client_models = [copy.deepcopy(model) for _ in range(num_parts)]
    wef_trackers = None
    if detecting_method in {"wef", "s2_wef"}:
        from module.method.qualitative_detecting.wef import WEFClientTracker

        tracker_weight_from_end = (
            args.s2_wef_weight_from_end
            if detecting_method == "s2_wef"
            else args.wef_weight_from_end
        )
        wef_trackers = [
            WEFClientTracker(weight_from_end=tracker_weight_from_end)
            for _ in range(num_parts)
        ]
        for tracker, client_model in zip(wef_trackers, client_models):
            tracker.initialize_from_model(client_model)
    # Keep legacy FedAvg arithmetic; higher-precision shares preserve half-mass
    # ties in the weighted median when client sample counts differ.
    weight_dtype = torch.float if aggregation_config.method == "fedavg" else torch.float64
    shard_sizes = torch.tensor([len(loader.X_train_parts[i]) for i in range(num_parts)], dtype=weight_dtype, device=device)
    weights = (shard_sizes / shard_sizes.sum())
    pass_observer = None
    if collect_history and detecting_method == "pass":
        pass_observer, detector_elapsed = _timed_qualitative_call(
            lambda: PASSAuditObserver(
                model, loader.X_train_parts, loader.y_train_parts,
                config=PASSConfig.from_args(args),
                forged_auditors=resolve_forged_auditors(
                    use_attack=use_attack, attacker_id=attacker_id,
                    attack_method=args.attack_method, use_lazy=args.use_lazy,
                ),
                resource_notes=attack_resource_notes(args.attack_method, use_attack, args.use_lazy),
            ),
            device=device,
        )
        pass_observer.detector_time_sec += detector_elapsed
    logfr_observer = None
    if collect_history and detecting_method == "log_based_free_rider_detection":
        logfr_observer, detector_elapsed = _timed_qualitative_call(
            lambda: LogBasedFreeRiderObserver(
                model, loader.X_val_parts, loader.y_val_parts,
                loader.X_test, loader.y_test,
                LogBasedFreeRiderConfig.from_args(args, seed=seed, trial_id=trial_id),
            ),
            device=device,
        )
        logfr_observer.detector_time_sec += detector_elapsed
    attack_time_log = []
    # [PATCH] Last-round SV caches
    state_t_minus_1 = None
    last_round_client_grads = None
    last_round_weights = None
    global_accuracies = []
    
    # only initialize when EDS is needed 
    round_client_gradients = []  # for EDS: List[List[torch.Tensor]]
    round_global_gradients = []  # for EDS: List[List[torch.Tensor]]
    round_wef_matrices = []  # for WEF: List[List[Tensor]]
    online_qualitative_enabled = bool(
        collect_history
        and detecting_type == "qualitative"
        and detecting_method in {"wef", "s2_wef"}
    )
    online_qualitative_aggregation_enabled = (
        _should_use_online_qualitative_aggregation(
            collect_history,
            detecting_type,
            detecting_method,
            args.qualitative_aggregation_mode,
        )
    )
    online_round_records = []
    online_suspicious_counts = {}
    online_decision_threshold = None
    online_detector_time_sec = 0.0
    previous_global_difference = None
    attack_global_gradient_history = []  # for ARIMA/Moirai: List[np.ndarray]
    lazy_rng_seed = seed + int(trial_id) * 100003 + int(attacker_id or 0) * 9973
    probabilistic_fake_rounds = frozenset()
    probabilistic_fake_budget = 0
    round_attack_activity = []
    if use_attack and args.use_lazy and args.lazy_schedule == "probabilistic":
        eligible_fake_rounds = np.arange(int(num_epoch), dtype=int)
        if args.attack_method == "adam_local_deviation":
            # Round 0 is reserved for Wang's mandatory public warm start.
            eligible_fake_rounds = eligible_fake_rounds[eligible_fake_rounds != 0]
        probabilistic_fake_rounds, probabilistic_fake_budget = (
            sample_probabilistic_fake_rounds(
                eligible_rounds=eligible_fake_rounds,
                num_rounds=int(num_epoch),
                attack_probability=float(args.attack_probability),
                seed=lazy_rng_seed,
            )
        )
        print(
            "[PROB_LAZY] "
            f"seed={lazy_rng_seed} target_probability={args.attack_probability} "
            f"fake_budget={probabilistic_fake_budget}/{len(eligible_fake_rounds)} "
            f"fake_rounds={sorted(probabilistic_fake_rounds)}",
            flush=True,
        )
    publish_realized_fake_rounds(
        realized_attack_rounds_out,
        probabilistic_fake_rounds,
    )
    if logfr_observer is not None and use_attack and args.use_lazy:
        first_round_activity = _resolve_round_attack_activity(
            0,
            client_ids=range(num_parts), use_attack=use_attack,
            attacker_id=attacker_id, use_lazy=True,
            lazy_schedule=args.lazy_schedule,
            lazy_honest_gap=args.lazy_honest_gap,
            lazy_honest_phase=args.lazy_honest_phase,
            probabilistic_fake_rounds=probabilistic_fake_rounds,
            public_warm_start=(args.attack_method == "adam_local_deviation"),
        )
        if first_round_activity["is_fake_round"]:
            raise ValueError(
                "Log-based lazy fake reports require an earlier honest report; "
                "the realized lazy schedule starts with a fake round."
            )

    for round_num in range(num_epoch):
            # Take snapshot right before entering the last R_FOR_SV rounds
        trig = (num_epoch - R_FOR_SV) - 1
        if num_epoch >= R_FOR_SV and round_num == trig:
            state_t_minus_R = copy.deepcopy(model.state_dict())
        current_round_client_grads = []
        round_client_base_parameters = []
        current_round_attack_activity = _resolve_round_attack_activity(
            round_num,
            client_ids=range(num_parts),
            use_attack=use_attack,
            attacker_id=attacker_id,
            attacker_ids=list(multi_attackers) if multi_attackers is not None else None,
            use_lazy=args.use_lazy,
            lazy_schedule=args.lazy_schedule,
            lazy_honest_gap=args.lazy_honest_gap,
            lazy_honest_phase=args.lazy_honest_phase,
            probabilistic_fake_rounds=probabilistic_fake_rounds,
            public_warm_start=bool(
                use_attack
                and attacker_id is not None
                and args.attack_method == "adam_local_deviation"
                and round_num == 0
            ),
        )
        

        if pass_observer is not None:
            _, detector_elapsed = _timed_qualitative_call(
                lambda: pass_observer.start_round(round_num, model), device=device,
            )
            pass_observer.detector_time_sec += detector_elapsed

        for i in range(num_parts):
            X_i, y_i = loader.X_train_parts[i], loader.y_train_parts[i]
            model_i = client_models[i].to(device)
            backup = copy.deepcopy(model_i)
            round_client_base_parameters.append(
                [parameter.detach().clone() for parameter in backup.parameters()]
            )
            if wef_trackers is not None:
                # S2-WEF is round-local; original WEF additionally keeps cumulative counts.
                wef_trackers[i].begin_round(model_i)

            is_attack_client = bool(use_attack and (
                i in multi_attackers if multi_attackers is not None else i == attacker_id
            ))
            client_attacker = multi_attackers.get(i) if multi_attackers is not None else attacker
            is_wang_attack = bool(
                is_attack_client and args.attack_method == "adam_local_deviation"
            )
            wang_public_warm_start = bool(is_wang_attack and round_num == 0)
            lazy_attack_selected = (
                current_round_attack_activity["lazy_attack_selected"]
                if is_attack_client
                else None
            )
            is_honest_round = bool(
                is_attack_client
                and current_round_attack_activity["is_honest_round"]
            )

            dkfr_fake_round = bool(is_attack_client and args.attack_method == "dkfr" and not is_honest_round)
            synthetic_fake_round = bool(
                is_attack_client
                and args.attack_method == "synthetic_replacement"
                and not is_honest_round
            )
            probabilistic_fake_round = bool(
                is_attack_client
                and args.use_lazy
                and args.lazy_schedule == "probabilistic"
                and not is_honest_round
                and not wang_public_warm_start
            )
            synthetic_replacement_time = None
            forecast_fake_round = bool(
                is_attack_client
                and (args.attack_method in FORECAST_FREE_RIDER_METHODS
                     or (getattr(args, 'rq1_strict_data_free', False)
                         and args.attack_method == 'advanced_scaled_delta'))
                and not is_honest_round
            )
            wang_fake_round = bool(
                is_wang_attack
                and not is_honest_round
                and not wang_public_warm_start
            )
            collect_wef_for_training = bool(
                wef_trackers is not None
                and (
                    not is_attack_client
                    or is_honest_round
                    or synthetic_fake_round
                    or wang_public_warm_start
                )
            )
            wef_tracker = (
                wef_trackers[i] if collect_wef_for_training else None
            )

            # Public warm start is the only Wang round that trains a model.  It
            # uses the disjoint pool created before client partitioning, never
            # the attacker's private shard or the evaluation set.
            if wang_public_warm_start:
                public_X, public_y = wang_public_data
                _fit_local_model_with_optional_wef(
                    model_i,
                    public_X,
                    public_y,
                    num_local_epochs,
                    wef_tracker=wef_tracker,
                )
                true_gradient = [
                    new.data - old.data
                    for new, old in zip(model_i.parameters(), backup.parameters())
                ]
            # Forecast/replay fake rounds never touch the attacker's private
            # shard. Their submitted update is generated entirely from public
            # broadcast history (or the configured fallback).
            elif forecast_fake_round:
                true_gradient = [
                    torch.zeros_like(param, device=device)
                    for param in model_i.parameters()
                ]
            # DKFR fake rounds are data-free: do not train on the attacker's real local shard.
            elif dkfr_fake_round:
                true_gradient = [torch.zeros_like(param, device=device) for param in model_i.parameters()]
            elif synthetic_fake_round:
                t0 = time.time()
                X_syn, y_syn = client_attacker.get_synthetic_data(
                    i,
                    X_i.to(device),
                    y_i.to(device),
                )
                _fit_local_model_with_optional_wef(
                    model_i,
                    X_syn,
                    y_syn,
                    num_local_epochs,
                    wef_tracker=wef_tracker,
                )
                true_gradient = [new.data - old.data for new, old in zip(model_i.parameters(), backup.parameters())]
                synthetic_replacement_time = time.time() - t0
            elif probabilistic_fake_round or wang_fake_round:
                true_gradient = [torch.zeros_like(param, device=device) for param in model_i.parameters()]
            else:
                _fit_local_model_with_optional_wef(
                    model_i,
                    X_i,
                    y_i,
                    num_local_epochs,
                    wef_tracker=wef_tracker,
                )
                if single_attacker_audit_out is not None:
                    single_attacker_audit_out['private_fit_calls_by_client'][str(i)] += 1
                true_gradient = [new.data - old.data for new, old in zip(model_i.parameters(), backup.parameters())]
            
            if is_attack_client:
                # Q2 (use_lazy=False): attacker always fakes.
                # Q3 (use_lazy=True): honest only on sparse rounds (1 honest / N fake).
                # log
                print(
                    f"[ROUND {round_num}] client={i} honest={bool(is_honest_round)} "
                    f"method={args.attack_method} "
                    f"hessian={'on' if args.use_hessian else 'off'} "
                    f"lambda_h={args.lambda_h if args.use_hessian else 'n/a'} "
                    f"lazy={'on' if args.use_lazy else 'off'} "
                    f"schedule={args.lazy_schedule if args.use_lazy else 'n/a'} "
                    f"attack_probability={args.attack_probability if args.use_lazy and args.lazy_schedule == 'probabilistic' else 'n/a'} "
                    f"fake_budget={probabilistic_fake_budget if args.use_lazy and args.lazy_schedule == 'probabilistic' else 'n/a'} "
                    f"selected_fake_round={lazy_attack_selected if lazy_attack_selected is not None else 'n/a'} "
                    f"private_training={'skipped' if forecast_fake_round else 'used_or_not_applicable'} "
                    f"public_warm_start={wang_public_warm_start} "
                    f"(honest_gap={args.lazy_honest_gap if args.use_lazy else 'n/a'}, "
                    f"honest_phase={args.lazy_honest_phase if args.use_lazy else 'n/a'}, "
                    f"fake_policy={args.lazy_fake_policy if args.use_lazy else 'n/a'})",
                    flush=True
                )

                if wang_public_warm_start:
                    t0 = time.time()
                    gradient = attacker.initialize_public_warm_start(backup, model_i)
                    if single_attacker_audit_out is not None:
                        single_attacker_audit_out['public_warm_start_rounds'].append(round_num)
                    attack_time_log.append(time.time() - t0)
                    print(
                        "[WANG_LOCAL_DEVIATION] round=0 submit public-warm-start delta "
                        "(in-domain held-out public pool)",
                        flush=True,
                    )
                elif is_honest_round:
                    gradient = true_gradient
                    print(f"[LAZY] round={round_num} client={i} submit honest", flush=True)
                else:
                    if args.use_lazy and args.lazy_fake_policy == 'zero':
                        gradient = [torch.zeros_like(p, device=p.device) for p in model.parameters()]
                        print(f"[LAZY] round={round_num} client={i} submit zero", flush=True)
                    elif args.attack_method == "adam_local_deviation":
                        t0 = time.time()
                        # ``model_i`` is the exact parameter base used by the
                        # unchanged online WEF/S2-WEF reconstruction path.  In
                        # ordinary FedAvg it equals the received global model.
                        gradient = client_attacker.get_fake_gradient(round_num, device, model_i)
                        attack_time_log.append(time.time() - t0)
                        hist_len = len(getattr(client_attacker, "global_gradient_history", []))
                        if single_attacker_audit_out is not None:
                            single_attacker_audit_out['fake_call_rounds'].append(round_num)
                            single_attacker_audit_out['fake_sources'].append('wang_local_parameter_deviation')
                        print(
                            f"[ATTACK] round={round_num} client={i} "
                            f"source=wang_local_parameter_deviation history_len={hist_len}",
                            flush=True,
                        )
                    elif args.attack_method == "dkfr":
                        t0 = time.time()
                        gradient = client_attacker.get_fake_gradient(round_num, device, model)
                        attack_time_log.append(time.time() - t0)
                        hist_len = len(getattr(client_attacker, "global_gradient_history", []))
                        if single_attacker_audit_out is not None:
                            single_attacker_audit_out['fake_call_rounds'].append(round_num)
                            single_attacker_audit_out['fake_sources'].append('data_free_kd')
                        print(f"[ATTACK] round={round_num} client={i} source=data_free_kd history_len={hist_len}", flush=True)
                    elif args.attack_method == "synthetic_replacement":
                        gradient = true_gradient
                        if synthetic_replacement_time is not None:
                            attack_time_log.append(synthetic_replacement_time)
                        hist_len = len(getattr(client_attacker, "global_gradient_history", []))
                        print(
                            f"[ATTACK] round={round_num} client={i} "
                            f"source=synthetic_replacement_train history_len={hist_len}",
                            flush=True,
                        )
                    else:
                        t0 = time.time()
                        gradient = client_attacker.get_fake_gradient(round_num, device, model_i)
                        attack_time_log.append(time.time() - t0)
                        if single_attacker_audit_out is not None:
                            single_attacker_audit_out['fake_call_rounds'].append(round_num)
                            single_attacker_audit_out['fake_sources'].append(
                                str(getattr(client_attacker, 'last_source', 'unspecified'))
                            )

                        # Prefer the attacker's runtime source marker when it
                        # exposes one (for example, to distinguish a predictor
                        # result from an internal fallback).
                        reported_source = getattr(client_attacker, "last_source", None)
                        if reported_source is not None:
                            src = str(reported_source)
                        elif args.attack_method == "random":
                            src = "lin_uniform"
                        elif args.attack_method == "fraboni_disguised":
                            src = "fraboni_gaussian"
                        elif args.attack_method == "replay":
                            src = "history_replay"
                        elif args.attack_method in [
                            "arima",
                            "arima_adaptive",
                            "scaled_delta",
                            "advanced_scaled_delta",
                            "moirai",
                            "moirai_online",
                            "arima_mnist_imdb_layerwise",
                            "arima_adaptive_mnist_imdb_layerwise",
                            "moirai_mnist_imdb_layerwise",
                            "moirai_online_mnist_imdb_layerwise",
                        ]:
                            src = "predict"
                        else:
                            src = "avg"
                        hist_len = len(getattr(client_attacker, "global_gradient_history", []))
                        print(f"[ATTACK] round={round_num} client={i} source={src} history_len={hist_len}", flush=True)
            else:
                # non-attacking period / no attacking：submit real gradient
                gradient = true_gradient

            # after choosing `gradient` (true or fake)
            if (
                detecting_method in {"wef", "s2_wef"}
                and wef_trackers is not None
                and is_attack_client
                and not collect_wef_for_training
            ):
                # Eq. (3): one-step counterfeit WEF scaled to the local epoch count.
                # Sequential AWCA trajectories are not exposed by the current attack API.
                wef_trackers[i].set_round_counterfeit_from_update(
                    gradient,
                    num_local_epochs,
                    accumulate=(detecting_method == "wef"),
                )
            _assert_submitted_update_contract(
                gradient,
                model,
                device,
                tag=f"round={round_num}/client={i}/submit",
            )
            _assert_all_grads_on_device([gradient], device, tag=f"round={round_num}/client={i}/submit")
            current_round_client_grads.append(gradient)




        if pass_observer is not None:
            _, detector_elapsed = _timed_qualitative_call(
                lambda: pass_observer.capture_submissions(
                    round_num, round_client_base_parameters, current_round_client_grads, model,
                ),
                device=device,
            )
            pass_observer.detector_time_sec += detector_elapsed
        if logfr_observer is not None:
            _, detector_elapsed = _timed_qualitative_call(
                lambda: logfr_observer.capture_submissions(
                    round_num, round_client_base_parameters,
                    current_round_client_grads, model,
                    activity=current_round_attack_activity,
                    attacker_id=attacker_id, use_attack=use_attack,
                    use_lazy=args.use_lazy, attack_method=args.attack_method,
                ),
                device=device,
            )
            logfr_observer.detector_time_sec += detector_elapsed

        round_attack_activity.append(current_round_attack_activity)
        if single_attacker_audit_out is not None:
            single_attacker_audit_out['round_activity'].append(dict(
                round_idx=round_num,
                active_attackers=list(current_round_attack_activity['active_attackers']),
            ))
        _assert_all_grads_on_device(current_round_client_grads, device, tag=f"round={round_num}/before_agg")
        _assert_model_params_on_device(model, device, tag=f"round={round_num}/before_agg")
        _assert_weights_on_device(weights, device, tag=f"round={round_num}/before_agg")

        current_round_wef_matrices = None
        online_group_model_parameters = None
        online_record = None
        if online_qualitative_enabled:
            def _snapshot_and_detect_online_round():
                round_wef_snapshot = [
                    tracker.round_snapshot()
                    if detecting_method == "s2_wef"
                    else tracker.snapshot()
                    for tracker in wef_trackers
                ]
                return round_wef_snapshot, _detect_qualitative_round_online(
                    round_num,
                    round_wef_snapshot,
                    previous_global_difference=previous_global_difference,
                    local_iterations=num_local_epochs,
                )

            (
                (current_round_wef_matrices, online_detection),
                detector_elapsed,
            ) = _timed_qualitative_call(
                _snapshot_and_detect_online_round,
                device=device,
            )
            online_detector_time_sec += detector_elapsed
            online_record, online_decision_threshold = online_detection
            online_round_records.append(online_record)
            for client_id in online_record["suspicious_clients"]:
                online_suspicious_counts[client_id] = (
                    online_suspicious_counts.get(client_id, 0) + 1
                )

        # Snapshot BEFORE consuming the final uploads. Each Shapley coalition
        # restores this same state and never updates the training optimizer.
        if round_num == num_epoch - 1 and last_round_aggregation_out is not None:
            last_round_aggregation_out.update(
                aggregation_config=aggregation_config.to_dict(),
                last_round_server_state=server_aggregator.state_dict(),
            )
        aggregated_gradient, online_group_model_parameters = _aggregate_round_updates(
            submitted_updates=current_round_client_grads,
            weights=weights,
            device=device,
            current_global_parameters=[
                parameter.detach().clone() for parameter in model.parameters()
            ],
            use_online_qualitative_aggregation=(
                online_qualitative_aggregation_enabled
            ),
            detecting_method=detecting_method,
            round_client_base_parameters=round_client_base_parameters,
            suspicious_clients=(
                online_record["suspicious_clients"]
                if online_record is not None
                else []
            ),
            server_aggregator=server_aggregator,
        )

        # This is the server's model step, including FedAdam's moments/scaling;
        # forecasts and dumps must not observe its internal pre-Adam mean.
        _assert_all_grads_on_device([aggregated_gradient], device, tag=f"round={round_num}/after_agg")
        _assert_model_params_on_device(model, device, tag=f"round={round_num}/after_agg")

        def publish_global_step(public_step):
            if use_attack:
                observers = multi_attackers.values() if multi_attackers is not None else [attacker]
                for observer in observers:
                    if getattr(observer, "supports_round_aware_recording", False):
                        observer.record_global_gradient(public_step, round_num=round_num)
                    else:
                        observer.record_global_gradient(public_step)
            if args.dump_grad_history:
                flat_global_grad = torch.cat([g.reshape(-1) for g in public_step]).detach().cpu().numpy()
                os.makedirs(os.path.join(args.dump_dir, dataset), exist_ok=True)
                csv_path = os.path.join(args.dump_dir, dataset, f'run_{trial_id}.csv')
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as f:
                        csv.writer(f).writerow(['t'] + [f'd{i}' for i in range(flat_global_grad.shape[0])])
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([round_num] + flat_global_grad.tolist())

        if aggregation_config.method == "fedavg":
            # Preserve the historical FedAvg observer timing and values.
            publish_global_step(aggregated_gradient)

        # update: aggregated gradient --> global model 
        # Cache last-round state and gradients before applying update
        if round_num == num_epoch - 1:
            # 1) snapshot of global model before applying round T update
            state_t_minus_1 = copy.deepcopy(model.state_dict())
            # 2) gradients submitted by each client in round T
            last_round_client_grads = {
                i: [g.detach().clone().to(device) for g in grad_list]
                for i, grad_list in enumerate(current_round_client_grads)
            }
            # 3) aggregation weights of round T (by shard size)
            last_round_weights = weights.tolist()

        _assert_model_params_on_device(model, device, tag=f"round={round_num}/before_apply")
        # Keep the actual global-model difference separate from submitted
        # attack updates.  For online WEF/S2-WEF, client models may diverge
        # after the detection decision, so load the appropriate group model
        # after updating the server's benign/global model.
        global_parameters_before_apply = [
            parameter.detach().clone() for parameter in model.parameters()
        ]
        for param, update in zip(model.parameters(), aggregated_gradient):
            param.data += update.data
        if aggregation_config.method != "fedavg":
            # Publish only what the broadcast models reveal, including floating
            # point rounding, rather than an internal optimizer step/aggregate.
            aggregated_gradient = [
                parameter.detach().clone() - previous
                for parameter, previous in zip(model.parameters(), global_parameters_before_apply)
            ]
            publish_global_step(aggregated_gradient)
        s2_global_model_difference = (
            [
                param.detach().clone() - previous
                for param, previous in zip(model.parameters(), global_parameters_before_apply)
            ]
            if detecting_method == "s2_wef"
            else None
        )
        if online_group_model_parameters is not None:
            for client_model, group_parameters in zip(
                client_models,
                online_group_model_parameters,
            ):
                for parameter, group_parameter in zip(
                    client_model.parameters(), group_parameters
                ):
                    parameter.data.copy_(group_parameter.data)
        else:
            for cm in client_models:
                cm.load_state_dict(model.state_dict())

        if detecting_method == "s2_wef":
            previous_global_difference = (
                [difference.detach().clone() for difference in s2_global_model_difference]
                if s2_global_model_difference is not None
                else None
            )

        # evaluation
        acc = model.score(loader.X_test, loader.y_test, ["accuracy"])[0]
        global_accuracies.append(acc)
        print(f"Round {round_num}: Global Accuracy = {acc:.4f}")
        if logfr_observer is not None:
            _, detector_elapsed = _timed_qualitative_call(
                lambda: logfr_observer.finish_round(round_num, model, acc),
                device=device,
            )
            logfr_observer.detector_time_sec += detector_elapsed

        if collect_history:
            if multi_attackers is not None:
                # STD-DAGMM needs submissions, but retaining 50 rounds on GPU
                # multiplies memory across concurrent tasks. Shapley only needs
                # the final-round cache already captured above.
                round_client_gradients.append(
                    [[g.detach().cpu().clone() for g in grads] for grads in current_round_client_grads]
                    if detecting_method == "std_dagmm" else []
                )
                round_global_gradients.append([])
                round_wef_matrices.append([])
                continue
            # Stores the client gradient for the current round
            round_client_gradients.append(current_round_client_grads)
            # S2-WEF receives the actual global-model difference. Other
            # qualitative callers retain the historical aggregated-gradient
            # record, and attack generation is untouched.
            if detecting_method == "s2_wef":
                round_global_gradients.append(s2_global_model_difference)
            else:
                round_global_gradients.append(
                    [g.detach().clone() for g in aggregated_gradient]
                )
            round_wef_matrices.append(
                current_round_wef_matrices
                if current_round_wef_matrices is not None
                else (
                    [
                        tracker.round_snapshot()
                        if detecting_method == "s2_wef"
                        else tracker.snapshot()
                        for tracker in wef_trackers
                    ]
                    if wef_trackers is not None
                    else []
                )
            )

    return_values = (model, attack_time_log, global_accuracies)
    if collect_history:
        return_values += (
            round_client_gradients,
            round_global_gradients,
            round_wef_matrices,
            round_attack_activity,
            logfr_observer.result(num_epoch) if logfr_observer is not None else (
                pass_observer.result(num_epoch) if pass_observer is not None else ({
                "suspicious_clients": list(
                    online_round_records[-1]["suspicious_clients"]
                    if online_round_records
                    else []
                ),
                "suspicious_counts": online_suspicious_counts,
                "round_records": online_round_records,
                "decision_threshold": online_decision_threshold,
                "effective_rounds": len(online_round_records),
                "detector_time_sec": online_detector_time_sec,
                "decision_mode": (
                    "online_round_isolation"
                    if online_qualitative_aggregation_enabled
                    else "passive_round_detection"
                ),
                "qualitative_aggregation_mode": args.qualitative_aggregation_mode,
            }
            if online_qualitative_enabled
            else None)),
        )
    return_values += (state_t_minus_1, last_round_client_grads, last_round_weights)
    
    return_values += (state_t_minus_R,)
    return return_values

if args.control_excluded_ids is not None:
    from module.method.freerider_control import run_control
    run_control(args=args, loader=loader, model=model,
                train=run_federated_learning, device=device,
                provenance=provenance_payload)
    raise SystemExit(0)

# hessian
provider = frobenius_scaled_identity_provider if args.use_hessian else None
lam = args.lambda_h if args.use_hessian else 1.0

moirai_online_parameters = (
    _effective_moirai_online_parameters(args, dataset, attack_method)
    if attack_method in MOIRAI_ONLINE_METHODS
    else None
)
arima_readiness_parameters = (
    _effective_arima_readiness_parameters(args, attack_method)
    if attack_method in ARIMA_METHODS
    else None
)
arima_adaptive_parameters = (
    _effective_arima_adaptive_parameters(args, dataset, attack_method)
    if attack_method in ADAPTIVE_ARIMA_METHODS
    else None
)
layerwise_parameters = {
    "selection_metric": args.layerwise_selection_metric,
    "selection_fraction": args.layerwise_selection_fraction,
    "selection_top_n": args.layerwise_selection_top_n,
    "pca_threshold": args.layerwise_pca_threshold,
    "pca_components": args.layerwise_pca_components,
}


#  6 attacking algorithms
fl_with_attacker_wall_sec = ""
fl_without_attacker_wall_sec = ""
quantitative_diagnostics_rows = []
requested_sampling_budget = (
    args.sampling_shapley_num_permutations
    if detecting_method == "sampling_shapley"
    else args.sampling_least_core_num_coalitions
    if detecting_method == "sampling_least_core"
    else None
)

if use_attack:
    layerwise_supported_datasets = ["mnist", "imdb", "tictactoe", "adult", "dota2"]
    if attack_method in [
        "arima_mnist_imdb_layerwise",
        "arima_adaptive_mnist_imdb_layerwise",
        "moirai_mnist_imdb_layerwise",
        "moirai_online_mnist_imdb_layerwise",
    ] and dataset not in layerwise_supported_datasets:
        raise ValueError(
            f"{attack_method} only supports dataset in {layerwise_supported_datasets}, got '{dataset}'"
        )

    if attack_method == "random":
        attacker = RandomAttack(radius=args.random_radius)
    elif attack_method == "replay":
        attacker = ReplayAttack(k=K, start_attack_round=WARMUP)
    elif attack_method == "fraboni_disguised":
        attacker = FraboniDisguisedAttack(
            noise_std=args.fraboni_noise_std,
            calibration_mode=args.fraboni_calibration_mode,
            calibration_window=args.fraboni_calibration_window,
            min_noise_std=args.fraboni_min_noise_std,
        )
    elif attack_method == "scaled_delta":
        attacker = ScaledDeltaAttack(
            random_round=WARMUP,
            lambda_h=lam,
            hessian_provider=provider,
            max_scale=args.scaled_delta_max_scale,
        )
    elif attack_method == "advanced_scaled_delta":
        attacker = AdvancedScaledDeltaAttack(
            num_clients=num_parts,
            random_round=WARMUP,
            c_const=args.scaled_delta_c_const,
            lambda_est=args.scaled_delta_lambda_est,
            noise_fraction=args.scaled_delta_noise_fraction,
            max_noise_ratio=args.scaled_delta_max_noise_ratio,
            lambda_h=lam,
            hessian_provider=provider,
            max_scale=args.scaled_delta_max_scale,
        )
    elif attack_method == "adam_local_deviation":
        attacker = WangLocalDeviationAttack(
            adam_lr=args.wang_adam_lr,
            beta1=args.wang_adam_beta1,
            beta2=args.wang_adam_beta2,
            adam_eps=args.wang_adam_eps,
        )
    elif attack_method == "dkfr":
        if args.rq1_strict_data_free:
            # Public MLP architecture only; no loader samples or estimated statistics.
            linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
            if not linear_layers:
                raise ValueError('Strict DKFR requires the public MLP input/output dimensions')
            input_dim = linear_layers[0].in_features
            # A single sigmoid output represents two classes in Adult/IMDB.
            num_classes = max(2, linear_layers[-1].out_features)
            feature_min = feature_max = feature_mean = feature_std = None
        else:
            input_dim = int(loader.X_train.shape[1])
            num_classes = 10 if dataset == "mnist" else 2
            feature_min = loader.X_train.detach().amin(dim=0).float()
            feature_max = loader.X_train.detach().amax(dim=0).float()
            feature_mean = loader.X_train.detach().mean(dim=0).float()
            feature_std = loader.X_train.detach().std(dim=0).clamp_min(1e-6).float()
        attacker = DKFRLiteAttack(
            base_model=model,
            dataset=dataset,
            input_dim=input_dim,
            num_classes=num_classes,
            feature_min=feature_min,
            feature_max=feature_max,
            feature_mean=feature_mean,
            feature_std=feature_std,
            synth_batch_size=args.dkfr_synth_batch_size,
            generator_steps=args.dkfr_generator_steps,
            kd_steps=args.dkfr_kd_steps,
            lr_g=args.dkfr_lr_g,
            lr_student=args.dkfr_lr_student,
            temperature=args.dkfr_temperature,
            lambda_kl_g=args.dkfr_lambda_kl_g,
            alpha_kl=args.dkfr_alpha_kl,
            alpha_cls=args.dkfr_alpha_cls,
            alpha_feat=args.dkfr_alpha_feat,
            hidden_dim=args.dkfr_hidden_dim,
            grad_clip=args.dkfr_grad_clip,
            max_update_norm=args.dkfr_max_update_norm,
            kl_cap=args.dkfr_kl_cap,
            student_kl_cap=args.dkfr_student_kl_cap,
            student_loss_cap=args.dkfr_student_loss_cap,
            teacher_conf_min=args.dkfr_teacher_conf_min,
            guard_reset_patience=args.dkfr_guard_reset_patience,
            mnist_tv_weight=args.dkfr_mnist_tv,
            mnist_pixel_reg_weight=args.dkfr_mnist_pixel_reg,
            mnist_logit_reg_weight=args.dkfr_mnist_logit_reg,
        )
    elif attack_method == "synthetic_replacement":
        attacker = SyntheticReplacementAttack(
            dataset=dataset,
            mixup_alpha=args.synthetic_mixup_alpha,
            noise_std=args.synthetic_noise_std,
            tabular_noise_std=args.synthetic_tabular_noise_std,
            tabular_flip_prob=args.synthetic_tabular_flip_prob,
            seed=args.synthetic_seed + trial_id,
        )
    elif attack_method == "arima":
        attacker = OfflineARIMAAttack(
            k=K, random_round=WARMUP, model_params=list(model.parameters()),
            lambda_h=lam, hessian_provider=provider,
            **arima_readiness_parameters,
        )
    elif attack_method == "arima_adaptive":
        attacker = AdaptiveARIMAAttack(
            k=K, random_round=WARMUP, model_params=list(model.parameters()),
            lambda_h=lam, hessian_provider=provider,
            **arima_adaptive_parameters,
            **arima_readiness_parameters,
        )
    elif attack_method == "moirai":
        attacker = OfflineMoiraiAttack(
            k=K, random_round=WARMUP, history_length=HISTORY,
            lambda_h=lam, hessian_provider=provider
        )
    elif attack_method == "moirai_online":
        attacker = OnlineMoiraiAttack(
            k=K, random_round=WARMUP, history_length=HISTORY,
            lambda_h=lam, hessian_provider=provider,
            **moirai_online_parameters,
        )
    elif attack_method == "arima_mnist_imdb_layerwise":
        attacker = LayerwiseOfflineARIMAAttackMNISTIMDB(
            k=K,
            random_round=WARMUP,
            model=model,
            lambda_h=lam,
            hessian_provider=provider,
            **arima_readiness_parameters,
            **layerwise_parameters,
        )
    elif attack_method == "arima_adaptive_mnist_imdb_layerwise":
        attacker = LayerwiseAdaptiveARIMAAttackMNISTIMDB(
            k=K,
            random_round=WARMUP,
            model=model,
            lambda_h=lam,
            hessian_provider=provider,
            **arima_adaptive_parameters,
            **arima_readiness_parameters,
            **layerwise_parameters,
        )
    elif attack_method == "moirai_mnist_imdb_layerwise":
        attacker = LayerwiseOfflineMoiraiAttackMNISTIMDB(
            k=K,
            random_round=WARMUP,
            model=model,
            history_length=HISTORY,
            lambda_h=lam,
            hessian_provider=provider,
            **layerwise_parameters,
        )
    elif attack_method == "moirai_online_mnist_imdb_layerwise":
        attacker = LayerwiseOnlineMoiraiAttackMNISTIMDB(
            k=K,
            random_round=WARMUP,
            model=model,
            history_length=HISTORY,
            lambda_h=lam,
            hessian_provider=provider,
            **moirai_online_parameters,
            **layerwise_parameters,
        )

    # elif attack_method == "arima":
    #     # attacker = ARIMAAttack(k=4, random_round=10)
    #     attacker = ARIMAAttack(k=4, random_round=10, model_params=list(model.parameters()))
    # elif attack_method == "moirai":
    #     attacker = MoiraiAttack(k=4, random_round=10)



    if args.attacker_ids is not None:
        from module.method.multi_freerider import run_experiment

        run_experiment(
            args=args, loader=loader, model=model, attack_template=attacker,
            train=run_federated_learning, detect=_run_qualitative_detection,
            attach_truth=_attach_round_attack_ground_truth, device=device,
            provenance=provenance_payload,
        )
        raise SystemExit(0)

    # Run FL with attacker
    rq1_context = None
    if args.rq1_only:
        from module.method.rq1_single_attacker import prepare_run as prepare_rq1_run, finish_run as finish_rq1_run
        rq1_context = prepare_rq1_run(args=args, loader=loader, model=model)
    realized_attack_rounds_with = set()
    aggregation_context_with = {}
    fl_with_attacker_started_at = time.perf_counter()
    model_with_attacker, attack_time_log_with_attacker, global_accuracies_log_with_attacker, *history_with = run_federated_learning(
        loader, model, attacker_id, attacker,
        use_attack=True,
        num_epoch=num_epoch,
        num_local_epochs=num_local_epochs,
        device=device,
        record_gradients=need_gradients,
        collect_history=need_history,
        public_data=adam_public_pool,
        realized_attack_rounds_out=realized_attack_rounds_with,
        last_round_aggregation_out=aggregation_context_with,
        single_attacker_audit_out=rq1_context['audit'] if rq1_context is not None else None,
    )
    fl_with_attacker_wall_sec = time.perf_counter() - fl_with_attacker_started_at
    # unpack last-round caches from history_with
    if need_history:
        (
            round_client_grads,
            round_global_grads,
            round_wef_matrices,
            round_attack_activity_with,
            online_qualitative_result_with,
            state_t_minus_1,
            last_round_client_grads,
            last_round_weights,
            state_t_minus_R,
        ) = history_with
    else:
        online_qualitative_result_with = None
        state_t_minus_1, last_round_client_grads, last_round_weights, state_t_minus_R = history_with

    if args.rq1_only:
        finish_rq1_run(
            args=args, loader=loader, model=model_with_attacker, attacker=attacker,
            run_context=rq1_context, last_round_state=state_t_minus_1,
            last_round_client_grads=last_round_client_grads, last_round_weights=last_round_weights,
            aggregation_context=aggregation_context_with, accuracy=global_accuracies_log_with_attacker,
            attack_times=attack_time_log_with_attacker, training_seconds=fl_with_attacker_wall_sec,
            provenance=provenance_payload,
        )
        raise SystemExit(0)

    # Run FL no attacker
    subset_loader, subset_ids = prepare_loader_without_attacker(loader, attacker_id)
    subset_model_kwargs = dict(
        seed=seed,
        num_epoch=num_epoch,
        lr=lr,
        batch_size=batch_size,
        hidden_layer_size=hidden_layer_size,
        device=device,
        dataset=dataset,
    )
    if dataset == "imdb":
        subset_model_kwargs["input_size"] = int(subset_loader.X_train.shape[1])
    model_no_attacker = return_model(model_name, **subset_model_kwargs)
    model_no_attacker.init_args = model.init_args  # keep hyperparams consistent
    aggregation_context_without = {}
    fl_without_attacker_started_at = time.perf_counter()
    model_without_attacker, attack_time_log_without_attacker, global_accuracies_log_without_attacker, *history_without = run_federated_learning(
        subset_loader, model_no_attacker, attacker_id=None, attacker=None,
        use_attack=False, num_epoch=num_epoch, num_local_epochs=num_local_epochs,
        device=device, record_gradients=need_gradients, collect_history=need_history,
        last_round_aggregation_out=aggregation_context_without,
    )
    fl_without_attacker_wall_sec = time.perf_counter() - fl_without_attacker_started_at
    if need_history:
        (
            round_client_grads_wo,
            round_global_grads_wo,
            round_wef_matrices_wo,
            round_attack_activity_without,
            online_qualitative_result_without,
            state_t_minus_1_wo,
            last_round_client_grads_wo,
            last_round_weights_wo,
            state_t_minus_R_wo,
        ) = history_without
    else:
        online_qualitative_result_without = None
        state_t_minus_1_wo, last_round_client_grads_wo, last_round_weights_wo, state_t_minus_R_wo = history_without


    # Contribution estimation
    contribs_w_minmax = [[]]
    contribs_wo_minmax = [[]]
    qualitative_result = None
    if detecting_type == "qualitative":
        if detecting_method in {"wef", "s2_wef", "pass", "log_based_free_rider_detection"}:
            qualitative_result = online_qualitative_result_with
        else:
            qualitative_result, detector_time_sec = _timed_qualitative_call(
                lambda: _run_qualitative_detection(
                    round_client_grads,
                    round_wef_matrices=round_wef_matrices,
                    round_global_gradients=round_global_grads,
                    ground_truth_attackers=None,
                ),
                device=device,
            )
            qualitative_result["detector_time_sec"] = detector_time_sec
        qualitative_result = _attach_round_attack_ground_truth(
            qualitative_result,
            round_attack_activity_with,
            score_evaluator=(
                QualitativeDetector.evaluate_round_scores
                if detecting_method == "std_dagmm"
                else None
            ),
        )
        suspicious_clients = qualitative_result["suspicious_clients"]
        suspicious_counts = qualitative_result["suspicious_counts"]
        round_records = qualitative_result["round_records"]
        threshold = qualitative_result["decision_threshold"]

        if round_records and detecting_method == "std_dagmm":
            last_record = round_records[-1]
            print(
                f"[std_dagmm Round {last_record['round_idx']}] "
                f"energies={last_record['client_scores']} "
                f"ranks={last_record['energy_ranks']} "
                f"auc={last_record['roc_auc']} "
                f"top_clients={last_record['top_clients']} "
                f"weights={last_record.get('weights', {})}"
            )
        elif round_records:
            last_record = next(
                (record for record in reversed(round_records) if record.get("decision_round") is not None),
                round_records[-1],
            ) if detecting_method == "pass" else round_records[-1]
            print(
                f"[{detecting_method} Round {last_record['round_idx']}] "
                f"scores={last_record['client_scores']} "
                f"suspicious={last_record['suspicious_clients']} "
                f"weights={last_record.get('weights', {})}"
            )
        if detecting_method == "std_dagmm":
            evaluation = qualitative_result.get("evaluation", {})
            print(
                "[std_dagmm Evaluation] "
                f"mean_round_auc={evaluation.get('mean_round_auc')} "
                f"std_round_auc={evaluation.get('std_round_auc')} "
                f"mean_average_precision={evaluation.get('mean_average_precision')} "
                f"mean_attacker_rank={evaluation.get('mean_attacker_rank')} "
                f"top1_hit_rate={evaluation.get('top1_hit_rate')} "
                f"valid_auc_rounds={evaluation.get('valid_auc_rounds')} "
                f"ranked_clients={qualitative_result.get('ranked_clients', [])} "
                "decision_mode=score_only"
            )
        else:
            attacker_hit = (
                "N/A" if detecting_method in {"pass", "log_based_free_rider_detection"}
                and not qualitative_result["final_valid_by_client"].get(attacker_id, False)
                else attacker_id in suspicious_clients
            )
            print(
                f"[{detecting_method} Detection] suspicious_counts={suspicious_counts} "
                f"threshold={threshold} final_suspicious={suspicious_clients} "
                f"attacker? {attacker_hit}"
            )
        contribs_w = [[float("nan")] * num_parts]
        contribs_wo = [[float("nan")] * (num_parts - 1)]
        contribs_w_minmax = [[float("nan")] * num_parts]
        contribs_wo_minmax = [[float("nan")] * (num_parts - 1)]
    else:
        # Quantitative detecting: compute contributions with and without attacker.
        quantitative_kwargs_with = dict(
            **aggregation_context_with,
            attacker=attacker,
            attacker_id=attacker_id,
            use_last_round_sv=args.use_last_round_sv,
            last_round_state=state_t_minus_1,
            last_round_client_grads=last_round_client_grads,
            last_round_weights=last_round_weights,
            mini_local_epochs=num_local_epochs,
            use_lazy=args.use_lazy,
            lazy_honest_gap=args.lazy_honest_gap,
            lazy_honest_phase=args.lazy_honest_phase,
            lazy_fake_policy=args.lazy_fake_policy,
            lazy_attack_rounds=(
                realized_attack_rounds_with
                if args.use_lazy and args.lazy_schedule == "probabilistic"
                else None
            ),
        )
        if not args.use_last_round_sv:
            quantitative_kwargs_with.update(
                base_state_dict=init_state,
                mini_rounds=num_epoch,
            )
        contrib_with = ContributionCalculator(
            loader, model_with_attacker, {},
            ["accuracy"],
            **quantitative_kwargs_with
        )
        sampling_call_kwargs_with = build_sampling_call_kwargs(
            detecting_method,
            num_parts=len(loader.X_train_parts),
            experiment_seed=seed,
            trial_id=trial_id,
            shapley_num_permutations=args.sampling_shapley_num_permutations,
            shapley_seed=args.sampling_shapley_seed,
            shapley_tolerance=args.sampling_shapley_tolerance,
            least_core_num_coalitions=args.sampling_least_core_num_coalitions,
            least_core_seed=args.sampling_least_core_seed,
        )
        contribs_w, contribution_time_with = contrib_with.get_contributions(
            **sampling_call_kwargs_with
        )

        quantitative_kwargs_without = dict(
            **aggregation_context_without,
            attacker=None,
            attacker_id=None,
            use_last_round_sv=args.use_last_round_sv,
            last_round_state=state_t_minus_1_wo,
            last_round_client_grads=last_round_client_grads_wo,
            last_round_weights=last_round_weights_wo,
            mini_local_epochs=num_local_epochs,
            use_lazy=False,
            lazy_honest_gap=args.lazy_honest_gap,
            lazy_honest_phase=args.lazy_honest_phase,
            lazy_fake_policy=args.lazy_fake_policy,
        )
        if not args.use_last_round_sv:
            quantitative_kwargs_without.update(
                base_state_dict=init_state,
                mini_rounds=num_epoch,
            )
        contrib_without = ContributionCalculator(
            subset_loader, model_no_attacker, {},
            ["accuracy"],
            **quantitative_kwargs_without
        )
        sampling_call_kwargs_without = build_sampling_call_kwargs(
            detecting_method,
            num_parts=len(subset_loader.X_train_parts),
            experiment_seed=seed,
            trial_id=trial_id,
            shapley_num_permutations=args.sampling_shapley_num_permutations,
            shapley_seed=args.sampling_shapley_seed,
            shapley_tolerance=args.sampling_shapley_tolerance,
            least_core_num_coalitions=args.sampling_least_core_num_coalitions,
            least_core_seed=args.sampling_least_core_seed,
        )
        contribs_wo, contribution_time_without = contrib_without.get_contributions(
            **sampling_call_kwargs_without
        )
        if detecting_method in {"sampling_shapley", "sampling_least_core"}:
            common_diagnostic_metadata = {
                "Dataset": dataset,
                "AttackMethod": attack_method,
                "AttackRegime": "lazy" if args.use_lazy else "pure",
                "Alpha": alpha,
                "AttackerID": attacker_id,
                "TrialID": trial_id,
                "DetectingMethod": detecting_method,
            }
            quantitative_diagnostics_rows.append(
                build_quantitative_diagnostic_row(
                    metadata={
                        **common_diagnostic_metadata,
                        "ClientNum": len(loader.X_train_parts),
                    },
                    world="with_attacker",
                    estimator=contrib_with,
                    requested_budget=requested_sampling_budget,
                    elapsed_seconds=contribution_time_with,
                )
            )
            quantitative_diagnostics_rows.append(
                build_quantitative_diagnostic_row(
                    metadata={
                        **common_diagnostic_metadata,
                        "ClientNum": len(subset_loader.X_train_parts),
                    },
                    world="without_attacker",
                    estimator=contrib_without,
                    requested_budget=requested_sampling_budget,
                    elapsed_seconds=contribution_time_without,
                )
            )
        contribs_w_minmax = _minmax_normalize_rows(contribs_w)
        contribs_wo_minmax = _minmax_normalize_rows(contribs_wo)


else:
    # === No attacker path ===
    aggregation_context_no_attack = {}
    fl_without_attacker_started_at = time.perf_counter()
    model_no_attack, attack_time_log_no_attack, global_accuracies_log_no_attack, *history_no = run_federated_learning(
        loader, model, attacker_id=None, attacker=None, use_attack=False,
        num_epoch=num_epoch, num_local_epochs=num_local_epochs,
        device=device, record_gradients=need_gradients, collect_history=need_history,
        last_round_aggregation_out=aggregation_context_no_attack,
    )
    fl_without_attacker_wall_sec = time.perf_counter() - fl_without_attacker_started_at
    if need_history:
        (
            round_client_grads_no,
            round_global_grads_no,
            round_wef_matrices_no,
            round_attack_activity_no,
            online_qualitative_result_no_attack,
            state_t_minus_1_no,
            last_round_client_grads_no,
            last_round_weights_no,
            state_t_minus_R_no,
        ) = history_no
    else:
        online_qualitative_result_no_attack = None
        (
            state_t_minus_1_no,
            last_round_client_grads_no,
            last_round_weights_no,
            state_t_minus_R_no,
        ) = history_no

    contribs_w_minmax = [[]]
    contribs_wo_minmax = [[]]
    qualitative_result = None
    if detecting_type == "quantitative":
        quantitative_kwargs_no_attack = dict(
            **aggregation_context_no_attack,
            use_last_round_sv=args.use_last_round_sv,
            last_round_state=state_t_minus_1_no,
            last_round_client_grads=last_round_client_grads_no,
            last_round_weights=last_round_weights_no,
            mini_local_epochs=num_local_epochs,
            use_lazy=False,
            lazy_honest_gap=args.lazy_honest_gap,
            lazy_honest_phase=args.lazy_honest_phase,
            lazy_fake_policy=args.lazy_fake_policy,
        )
        if not args.use_last_round_sv:
            quantitative_kwargs_no_attack.update(
                base_state_dict=init_state,
                mini_rounds=num_epoch,
            )
        contrib_without = ContributionCalculator(
            loader, model_no_attack, {},
            ["accuracy"],
            **quantitative_kwargs_no_attack
        )
        sampling_call_kwargs_no_attack = build_sampling_call_kwargs(
            detecting_method,
            num_parts=len(loader.X_train_parts),
            experiment_seed=seed,
            trial_id=trial_id,
            shapley_num_permutations=args.sampling_shapley_num_permutations,
            shapley_seed=args.sampling_shapley_seed,
            shapley_tolerance=args.sampling_shapley_tolerance,
            least_core_num_coalitions=args.sampling_least_core_num_coalitions,
            least_core_seed=args.sampling_least_core_seed,
        )
        contribs_wo, contribution_time_without = contrib_without.get_contributions(
            **sampling_call_kwargs_no_attack
        )
        if detecting_method in {"sampling_shapley", "sampling_least_core"}:
            quantitative_diagnostics_rows.append(
                build_quantitative_diagnostic_row(
                    metadata={
                        "Dataset": dataset,
                        "AttackMethod": attack_method,
                        "AttackRegime": "benign",
                        "Alpha": alpha,
                        "ClientNum": len(loader.X_train_parts),
                        "AttackerID": "N/A",
                        "TrialID": trial_id,
                        "DetectingMethod": detecting_method,
                    },
                    world="without_attacker",
                    estimator=contrib_without,
                    requested_budget=requested_sampling_budget,
                    elapsed_seconds=contribution_time_without,
                )
            )
        contribs_wo_minmax = _minmax_normalize_rows(contribs_wo)
    else:
        # Qualitative methods also run in benign-only experiments so false
        # positives can be measured rather than silently skipping detection.
        if detecting_method in {"wef", "s2_wef", "pass", "log_based_free_rider_detection"}:
            qualitative_result = online_qualitative_result_no_attack
        else:
            qualitative_result, detector_time_sec = _timed_qualitative_call(
                lambda: _run_qualitative_detection(
                    round_client_grads_no,
                    round_wef_matrices=round_wef_matrices_no,
                    round_global_gradients=round_global_grads_no,
                    ground_truth_attackers=[],
                ),
                device=device,
            )
            qualitative_result["detector_time_sec"] = detector_time_sec
        qualitative_result = _attach_round_attack_ground_truth(
            qualitative_result,
            round_attack_activity_no,
            score_evaluator=(
                QualitativeDetector.evaluate_round_scores
                if detecting_method == "std_dagmm"
                else None
            ),
        )
        contribs_wo = [[float("nan")] * num_parts]
        contribs_wo_minmax = [[float("nan")] * num_parts]

detecting_suffix = f"{detecting_type}_{detecting_method}"
contribution_dir = os.path.join(results_root, f"Detecting_{detecting_suffix}")
time_dir = os.path.join(results_root, f"Time_{detecting_suffix}")
accuracy_dir = os.path.join(results_root, f"Accuracy_{detecting_suffix}")

os.makedirs(contribution_dir, exist_ok=True)
os.makedirs(time_dir, exist_ok=True)
os.makedirs(accuracy_dir, exist_ok=True)

if detecting_method in {"sampling_shapley", "sampling_least_core"}:
    write_quantitative_diagnostics(
        os.path.join(
            contribution_dir,
            f"quantitative_diagnostics_trial{trial_id}.csv",
        ),
        quantitative_diagnostics_rows,
    )

if detecting_type == "qualitative" and qualitative_result is not None:
    _write_qualitative_detection_csv(
        os.path.join(
            contribution_dir,
            f"round_client_detection_trial{trial_id}.csv",
        ),
        qualitative_result,
        {
            "TaskID": (
                f"{dataset}_{attack_method}_{detecting_method}_alpha{alpha}_"
                f"clients{num_parts}_attacker{attacker_id if use_attack else 'none'}_"
                f"trial{trial_id}"
            ),
            "CampaignComplete": 1,
            "Dataset": dataset,
            "AttackMethod": attack_method,
            "AttackRegime": "lazy" if args.use_lazy else "pure",
            "DetectingMethod": detecting_method,
            "QualitativeAggregationMode": args.qualitative_aggregation_mode,
            "Alpha": alpha,
            "ClientNum": num_parts,
            "AttackerID": attacker_id if use_attack else "N/A",
            "TrialID": trial_id,
            "UseHessian": args.use_hessian,
            "LambdaH": args.lambda_h,
            "UseLazy": args.use_lazy,
            "LazySchedule": args.lazy_schedule,
            "LazyHonestGap": args.lazy_honest_gap,
            "LazyHonestPhase": args.lazy_honest_phase,
            "LazyFakePolicy": args.lazy_fake_policy,
        },
    )



if detecting_method == "pass" and qualitative_result is not None:
    qualitative_result["experiment"] = {
        "dataset": dataset, "attack_method": attack_method, "use_attack": use_attack,
        "use_lazy": args.use_lazy, "trial_id": trial_id,
        "original_client_ids": list(range(num_parts)),
    }
    write_pass_artifacts(contribution_dir, trial_id, qualitative_result)
    print(f"[PASS coverage] {qualitative_result['coverage']}")
    if use_attack:
        pass_control = _attach_round_attack_ground_truth(
            online_qualitative_result_without, round_attack_activity_without,
        )
        pass_control["experiment"] = {
            **qualitative_result["experiment"], "use_attack": False,
            "control": "existing_remove_attacker_run", "original_client_ids": list(subset_ids),
        }
        write_pass_artifacts(contribution_dir, trial_id, pass_control, suffix="_without_attacker")

if detecting_method == "log_based_free_rider_detection" and qualitative_result is not None:
    qualitative_result["validation_split"] = logfr_validation_metadata
    qualitative_result["experiment"] = {
        "dataset": dataset, "attack_method": attack_method,
        "use_attack": use_attack, "use_lazy": args.use_lazy,
        "trial_id": trial_id, "alpha": alpha,
        "original_client_ids": list(range(num_parts)),
    }
    write_log_based_free_rider_artifacts(contribution_dir, trial_id, qualitative_result)
    if use_attack:
        logfr_control = _attach_round_attack_ground_truth(
            online_qualitative_result_without, round_attack_activity_without,
        )
        logfr_control["validation_split"] = {
            "scope": "parent_run_split",
            "active_original_client_ids": list(subset_ids),
            "parent_run": logfr_validation_metadata,
        }
        logfr_control["experiment"] = {
            **qualitative_result["experiment"], "use_attack": False,
            "control": "existing_remove_attacker_run",
            "original_client_ids": list(subset_ids),
        }
        write_log_based_free_rider_artifacts(
            contribution_dir, trial_id, logfr_control, suffix="_without_attacker",
        )

with open(os.path.join(contribution_dir, f"combined_contributions_trial{trial_id}.csv"), "a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow([
            "Dataset","AttackMethod","Alpha","ClientNum","AttackerID","TrialID",
            "UseAttack","DetectingType","DetectingMethod",
            "UseHessian","UseLazy","LazySchedule","AttackProbability",
            "LazyHonestGap","LazyHonestPhase","LazyFakePolicy",
            "ClientID",
            "WithAttacker","WithoutAttacker","WithAttackerMinMax","WithoutAttackerMinMax"
        ])

    for i in range(num_parts):
        if use_attack:
            w_val = contribs_w[0][i]
            w_val_minmax = contribs_w_minmax[0][i]
            if i == attacker_id:
                wo_val = "N/A"
                wo_val_minmax = "N/A"
            else:
                j = i if i < attacker_id else i - 1
                wo_val = contribs_wo[0][j]
                wo_val_minmax = contribs_wo_minmax[0][j]
        else:
            w_val = "N/A"
            w_val_minmax = "N/A"
            wo_val = contribs_wo[0][i]
            wo_val_minmax = contribs_wo_minmax[0][i]

        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, detecting_type, detecting_method,
            args.use_hessian, args.use_lazy, args.lazy_schedule, args.attack_probability,
            args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy,
            i,
            w_val, wo_val, w_val_minmax, wo_val_minmax
        ])


with open(os.path.join(accuracy_dir, f"global_accuracy_log_trial{trial_id}.csv"), "a", newline="") as f:
    writer = csv.writer(f)

    if f.tell() == 0:
        writer.writerow([
            "Dataset","AttackMethod","Alpha","ClientNum","AttackerID","TrialID",
            "UseAttack","DetectingType","DetectingMethod",
            "UseHessian","UseLazy","LazySchedule","AttackProbability",
            "LazyHonestGap","LazyHonestPhase","LazyFakePolicy",
            "Round","WithAttackerAccuracy","WithoutAttackerAccuracy"
        ])

    for i in range(num_epoch):
        if use_attack:
            acc_with = global_accuracies_log_with_attacker[i] if i < len(global_accuracies_log_with_attacker) else ""
            acc_without = global_accuracies_log_without_attacker[i] if i < len(global_accuracies_log_without_attacker) else ""
        else:
            acc_with = "N/A"
            acc_without = global_accuracies_log_no_attack[i] if i < len(global_accuracies_log_no_attack) else ""

        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, detecting_type, detecting_method,
            args.use_hessian, args.use_lazy, args.lazy_schedule, args.attack_probability,
            args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy,
            i, acc_with, acc_without
        ])

if detecting_type == "qualitative":
    with open(os.path.join(contribution_dir, f"{detecting_method}_detection_trial{trial_id}.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
                "UseAttack", "DetectingType", "DetectingMethod",
                "QualitativeAggregationMode",
                "UseHessian", "UseLazy", "LazySchedule", "AttackProbability",
                "LazyHonestGap", "LazyHonestPhase", "LazyFakePolicy",
                "DetectedSuspiciousClients", "DetectedAttackerHit",
                "SuspiciousCounts", "DecisionThreshold", "EffectiveRounds",
                "LastRoundIdx", "LastRoundScores", "LastRoundClientSTDs",
                "LastRoundClientMeans", "LastRoundMeanEnergy",
                "LastRoundSuspiciousClients", "LastRoundClusterCenters",
                "LastRoundClusterSeparation", "LastRoundEnergyThreshold",
                "LastRoundFinalLoss", "LastRoundWeights",
                "LastRoundWEFEuclideanDistances", "LastRoundWEFCosineSimilarities",
                "LastRoundWEFAverageFrequencies", "LastRoundWEFMatrixSums",
                "LastRoundWEFReputationThreshold",
                "LastRoundS2WEFGammaScores", "LastRoundS2WEFDeviationScores",
                "LastRoundS2WEFSimulatedSum", "LastRoundS2WEFClusterCount",
                "LastRoundS2WEFSilhouette", "LastRoundS2WEFMergeJump",
                "LastRoundS2WEFGammaVote", "LastRoundS2WEFDeviationVote"
            ])

        last_round_idx = ""
        last_round_scores = ""
        last_round_stds = ""
        last_round_means = ""
        last_round_mean_energy = ""
        last_round_suspicious = ""
        last_round_cluster_centers = ""
        last_round_cluster_separation = ""
        last_round_energy_threshold = ""
        last_round_final_loss = ""
        last_round_weights = ""
        last_round_wef_euclidean = ""
        last_round_wef_cosine = ""
        last_round_wef_average = ""
        last_round_wef_sums = ""
        last_round_wef_threshold = ""
        last_round_s2_gamma = ""
        last_round_s2_deviation = ""
        last_round_s2_simulated_sum = ""
        last_round_s2_cluster_count = ""
        last_round_s2_silhouette = ""
        last_round_s2_merge_jump = ""
        last_round_s2_gamma_vote = ""
        last_round_s2_deviation_vote = ""
        suspicious_clients = []
        suspicious_counts = {}
        threshold = ""
        effective_rounds = ""
        decision_mode = ""

        if qualitative_result is not None:
            suspicious_clients = qualitative_result.get("suspicious_clients", [])
            suspicious_counts = qualitative_result.get("suspicious_counts", {})
            threshold = qualitative_result.get("decision_threshold", "")
            effective_rounds = qualitative_result.get("effective_rounds", "")
            decision_mode = qualitative_result.get("decision_mode", "")
            round_records = qualitative_result.get("round_records", [])
            if round_records:
                last_record = next(
                    (record for record in reversed(round_records) if record.get("decision_round") is not None),
                    round_records[-1],
                ) if detecting_method == "pass" else round_records[-1]
                last_round_idx = last_record.get("round_idx", "")
                last_round_scores = last_record.get("client_scores", {})
                last_round_stds = last_record.get("client_stds", {})
                last_round_means = last_record.get("client_means", {})
                last_round_mean_energy = last_record.get("mean_energy", "")
                last_round_suspicious = last_record.get("suspicious_clients", [])
                last_round_cluster_centers = last_record.get("cluster_centers", [])
                last_round_cluster_separation = last_record.get("cluster_separation", "")
                last_round_energy_threshold = last_record.get("energy_threshold", "")
                last_round_final_loss = last_record.get("final_loss", "")
                last_round_weights = last_record.get("weights", {})
                last_round_wef_euclidean = last_record.get("euclidean_distances", {})
                last_round_wef_cosine = last_record.get("cosine_similarities", {})
                last_round_wef_average = last_record.get("average_frequencies", {})
                last_round_wef_sums = last_record.get("wef_matrix_sums", {})
                last_round_wef_threshold = last_record.get("reputation_threshold", "")
                last_round_s2_gamma = last_record.get("gamma_scores", {})
                last_round_s2_deviation = last_record.get("deviation_scores", {})
                last_round_s2_simulated_sum = last_record.get("simulated_wef_sum", "")
                last_round_s2_cluster_count = last_record.get("cluster_count", "")
                last_round_s2_silhouette = last_record.get("silhouette_score", "")
                last_round_s2_merge_jump = last_record.get("merge_jump_ratio", "")
                last_round_s2_gamma_vote = last_record.get("gamma_vote", "")
                last_round_s2_deviation_vote = last_record.get("deviation_vote", "")

        detected_attacker_hit = (
            "N/A"
            if not use_attack or decision_mode == "score_only" or (
                detecting_method in {"pass", "log_based_free_rider_detection"}
                and not qualitative_result.get("final_valid_by_client", {}).get(attacker_id, False)
            )
            else int(attacker_id in suspicious_clients)
        )
        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, detecting_type, detecting_method,
            args.qualitative_aggregation_mode,
            args.use_hessian, args.use_lazy, args.lazy_schedule, args.attack_probability,
            args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy,
            str(suspicious_clients), detected_attacker_hit,
            str(suspicious_counts), threshold, effective_rounds,
            last_round_idx, str(last_round_scores), str(last_round_stds),
            str(last_round_means), last_round_mean_energy,
            str(last_round_suspicious), str(last_round_cluster_centers),
            last_round_cluster_separation, last_round_energy_threshold,
            last_round_final_loss, str(last_round_weights),
            str(last_round_wef_euclidean), str(last_round_wef_cosine),
            str(last_round_wef_average), str(last_round_wef_sums),
            last_round_wef_threshold,
            str(last_round_s2_gamma), str(last_round_s2_deviation),
            last_round_s2_simulated_sum, last_round_s2_cluster_count,
            last_round_s2_silhouette, last_round_s2_merge_jump,
            last_round_s2_gamma_vote, last_round_s2_deviation_vote
        ])

    if detecting_method == "std_dagmm" and qualitative_result is not None:
        score_path = os.path.join(
            contribution_dir,
            f"std_dagmm_scores_trial{trial_id}.csv",
        )
        with open(score_path, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "Dataset", "AttackMethod", "Alpha", "ClientNum",
                    "AttackerID", "TrialID", "UseAttack",
                    "UseLazy", "LazySchedule", "AttackProbability",
                    "Round", "ClientID", "Energy", "UpdateSTD",
                    "IsDesignatedAttacker", "GroundTruthActiveAttack",
                    "EnergyRank", "EnergyPercentile",
                    "IsTopEnergy", "RoundROCAUC", "RoundAveragePrecision",
                    "RoundAttackerMeanRank", "RoundTop1Hit",
                ])

            for record in qualitative_result.get("round_records", []):
                scores = record.get("client_scores", {})
                client_stds = record.get("client_stds", {})
                designated_labels = record.get("designated_attacker_labels", {})
                labels = record.get("ground_truth_labels", {})
                ranks = record.get("energy_ranks", {})
                percentiles = record.get("energy_percentiles", {})
                top_clients = set(record.get("top_clients", []))
                for client_id in sorted(scores):
                    writer.writerow([
                        dataset, attack_method, alpha, num_parts,
                        attacker_id, trial_id, use_attack,
                        args.use_lazy, args.lazy_schedule, args.attack_probability,
                        record.get("round_idx", ""), client_id,
                        scores.get(client_id, ""), client_stds.get(client_id, ""),
                        designated_labels.get(client_id, ""),
                        labels.get(client_id, ""), ranks.get(client_id, ""),
                        percentiles.get(client_id, ""), int(client_id in top_clients),
                        record.get("roc_auc", ""),
                        record.get("average_precision", ""),
                        record.get("attacker_mean_rank", ""),
                        record.get("top1_hit", ""),
                    ])

        summary_path = os.path.join(
            contribution_dir,
            f"std_dagmm_auc_summary_trial{trial_id}.csv",
        )
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "Dataset", "AttackMethod", "Alpha", "ClientNum",
                    "AttackerID", "TrialID", "UseAttack",
                    "UseLazy", "LazySchedule", "AttackProbability",
                    "DecisionMode", "EffectiveRounds", "ValidAUCRounds",
                    "MeanRoundAUC", "StdRoundAUC", "MeanAveragePrecision",
                    "MeanAttackerRank", "Top1HitRate", "RankedClients",
                    "ClientMeanPercentiles", "ClientMeanRanks",
                ])

            evaluation = qualitative_result.get("evaluation", {})
            writer.writerow([
                dataset, attack_method, alpha, num_parts,
                attacker_id, trial_id, use_attack,
                args.use_lazy, args.lazy_schedule, args.attack_probability,
                qualitative_result.get("decision_mode", "score_only"),
                qualitative_result.get("effective_rounds", ""),
                evaluation.get("valid_auc_rounds", ""),
                evaluation.get("mean_round_auc", ""),
                evaluation.get("std_round_auc", ""),
                evaluation.get("mean_average_precision", ""),
                evaluation.get("mean_attacker_rank", ""),
                evaluation.get("top1_hit_rate", ""),
                str(qualitative_result.get("ranked_clients", [])),
                str(qualitative_result.get("client_mean_percentiles", {})),
                str(qualitative_result.get("client_mean_ranks", {})),
            ])


if detecting_type == "qualitative":
    task_id = (
        f"{dataset}_{attack_method}_{detecting_method}_alpha{alpha}_"
        f"clients{num_parts}_attacker{attacker_id if use_attack else 'none'}_"
        f"trial{trial_id}"
    )
    qualitative_time_row = _build_qualitative_time_row(
        {
            "TaskID": task_id,
            "CampaignComplete": 1,
            "Dataset": dataset,
            "AttackMethod": attack_method,
            "AttackRegime": "lazy" if args.use_lazy else "pure",
            "Alpha": alpha,
            "ClientNum": num_parts,
            "AttackerID": attacker_id if use_attack else "N/A",
            "TrialID": trial_id,
            "UseHessian": args.use_hessian,
            "LambdaH": args.lambda_h if args.use_hessian else "N/A",
            "UseLazy": args.use_lazy,
            "LazySchedule": args.lazy_schedule,
            "LazyHonestGap": args.lazy_honest_gap,
            "LazyHonestPhase": args.lazy_honest_phase,
            "LazyFakePolicy": args.lazy_fake_policy,
            "DetectingMethod": detecting_method,
            "QualitativeAggregationMode": args.qualitative_aggregation_mode,
        },
        use_attack=use_attack,
        attack_time_log_with_attacker=(
            attack_time_log_with_attacker if use_attack else []
        ),
        round_attack_activity_with=(
            round_attack_activity_with if use_attack else []
        ),
        qualitative_result_with=qualitative_result if use_attack else None,
        qualitative_result_without=(
            online_qualitative_result_without if use_attack else qualitative_result
        ),
        fl_with_attacker_wall_sec=fl_with_attacker_wall_sec,
        fl_without_attacker_wall_sec=fl_without_attacker_wall_sec,
        task_wall_sec=time.perf_counter() - task_wall_started_at,
    )

    with open(
        os.path.join(time_dir, f"attacking_time_log_trial{trial_id}.csv"),
        "a",
        newline="",
    ) as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(list(qualitative_time_row))
        writer.writerow(list(qualitative_time_row.values()))
else:
    with open(os.path.join(time_dir, f"attacking_time_log_trial{trial_id}.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "Dataset","AttackMethod","Alpha","ClientNum","AttackerID","TrialID",
                "UseAttack","DetectingType","DetectingMethod",
                "UseHessian","UseLazy","LazySchedule","AttackProbability",
                "LazyHonestGap","LazyHonestPhase","LazyFakePolicy",
                "WithAttackerSec","WithoutAttackerSec"
            ])

        base = [
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, detecting_type, detecting_method,
            args.use_hessian, args.use_lazy, args.lazy_schedule, args.attack_probability,
            args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy
        ]

        if use_attack:
            writer.writerow(base + [sum(attack_time_log_with_attacker), sum(attack_time_log_without_attacker)])
        else:
            writer.writerow(base + ["N/A", "N/A"])

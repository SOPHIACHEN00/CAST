"""Opt-in reporting for the original FLCE single-attacker training path."""
from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from module.method.fl_aggregation import AggregationConfig


def validate_config(args: Namespace) -> None:
    """Keep the dedicated RQ1 experiment separate from every existing mode."""
    alphas = {'adult': 1000, 'mnist': 10, 'imdb': 10}
    strict = getattr(args, 'rq1_strict_data_free', False)
    if strict and (not args.rq1_only or args.attack_method not in {'advanced_scaled_delta', 'dkfr'}):
        raise ValueError('Strict data-free RQ1 supports advanced_scaled_delta and dkfr only')
    if args.attack_method == 'advanced_scaled_delta' and not strict:
        raise ValueError('Advanced scaled-delta RQ1 requires explicit strict data-free mode')
    if args.attacker_ids is not None or args.control_excluded_ids is not None:
        raise ValueError('--rq1_only requires singular --attacker_id and no removal control')
    if not args.use_attack or args.use_lazy or args.client_num != 6:
        raise ValueError('--rq1_only requires one pure attacker among 6 clients')
    if not 0 <= args.attacker_id < 6:
        raise ValueError('RQ1 attacker_id must be one of 0,1,2,3,4,5')
    if args.dataset not in alphas or args.alpha != alphas[args.dataset]:
        raise ValueError('RQ1 requires first-column alpha: Adult=1000, MNIST=10, IMDB=10')
    suffix = '' if args.dataset == 'adult' else '_mnist_imdb_layerwise'
    if args.attack_method not in {'arima' + suffix, 'moirai' + suffix, 'arima_adaptive' + suffix, 'moirai_online' + suffix, 'adam_local_deviation', 'dkfr', 'advanced_scaled_delta'}:
        raise ValueError('Unsupported RQ1 attack variant')
    if args.num_epoch != 50 or args.num_local_epochs != 1:
        raise ValueError('RQ1 requires 50 rounds and 1 local epoch')
    if args.detecting_method != 'shapley' or not args.use_last_round_sv:
        raise ValueError('RQ1 requires exact last-round Shapley')
    if args.qualitative_aggregation_mode != 'passive' or args.dump_grad_history:
        raise ValueError('RQ1 requires passive aggregation and no per-task CSV dumps')
    if args.seed != 42 or args.trial_id != 0:
        raise ValueError('This RQ1 sweep fixes seed=42 and trial_id=0; it varies attacker position')
    if not args.use_hessian or args.lambda_h != 0.9 or args.random_round != 10:
        raise ValueError('RQ1 preserves Hessian lambda=0.9 and 10 fake warmup rounds')
    AggregationConfig.from_args(args)


def configure_worker_threads() -> int:
    threads = int(os.environ.get('FLCE_CPU_THREADS_PER_TASK', '15'))
    if threads != 15:
        raise ValueError('This RQ1 experiment fixes 15 CPU compute threads per task')
    torch.set_num_threads(threads)
    return torch.get_num_threads()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare_run(*, args: Namespace, loader: Any, model: Any) -> dict:
    """Capture pairing evidence before the existing single-attacker loop starts."""
    validate_config(args)
    target = Path(args.results_root) / 'rq1_result.json'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite completed RQ1 result: {target}')
    sizes = [len(x) for x in loader.X_train_parts]
    if len(sizes) != 6 or any(size <= 0 for size in sizes):
        raise ValueError('RQ1 requires six nonempty client shards')
    fingerprints = dict(
        shards=[dict(client_id=i, size=size, x_sha256=_tensor_hash(loader.X_train_parts[i]),
                     y_sha256=_tensor_hash(loader.y_train_parts[i])) for i, size in enumerate(sizes)],
        test_x_sha256=_tensor_hash(loader.X_test), test_y_sha256=_tensor_hash(loader.y_test),
        initial_model={name: _tensor_hash(value) for name, value in model.state_dict().items()},
    )
    return dict(target=target, fingerprints=fingerprints, audit={},
                cpu_threads=configure_worker_threads(), started_at=time.perf_counter())


def rq1_metrics(contributions: Sequence[float], attacker_id: int) -> dict:
    values = [float(value) for value in contributions]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Contributions must be finite')
    denominator = math.fsum(values)
    attacker = values[attacker_id]
    ratio = attacker / denominator if denominator != 0.0 else None
    if ratio is not None and not math.isfinite(ratio):
        raise ValueError('Nonfinite RQ1 ratio')
    return dict(attacker_contribution=attacker, total_contribution=denominator,
                attacker_ratio=ratio,
                ratio_status='valid' if ratio is not None else 'zero_denominator')


def finish_run(*, args: Namespace, loader: Any, model: Any, attacker: Any,
               run_context: dict, last_round_state: dict, last_round_client_grads: dict,
               last_round_weights: list, aggregation_context: dict,
               accuracy: Sequence[float], attack_times: Sequence[float],
               training_seconds: float, provenance: dict) -> dict:
    """Evaluate cached submissions and publish one JSON; never train a control."""
    from module.method.quantitative_detecting.shapley_value import ShapleyValue

    validate_config(args)
    audit = run_context['audit']
    warmup = [0] if args.attack_method == 'adam_local_deviation' else []
    expected = list(range(len(warmup), 50))
    if audit.get('public_warm_start_rounds', []) != warmup:
        raise RuntimeError('RQ1 public warm-start audit mismatch')
    if audit.get('fake_call_rounds') != expected:
        raise RuntimeError('RQ1 fake-gradient calls do not match the attack schedule')
    activity = audit.get('round_activity', [])
    if ([row['round_idx'] for row in activity] != list(range(50))
            or any(row['active_attackers'] != ([] if row['round_idx'] in warmup else [args.attacker_id]) for row in activity)):
        raise RuntimeError('RQ1 round labels do not describe a single pure attacker')
    private_fits = audit.get('private_fit_calls_by_client', {})
    expected_fits = {str(i): 0 if i == args.attacker_id else 50 for i in range(6)}
    if private_fits != expected_fits:
        raise RuntimeError('RQ1 private-training audit does not match 0 attacker / 50 honest fits')
    accuracy = [float(value) for value in accuracy]
    if len(accuracy) != 50 or not all(math.isfinite(value) and 0 <= value <= 1 for value in accuracy):
        raise ValueError('Incomplete or invalid accuracy trajectory')
    config = AggregationConfig.from_args(args)
    if aggregation_context.get('aggregation_config') != config.to_dict():
        raise ValueError('Training and RQ1 aggregation configurations differ')
    if aggregation_context['last_round_server_state']['rounds'] != 49:
        raise ValueError('RQ1 requires the server state before the final round')

    started = time.perf_counter()
    final_state = copy.deepcopy(model.state_dict())
    try:
        calculator = ShapleyValue(
            loader, model, {}, ['accuracy'], attacker=attacker, attacker_id=args.attacker_id,
            use_last_round_sv=True, last_round_state=last_round_state,
            last_round_client_grads=last_round_client_grads,
            last_round_weights=last_round_weights, **aggregation_context,
        )
        values, _ = calculator.get_contributions()
        contributions = [float(value) for value in values[0]]
        grand = float(calculator.evaluate_subset_world(set(range(6)), 'accuracy'))
        if len(contributions) != 6 or not math.isclose(grand, accuracy[-1], rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError('RQ1 grand coalition does not reproduce the trained model')
        metrics = rq1_metrics(contributions, args.attacker_id)
        if not math.isclose(metrics['total_contribution'], grand, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError('Exact Shapley efficiency check failed')
        coalition_count = len(calculator.cache)
        if coalition_count != 64:
            raise RuntimeError('Expected all 64 coalitions for six clients')
    finally:
        model.load_state_dict(final_state)
    if torch.device(model.device).type == 'cuda':
        torch.cuda.synchronize(model.device)
    evaluation_seconds = time.perf_counter() - started
    metadata = {name: getattr(args, name) for name in (
        'dataset', 'alpha', 'client_num', 'attacker_id', 'attack_method', 'detecting_method',
        'trial_id', 'seed', 'num_epoch', 'num_local_epochs', 'use_hessian', 'lambda_h',
    )}
    metadata.update(rq1_only=True, attacker_count=1, regime='pure', fl_aggregate=config.method,
                    rq1_strict_data_free=getattr(args, 'rq1_strict_data_free', False),
                    fl_aggregation=config.to_dict(), cpu_threads_per_task=run_context['cpu_threads'],
                    qualitative_aggregation_mode='passive', replicate_axis='attacker_position')
    artifact = dict(
        schema_version=1, metadata=metadata, parameters=vars(args), provenance=provenance,
        fingerprints=run_context['fingerprints'], accuracy=accuracy, contributions=contributions,
        shard_sizes=[len(x) for x in loader.X_train_parts], shard_weights=last_round_weights,
        rq1=metrics,
        audit=dict(audit, fake_rounds=expected, honest_rounds=[], public_warm_start_rounds=warmup),
        evaluation=dict(utility='accuracy', shapley_scope='actual_last_round_submissions',
                        empty_coalition_utility=0.0, final_round_idx=49, grand_coalition_utility=grand,
                        coalitions_evaluated=coalition_count, server_state_rounds_before_last=49),
        timing=dict(training_sec=training_seconds, evaluation_sec=evaluation_seconds,
                    attack_host_sec=sum(attack_times), total_sec=time.perf_counter() - run_context['started_at']),
    )
    target = run_context['target']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(f'[RQ1 SINGLE] Completed: {target}', flush=True)
    return artifact

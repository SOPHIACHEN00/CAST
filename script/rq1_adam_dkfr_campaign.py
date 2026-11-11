#!/usr/bin/env python3
"""Single-attacker RQ1 sweep: preview by default; dataset CSVs from audited JSON.

Independent Adam/DKFR supplement. Standard library only. Public warm start is not a fake round.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
FL_METHODS = ('fedavg', 'fedadam', 'weighted_median')
ATTACK_FAMILIES = ('adam_local_deviation', 'dkfr')
PROFILES = {
    'adult': dict(alpha=1000, arima_var='0.005', arima_err='0.08', moirai_lr='0.003', moirai_err='0.12'),
    'mnist': dict(alpha=10, arima_var='0.05', arima_err='0.05', moirai_lr='0.003', moirai_err='0.06'),
    'imdb': dict(alpha=10, arima_var='0.10', arima_err='0.015', moirai_lr='0.01', moirai_err='0.025'),
}
SERVER = dict(server_lr=0.01, server_beta1=0.9, server_beta2=0.99, server_tau=0.001)
META = ['task_id', 'dataset', 'alpha', 'client_num', 'attacker_id', 'attack_family',
        'attack_method', 'fl_aggregate', *SERVER, 'seed', 'trial_id', 'regime', 'detecting_method']
METRICS = ['attacker_contribution', 'total_contribution', 'attacker_ratio',
           'honest_contribution_mean', 'attacker_data_fraction', 'final_accuracy']
DETAIL = METRICS + ['ratio_status', 'attacker_sample_count', 'fake_count', 'honest_count', 'public_warm_start_count', 'cpu_threads_per_task']
GROUP = [name for name in META if name not in ('task_id', 'attacker_id')]
TABLE_COLUMNS = {
    'rq1_detail': META + DETAIL,
    'rq1_mean_std': GROUP + ['replicate_axis', 'n_expected', 'n_completed', 'status',
                            'attacker_ids_completed', 'attacker_ids_missing', 'std_ddof']
                     + [f'{metric}_{stat}' for metric in METRICS for stat in ('mean', 'std', 'n')],
    'contributions': META + ['client_id', 'is_attacker', 'shard_size', 'shard_weight', 'shapley_value', 'scope'],
    'accuracy': META + ['round_idx', 'accuracy'],
    'time': META + ['training_sec', 'evaluation_sec', 'attack_host_sec', 'total_sec'],
    'audit': META + ['status', 'reason', 'fake_count', 'honest_count', 'public_warm_start_count', 'attacker_private_fit_count',
                     'fake_sources', 'result_path', 'log_path'],
}
SOURCE_FILES = (
    'FLCE_updated.py', 'config.py', 'module/model/net.py', 'script/datapre.py',
    'module/method/rq1_single_attacker.py', 'module/method/fl_aggregation.py',
    'module/method/quantitative_detecting/coalition_value.py',
    'module/method/quantitative_detecting/shapley_value.py',
    'module/method/attack_updated_nonlayerwise.py', 'module/method/attack_updated_layerwise.py',
    'script/rq1_adam_dkfr_campaign.py', 'script/run_rq1_adam_dkfr.sh',
)


def build_tasks(datasets: Sequence[str] = tuple(PROFILES)) -> list[dict]:
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - PROFILES.keys():
        raise ValueError('Select distinct datasets from adult, mnist, imdb')
    tasks = []
    # Interleave datasets/methods so one shared queue uses every GPU for all data.
    for attacker in range(6):
        for dataset in datasets:
            suffix = ''
            for family in ATTACK_FAMILIES:
                for method in FL_METHODS:
                    tasks.append(dict(
                        task_id=f'{dataset}_{family}_{method}_n6_attacker{attacker}_seed42_trial0',
                        dataset=dataset, alpha=PROFILES[dataset]['alpha'], client_num=6,
                        attacker_id=attacker, attack_family=family, attack_method=family + suffix,
                        fl_aggregate=method, **SERVER, seed=42, trial_id=0,
                        regime='pure', detecting_method='shapley',
                    ))
    return tasks


def aggregation_config(task: dict) -> dict:
    return dict(method=task['fl_aggregate'], server_lr=task['server_lr'],
                beta1=task['server_beta1'], beta2=task['server_beta2'], tau=task['server_tau'])


def result_path(task: dict, task_dir: Path) -> Path:
    """Mirror the entry's automatic suffix; checked against its actual config in tests."""
    method = task['fl_aggregate']
    if method == 'fedadam':
        config = aggregation_config(task)
        suffix = (f"fedadam_lr{config['server_lr']}_b1{config['beta1']}"
                  f"_b2{config['beta2']}_tau{config['tau']}")
    else:
        suffix = 'weighted_median' if method == 'weighted_median' else ''
    return Path(task_dir) / suffix / 'rq1_result.json'


def command_for(task: dict, python_bin: str, root: Path, task_dir: Path) -> list[str]:
    profile = PROFILES[task['dataset']]
    argv = [str(python_bin), str(Path(root) / 'FLCE_updated.py'), '--rq1_only']
    for key in ('dataset', 'alpha', 'client_num', 'attacker_id', 'attack_method', 'fl_aggregate',
                'detecting_method', 'seed', 'trial_id', *SERVER):
        argv += ['--' + key, str(task[key])]
    argv += ['--use_attack', '--num_epoch', '50', '--num_local_epochs', '1', '--random_round', '10',
             '--use_last_round_sv', '--qualitative_aggregation_mode', 'passive',
             '--use_hessian', '--lambda_h', '0.9', '--results_root', str(task_dir)]
    if task['attack_family'] == 'adam_local_deviation':
        argv += ['--adam_public_fraction', '0.05', '--adam_public_seed', '20260726',
                 '--wang_adam_lr', '0.008', '--wang_adam_beta1', '0.9',
                 '--wang_adam_beta2', '0.999', '--wang_adam_eps', '1e-8']
    else:
        argv += ['--dkfr_synth_batch_size', '64', '--dkfr_generator_steps', '5',
                 '--dkfr_kd_steps', '5', '--dkfr_lr_g', '0.001', '--dkfr_lr_student', '0.001',
                 '--dkfr_temperature', '2.0', '--dkfr_hidden_dim', '128', '--dkfr_grad_clip', '5.0']
    return argv


def worker_environment(gpu: str, cpu_threads: int) -> dict[str, str]:
    if cpu_threads != 15:
        raise ValueError('RQ1 fixes 15 CPU compute threads per task')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    for key in ('FLCE_CPU_THREADS_PER_TASK', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = str(cpu_threads)
    return env


def dispatch(tasks: Sequence, gpus: Sequence[str], tasks_per_gpu: int,
             job: Callable, stop: threading.Event | None = None) -> list:
    if len(set(gpus)) != len(gpus) or not gpus or tasks_per_gpu < 1:
        raise ValueError('Distinct GPUs and positive slot counts are required')
    stop = stop or threading.Event()
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    def worker(gpu):
        completed = []
        while not stop.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                break
            completed.append(job(task, gpu))
        return completed
    slots = (list(gpus) * tasks_per_gpu)[:len(tasks)]
    if not slots:
        return []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in slots]
        return [value for future in futures for value in future.result()]


def _close(left: float, right: float) -> bool:
    return math.isfinite(float(left)) and math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8)


def validate_result(task: dict, result: dict) -> None:
    if not isinstance(result, dict) or result.get('schema_version') != 1:
        raise ValueError('Unsupported RQ1 result schema')
    for section in ('metadata', 'audit', 'evaluation', 'rq1', 'fingerprints', 'timing'):
        if not isinstance(result.get(section), dict):
            raise ValueError(f'Invalid RQ1 result section: {section}')
    metadata = result['metadata']
    for key in ('dataset', 'alpha', 'client_num', 'attacker_id', 'attack_method', 'fl_aggregate',
                'seed', 'trial_id', 'regime', 'detecting_method'):
        if metadata.get(key) != task[key]:
            raise ValueError(f'Result metadata mismatch: {key}')
    fixed = dict(rq1_only=True, attacker_count=1, num_epoch=50, num_local_epochs=1,
                 use_hessian=True, lambda_h=0.9, qualitative_aggregation_mode='passive', cpu_threads_per_task=15)
    if any(metadata.get(key) != value for key, value in fixed.items()):
        raise ValueError('Result does not match the frozen RQ1 settings')
    if metadata['fl_aggregation'] != aggregation_config(task):
        raise ValueError('Server aggregation or hyperparameters differ')
    accuracy, values = result['accuracy'], result['contributions']
    if len(accuracy) != 50 or not all(math.isfinite(x) and 0 <= x <= 1 for x in accuracy):
        raise ValueError('Incomplete or invalid accuracy trajectory')
    if len(values) != 6 or not all(math.isfinite(x) for x in values):
        raise ValueError('Expected six finite raw Shapley values')
    sizes, weights = result['shard_sizes'], result['shard_weights']
    if (len(sizes) != 6 or len(weights) != 6 or any(not isinstance(n, int) or n <= 0 for n in sizes)
            or any(not _close(w, n / sum(sizes)) for w, n in zip(weights, sizes))):
        raise ValueError('Shard counts/weights do not match sample-size weighting')
    audit, attacker = result['audit'], task['attacker_id']
    warmup = [0] if task['attack_method'] == 'adam_local_deviation' else []
    rounds = list(range(len(warmup), 50))
    if audit.get('public_warm_start_rounds') != warmup:
        raise ValueError('Public warm-start audit mismatch')
    if (audit['fake_call_rounds'] != rounds or audit['fake_rounds'] != rounds or audit['honest_rounds'] != []
            or len(audit['fake_sources']) != len(rounds)
            or audit['private_fit_calls_by_client'] != {str(i): 0 if i == attacker else 50 for i in range(6)}
            or audit['round_activity'] != [dict(round_idx=r, active_attackers=[] if r in warmup else [attacker]) for r in range(50)]):
        raise ValueError('Actual pure/private-training audit failed')
    evaluation = result['evaluation']
    fixed_eval = dict(shapley_scope='actual_last_round_submissions', coalitions_evaluated=64,
                      empty_coalition_utility=0.0, final_round_idx=49, server_state_rounds_before_last=49)
    if any(evaluation.get(key) != value for key, value in fixed_eval.items()):
        raise ValueError('Wrong coalition evaluation scope or state')
    total = math.fsum(values)
    if not _close(total, accuracy[-1]) or not _close(evaluation['grand_coalition_utility'], accuracy[-1]):
        raise ValueError('Grand-coalition/efficiency check failed')
    metric = result['rq1']
    ratio = values[attacker] / total if total != 0 else None
    if (not _close(metric['attacker_contribution'], values[attacker]) or not _close(metric['total_contribution'], total)
            or (ratio is None and (metric['attacker_ratio'] is not None or metric['ratio_status'] != 'zero_denominator'))
            or (ratio is not None and (metric['ratio_status'] != 'valid' or not _close(metric['attacker_ratio'], ratio)))):
        raise ValueError('Stored RQ1 ratio does not match signed raw Shapley values')
    fingerprints = result['fingerprints']
    if ([row['client_id'] for row in fingerprints['shards']] != list(range(6))
            or [row['size'] for row in fingerprints['shards']] != sizes
            or any(not row['x_sha256'] or not row['y_sha256'] for row in fingerprints['shards'])
            or not fingerprints['initial_model'] or not fingerprints['test_x_sha256'] or not fingerprints['test_y_sha256']):
        raise ValueError('Incomplete pairing fingerprints')
    for key in TABLE_COLUMNS['time'][len(META):]:
        if not math.isfinite(result['timing'][key]) or result['timing'][key] < 0:
            raise ValueError('Invalid task timing')


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge_results(tasks: Sequence[dict], campaign_dir: Path) -> dict[str, int]:
    directory = Path(campaign_dir)
    datasets = list(dict.fromkeys(task['dataset'] for task in tasks))
    expected = {task['task_id']: task for task in build_tasks(datasets)}
    if len(tasks) != len(expected) or {task['task_id'] for task in tasks} != set(expected):
        raise ValueError('Manifest must contain the complete distinct attacker-position matrix')
    if any(expected[task['task_id']] != task for task in tasks):
        raise ValueError('Task matrix differs from the approved RQ1 settings')
    # One writer, including manual --merge-only invocations during a running campaign.
    lock = directory / '.rq1_merge.lock'
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        return _merge_locked(tasks, directory, datasets)
    finally:
        os.close(lock_fd)
        lock.unlink()


def _merge_locked(tasks: Sequence[dict], directory: Path, datasets: list[str]) -> dict[str, int]:
    valid, errors = {}, {}
    for task in tasks:
        task_dir = directory / 'tasks' / task['task_id']
        try:
            status = json.loads((task_dir / 'status.json').read_text(encoding='utf-8'))
            if status['returncode'] != 0:
                raise ValueError(f"Worker exit {status['returncode']}: {status.get('error', '')}")
            result = json.loads(result_path(task, task_dir).read_text(encoding='utf-8'))
            validate_result(task, result)
            valid[task['task_id']] = result
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            errors[task['task_id']] = str(exc)
    # Do not guess which partition/initialization is authoritative if pairing fails.
    for dataset, family in ((d, f) for d in datasets for f in ATTACK_FAMILIES):
        ids = [task['task_id'] for task in tasks if task['dataset'] == dataset and task['attack_family'] == family and task['task_id'] in valid]
        fingerprints = {json.dumps(valid[identity]['fingerprints'], sort_keys=True) for identity in ids}
        if len(fingerprints) > 1:
            for identity in ids:
                errors[identity] = 'Pairing mismatch: data shards/test set/initial model differ within dataset/attack family'
                del valid[identity]
    tables = {dataset: {name: [] for name in TABLE_COLUMNS} for dataset in datasets}
    for task in tasks:
        base = {key: task[key] for key in META}
        table = tables[task['dataset']]
        task_dir = directory / 'tasks' / task['task_id']
        audit_row = dict(base, status='complete' if task['task_id'] in valid else 'failed_or_missing',
                         reason=errors.get(task['task_id'], ''), result_path=str(result_path(task, task_dir)),
                         log_path=str(task_dir / 'stdout.log'))
        if task['task_id'] not in valid:
            table['audit'].append(audit_row)
            continue
        result = valid[task['task_id']]
        audit, attacker = result['audit'], task['attacker_id']
        audit_row.update(fake_count=len(audit['fake_rounds']), honest_count=0, public_warm_start_count=len(audit['public_warm_start_rounds']), attacker_private_fit_count=0,
                         fake_sources=json.dumps(dict(Counter(audit['fake_sources'])), sort_keys=True))
        table['audit'].append(audit_row)
        values = result['contributions']
        table['rq1_detail'].append(dict(base, **result['rq1'], final_accuracy=result['accuracy'][-1],
            honest_contribution_mean=statistics.mean(value for i, value in enumerate(values) if i != attacker),
            attacker_sample_count=result['shard_sizes'][attacker], attacker_data_fraction=result['shard_weights'][attacker],
            fake_count=len(audit['fake_rounds']), honest_count=0, public_warm_start_count=len(audit['public_warm_start_rounds']), cpu_threads_per_task=result['metadata']['cpu_threads_per_task']))
        for i, value in enumerate(values):
            table['contributions'].append(dict(base, client_id=i, is_attacker=int(i == attacker),
                shard_size=result['shard_sizes'][i], shard_weight=result['shard_weights'][i], shapley_value=value,
                scope='actual_last_round_submissions'))
        table['accuracy'].extend(dict(base, round_idx=i, accuracy=value) for i, value in enumerate(result['accuracy']))
        table['time'].append(dict(base, **result['timing']))
    for dataset, table in tables.items():
        groups = {}
        for task in tasks:
            if task['dataset'] == dataset:
                key = tuple(task[column] for column in GROUP)
                groups.setdefault(key, [])
        for row in table['rq1_detail']:
            groups[tuple(row[column] for column in GROUP)].append(row)
        for key, observations in groups.items():
            ids = sorted(row['attacker_id'] for row in observations)
            group = dict(zip(GROUP, key))
            group.update(replicate_axis='attacker_position', n_expected=6, n_completed=len(ids),
                         status='complete' if len(ids) == 6 else 'partial' if ids else 'missing',
                         attacker_ids_completed=json.dumps(ids), attacker_ids_missing=json.dumps(sorted(set(range(6)) - set(ids))),
                         std_ddof=1)
            for metric in METRICS:
                numbers = [row[metric] for row in observations if row[metric] is not None]
                group.update({f'{metric}_mean': statistics.mean(numbers) if numbers else None,
                              f'{metric}_std': statistics.stdev(numbers) if len(numbers) > 1 else None,
                              f'{metric}_n': len(numbers)})
            table['rq1_mean_std'].append(group)
        output = directory / 'merged' / dataset
        output.mkdir(parents=True, exist_ok=True)
        for name, rows in table.items():
            _write_csv(output / f'{name}.csv', TABLE_COLUMNS[name], rows)
    return dict(complete=len(valid), failed=len(errors))


@contextmanager
def gpu_locks(gpus: Sequence[str]):
    import fcntl  # Only the explicitly requested Linux run reaches this code.
    handles = []
    try:
        for gpu in sorted(gpus):
            # Coordinate with the existing campaigns without importing their code.
            path = Path(tempfile.gettempdir()) / f'flce_multi_{os.getuid()}_gpu_{gpu}.lock'
            handle = path.open('a')
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'GPU {gpu} is reserved by another campaign') from exc
        yield
    finally:
        for handle in handles:
            handle.close()


def run_campaign(args: argparse.Namespace, tasks: list[dict], directory: Path) -> int:
    if os.name != 'posix':
        raise ValueError('GPU execution requires Linux; use --dry-run for local previews')
    stop = threading.Event()
    with gpu_locks(args.gpus):
        probe = subprocess.run([args.python, '-c',
            'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4; '
            'print([torch.cuda.get_device_name(i) for i in range(4)])'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(args.gpus)),
            check=True, text=True, capture_output=True)
        print('GPU preflight:', probe.stdout.strip(), flush=True)
        directory.mkdir(parents=True, exist_ok=False)
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
        commands = {task['task_id']: command_for(task, args.python, ROOT, directory / 'tasks' / task['task_id']) for task in tasks}
        _write_json(directory / 'manifest.json', dict(schema_version=1, experiment='rq1_single_adam_dkfr_fl_aggregation',
            tasks=tasks, settings=vars(args), code_sha256=hashes, commands=commands,
            result_paths={task['task_id']: str(result_path(task, directory / 'tasks' / task['task_id'])) for task in tasks}))
        def job(task, gpu):
            task_dir = directory / 'tasks' / task['task_id']
            task_dir.mkdir(parents=True, exist_ok=False)
            status = dict(task_id=task['task_id'], gpu=gpu, returncode=1)
            try:
                with (task_dir / 'stdout.log').open('w', encoding='utf-8') as log:
                    print(f"[START gpu={gpu}] {task['task_id']}", flush=True)
                    process = subprocess.Popen(commands[task['task_id']], cwd=ROOT,
                        env=worker_environment(gpu, args.cpu_threads), stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    while process.poll() is None:
                        if stop.wait(0.5):
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            break
                    status['returncode'] = process.returncode
            except Exception as exc:
                status['error'] = str(exc)
            _write_json(task_dir / 'status.json', status)
            print(f"[END exit={status['returncode']}] {task['task_id']}", flush=True)
            return status
        previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            dispatch(tasks, args.gpus, args.tasks_per_gpu, job, stop)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        counts = merge_results(tasks, directory)
        print(f'Merged {counts}; dataset CSVs: {directory / "merged"}', flush=True)
        return int(bool(counts['failed']))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--datasets', nargs='+', choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument('--partition', choices=['first'], default='first')
    parser.add_argument('--client-num', type=int, default=6)
    parser.add_argument('--attacker-sweep', nargs='+', type=int, default=list(range(6)))
    parser.add_argument('--fl-methods', nargs='+', choices=FL_METHODS, default=list(FL_METHODS))
    parser.add_argument('--attack-families', nargs='+', choices=ATTACK_FAMILIES, default=list(ATTACK_FAMILIES))
    parser.add_argument('--regime', choices=['pure'], default='pure')
    parser.add_argument('--detecting-method', choices=['shapley'], default='shapley')
    parser.add_argument('--use-last-round-sv', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--qualitative-aggregation-mode', choices=['passive'], default='passive')
    parser.add_argument('--num-epoch', type=int, default=50)
    parser.add_argument('--num-local-epochs', type=int, default=1)
    parser.add_argument('--random-round', type=int, default=10)
    parser.add_argument('--use-hessian', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lambda-h', type=float, default=0.9)
    for name, default in SERVER.items():
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=default)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial-id', type=int, default=0)
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '2', '3'])
    parser.add_argument('--tasks-per-gpu', type=int, default=15)
    parser.add_argument('--cpu-threads', type=int, default=15)
    parser.add_argument('--output-root', default=str(ROOT / 'Results_rq1_adam_dkfr'))
    parser.add_argument('--print-all-commands', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--merge-only', metavar='CAMPAIGN_DIRECTORY')
    args = parser.parse_args(argv)
    fixed = dict(client_num=6, num_epoch=50, num_local_epochs=1, random_round=10, use_hessian=True,
                 lambda_h=0.9, seed=42, trial_id=0, tasks_per_gpu=15, cpu_threads=15, use_last_round_sv=True, **SERVER)
    if any(getattr(args, name) != value for name, value in fixed.items()):
        raise ValueError('RQ1 settings are frozen to N=6, 50 pure rounds, seed=42, trial=0, 15 tasks/GPU and 15 CPU threads')
    if args.attacker_sweep != list(range(6)) or args.fl_methods != list(FL_METHODS) or args.attack_families != list(ATTACK_FAMILIES):
        raise ValueError('Use the complete six-position, three-FL, two-attack sweep in its declared order')
    if args.gpus != ['0', '1', '2', '3']:
        raise ValueError('This experiment uses GPUs 0,1,2,3')
    build_tasks(args.datasets)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.merge_only:
            directory = Path(args.merge_only).resolve()
            manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('experiment') != 'rq1_single_adam_dkfr_fl_aggregation':
                raise ValueError('Not a single-attacker RQ1 manifest')
            outcome = merge_results(manifest['tasks'], directory)
            print(outcome)
            return int(bool(outcome['failed']))
        tasks = build_tasks(args.datasets)
        name = 'campaign_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        directory = Path(args.output_root).resolve() / (name if args.run else 'PREVIEW')
        print(f'{len(tasks)} tasks | GPUs=0,1,2,3 | 15 tasks/GPU | 15 CPU threads/task | max 60 concurrent')
        print('Original single --attacker_id path; N=6; Adam=1 public warm start +49 fake; DKFR=50 fake; no private attacker fits; exact last-round Shapley.')
        print('Seed=42, trial=0; sweep all attacker positions 0..5; averages describe attacker positions.')
        for dataset in args.datasets:
            print(f"{dataset}: alpha={PROFILES[dataset]['alpha']}, 36 tasks")
        for task in tasks if args.print_all_commands else tasks[:3]:
            print('COMMAND ' + shlex.join(command_for(task, args.python, ROOT, directory / 'tasks' / task['task_id'])))
        print(f'Dataset CSVs: {directory}/merged/<dataset>/')
        if not args.run:
            print('DRY RUN: no directories created; no training or GPU probe started.')
            return 0
        return run_campaign(args, tasks, directory)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

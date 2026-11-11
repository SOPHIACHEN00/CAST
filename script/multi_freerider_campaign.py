#!/usr/bin/env python3
"""Three balanced attacker groups, four-GPU queue and dataset CSVs.

Standard library only. Without --run this prints a plan and creates no files.
Each worker writes JSON/logs; only the coordinator writes public CSVs.
Defaults preserve the original 216 tasks; Uniform + Shapley selects 108 tasks.
"""
import argparse
from collections.abc import Callable, Sequence
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
import random
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
GROUP_IDS = (0, 1, 2)
FIXED_SEED = 42
DEFAULT_SELECTION_SEED = 2026
PROFILES = {
    'adult': dict(alpha=1000, arima_var='0.005', arima_err='0.08', moirai_lr='0.003', moirai_err='0.12'),
    'mnist': dict(alpha=10, arima_var='0.05', arima_err='0.05', moirai_lr='0.003', moirai_err='0.06'),
    'imdb': dict(alpha=10, arima_var='0.10', arima_err='0.015', moirai_lr='0.01', moirai_err='0.025'),
}
PARTITION_ALPHAS = {
    'first': {dataset: profile['alpha'] for dataset, profile in PROFILES.items()},
    'uniform': {'adult': 20000000, 'mnist': 10000000, 'imdb': 200000},
}
DETECTORS = ('shapley', 'std_dagmm')
ATTACK_FAMILIES = ('moirai_online', 'arima_adaptive', 'moirai', 'arima')
DEFAULT_ATTACKS = ('moirai_online', 'arima_adaptive')
META_COLUMNS = ['task_id', 'dataset', 'alpha', 'client_num', 'attacker_count', 'attacker_ids',
                'attack_method', 'regime', 'detecting_method', 'trial_id', 'seed', 'num_epoch',
                'fake_rounds_per_attacker', 'real_rounds_per_attacker']
CSV_META_COLUMNS = META_COLUMNS + ['attacker_selection_seed', 'replicate_axis']
CSV_COLUMNS = {
    'accuracy': ['round_idx', 'accuracy'],
    'contributions': ['client_id', 'is_attacker', 'shard_size', 'shard_weight', 'shapley_value', 'scope'],
    'detection': ['round_idx', 'client_id', 'is_designated_attacker', 'is_fake', 'energy',
                  'energy_rank', 'energy_threshold', 'predicted_suspicious', 'roc_auc',
                  'tp', 'fp', 'fn', 'tn', 'tpr', 'fpr'],
    'summary': ['final_accuracy', 'attacker_sample_count', 'attacker_data_fraction',
                'attacker_shapley_sum', 'attacker_shapley_mean', 'honest_shapley_mean',
                'mean_round_auc', 'std_round_auc', 'valid_auc_rounds', 'mean_average_precision',
                'mean_attacker_rank', 'top1_hit_rate', 'tp', 'fp', 'fn', 'tn', 'tpr', 'fpr',
                'fpr_fake_rounds'],
    'time': ['fl_sec', 'evaluation_sec', 'attack_host_sec', 'total_sec'],
    'audit': ['status', 'reason', 'actual_attacker_counts', 'log_path'],
}
GROUP_COLUMNS = [key for key in CSV_META_COLUMNS if key not in ('task_id', 'trial_id', 'attacker_ids')]
AGGREGATE_COLUMNS = GROUP_COLUMNS + [
    'n_expected', 'n_planned', 'n_completed', 'status', 'group_ids_expected',
    'group_ids_planned', 'group_ids_completed', 'group_ids_missing', 'std_ddof',
] + [f'{metric}_{stat}' for metric in CSV_COLUMNS['summary'] for stat in ('mean', 'std', 'n')]


def selection_design(datasets: Sequence[str], attacker_selection_seed: int) -> dict:
    """Local RNG only: balance client coverage, then nest k=1/2/3 within each group."""
    if not isinstance(attacker_selection_seed, int) or attacker_selection_seed < 0:
        raise ValueError('Attacker selection seed must be a nonnegative integer')
    design = dict(name='balanced_nested_attacker_groups_v1', attacker_selection_seed=attacker_selection_seed,
                  training_seed=FIXED_SEED, trial_id_axis='attacker_assignment',
                  expected_group_ids=list(GROUP_IDS), std_ddof=1, datasets={})
    for dataset in datasets:
        order = list(range(8))
        random.Random(f'{attacker_selection_seed}:{dataset}').shuffle(order)
        groups = []
        for trial in GROUP_IDS:
            triple = [order[(3 * trial + j) % 8] for j in range(3)]
            groups.append(dict(trial_id=trial, attacker_ids_by_count={str(k): triple[:k] for k in (1, 2, 3)}))
        design['datasets'][dataset] = dict(permutation=order, groups=groups)
    return design


def build_tasks(datasets=('adult', 'mnist', 'imdb'), trials=GROUP_IDS, seed=FIXED_SEED,
                attacker_selection_seed=DEFAULT_SELECTION_SEED, partition='first',
                detectors=DETECTORS, attacks=DEFAULT_ATTACKS) -> list[dict]:
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - PROFILES.keys():
        raise ValueError('Choose distinct datasets from adult, mnist, imdb')
    if not trials or len(set(trials)) != len(trials) or set(trials) - set(GROUP_IDS):
        raise ValueError('Group/trial IDs must be a distinct subset of 0, 1, 2')
    if seed != FIXED_SEED:
        raise ValueError('Data partition, training and detector seed must remain fixed at 42')
    if partition not in PARTITION_ALPHAS:
        raise ValueError('Partition must be first or uniform')
    if not detectors or len(set(detectors)) != len(detectors) or set(detectors) - set(DETECTORS):
        raise ValueError('Choose distinct detectors from shapley, std_dagmm')
    design = selection_design(datasets, attacker_selection_seed)
    if not attacks or len(set(attacks)) != len(attacks) or set(attacks) - set(ATTACK_FAMILIES):
        raise ValueError('Choose distinct supported attack families')
    tasks = []
    for trial in trials:
        for dataset in datasets:
            suffix = '' if dataset == 'adult' else '_mnist_imdb_layerwise'
            for count in (1, 2, 3):
                for family in attacks:
                    for regime in ('pure', 'lazy'):
                        for detector in detectors:
                            ids = design['datasets'][dataset]['groups'][trial]['attacker_ids_by_count'][str(count)][:]
                            task_id = f'{dataset}_{family}_{regime}_k{count}_ids' + '-'.join(map(str, ids))
                            task_id += f'_{detector}_trial{trial}_seed{seed}'
                            if partition == 'uniform':
                                task_id += '_uniform'
                            tasks.append(dict(task_id=task_id, dataset=dataset, alpha=PARTITION_ALPHAS[partition][dataset],
                                client_num=8, attacker_count=count, attacker_ids=ids, attack_method=family + suffix,
                                regime=regime, detecting_method=detector, trial_id=trial, seed=seed,
                                num_epoch=50, num_local_epochs=1,
                                fake_rounds_per_attacker=50 if regime == 'pure' else 41,
                                real_rounds_per_attacker=0 if regime == 'pure' else 9))
    return tasks


def command_for(task: dict, python_bin: str, root: Path, task_dir: Path) -> list[str]:
    profile = PROFILES[task['dataset']]
    argv = [str(python_bin), '-u', str(Path(root) / 'FLCE_updated.py')]
    for key in ('dataset', 'alpha', 'client_num', 'attack_method', 'detecting_method',
                'trial_id', 'seed', 'num_epoch', 'num_local_epochs'):
        argv += ['--' + key, str(task[key])]
    argv += ['--attacker_ids', *map(str, task['attacker_ids']), '--results_root', str(task_dir),
             '--use_attack', '--use_hessian', '--lambda_h', '0.9', '--random_round', '10',
             '--qualitative_aggregation_mode', 'passive', '--use_last_round_sv',
             '--std_dagmm_seed', str(task['seed'])]
    if task['regime'] == 'lazy':
        argv += ['--use_lazy', '--lazy_schedule', 'fixed', '--lazy_honest_gap', '6',
                 '--lazy_honest_phase', '0', '--lazy_fake_policy', 'attack']
    if task['attack_method'].startswith('arima'):
        argv += ['--arima_var_window', '3', '--arima_var_threshold', profile['arima_var']]
    if task['attack_method'].startswith('arima_adaptive'):
        argv += ['--arima_err_threshold', profile['arima_err'], '--arima_min_rounds_after_start', '4',
                 '--arima_retrain_interval', '15', '--arima_err_ema_beta', '0.9', '--arima_trim_history']
    elif task['attack_method'].startswith('moirai_online'):
        argv += ['--moirai_online_tuning', '--moirai_tune_lr', profile['moirai_lr'],
                 '--moirai_tune_steps', '1', '--moirai_tune_start_round', '10', '--moirai_adapt_interval', '4',
                 '--moirai_residual_decay', '0.95', '--moirai_residual_clip', '5.0',
                 '--moirai_err_ema_beta', '0.9', '--moirai_err_threshold', profile['moirai_err'],
                 '--moirai_min_rounds_after_start', '4', '--moirai_trim_history']
    if task['dataset'] != 'adult':
        argv += ['--layerwise_selection_metric', 'normalized_rms', '--layerwise_selection_fraction', '0.5',
                 '--layerwise_pca_threshold', '512', '--layerwise_pca_components', '8']
    return argv


def worker_environment(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
               FLCE_CPU_THREADS_PER_TASK=str(cpu_threads))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env[key] = str(cpu_threads)
    return env


def dispatch(tasks: Sequence, gpus: Sequence[str], tasks_per_gpu: int,
             job: Callable, stop: threading.Event | None = None) -> list:
    """One shared task queue; each worker owns exactly one GPU slot."""
    if not gpus or len(set(gpus)) != len(gpus) or tasks_per_gpu < 1:
        raise ValueError('Distinct GPUs and positive tasks_per_gpu are required')
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
    # Interleave GPU slots so even a single-group preview/run uses all four devices.
    slots = (list(gpus) * tasks_per_gpu)[:len(tasks)]
    if not slots:
        return []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in slots]
        return [result for future in futures for result in future.result()]


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_result(task: dict, result: dict) -> None:
    if result.get('schema_version') != 1:
        raise ValueError('Unsupported result schema')
    for key, expected in task.items():
        if key != 'task_id' and result['metadata'].get(key) != expected:
            raise ValueError(f'Result metadata mismatch: {key}')
    if len(result['accuracy']) != 50 or not all(math.isfinite(v) for v in result['accuracy']):
        raise ValueError('Incomplete/nonfinite accuracy trajectory')
    sizes, weights = result['shard_sizes'], result['shard_weights']
    if (len(sizes) != 8 or len(weights) != 8
            or not all(isinstance(n, int) and n >= 0 for n in sizes) or sum(sizes) <= 0
            or not all(math.isfinite(w) and w >= 0 for w in weights)
            or not all(math.isclose(w, n / sum(sizes), rel_tol=1e-7, abs_tol=1e-9)
                       for w, n in zip(weights, sizes))):
        raise ValueError('Invalid shard sizes/weights for attacker data coverage')
    audit = result['attackers']
    if sorted(a['client_id'] for a in audit) != sorted(task['attacker_ids']):
        raise ValueError('Attacker audit IDs do not match the task')
    if any(a['fake_count'] != task['fake_rounds_per_attacker'] or
           a['real_count'] != task['real_rounds_per_attacker'] for a in audit):
        raise ValueError('Actual pure/lazy counts do not match the task')
    if task['detecting_method'] == 'shapley':
        if len(result['contributions']) != 8 or not all(math.isfinite(v) for v in result['contributions']):
            raise ValueError('Incomplete/nonfinite Shapley values')
    else:
        if len(result['detection']) != 50 or result['evaluation']['valid_auc_rounds'] != task['fake_rounds_per_attacker']:
            raise ValueError('Incomplete STD-DAGMM rounds/AUC coverage')
        for r, record in enumerate(result['detection']):
            if record['round_idx'] != r or set(record['client_scores']) != set(map(str, range(8))):
                raise ValueError('Incomplete STD-DAGMM energies')
            if not all(math.isfinite(v) for v in record['client_scores'].values()):
                raise ValueError('Nonfinite STD-DAGMM energies')
            real = task['regime'] == 'lazy' and r % 6 == 0
            expected = {str(i): int(not real and i in task['attacker_ids']) for i in range(8)}
            if record['ground_truth_labels'] != expected or (record['roc_auc'] is None) != real:
                raise ValueError('STD-DAGMM labels/AUC do not match actual fake rounds')


def aggregate_summaries(planned: Sequence[dict], completed: Sequence[dict]) -> list[dict]:
    """Descriptive sample SD across assignments; missing observations never become zero."""
    groups = {}
    for row in planned:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        if key not in groups:
            groups[key] = dict(base={column: row[column] for column in GROUP_COLUMNS}, planned=[], completed=[])
        groups[key]['planned'].append(row['trial_id'])
    for row in completed:
        groups[tuple(row[column] for column in GROUP_COLUMNS)]['completed'].append(row)
    output = []
    for group in groups.values():
        observations = group['completed']
        ids = sorted(row['trial_id'] for row in observations)
        missing = sorted(set(GROUP_IDS) - set(ids))
        row = dict(group['base'], n_expected=len(GROUP_IDS), n_planned=len(group['planned']),
                   n_completed=len(ids), status='complete' if not missing else 'partial' if ids else 'missing',
                   std_ddof=1)
        for label, values in [('expected', GROUP_IDS), ('planned', sorted(group['planned'])),
                              ('completed', ids), ('missing', missing)]:
            row[f'group_ids_{label}'] = json.dumps(values, separators=(',', ':'))
        for metric in CSV_COLUMNS['summary']:
            values = [item[metric] for item in observations
                      if isinstance(item.get(metric), (int, float)) and math.isfinite(item[metric])]
            row[f'{metric}_n'] = len(values)
            row[f'{metric}_mean'] = statistics.mean(values) if values else None
            row[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else None
        output.append(row)
    return output


def merge_results(tasks: Sequence[dict], campaign_dir: Path,
                  attacker_selection_seed: int = DEFAULT_SELECTION_SEED,
                  partition: str = 'first', detectors: Sequence[str] = DETECTORS,
                  attacks: Sequence[str] = DEFAULT_ATTACKS) -> dict[str, int]:
    campaign_dir = Path(campaign_dir)
    datasets = list(dict.fromkeys(t['dataset'] for t in tasks))
    expected = {t['task_id']: t for t in build_tasks(datasets, attacker_selection_seed=attacker_selection_seed,
                                                    partition=partition, detectors=detectors, attacks=attacks)}
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate tasks would count an assignment more than once')
    if any(expected.get(t['task_id']) != t for t in tasks):
        raise ValueError('Task assignment does not match the fixed-seed balanced selection design')
    tables = {dataset: {name: [] for name in CSV_COLUMNS} for dataset in datasets}
    planned = {dataset: [] for dataset in datasets}
    counts = dict(complete=0, failed=0)
    for task in tasks:
        table = tables[task['dataset']]
        base = {key: task[key] for key in META_COLUMNS}
        base['attacker_ids'] = json.dumps(task['attacker_ids'], separators=(',', ':'))
        base.update(attacker_selection_seed=attacker_selection_seed, replicate_axis='attacker_assignment')
        planned[task['dataset']].append(base)
        task_dir = campaign_dir / 'tasks' / task['task_id']
        try:
            status = json.loads((task_dir / 'status.json').read_text(encoding='utf-8'))
            if status['returncode'] != 0:
                raise ValueError(f"Worker exit {status['returncode']}: {status.get('error', '')}")
            result = json.loads((task_dir / 'result.json').read_text(encoding='utf-8'))
            validate_result(task, result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            table['audit'].append(dict(base, status='failed_or_missing', reason=str(exc),
                                       log_path=str(task_dir / 'stdout.log')))
            counts['failed'] += 1
            continue
        counts['complete'] += 1
        table['audit'].append(dict(base, status='complete', reason='',
            actual_attacker_counts=json.dumps([
                {key: a[key] for key in ('client_id', 'fake_count', 'real_count')}
                for a in result['attackers']], separators=(',', ':')),
            log_path=str(task_dir / 'stdout.log')))
        for r, accuracy in enumerate(result['accuracy']):
            table['accuracy'].append(dict(base, round_idx=r, accuracy=accuracy))
        summary = dict(base, final_accuracy=result['accuracy'][-1],
                       attacker_sample_count=sum(result['shard_sizes'][i] for i in task['attacker_ids']),
                       attacker_data_fraction=sum(result['shard_weights'][i] for i in task['attacker_ids']))
        if result['contributions']:
            for i, value in enumerate(result['contributions']):
                table['contributions'].append(dict(base, client_id=i, is_attacker=int(i in task['attacker_ids']),
                    shard_size=result['shard_sizes'][i], shard_weight=result['shard_weights'][i],
                    shapley_value=value, scope='actual_last_round_submissions'))
            attack_values = [result['contributions'][i] for i in task['attacker_ids']]
            honest_values = [v for i, v in enumerate(result['contributions']) if i not in task['attacker_ids']]
            summary.update(attacker_shapley_sum=sum(attack_values),
                           attacker_shapley_mean=sum(attack_values)/len(attack_values),
                           honest_shapley_mean=sum(honest_values)/len(honest_values))
        for record in result['detection']:
            for i in range(8):
                table['detection'].append(dict(base, round_idx=record['round_idx'], client_id=i,
                    is_designated_attacker=int(i in task['attacker_ids']),
                    is_fake=record['ground_truth_labels'][str(i)], energy=record['client_scores'][str(i)],
                    energy_rank=record['energy_ranks'][str(i)], energy_threshold=record['energy_threshold'],
                    predicted_suspicious=int(i in record['suspicious_clients']),
                    **{key: record.get(key) for key in ('roc_auc', 'tp', 'fp', 'fn', 'tn', 'tpr', 'fpr')}))
        summary.update({k: v for k, v in result['evaluation'].items() if k in CSV_COLUMNS['summary']})
        table['summary'].append(summary)
        table['time'].append(dict(base, **result['timing']))
    for dataset, tables_for_dataset in tables.items():
        tables_for_dataset['summary_mean_std'] = aggregate_summaries(planned[dataset], tables_for_dataset['summary'])
        directory = campaign_dir / 'merged' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in tables_for_dataset.items():
            target = directory / f'{name}.csv'
            temporary = target.with_suffix('.csv.tmp')
            with temporary.open('w', newline='', encoding='utf-8') as stream:
                columns = AGGREGATE_COLUMNS if name == 'summary_mean_std' else CSV_META_COLUMNS + CSV_COLUMNS[name]
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(target)
    return counts


@contextmanager
def gpu_locks(gpus):
    import fcntl  # Server launcher is Linux; preview and merge work on any OS.
    locks = []
    try:
        for gpu in sorted(gpus):
            path = Path(tempfile.gettempdir()) / f'flce_multi_{os.getuid()}_gpu_{gpu}.lock'
            lock = path.open('a')
            locks.append(lock)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'GPU {gpu} is already assigned to another multi-free-rider campaign') from exc
        yield
    finally:
        for lock in locks:
            lock.close()


def run_campaign(args: argparse.Namespace, tasks: Sequence[dict], campaign_dir: Path) -> int:
    if os.name != 'posix':
        raise ValueError('Real GPU runs require the Linux server; use --dry-run locally')
    stop = threading.Event()
    with gpu_locks(args.gpus):
        probe = subprocess.run([args.python, '-c',
            'import torch; assert torch.cuda.is_available(); '
            f'assert torch.cuda.device_count() == {len(args.gpus)}; '
            'print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(args.gpus)), check=True, text=True, capture_output=True)
        print('GPU preflight:', probe.stdout.strip(), flush=True)
        campaign_dir.mkdir(parents=True, exist_ok=False)
        hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT / 'FLCE_updated.py', Path(__file__), ROOT / 'module/method/multi_freerider.py',
            ROOT / 'module/method/attack_updated_nonlayerwise.py', ROOT / 'module/method/attack_updated_layerwise.py',
            ROOT / 'module/method/quantitative_detecting/coalition_value.py',
            ROOT / 'module/method/quantitative_detecting/shapley_value.py',
            ROOT / 'module/method/qualitative_detecting/std_dagmm.py',
            ROOT / 'module/method/qualitative_detecting/std_dagmm_old_v1.py',
            ROOT / 'script/run_multi_freerider.sh', ROOT / 'script/run_uniform_freerider_rq1_rq2.sh')}
        write_json(campaign_dir / 'manifest.json', dict(tasks=tasks, settings=vars(args), code_sha256=hashes,
            selection_design=selection_design(args.datasets, args.attacker_selection_seed),
            commands={t['task_id']: command_for(t, args.python, ROOT, campaign_dir / 'tasks' / t['task_id']) for t in tasks}))
        def job(task, gpu):
            directory = campaign_dir / 'tasks' / task['task_id']
            directory.mkdir(parents=True)
            status = dict(task_id=task['task_id'], gpu=gpu, returncode=1)
            try:
                with (directory / 'stdout.log').open('w', encoding='utf-8') as log:
                    command = command_for(task, args.python, ROOT, directory)
                    print(f"[START gpu={gpu}] {task['task_id']}", flush=True)
                    process = subprocess.Popen(command, cwd=ROOT, env=worker_environment(gpu, args.cpu_threads),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
            write_json(directory / 'status.json', status)
            print(f"[END exit={status['returncode']}] {task['task_id']}", flush=True)
            return status
        previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            dispatch(tasks, args.gpus, args.tasks_per_gpu, job, stop=stop)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        outcome = merge_results(tasks, campaign_dir, args.attacker_selection_seed, args.partition, args.detectors, args.attacks)
        print(f'Merged: {outcome}; CSVs: {campaign_dir / "merged"}', flush=True)
        return 1 if outcome['failed'] or stop.is_set() else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=list(PROFILES))
    parser.add_argument('--partition', choices=tuple(PARTITION_ALPHAS), default='first')
    parser.add_argument('--detectors', nargs='+', choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument('--attacks', nargs='+', choices=ATTACK_FAMILIES, default=list(DEFAULT_ATTACKS))
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '2', '3'])
    parser.add_argument('--tasks-per-gpu', type=int, default=15)
    parser.add_argument('--cpu-threads', type=int, default=15)
    parser.add_argument('--trials', nargs='+', type=int, default=list(GROUP_IDS),
                        help='Attacker assignment group IDs (0/1/2), not training-seed repetitions')
    parser.add_argument('--seed', type=int, default=FIXED_SEED, help='Fixed data/training/detector seed: 42 only')
    parser.add_argument('--attacker-selection-seed', type=int, default=DEFAULT_SELECTION_SEED,
                        help='Independent RNG used only to choose attacker IDs')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output-root', help='Default separates first-column and Uniform campaigns')
    parser.add_argument('--print-limit', type=int, default=6)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--merge-only', metavar='CAMPAIGN_DIRECTORY')
    args = parser.parse_args(argv)
    if args.merge_only:
        directory = Path(args.merge_only).resolve()
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        selection_seed = manifest.get('settings', {}).get('attacker_selection_seed')
        if selection_seed is None:
            parser.error('This manifest predates balanced assignments; use its backed-up launcher to merge it')
        outcome = merge_results(manifest['tasks'], directory, selection_seed,
                                manifest['settings'].get('partition', 'first'),
                                manifest['settings'].get('detectors', DETECTORS),
                                manifest['settings'].get('attacks', DEFAULT_ATTACKS))
        print(outcome)
        return int(bool(outcome['failed']))
    if args.tasks_per_gpu < 1 or args.cpu_threads < 1 or args.print_limit < 0:
        parser.error('Resource counts must be positive and print-limit nonnegative')
    if len(set(args.gpus)) != len(args.gpus) or not all(g.isdigit() for g in args.gpus):
        parser.error('GPU IDs must be distinct nonnegative integer indices')
    try:
        tasks = build_tasks(args.datasets, args.trials, args.seed, args.attacker_selection_seed,
                            args.partition, args.detectors, args.attacks)
    except ValueError as exc:
        parser.error(str(exc))
    name = 'campaign_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if args.output_root is None:
        suffix = '_uniform' if args.partition == 'uniform' else ''
        args.output_root = str(ROOT / ('Results_multi_freerider' + suffix))
    directory = Path(args.output_root).resolve() / name
    concurrency = len(args.gpus) * args.tasks_per_gpu
    print(f'{len(tasks)} tasks | GPUs={args.gpus} | {args.tasks_per_gpu} tasks/GPU | '
          f'{args.cpu_threads} CPU threads/task | maximum {concurrency} concurrent tasks')
    print('N=8 including attackers; k=1,2,3; pure=50 fake; lazy=9 real + 41 fake (fixed 1 real/5 fake).')
    print(f'Partition={args.partition}; detectors={args.detectors}')
    print(f'Fixed data/training/detector seed=42; attacker selection_seed={args.attacker_selection_seed}; '
          f'trial_id=assignment group {args.trials}.')
    design = selection_design(args.datasets, args.attacker_selection_seed)
    for dataset in args.datasets:
        print(f"{dataset}: alpha={PARTITION_ALPHAS[args.partition][dataset]}, {sum(t['dataset'] == dataset for t in tasks)} tasks")
        for trial in args.trials:
            ids = design['datasets'][dataset]['groups'][trial]['attacker_ids_by_count']
            print(f"  group {trial}: k1={ids['1']}, k2={ids['2']}, k3={ids['3']}")
    for task in tasks[:args.print_limit]:
        print(shlex.join(command_for(task, args.python, ROOT, directory / 'tasks' / task['task_id'])))
    print(f'Public CSVs: {directory}/merged/<dataset>/{{{",".join(CSV_COLUMNS)}}}.csv')
    print('Plus summary_mean_std.csv: three assignment groups; sample SD (ddof=1); missing groups are flagged.')
    if not args.run:
        print('DRY RUN: no directories created; no experiments started.')
        return 0
    return run_campaign(args, tasks, directory)


if __name__ == '__main__':
    raise SystemExit(main())

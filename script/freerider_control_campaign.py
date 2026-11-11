#!/usr/bin/env python3
"""27 RQ2 controls; default is a file-free preview. Never rerun attacked trajectories."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading

from multi_freerider_campaign import PARTITION_ALPHAS, selection_design, dispatch, worker_environment, gpu_locks, write_json

ROOT = Path(__file__).resolve().parents[1]


def tasks_for(datasets, partition: str = 'first') -> list[dict]:
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - {'adult', 'mnist', 'imdb'}:
        raise ValueError('Choose distinct adult/mnist/imdb datasets')
    if partition not in PARTITION_ALPHAS:
        raise ValueError('Partition must be first or uniform')
    plan = selection_design(datasets, 2026)
    tasks = []
    for dataset in datasets:
        for group in plan['datasets'][dataset]['groups']:
            for k in (1, 2, 3):
                ids = sorted(group['attacker_ids_by_count'][str(k)])
                task_id = f"{dataset}_group{group['trial_id']}_k{k}_exclude"+'-'.join(map(str, ids))
                if partition == 'uniform':
                    task_id += '_uniform'
                tasks.append(dict(task_id=task_id,
                    dataset=dataset, alpha=PARTITION_ALPHAS[partition][dataset],
                    trial_id=group['trial_id'], attacker_count=k, excluded_ids=ids))
    return tasks


def command_for(task, python, directory):
    return [python, '-u', str(ROOT/'FLCE_updated.py'), '--dataset', task['dataset'],
        '--alpha', str(task['alpha']), '--client_num', '8', '--seed', '42',
        '--trial_id', str(task['trial_id']), '--num_epoch', '50', '--num_local_epochs', '1',
        '--attack_method', 'random', '--detecting_method', 'shapley', '--use_last_round_sv',
        '--qualitative_aggregation_mode', 'passive', '--use_hessian', '--lambda_h', '0.9',
        '--control_excluded_ids', *map(str, task['excluded_ids']), '--results_root', str(directory)]


def merge(tasks, directory, partition: str = 'first') -> int:
    datasets = list(dict.fromkeys(t['dataset'] for t in tasks))
    expected = {t['task_id']: t for t in tasks_for(datasets, partition)}
    if len(tasks) != len(expected) or {t['task_id']: t for t in tasks} != expected:
        raise ValueError('Control manifest must contain all three groups at the selected partition')
    tables = {d: {'control_contributions': [], 'control_accuracy': [], 'control_audit': []}
              for d in dict.fromkeys(t['dataset'] for t in tasks)}
    failed = 0
    for t in tasks:
        path = directory/'tasks'/t['task_id']
        base = dict(t, excluded_ids=json.dumps(t['excluded_ids']), seed=42, original_client_num=8,
                    control_client_num=8-t['attacker_count'], attacker_selection_seed=2026)
        try:
            status = json.loads((path/'status.json').read_text())
            if status['returncode'] != 0:
                raise ValueError(f"Worker exit {status['returncode']}")
            result = json.loads((path/'control_result.json').read_text())
            m = result['metadata']
            for key in ('dataset', 'alpha', 'trial_id', 'excluded_ids'):
                if m[key] != t[key]:
                    raise ValueError(f'Metadata mismatch: {key}')
            kept = sorted(set(range(8))-set(t['excluded_ids']))
            if (m['seed'] != 42 or m['original_client_num'] != 8 or m['num_epoch'] != 50
                    or m['num_local_epochs'] != 1 or m['shapley_round'] != 49
                    or m['use_attack'] or m['use_lazy'] or m['original_client_ids'] != kept):
                raise ValueError('Invalid control configuration')
            rows = result['contributions']
            if sorted(r['client_id'] for r in rows) != kept or not all(math.isfinite(r['shapley_value']) for r in rows):
                raise ValueError('Missing/nonfinite honest SV')
            if len(result['accuracy']) != 50 or not all(math.isfinite(v) for v in result['accuracy']):
                raise ValueError('Missing/nonfinite accuracy')
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failed += 1
            tables[t['dataset']]['control_audit'].append(dict(base, status='failed_or_missing', reason=str(exc)))
            continue
        tables[t['dataset']]['control_audit'].append(dict(base, status='complete', reason=''))
        for row in rows:
            tables[t['dataset']]['control_contributions'].append(dict(base, **row))
        for index, value in enumerate(result['accuracy']):
            tables[t['dataset']]['control_accuracy'].append(dict(base, round_idx=index, accuracy=value))
    meta = ['task_id','dataset','alpha','trial_id','attacker_count','excluded_ids','seed',
            'original_client_num','control_client_num','attacker_selection_seed']
    extra = dict(control_contributions=['client_id','shapley_value','shard_size','aggregation_weight'],
                 control_accuracy=['round_idx','accuracy'], control_audit=['status','reason'])
    for dataset, kinds in tables.items():
        out = directory/'merged'/dataset
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in kinds.items():
            target = out/(name+'.csv'); tmp = target.with_suffix('.csv.tmp')
            with tmp.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=meta+extra[name]); writer.writeheader(); writer.writerows(rows)
            tmp.replace(target)
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--merge-only', type=Path)
    p.add_argument('--datasets', nargs='+', default=['adult','mnist','imdb'])
    p.add_argument('--partition', choices=tuple(PARTITION_ALPHAS), default='first')
    p.add_argument('--gpus', nargs='+', default=['0','1','2','3'])
    p.add_argument('--tasks-per-gpu', type=int, default=15)
    p.add_argument('--cpu-threads', type=int, default=15)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--output-root', type=Path, help='Default separates first-column and Uniform controls')
    a = p.parse_args()
    if a.merge_only:
        manifest = json.loads((a.merge_only/'manifest.json').read_text())
        return int(bool(merge(manifest['tasks'], a.merge_only,
                              manifest.get('settings', {}).get('partition', 'first'))))
    if a.tasks_per_gpu < 1 or a.cpu_threads < 1 or len(set(a.gpus)) != len(a.gpus) or not all(g.isdigit() for g in a.gpus):
        p.error('Use unique nonnegative GPU indices and positive resource counts')
    tasks = tasks_for(a.datasets, a.partition)
    if a.output_root is None:
        suffix = '_uniform' if a.partition == 'uniform' else ''
        a.output_root = ROOT / ('Results_freerider_controls' + suffix)
    directory = a.output_root.resolve()/('controls_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    print(f'{len(tasks)} honest-only controls; GPUs={a.gpus}; slots/GPU={a.tasks_per_gpu}; CPU threads/task={a.cpu_threads}')
    print('Fixed original N=8, seed=42, selection seed=2026; 50 honest rounds; no old baseline reuse.')
    print(f'Partition={a.partition}; alpha={PARTITION_ALPHAS[a.partition]}')
    for t in tasks:
        print(shlex.join(command_for(t, a.python, directory/'tasks'/t['task_id'])))
    if not a.run:
        print('DRY RUN: no files or training processes created.')
        return 0
    if os.name != 'posix':
        p.error('Real runs require Linux; preview is supported locally')
    stop = threading.Event()
    with gpu_locks(a.gpus):
        subprocess.run([a.python,'-c','import torch; assert torch.cuda.is_available(); '
                        f'assert torch.cuda.device_count()=={len(a.gpus)}'],
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(a.gpus)),check=True)
        directory.mkdir(parents=True, exist_ok=False)
        files = ['FLCE_updated.py','config.py','module/method/freerider_control.py',
                 'module/method/multi_freerider.py','script/freerider_control_campaign.py',
                 'script/multi_freerider_campaign.py','script/run_freerider_controls.sh',
                 'script/run_uniform_freerider_rq1_rq2.sh']
        write_json(directory/'manifest.json',dict(tasks=tasks, settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            selection_design=selection_design(a.datasets,2026), reuse_existing=False,
            code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}))
        def job(t, gpu):
            out = directory/'tasks'/t['task_id'];out.mkdir(parents=True)
            status = dict(task_id=t['task_id'],gpu=gpu,returncode=1)
            try:
                with (out/'stdout.log').open('w') as log:
                    print(f"[START gpu={gpu}] {t['task_id']}",flush=True)
                    proc = subprocess.Popen(command_for(t,a.python,out),cwd=ROOT,
                        env=worker_environment(gpu,a.cpu_threads),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while proc.poll() is None:
                        if stop.wait(.5):
                            try: os.killpg(proc.pid,signal.SIGTERM)
                            except ProcessLookupError: pass
                            try: proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                            break
                    status['returncode']=proc.returncode
            except Exception as exc:
                status['error']=str(exc)
            write_json(out/'status.json',status)
            print(f"[END exit={status['returncode']}] {t['task_id']}",flush=True)
            return status
        handlers = {s:signal.signal(s,lambda *_:stop.set()) for s in (signal.SIGINT,signal.SIGTERM)}
        try: dispatch(tasks,a.gpus,a.tasks_per_gpu,job,stop)
        finally:
            for sig,handler in handlers.items(): signal.signal(sig,handler)
        failed = merge(tasks,directory,a.partition)
        print(f'Merged controls: {len(tasks)-failed} complete, {failed} failed/missing; {directory}',flush=True)
        return int(bool(failed or stop.is_set()))


if __name__ == '__main__':
    raise SystemExit(main())

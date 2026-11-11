"""Independent campaign contracts: singular entry, paired matrix, bounded queue, CSVs."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from test_rq1_single_attacker import actual_parser, api


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'script/rq1_fl_aggregation_campaign.py'


def campaign():
    assert SCRIPT.exists(), 'Independent single-attacker RQ1 campaign is missing'
    spec = importlib.util.spec_from_file_location('rq1_campaign_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(argv, name):
    return argv[argv.index(name) + 1]


def fixture_result(task):
    values = [-0.02 if i == task['attacker_id'] else 0.104 for i in range(6)]
    return dict(
        schema_version=1,
        metadata={**{key: task[key] for key in ('dataset', 'alpha', 'client_num', 'attacker_id',
                    'attack_method', 'fl_aggregate', 'seed', 'trial_id')},
                  'rq1_only': True, 'attacker_count': 1, 'regime': 'pure',
                  'detecting_method': 'shapley', 'num_epoch': 50, 'num_local_epochs': 1,
                  'use_hessian': True, 'lambda_h': 0.9, 'qualitative_aggregation_mode': 'passive',
                  'cpu_threads_per_task': 15, 'fl_aggregation': campaign().aggregation_config(task)},
        fingerprints=dict(shards=[dict(client_id=i, size=10, x_sha256=str(i), y_sha256=str(i)) for i in range(6)],
                          test_x_sha256='test_x', test_y_sha256='test_y', initial_model={'weight': 'initial'}),
        accuracy=[0.5] * 50, contributions=values, shard_sizes=[10] * 6, shard_weights=[1 / 6] * 6,
        audit=dict(fake_call_rounds=list(range(50)), fake_rounds=list(range(50)), honest_rounds=[],
                   fake_sources=['fallback'] * 50,
                   private_fit_calls_by_client={str(i): 0 if i == task['attacker_id'] else 50 for i in range(6)},
                   round_activity=[dict(round_idx=r, active_attackers=[task['attacker_id']]) for r in range(50)]),
        evaluation=dict(shapley_scope='actual_last_round_submissions', coalitions_evaluated=64,
                        empty_coalition_utility=0.0, grand_coalition_utility=0.5,
                        final_round_idx=49, server_state_rounds_before_last=49),
        rq1=dict(attacker_contribution=-0.02, total_contribution=0.5, attacker_ratio=-0.04, ratio_status='valid'),
        timing=dict(training_sec=1.0, evaluation_sec=0.2, attack_host_sec=0.1, total_sec=1.2),
    )


def populate(c, tasks, directory):
    for task in tasks:
        task_dir = directory / 'tasks' / task['task_id']
        path = c.result_path(task, task_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fixture_result(task)), encoding='utf-8')
        (task_dir / 'status.json').write_text(json.dumps(dict(returncode=0)), encoding='utf-8')


def test_matrix_and_every_command_use_original_single_attacker(monkeypatch, tmp_path):
    c = campaign()
    tasks = c.build_tasks()
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 108
    parser = actual_parser(monkeypatch)
    single = api(monkeypatch)
    from module.method.fl_aggregation import AggregationConfig
    for task in tasks:
        argv = c.command_for(task, sys.executable, ROOT, tmp_path / task['task_id'])
        assert '--attacker_ids' not in argv and '--use_lazy' not in argv
        args = parser.parse_args(argv[2:])
        single.validate_config(args)
        assert args.rq1_only and args.attacker_id == task['attacker_id']
        assert args.client_num == 6 and args.seed == 42 and args.trial_id == 0
        expected = Path(args.results_root) / AggregationConfig.from_args(args).output_subdirectory() / 'rq1_result.json'
        assert c.result_path(task, tmp_path / task['task_id']) == expected
    for dataset, alpha in [('adult', 1000), ('mnist', 10), ('imdb', 10)]:
        subset = [t for t in tasks if t['dataset'] == dataset]
        assert len(subset) == 36 and {t['alpha'] for t in subset} == {alpha}
        for fl in c.FL_METHODS:
            for family in c.ATTACK_FAMILIES:
                assert sorted(t['attacker_id'] for t in subset if t['fl_aggregate'] == fl and t['attack_family'] == family) == list(range(6))


def test_worker_environment_and_frozen_resources():
    c = campaign()
    env = c.worker_environment('2', 15)
    assert env['CUDA_VISIBLE_DEVICES'] == '2'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'FLCE_CPU_THREADS_PER_TASK'):
        assert env[name] == '15'
    with pytest.raises(ValueError):
        c.parse_args(['--cpu-threads', '14'])
    with pytest.raises(ValueError):
        c.parse_args(['--attacker-sweep', '1'])


def test_shared_queue_never_exceeds_fifteen_per_gpu():
    c = campaign()
    lock = threading.Lock()
    active = {str(i): 0 for i in range(4)}
    peaks = dict(active)
    seen = []
    def job(task, gpu):
        with lock:
            active[gpu] += 1
            peaks[gpu] = max(peaks[gpu], active[gpu])
        time.sleep(0.01)
        with lock:
            seen.append(task)
            active[gpu] -= 1
        return task
    assert sorted(c.dispatch(list(range(108)), ['0', '1', '2', '3'], 15, job)) == list(range(108))
    assert sorted(seen) == list(range(108)) and all(0 < n <= 15 for n in peaks.values())


def test_dry_run_creates_no_output_or_training_process(tmp_path):
    c = campaign()
    target = tmp_path / 'must_not_exist'
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--dry-run', '--datasets', 'adult',
                             '--output-root', str(target), '--print-all-commands'],
                            cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not target.exists()
    assert result.stdout.count('COMMAND ') == 36
    assert '--attacker_ids' not in result.stdout and '36 tasks' in result.stdout


def test_dataset_csvs_preserve_signed_rq1_and_attacker_position_means(tmp_path):
    c = campaign()
    tasks = c.build_tasks(['adult'])
    populate(c, tasks, tmp_path)
    counts = c.merge_results(tasks, tmp_path)
    assert counts == dict(complete=36, failed=0)
    directory = tmp_path / 'merged/adult'
    expected = dict(rq1_detail=36, rq1_mean_std=6, contributions=216, accuracy=1800, time=36, audit=36)
    for name, count in expected.items():
        with (directory / f'{name}.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == count
        if name == 'rq1_detail':
            assert all(float(row['attacker_ratio']) == pytest.approx(-0.04) for row in rows)
        if name == 'rq1_mean_std':
            assert all(row['n_completed'] == '6' and row['replicate_axis'] == 'attacker_position' for row in rows)
            assert all(float(row['attacker_ratio_mean']) == pytest.approx(-0.04) for row in rows)
    assert not list((tmp_path / 'tasks').rglob('*.csv'))


def test_partial_and_corrupt_results_are_audited_not_zero_filled(tmp_path):
    c = campaign()
    tasks = c.build_tasks(['adult'])
    populate(c, tasks, tmp_path)
    missing = c.result_path(tasks[0], tmp_path / 'tasks' / tasks[0]['task_id'])
    missing.unlink()
    invalid = c.result_path(tasks[1], tmp_path / 'tasks' / tasks[1]['task_id'])
    data = json.loads(invalid.read_text())
    data['audit']['fake_call_rounds'] = list(range(49))
    invalid.write_text(json.dumps(data))
    assert c.merge_results(tasks, tmp_path) == dict(complete=34, failed=2)
    with (tmp_path / 'merged/adult/rq1_mean_std.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert sum(int(row['n_completed']) for row in rows) == 34
    assert any(row['status'] == 'partial' for row in rows)


def test_cross_fl_pairing_and_recomputed_ratio_are_enforced(tmp_path):
    c = campaign()
    tasks = c.build_tasks(['adult'])
    populate(c, tasks, tmp_path)
    path = c.result_path(tasks[0], tmp_path / 'tasks' / tasks[0]['task_id'])
    data = json.loads(path.read_text())
    data['rq1']['attacker_ratio'] = 0.8
    with pytest.raises(ValueError):
        c.validate_result(tasks[0], data)
    data['rq1']['attacker_ratio'] = -0.04
    data['fingerprints']['test_x_sha256'] = 'unpaired'
    path.write_text(json.dumps(data))
    outcome = c.merge_results(tasks, tmp_path)
    assert outcome['failed'] > 0
    with (tmp_path / 'merged/adult/audit.csv').open(newline='') as handle:
        assert any('pair' in row['reason'].lower() for row in csv.DictReader(handle))


@pytest.mark.parametrize('bad_section', ['root', 'metadata'])
def test_malformed_result_is_audited_without_aborting_dataset(tmp_path, bad_section):
    c = campaign()
    tasks = c.build_tasks(['adult'])
    populate(c, tasks, tmp_path)
    malformed = fixture_result(tasks[0])
    if bad_section == 'root':
        malformed = []
    else:
        malformed['metadata'] = []
    path = c.result_path(tasks[0], tmp_path / 'tasks' / tasks[0]['task_id'])
    path.write_text(json.dumps(malformed), encoding='utf-8')
    assert c.merge_results(tasks, tmp_path) == dict(complete=35, failed=1)

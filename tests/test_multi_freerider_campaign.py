"""Campaign matrix, dry-run, bounded GPU dispatch and dataset-level CSV checks."""
import csv
import argparse
import ast
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'script/multi_freerider_campaign.py'
EXPECTED_TRIPLES = {
    'adult': [[1, 4, 7], [0, 6, 3], [2, 5, 1]],
    'mnist': [[3, 4, 6], [7, 2, 0], [1, 5, 3]],
    'imdb': [[4, 6, 7], [2, 5, 3], [0, 1, 4]],
}


def campaign():
    assert SCRIPT.exists(), 'The dedicated campaign launcher is missing'
    spec = importlib.util.spec_from_file_location('multi_campaign_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(argv, name):
    return argv[argv.index(name) + 1]


def test_matrix_exactly_matches_frozen_experiment(tmp_path):
    c = campaign()
    tasks = c.build_tasks()
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 216
    for dataset, alpha in [('adult', 1000), ('mnist', 10), ('imdb', 10)]:
        selected = [t for t in tasks if t['dataset'] == dataset]
        assert len(selected) == 72
        assert {t['attacker_count'] for t in selected} == {1, 2, 3}
        for t in selected:
            assert t['alpha'] == alpha and t['client_num'] == 8 and t['num_epoch'] == 50
            assert t['attacker_ids'] == EXPECTED_TRIPLES[dataset][t['trial_id']][:t['attacker_count']]
            argv = c.command_for(t, sys.executable, ROOT, tmp_path / t['task_id'])
            assert '--attacker_id' not in argv
            assert option(argv, '--seed') == option(argv, '--std_dagmm_seed') == '42'
            assert option(argv, '--detecting_method') in {'shapley', 'std_dagmm'}
            assert option(argv, '--qualitative_aggregation_mode') == 'passive'
            assert '--use_last_round_sv' in argv and '--use_hessian' in argv
            assert ('layerwise' in option(argv, '--attack_method')) == (dataset != 'adult')
            assert ('--use_lazy' in argv) == (t['regime'] == 'lazy')
            if t['regime'] == 'lazy':
                assert option(argv, '--lazy_honest_gap') == '6'
                assert option(argv, '--lazy_honest_phase') == '0'
                assert t['fake_rounds_per_attacker'] == 41
            else:
                assert t['fake_rounds_per_attacker'] == 50
    # Pairing across evaluators keeps FL/attack seeds identical.
    assert {t['seed'] for t in tasks} == {42}
    assert len(c.build_tasks(trials=[0, 1])) == 144


def test_selection_is_balanced_reproducible_and_separate_from_training_rng():
    c = campaign()
    initial_rng = random.getstate()
    for selection_seed in (2026, 2027, 0, 1, 8):
        tasks = c.build_tasks(attacker_selection_seed=selection_seed)
        for dataset in ('adult', 'mnist', 'imdb'):
            selected = [t for t in tasks if t['dataset'] == dataset]
            assert selected == c.build_tasks(datasets=[dataset], attacker_selection_seed=selection_seed)
            for k, expected_counts in [(1, [0, 0, 0, 0, 0, 1, 1, 1]),
                                       (2, [0, 0, 1, 1, 1, 1, 1, 1]),
                                       (3, [1, 1, 1, 1, 1, 1, 1, 2])]:
                assignments = {t['trial_id']: t['attacker_ids'] for t in selected if t['attacker_count'] == k}
                assert len(assignments) == 3
                assert all(len(ids) == len(set(ids)) == k for ids in assignments.values())
                counts = Counter(i for ids in assignments.values() for i in ids)
                assert sorted(counts[i] for i in range(8)) == expected_counts
            assert {t['seed'] for t in selected} == {42}
    assert random.getstate() == initial_rng
    default = c.build_tasks()
    assert c.build_tasks(trials=[2]) == [t for t in default if t['trial_id'] == 2]
    assert any(a['attacker_ids'] != b['attacker_ids'] for a, b in
               zip(default, c.build_tasks(attacker_selection_seed=2027)))


@pytest.mark.parametrize('kwargs', [{'seed': 43}, {'trials': [3]}, {'trials': [0, 0]},
                                    {'trials': []}, {'attacker_selection_seed': -1}])
def test_campaign_rejects_changed_training_seed_and_invalid_group_design(kwargs):
    with pytest.raises(ValueError):
        campaign().build_tasks(**kwargs)


def test_every_command_parses_with_the_real_entrypoint_and_passes_its_guard(tmp_path):
    c = campaign()
    parser = argparse.ArgumentParser()
    tree = ast.parse((ROOT / 'FLCE_updated.py').read_text(encoding='utf-8'))
    arguments = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
                 and n.value.func.value.id == 'parser' and n.value.func.attr == 'add_argument']
    exec(compile(ast.Module(body=arguments, type_ignores=[]), 'actual_cli_arguments', 'exec'),
         dict(parser=parser, argparse=argparse))
    spec = importlib.util.spec_from_file_location('multi_config_under_test', ROOT / 'module/method/multi_freerider.py')
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    for task in c.build_tasks():
        args = parser.parse_args(c.command_for(task, sys.executable, ROOT, tmp_path / task['task_id'])[3:])
        implementation.validate_config(args)
    args.lazy_honest_gap = 3
    with pytest.raises(ValueError, match='1 real'):
        implementation.validate_config(args)


def test_preview_creates_no_files_or_training_processes(tmp_path):
    campaign()
    output = tmp_path / 'must_not_exist'
    process = subprocess.run([sys.executable, str(SCRIPT), '--output-root', str(output),
                              '--dry-run', '--print-limit', '1'], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert '216 tasks' in process.stdout and '60' in process.stdout
    assert 'selection_seed=2026' in process.stdout
    assert 'k3=[1, 4, 7]' in process.stdout
    assert 'summary_mean_std.csv' in process.stdout
    assert not output.exists()


def test_shared_queue_obeys_each_gpu_limit_and_cpu_environment():
    c = campaign()
    active, peak, observed = {}, {}, set()
    lock = threading.Lock()
    def job(task, gpu):
        with lock:
            active[gpu] = active.get(gpu, 0) + 1
            peak[gpu] = max(peak.get(gpu, 0), active[gpu])
            observed.add(task)
        time.sleep(0.01)
        with lock:
            active[gpu] -= 1
        return {'task_id': task, 'returncode': 0}
    results = c.dispatch(list(range(72)), ['0', '1', '2', '3'], 15, job)
    assert len(results) == len(observed) == 72
    assert set(peak) == {'0', '1', '2', '3'}
    assert all(value <= 15 for value in peak.values())
    env = c.worker_environment('3', 15)
    assert env['CUDA_VISIBLE_DEVICES'] == '3'
    assert env['FLCE_CPU_THREADS_PER_TASK'] == '15'
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert env[key] == '15'


def write_fixture(c, task, root, *, failed=False, mismatch=False):
    directory = root / 'tasks' / task['task_id']
    directory.mkdir(parents=True)
    (directory / 'status.json').write_text(json.dumps({'returncode': 1 if failed else 0}), encoding='utf-8')
    metadata = {k: v for k, v in task.items() if k != 'task_id'}
    if mismatch:
        metadata['attacker_count'] = 99
    records = []
    if task['detecting_method'] == 'std_dagmm':
        for r in range(50):
            active = [] if task['regime'] == 'lazy' and r % 6 == 0 else task['attacker_ids']
            records.append(dict(round_idx=r, client_scores={i: float(i) for i in range(8)},
                ground_truth_labels={i: int(i in active) for i in range(8)},
                suspicious_clients=[4, 5, 6, 7], roc_auc=0.4 if active else None,
                energy_ranks={i: 8-i for i in range(8)}, energy_threshold=3.5,
                tp=0, fp=4, fn=len(active), tn=4-len(active), tpr=0 if active else None, fpr=0.5))
    artifact = dict(schema_version=1, metadata=metadata, accuracy=[0.8] * 50,
        contributions=[0.1] * 8 if task['detecting_method'] == 'shapley' else [], detection=records,
        evaluation={'valid_auc_rounds': task['fake_rounds_per_attacker']} if records else {},
        timing={'fl_sec': 1, 'evaluation_sec': 2, 'total_sec': 3, 'attack_host_sec': 0.1},
        attackers=[dict(client_id=i, fake_count=task['fake_rounds_per_attacker'],
                        real_count=task['real_rounds_per_attacker']) for i in task['attacker_ids']],
        shard_weights=[0.125]*8, shard_sizes=[10]*8)
    (directory / 'result.json').write_text(json.dumps(artifact), encoding='utf-8')
    return artifact


def read_csv(root, name, dataset='adult'):
    with (root / 'merged' / dataset / (name + '.csv')).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_merger_groups_by_dataset_and_keeps_invalid_tasks_in_audit(tmp_path):
    c = campaign()
    tasks = c.build_tasks(datasets=['adult'])
    for index, task in enumerate(tasks):
        write_fixture(c, task, tmp_path, failed=index == 0, mismatch=index == 1)
    outcome = c.merge_results(tasks, tmp_path)
    assert outcome['complete'] == 70 and outcome['failed'] == 2
    csv_paths = list(tmp_path.rglob('*.csv'))
    assert len(csv_paths) == 7
    assert all(path.parent == tmp_path / 'merged/adult' for path in csv_paths)
    def read(name):
        with (tmp_path / 'merged/adult' / name).open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    assert len(read('audit.csv')) == 72
    assert len(read('accuracy.csv')) == 70 * 50
    assert len(read('summary.csv')) == 70
    assert all(row['attacker_count'] in {'1', '2', '3'} for row in read('summary.csv'))
    lazy_honest = [row for row in read('detection.csv') if row['regime'] == 'lazy' and row['round_idx'] == '0']
    assert lazy_honest and all(row['roc_auc'] == '' for row in lazy_honest)
    grouped = read('summary_mean_std.csv')
    assert len(grouped) == 24
    assert Counter(row['status'] for row in grouped) == {'complete': 22, 'partial': 2}
    assert {row['n_expected'] for row in grouped} == {'3'}


def test_dataset_summary_averages_groups_not_clients_or_rounds(tmp_path):
    c = campaign()
    tasks = c.build_tasks(datasets=['adult'])
    for task in tasks:
        artifact = write_fixture(c, task, tmp_path)
        group = task['trial_id']
        artifact['accuracy'] = [[0.6, 0.8, 1.0][group]] * 50
        artifact['shard_sizes'] = [10, 20, 30, 40, 50, 60, 70, 80]
        artifact['shard_weights'] = [v / 360 for v in artifact['shard_sizes']]
        if artifact['contributions']:
            artifact['contributions'] = [[0.1, 0.2, 0.3][group]] * 8
        else:
            # Missing metrics stay missing; within-run SD is not across-group SD.
            artifact['evaluation'].update(mean_round_auc=[0.4, 0.6, None][group],
                                          std_round_auc=[0.05, 0.10, 0.15][group])
        (tmp_path / 'tasks' / task['task_id'] / 'result.json').write_text(json.dumps(artifact), encoding='utf-8')
    assert c.merge_results(tasks, tmp_path) == {'complete': 72, 'failed': 0}
    raw = read_csv(tmp_path, 'summary')
    grouped = read_csv(tmp_path, 'summary_mean_std')
    assert len(raw) == 72 and len(grouped) == 24
    assert {row['trial_id'] for row in raw} == {'0', '1', '2'}
    assert {row['seed'] for row in raw} == {'42'}
    assert {row['attacker_selection_seed'] for row in raw} == {'2026'}
    for row in grouped:
        assert row['status'] == 'complete' and row['n_completed'] == row['n_expected'] == '3'
        assert row['final_accuracy_n'] == '3'
        assert float(row['final_accuracy_mean']) == pytest.approx(0.8)
        assert float(row['final_accuracy_std']) == pytest.approx(0.2)  # sample SD, ddof=1
        assert row['group_ids_completed'] == '[0,1,2]'
        if row['detecting_method'] == 'shapley':
            assert row['mean_round_auc_mean'] == row['mean_round_auc_std'] == ''
            assert row['mean_round_auc_n'] == '0'
            assert float(row['attacker_shapley_mean_mean']) == pytest.approx(0.2)
            assert float(row['attacker_shapley_mean_std']) == pytest.approx(0.1)
            assert float(row['attacker_shapley_sum_mean']) == pytest.approx(0.2 * int(row['attacker_count']))
        else:
            assert row['mean_round_auc_n'] == '2'
            assert float(row['mean_round_auc_mean']) == pytest.approx(0.5)
            assert float(row['mean_round_auc_std']) == pytest.approx(0.02 ** 0.5)
            assert float(row['std_round_auc_mean']) == pytest.approx(0.1)
    k1 = next(row for row in grouped if row['attacker_count'] == '1')
    # Adult k1 attackers 1,0,2 own 20,10,30 samples respectively.
    assert float(k1['attacker_sample_count_mean']) == 20
    assert float(k1['attacker_sample_count_std']) == 10
    assert float(k1['attacker_data_fraction_mean']) == pytest.approx(20 / 360)


@pytest.mark.parametrize('missing,n,mean,sd', [([2], 2, 0.7, 0.02 ** 0.5),
                                             ([1, 2], 1, 0.6, None),
                                             ([0, 1, 2], 0, None, None)])
def test_missing_groups_are_counted_without_zero_imputation(tmp_path, missing, n, mean, sd):
    c = campaign()
    tasks = [t for t in c.build_tasks(datasets=['adult'])
             if t['attack_method'] == 'moirai_online' and t['attacker_count'] == 1
             and t['regime'] == 'pure' and t['detecting_method'] == 'shapley']
    for task in tasks:
        if task['trial_id'] in missing:
            continue
        artifact = write_fixture(c, task, tmp_path)
        artifact['accuracy'] = [[0.6, 0.8, 1.0][task['trial_id']]] * 50
        (tmp_path / 'tasks' / task['task_id'] / 'result.json').write_text(json.dumps(artifact), encoding='utf-8')
    c.merge_results(tasks, tmp_path)
    row, = read_csv(tmp_path, 'summary_mean_std')
    assert row['n_expected'] == row['n_planned'] == '3'
    assert row['n_completed'] == row['final_accuracy_n'] == str(n)
    assert json.loads(row['group_ids_missing']) == missing
    assert row['status'] == ('partial' if n else 'missing')
    assert row['final_accuracy_mean'] == '' if mean is None else float(row['final_accuracy_mean']) == pytest.approx(mean)
    assert row['final_accuracy_std'] == '' if sd is None else float(row['final_accuracy_std']) == pytest.approx(sd)


def test_single_group_preview_never_claims_three_group_completion(tmp_path):
    c = campaign()
    task = c.build_tasks(datasets=['adult'], trials=[2])[0]
    write_fixture(c, task, tmp_path)
    c.merge_results([task], tmp_path)
    row, = read_csv(tmp_path, 'summary_mean_std')
    assert row['n_expected'] == '3' and row['n_planned'] == row['n_completed'] == '1'
    assert row['status'] == 'partial' and row['group_ids_missing'] == '[0,1]'
    assert row['final_accuracy_std'] == ''


@pytest.mark.parametrize('invalid', ['duplicate', 'wrong_selection_seed'])
def test_merger_rejects_duplicate_groups_or_wrong_assignment_provenance(tmp_path, invalid):
    c = campaign()
    task = c.build_tasks(datasets=['adult'])[0]
    tasks = [task, task] if invalid == 'duplicate' else [task]
    with pytest.raises(ValueError, match='assignment|Duplicate'):
        c.merge_results(tasks, tmp_path, attacker_selection_seed=2027 if invalid == 'wrong_selection_seed' else 2026)
    assert not (tmp_path / 'merged').exists()


def test_merge_only_uses_manifest_selection_seed(tmp_path):
    c = campaign()
    tasks = c.build_tasks(datasets=['imdb'], attacker_selection_seed=2027)
    for task in tasks:
        write_fixture(c, task, tmp_path)
    manifest = dict(tasks=tasks, settings={'attacker_selection_seed': 2027})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    process = subprocess.run([sys.executable, str(SCRIPT), '--merge-only', str(tmp_path)],
                             text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    raw = read_csv(tmp_path, 'summary', 'imdb')
    grouped = read_csv(tmp_path, 'summary_mean_std', 'imdb')
    assert len(raw) == 72 and len(grouped) == 24
    assert {row['attacker_selection_seed'] for row in raw + grouped} == {'2027'}


def test_invalid_shards_are_audited_instead_of_entering_average(tmp_path):
    c = campaign()
    task = c.build_tasks(datasets=['adult'])[0]
    artifact = write_fixture(c, task, tmp_path)
    artifact['shard_sizes'] = [10]
    (tmp_path / 'tasks' / task['task_id'] / 'result.json').write_text(json.dumps(artifact), encoding='utf-8')
    assert c.merge_results([task], tmp_path) == {'complete': 0, 'failed': 1}
    assert read_csv(tmp_path, 'summary') == []
    row, = read_csv(tmp_path, 'summary_mean_std')
    assert row['n_completed'] == '0' and row['attacker_sample_count_mean'] == ''

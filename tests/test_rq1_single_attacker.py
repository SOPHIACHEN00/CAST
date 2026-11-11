"""Opt-in single-attacker RQ1 behavior, using the real FL loop and evaluator."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from test_fl_aggregation import aggregation_module
from test_multi_freerider import ToyAttack, ToyModel, training_namespace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api(monkeypatch):
    aggregation_module(monkeypatch)
    path = ROOT / 'module/method/rq1_single_attacker.py'
    assert path.exists(), 'Single-attacker RQ1 result helper is missing'
    name = 'module.method.rq1_single_attacker'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def actual_parser(monkeypatch):
    aggregation = aggregation_module(monkeypatch)
    tree = ast.parse((ROOT / 'FLCE_updated.py').read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                    and fn.value.id == 'parser' and fn.attr == 'add_argument'):
                nodes.append(node)
            elif isinstance(fn, ast.Name) and fn.id == 'add_aggregation_arguments':
                nodes.append(node)
    ns = dict(argparse=argparse, add_aggregation_arguments=aggregation.add_aggregation_arguments)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual CLI>', 'exec'), ns)
    return ns['parser']


def arguments(monkeypatch, tmp_path, method='fedavg', attacker=0):
    parser = actual_parser(monkeypatch)
    assert '--rq1_only' in parser._option_string_actions, 'Explicit single-attacker RQ1 mode is missing'
    return parser.parse_args([
        '--rq1_only', '--dataset', 'adult', '--alpha', '1000', '--client_num', '6',
        '--attacker_id', str(attacker), '--attack_method', 'moirai_online', '--use_attack',
        '--detecting_method', 'shapley', '--use_last_round_sv', '--fl_aggregate', method,
        '--qualitative_aggregation_mode', 'passive', '--use_hessian', '--lambda_h', '0.9',
        '--results_root', str(tmp_path),
    ])


def loader():
    return SimpleNamespace(
        X_train_parts=[torch.full((i + 2,), float(i)) for i in range(6)],
        y_train_parts=[torch.zeros(i + 2) for i in range(6)],
        X_test=torch.zeros(3), y_test=torch.zeros(3),
    )


class ScoreModel(ToyModel):
    def score(self, x, y, metrics):
        return [torch.sigmoid(self.weight / 100).item()]


def test_default_single_attacker_behavior_is_opt_in(monkeypatch):
    parser = actual_parser(monkeypatch)
    assert '--rq1_only' in parser._option_string_actions
    args = parser.parse_args(['--client_num', '6', '--attacker_id', '5'])
    assert args.rq1_only is False and args.attacker_id == 5 and args.attacker_ids is None


@pytest.mark.parametrize('changes', [
    {'attacker_ids': [0]}, {'control_excluded_ids': [0]}, {'client_num': 8},
    {'use_lazy': True}, {'use_attack': False}, {'num_epoch': 49},
    {'num_local_epochs': 2}, {'detecting_method': 'sampling_shapley'},
    {'use_last_round_sv': False}, {'attacker_id': 6}, {'attacker_id': -1},
    {'alpha': 20000000}, {'attack_method': 'random'}, {'dump_grad_history': True},
    {'qualitative_aggregation_mode': 'paper_online'},
])
def test_rq1_rejects_out_of_scope_modes(monkeypatch, tmp_path, changes):
    args = arguments(monkeypatch, tmp_path)
    module = api(monkeypatch)
    args.__dict__.update(changes)
    with pytest.raises(ValueError):
        module.validate_config(args)


@pytest.mark.parametrize('method', ['fedavg', 'fedadam', 'weighted_median'])
@pytest.mark.parametrize('attacker_id', [0, 5])
def test_fifty_fake_rounds_original_single_path_and_exact_rq1(monkeypatch, tmp_path, method, attacker_id):
    args = arguments(monkeypatch, tmp_path, method, attacker_id)
    module = api(monkeypatch)
    module.validate_config(args)
    data, model, attack = loader(), ScoreModel(), ToyAttack()
    context = module.prepare_run(args=args, loader=data, model=model)
    ns = training_namespace()
    ns['args'] = args
    aggregation_context = {}
    ToyModel.trained_clients = []
    result = ns['run_federated_learning'](
        data, model, attacker_id, attack, True, 50, 1, torch.device('cpu'),
        last_round_aggregation_out=aggregation_context,
        single_attacker_audit_out=context['audit'],
    )
    assert ToyModel.trained_clients.count(attacker_id) == 0
    assert all(ToyModel.trained_clients.count(i) == 50 for i in range(6) if i != attacker_id)
    final_state = copy.deepcopy(model.state_dict())
    before_evaluation_calls = len(ToyModel.trained_clients)
    artifact = module.finish_run(
        args=args, loader=data, model=model, attacker=attack, run_context=context,
        last_round_state=result[3], last_round_client_grads=result[4],
        last_round_weights=result[5], aggregation_context=aggregation_context,
        accuracy=result[2], attack_times=result[1], training_seconds=0.1, provenance={},
    )
    assert len(ToyModel.trained_clients) == before_evaluation_calls
    torch.testing.assert_close(model.weight, final_state['weight'])
    assert artifact['metadata']['attacker_id'] == attacker_id
    assert artifact['metadata']['client_num'] == 6
    assert artifact['metadata']['fl_aggregate'] == method
    assert artifact['metadata']['cpu_threads_per_task'] == 15
    assert artifact['audit']['fake_rounds'] == list(range(50))
    assert artifact['audit']['honest_rounds'] == []
    assert artifact['audit']['private_fit_calls_by_client'][str(attacker_id)] == 0
    assert len(artifact['contributions']) == 6
    assert sum(artifact['contributions']) == pytest.approx(result[2][-1], abs=1e-6)
    assert artifact['evaluation']['coalitions_evaluated'] == 64
    assert artifact['rq1']['attacker_ratio'] == pytest.approx(
        artifact['contributions'][attacker_id] / sum(artifact['contributions']))
    assert json.loads((tmp_path / 'rq1_result.json').read_text())['rq1'] == artifact['rq1']
    assert not list(tmp_path.rglob('*.csv'))
    assert len(context['fingerprints']['shards']) == 6
    # Validate a real producer artifact with the independent campaign consumer.
    from test_rq1_fl_aggregation_campaign import campaign
    consumer = campaign()
    task = next(task for task in consumer.build_tasks(['adult'])
                if task['fl_aggregate'] == method and task['attacker_id'] == attacker_id
                and task['attack_family'] == 'moirai_online')
    consumer.validate_result(task, artifact)


def test_signed_ratio_is_not_clipped_and_zero_denominator_is_missing(monkeypatch):
    module = api(monkeypatch)
    assert module.rq1_metrics([-0.1, 0.2, 0.3], 0)['attacker_ratio'] == pytest.approx(-0.25)
    result = module.rq1_metrics([0.1, -0.1], 0)
    assert result['attacker_ratio'] is None and result['ratio_status'] == 'zero_denominator'


@pytest.mark.parametrize('rq1_only', [True, False])
def test_entry_rq1_exit_and_default_control_path(monkeypatch, tmp_path, rq1_only):
    args = arguments(monkeypatch, tmp_path, 'fedadam', 3)
    args.rq1_only = rq1_only
    api(monkeypatch)
    ns = training_namespace()
    ns.update(args=args, loader=loader(), model=ScoreModel(), attacker=ToyAttack(),
              attacker_id=3, num_epoch=50, num_local_epochs=1, device=torch.device('cpu'),
              need_gradients=False, need_history=False, adam_public_pool=None, provenance_payload={})
    tree = ast.parse((ROOT / 'FLCE_updated.py').read_text(encoding='utf-8'))
    block = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == 'use_attack'
                 and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'aggregation_context_with'
                                                           for t in x.targets) for x in n.body))
    start = next(i for i, n in enumerate(block.body) if isinstance(n, ast.If)
                 and ast.unparse(n.test) == 'args.attacker_ids is not None')
    class ReachedLegacyControl(Exception):
        pass
    def reached_control(*_args, **_kwargs):
        raise ReachedLegacyControl()
    ns['prepare_loader_without_attacker'] = reached_control
    expected = SystemExit if rq1_only else ReachedLegacyControl
    with pytest.raises(expected) as stopped:
        exec(compile(ast.Module(body=block.body[start:], type_ignores=[]), '<actual single entry>', 'exec'), ns)
    if rq1_only:
        assert stopped.value.code == 0
    assert (tmp_path / 'rq1_result.json').exists() == rq1_only
    assert not list(tmp_path.rglob('*.csv'))

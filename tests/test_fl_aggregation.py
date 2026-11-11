"""Numerical contracts and actual last-round evaluation, without dataset downloads."""
import argparse
import ast
import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_multi_freerider import ToyAttack, ToyModel, training_namespace


ROOT = Path(__file__).resolve().parents[1]


def aggregation_module(monkeypatch):
    path = ROOT / 'module/method/fl_aggregation.py'
    assert path.exists(), 'Configurable sample-weighted aggregation is missing'
    name = 'module.method.fl_aggregation'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    # Bypass the package's dataset/torchvision imports; load real aggregation,
    # Measure, Shapley and runner functions under the actual package paths.
    for package_name, folder in [('module', 'module'), ('module.method', 'module/method')]:
        package = ModuleType(package_name)
        package.__path__ = [str(ROOT / folder)]
        monkeypatch.setitem(sys.modules, package_name, package)
    data = ModuleType('module.data_manager')
    data.DataManager = object
    monkeypatch.setitem(sys.modules, 'module.data_manager', data)
    return module


def test_fedavg_keeps_legacy_float32_arithmetic(monkeypatch):
    api = aggregation_module(monkeypatch)
    updates = [[torch.tensor([1., -3.])], [torch.tensor([5., 7.])]]
    weights = torch.tensor([0.25, 0.75])
    server = api.ServerAggregator(api.AggregationConfig())
    actual = server.step(updates, weights, [torch.zeros(2)])[0]
    assert torch.equal(actual, torch.tensor([4., 4.5]))
    assert torch.equal(updates[0][0], torch.tensor([1., -3.]))


@pytest.mark.parametrize('weights,expected', [
    ([0.6, 0.1, 0.1, 0.2], [1., 9.]),
    ([0.25] * 4, [2.5, 3.5]),
    ([0.5, 0., 0., 0.5], [2.5, 6.]),
])
def test_weighted_median_uses_mass_and_midpoint_ties(monkeypatch, weights, expected):
    api = aggregation_module(monkeypatch)
    updates = [[torch.tensor(x)] for x in ([1., 9.], [2., 1.], [3., 4.], [4., 3.])]
    server = api.ServerAggregator(api.AggregationConfig(method='weighted_median'))
    actual = server.step(updates, weights, [torch.zeros(2)])[0]
    torch.testing.assert_close(actual, torch.tensor(expected))
    # Invariant under reordering clients together with their sample weights.
    reverse = server.step(updates[::-1], weights[::-1], [torch.zeros(2)])[0]
    torch.testing.assert_close(reverse, actual)


def test_fedadam_uses_weighted_delta_and_retains_moments(monkeypatch):
    api = aggregation_module(monkeypatch)
    config = api.AggregationConfig(method='fedadam', server_lr=2., beta1=0.5, beta2=0.5, tau=1.)
    server = api.ServerAggregator(config)
    first = server.step([[torch.tensor([1.])], [torch.tensor([5.])]], [0.75, 0.25], [torch.zeros(1)])
    # Weighted delta=2, m=1, v=2.5 (v_initial=tau**2), no bias correction.
    assert first[0].item() == pytest.approx(2 / (math.sqrt(2.5) + 1))
    snapshot = server.state_dict()
    second = server.step([[torch.tensor([-2.])]], [1.], [torch.zeros(1)])
    # m=-0.5, v=3.25: resetting moments or averaging moments per client fails.
    assert second[0].item() == pytest.approx(-1 / (math.sqrt(3.25) + 1))
    torch.testing.assert_close(snapshot['m'][0], torch.tensor([1.]))
    restored = api.ServerAggregator(config, state=snapshot)
    torch.testing.assert_close(restored.step([[torch.tensor([-2.])]], [1.], [torch.zeros(1)])[0], second[0])
    snapshot['m'][0].fill_(99)
    assert restored.state_dict()['m'][0].item() == pytest.approx(-0.5)


@pytest.mark.parametrize('bad', [dict(server_lr=0), dict(beta1=1), dict(beta2=-1), dict(tau=float('nan'))])
def test_invalid_server_hyperparameters_fail_early(monkeypatch, bad):
    api = aggregation_module(monkeypatch)
    with pytest.raises(ValueError):
        api.AggregationConfig(method='fedadam', **bad)


def test_cli_defaults_alias_and_separate_results(monkeypatch):
    api = aggregation_module(monkeypatch)
    parser = argparse.ArgumentParser()
    api.add_aggregation_arguments(parser)
    base = api.AggregationConfig.from_args(parser.parse_args([]))
    median = api.AggregationConfig.from_args(parser.parse_args(['--fl_aggregate', 'median']))
    adam = api.AggregationConfig.from_args(parser.parse_args(['--aggregation_method', 'fedadam', '--server_lr', '0.2']))
    assert base.method == 'fedavg' and base.output_subdirectory() == ''
    assert median.method == 'weighted_median'
    assert adam.server_lr == 0.2
    assert median.output_subdirectory() != adam.output_subdirectory()
    assert adam.output_subdirectory() != api.AggregationConfig(method='fedadam').output_subdirectory()


def make_loader():
    return SimpleNamespace(X_train_parts=[torch.zeros(3), torch.ones(1)],
        y_train_parts=[torch.zeros(3), torch.zeros(1)], X_test=torch.zeros(1), y_test=torch.zeros(1))


@pytest.mark.parametrize('method', ['fedavg', 'fedadam', 'weighted_median'])
@pytest.mark.parametrize('multi', [False, True])
def test_last_round_shapley_replays_aggregator_without_training(monkeypatch, method, multi):
    api = aggregation_module(monkeypatch)
    from module.method.quantitative_detecting.shapley_value import ShapleyValue
    ns = training_namespace()
    ns['args'].fl_aggregate = method
    ns['args'].server_lr = 0.2
    loader, model, attack = make_loader(), ToyModel(), ToyAttack()
    context = {}
    ToyModel.trained_clients = []
    result = ns['run_federated_learning'](loader, model, None if multi else 1,
        {1: attack} if multi else attack, True, 3, 1, torch.device('cpu'),
        collect_history=True, last_round_aggregation_out=context)
    final_value = float(result[0].weight.item())
    if method == 'fedavg':
        assert final_value == pytest.approx(1.5)  # 3 * (0.75 * 1 + 0.25 * -1)
    elif method == 'weighted_median':
        assert final_value == pytest.approx(3.)  # the honest client holds 75% weight
    last_model, last_grads, last_weights = result[8:11]
    before = copy.deepcopy(context['last_round_server_state'])
    calculator = ShapleyValue(loader, ToyModel(), {}, ['accuracy'], use_last_round_sv=True,
        last_round_state=last_model, last_round_client_grads=last_grads,
        last_round_weights=last_weights, **context)
    calls = len(ToyModel.trained_clients)
    grand = calculator.evaluate_subset_world({0, 1}, 'accuracy')
    assert grand == pytest.approx(final_value, abs=1e-6)
    singleton = calculator.evaluate_subset_world({1}, 'accuracy')
    calculator.evaluate_subset_world({0}, 'accuracy')
    calculator.cache = {str(set()): [0.]}
    assert calculator.evaluate_subset_world({1}, 'accuracy') == pytest.approx(singleton)
    values, _ = calculator.get_contributions()
    assert sum(values[0]) == pytest.approx(final_value, abs=1e-6)
    assert len(ToyModel.trained_clients) == calls  # no coalition retraining
    assert 1 not in ToyModel.trained_clients  # both attacker-input paths skip private training
    if method == 'fedadam':
        torch.testing.assert_close(context['last_round_server_state']['m'][0], before['m'][0])
    # Forecasts see the server step, not FedAdam's internal weighted mean (0.5).
    observed = sum(record[0].item() for record in attack.global_gradient_history)
    assert observed == pytest.approx(final_value, abs=1e-6)
    if method == 'fedadam':
        assert attack.global_gradient_history[0][0].item() != pytest.approx(0.5)


def test_fedadam_last_round_requires_pre_round_state(monkeypatch):
    api = aggregation_module(monkeypatch)
    from module.method.quantitative_detecting.shapley_value import ShapleyValue
    with pytest.raises(ValueError, match='state'):
        ShapleyValue(make_loader(), ToyModel(), {}, ['accuracy'], use_last_round_sv=True,
            aggregation_config=api.AggregationConfig(method='fedadam'),
            last_round_state=ToyModel().state_dict(),
            last_round_client_grads={0: [torch.ones(1)], 1: [torch.zeros(1)]},
            last_round_weights=[0.75, 0.25])


@pytest.mark.parametrize('method', ['fedadam', 'weighted_median'])
def test_multi_attacker_result_uses_its_actual_last_round_utility(monkeypatch, tmp_path, method):
    aggregation_module(monkeypatch)
    from test_multi_freerider import helper
    ns = training_namespace()
    args = ns['args']
    args.__dict__.update(dataset='adult', alpha=1000, client_num=8, attacker_ids=[1, 2],
        num_epoch=50, num_local_epochs=1, seed=42, trial_id=0, detecting_method='shapley',
        results_root=str(tmp_path), fl_aggregate=method)
    loader = SimpleNamespace(X_train_parts=[torch.full((20 if i == 0 else 1,), float(i)) for i in range(8)],
        y_train_parts=[torch.zeros(20 if i == 0 else 1) for i in range(8)],
        X_test=torch.zeros(1), y_test=torch.zeros(1))
    result = helper().run_experiment(args=args, loader=loader, model=ToyModel(), attack_template=ToyAttack(),
        train=ns['run_federated_learning'], detect=None,
        attach_truth=ns['_attach_round_attack_ground_truth'], device=torch.device('cpu'), provenance={})
    assert sum(result['contributions']) == pytest.approx(result['accuracy'][-1], abs=1e-5)
    assert result['metadata'].get('fl_aggregate') == method
    assert result['evaluation']['shapley_scope'] == 'actual_last_round_submissions'
    assert result['metadata']['attacker_ids'] == [1, 2]


def test_honest_removal_control_uses_same_weighted_median(monkeypatch, tmp_path):
    aggregation_module(monkeypatch)
    from module.method.freerider_control import run_control
    ns = training_namespace()
    args = ns['args']
    args.__dict__.update(dataset='adult', alpha=1000, client_num=8, control_excluded_ids=[1, 2],
        num_epoch=50, num_local_epochs=1, seed=42, trial_id=0, detecting_method='shapley',
        results_root=str(tmp_path), fl_aggregate='weighted_median')
    loader = SimpleNamespace(X_train_parts=[torch.full((20 if i == 0 else 1,), float(i)) for i in range(8)],
        y_train_parts=[torch.zeros(20 if i == 0 else 1) for i in range(8)],
        X_test=torch.zeros(1), y_test=torch.zeros(1))
    run_control(args=args, loader=loader, model=ToyModel(), train=ns['run_federated_learning'],
                device=torch.device('cpu'), provenance={})
    import json
    result = json.loads((tmp_path / 'control_result.json').read_text())
    assert sum(row['shapley_value'] for row in result['contributions']) == pytest.approx(result['accuracy'][-1])
    assert result['metadata'].get('fl_aggregate') == 'weighted_median'
    assert [row['client_id'] for row in result['contributions']] == [0, 3, 4, 5, 6, 7]


@pytest.mark.parametrize('method', ['fedavg', 'fedadam', 'weighted_median'])
def test_real_cli_preserves_single_and_multiple_attacker_flags(monkeypatch, method):
    api = aggregation_module(monkeypatch)
    tree = ast.parse((ROOT / 'FLCE_updated.py').read_text(encoding='utf-8'))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets):
            body.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == 'parser' and func.attr == 'add_argument'):
                body.append(node)
            elif isinstance(func, ast.Name) and func.id == 'add_aggregation_arguments':
                body.append(node)
    namespace = dict(argparse=argparse, add_aggregation_arguments=api.add_aggregation_arguments)
    exec(compile(ast.Module(body=body, type_ignores=[]), '<real FLCE CLI>', 'exec'), namespace)
    single = namespace['parser'].parse_args(['--fl_aggregate', method, '--attacker_id', '2'])
    multi = namespace['parser'].parse_args(['--fl_aggregate', method, '--attacker_ids', '1', '3', '5'])
    assert single.attacker_id == 2 and single.attacker_ids is None
    assert multi.attacker_ids == [1, 3, 5]
    assert single.use_last_round_sv and multi.use_last_round_sv
    assert single.fl_aggregate == multi.fl_aggregate == method


def test_weighted_median_covers_chunk_boundary_and_unequal_shape(monkeypatch):
    api = aggregation_module(monkeypatch)
    parameters = [torch.zeros(65539), torch.zeros(2, 3)]
    updates = [[torch.zeros_like(p) for p in parameters], [torch.ones_like(p) for p in parameters]]
    result = api.ServerAggregator(api.AggregationConfig(method='weighted_median')).step(updates, [0.5, 0.5], parameters)
    for actual, parameter in zip(result, parameters):
        torch.testing.assert_close(actual, torch.full_like(parameter, 0.5))


def test_first_round_fedadam_coalition_reuses_initial_state(monkeypatch):
    aggregation_module(monkeypatch)
    from module.method.quantitative_detecting.shapley_value import ShapleyValue
    ns = training_namespace()
    ns['args'].fl_aggregate = 'fedadam'
    context = {}
    result = ns['run_federated_learning'](make_loader(), ToyModel(), None, None, False,
        1, 1, torch.device('cpu'), last_round_aggregation_out=context)
    assert context['last_round_server_state']['rounds'] == 0
    calculator = ShapleyValue(make_loader(), ToyModel(), {}, ['accuracy'], use_last_round_sv=True,
        last_round_state=result[3], last_round_client_grads=result[4], last_round_weights=result[5], **context)
    assert calculator.evaluate_subset_world({0, 1}, 'accuracy') == pytest.approx(result[2][-1])


def test_fedadam_forecast_and_dump_only_see_broadcast_model_difference(monkeypatch, tmp_path):
    aggregation_module(monkeypatch)
    import csv
    import os
    ns = training_namespace()
    ns.update(csv=csv, os=os, dataset='fixture')
    ns['args'].__dict__.update(fl_aggregate='fedadam', dump_grad_history=True, dump_dir=str(tmp_path))
    model, attack = ToyModel(), ToyAttack()
    with torch.no_grad():
        model.weight.fill_(1e8)
    ns['run_federated_learning'](make_loader(), model, 1, attack, True, 1, 1, torch.device('cpu'))
    # A nonzero internal Adam step is below float32 resolution here. It cannot
    # be inferred from the broadcast models and must not leak into predictions.
    assert model.weight.item() == 1e8
    assert attack.global_gradient_history[0][0].item() == 0.
    with (tmp_path / 'fixture/run_0.csv').open(newline='') as handle:
        row = list(csv.DictReader(handle))[0]
    assert float(row['d0']) == 0.

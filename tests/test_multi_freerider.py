"""Behavior checks without loading/downloading any of the experiment datasets."""
import ast
import copy
import importlib.util
import random
import sys
import time
import typing
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_cpu_fixture():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def helper():
    path = ROOT / "module/method/multi_freerider.py"
    assert path.exists(), "The dedicated multi-free-rider implementation is missing"
    spec = importlib.util.spec_from_file_location("multi_freerider_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_enforces_requested_torch_threads(monkeypatch):
    monkeypatch.setenv('FLCE_CPU_THREADS_PER_TASK', '15')
    assert helper().configure_worker_threads() == 15
    assert torch.get_num_threads() == 15


def training_namespace(lazy=False, detector="shapley"):
    # Load the real, dataset-independent server dependency of the AST runner.
    name = '_multi_test_fl_aggregation'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'module/method/fl_aggregation.py')
        aggregation = importlib.util.module_from_spec(spec)
        sys.modules[name] = aggregation
        spec.loader.exec_module(aggregation)
    aggregation = sys.modules[name]
    tree = ast.parse((ROOT / "FLCE_updated.py").read_text(encoding="utf-8"))
    names = {
        "_resolve_round_attack_activity", "_attach_round_attack_ground_truth",
        "_should_use_online_qualitative_aggregation", "_aggregate_round_updates",
        "_fit_local_model_with_optional_wef", "run_federated_learning",
        "_assert_submitted_update_contract", "_assert_all_grads_on_device",
        "_assert_model_params_on_device", "_assert_weights_on_device",
    }
    ns = dict(torch=torch, np=np, typing=typing, copy=copy, time=time,
              AggregationConfig=aggregation.AggregationConfig, ServerAggregator=aggregation.ServerAggregator,
              seed=42, trial_id=0, R_FOR_SV=20,
              detecting_method=detector,
              detecting_type="qualitative" if detector == "std_dagmm" else "quantitative",
              FORECAST_FREE_RIDER_METHODS={"moirai_online"},
              publish_realized_fake_rounds=lambda output, rounds: None,
              args=SimpleNamespace(attack_method="moirai_online", use_lazy=lazy,
                  lazy_schedule="fixed", lazy_honest_gap=6, lazy_honest_phase=0,
                  lazy_fake_policy="attack", attack_probability=0.82,
                  qualitative_aggregation_mode="passive", use_hessian=True,
                  lambda_h=0.9, dump_grad_history=False))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / "FLCE_updated.py"), "exec"), ns)
    return ns


class ToyModel(torch.nn.Module):
    trained_clients = []

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")
        self.num_epoch = 50

    def fit(self, x, y, **kwargs):
        self.trained_clients.append(int(x[0].item()))
        with torch.no_grad():
            self.weight.add_(float(x[0].item()) + 1)

    def score(self, x, y, metrics):
        return [float(self.weight.item())]


class ToyAttack:
    supports_round_aware_recording = True

    def __init__(self):
        self.global_gradient_history = []
        self.last_source = "test_fake"

    def get_fake_gradient(self, round_num, device, model):
        return [torch.full_like(p, -1) for p in model.parameters()]

    def record_global_gradient(self, gradients, round_num=None):
        self.global_gradient_history.append([g.clone() for g in gradients])


class RandomAttack(ToyAttack):
    def get_fake_gradient(self, round_num, device, model):
        return [torch.tensor([random.random(), np.random.random(), torch.rand(()).item()])]


@pytest.mark.parametrize("lazy,expected_real,expected_fake", [(False, 0, 50), (True, 9, 41)])
@pytest.mark.parametrize("ids", [[1], [1, 2], [1, 2, 3]])
def test_fifty_round_schedule_labels_all_attackers(lazy, expected_real, expected_fake, ids):
    resolve = training_namespace(lazy)["_resolve_round_attack_activity"]
    activities = [resolve(r, client_ids=range(8), use_attack=True, attacker_id=None,
        attacker_ids=ids, use_lazy=lazy, lazy_schedule="fixed", lazy_honest_gap=6,
        lazy_honest_phase=0, probabilistic_fake_rounds=frozenset()) for r in range(50)]
    for client in range(8):
        assert sum(a["ground_truth_labels"][client] for a in activities) == (expected_fake if client in ids else 0)
        assert all(a["designated_attacker_labels"][client] == int(client in ids) for a in activities)
    assert sum(a["is_honest_round"] for a in activities) == expected_real
    assert activities[-1]["active_attackers"] == ids


def test_independent_rng_state_order_and_public_history():
    make = helper().make_attackers
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    left = make(RandomAttack(), [1, 2, 3], seed=42, device=torch.device("cpu"))
    right = make(RandomAttack(), [3, 1], seed=42, device=torch.device("cpu"))
    outputs = {i: left[i].get_fake_gradient(0, "cpu", ToyModel())[0] for i in [1, 2, 3]}
    for i in [3, 1]:
        assert torch.equal(outputs[i], right[i].get_fake_gradient(0, "cpu", ToyModel())[0])
    assert not torch.equal(outputs[1], outputs[2])
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    public = [torch.ones(1)]
    for attack in left.values():
        attack.record_global_gradient(public, round_num=0)
    left[1].global_gradient_history[0][0].zero_()
    assert left[2].global_gradient_history[0][0].item() == 1
    assert public[0].item() == 1


@pytest.mark.parametrize("lazy,expected_real,expected_fake", [(False, 0, 50), (True, 9, 41)])
def test_actual_training_loop_skips_private_data_and_keeps_eight_clients(lazy, expected_real, expected_fake):
    ns = training_namespace(lazy)
    ids = [1, 2, 3]
    attacks = helper().make_attackers(ToyAttack(), ids, seed=42, device=torch.device("cpu"))
    loader = SimpleNamespace(X_train_parts=[torch.tensor([i]) for i in range(8)],
        y_train_parts=[torch.zeros(1) for _ in range(8)], X_test=torch.zeros(1), y_test=torch.zeros(1))
    ToyModel.trained_clients = []
    result = ns["run_federated_learning"](loader, ToyModel(), None, attacks, True,
        50, 1, torch.device("cpu"), collect_history=True)
    for i in range(8):
        assert ToyModel.trained_clients.count(i) == (expected_real if i in ids else 50)
    for attack in attacks.values():
        assert len(attack.fake_calls) == expected_fake
        assert len(attack.global_gradient_history) == 50
    assert len(result[2]) == 50
    assert set(result[9]) == set(range(8))  # cached actual final submissions
    assert result[10] == [0.125] * 8
    expected_increment = (sum(i + 1 for i in range(8) if i not in ids) - len(ids)) / 8
    if not lazy:
        assert result[2][-1] == pytest.approx(50 * expected_increment)


def test_legacy_single_attacker_loop_still_works():
    ns = training_namespace(False)
    loader = SimpleNamespace(X_train_parts=[torch.tensor([i]) for i in range(8)],
        y_train_parts=[torch.zeros(1) for _ in range(8)], X_test=torch.zeros(1), y_test=torch.zeros(1))
    ToyModel.trained_clients = []
    result = ns["run_federated_learning"](loader, ToyModel(), 1, ToyAttack(), True,
        2, 1, torch.device("cpu"), collect_history=True)
    assert ToyModel.trained_clients.count(1) == 0
    assert result[6][0]["active_attackers"] == [1]


def evaluator_modules(monkeypatch):
    # Avoid dataset loader imports (torchvision/downloads), use the real evaluators.
    for name, path in [("module", ROOT / "module"), ("module.method", ROOT / "module/method")]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    data = ModuleType("module.data_manager")
    data.DataManager = object
    monkeypatch.setitem(sys.modules, "module.data_manager", data)
    from module.method.qualitative_detecting.std_dagmm import STDDAGMMConfig, STDDAGMMDetector
    return STDDAGMMConfig, STDDAGMMDetector


@pytest.mark.parametrize("method,lazy", [("shapley", False), ("std_dagmm", True)])
def test_new_result_path_uses_real_evaluators_without_per_task_csv(tmp_path, monkeypatch, method, lazy):
    Config, Detector = evaluator_modules(monkeypatch)
    ns = training_namespace(lazy, method)
    args = ns["args"]
    args.__dict__.update(dataset="adult", alpha=1000, client_num=8, attacker_ids=[1, 2, 3],
                         num_epoch=50, num_local_epochs=1, seed=42, trial_id=0,
                         detecting_method=method, results_root=str(tmp_path))
    loader = SimpleNamespace(X_train_parts=[torch.tensor([i]) for i in range(8)],
        y_train_parts=[torch.zeros(1) for _ in range(8)], X_test=torch.zeros(1), y_test=torch.zeros(1))
    def detect(history):
        return Detector.detect_free_riders_from_history(history, config=Config(epochs=1), device="cpu")
    result = helper().run_experiment(args=args, loader=loader, model=ToyModel(), attack_template=ToyAttack(),
        train=ns["run_federated_learning"], detect=detect,
        attach_truth=ns["_attach_round_attack_ground_truth"], device=torch.device("cpu"), provenance={})
    assert (tmp_path / "result.json").exists()
    assert not list(tmp_path.rglob("*.csv"))
    assert result["metadata"]["attacker_ids"] == [1, 2, 3]
    assert len(result["accuracy"]) == 50
    assert all(a["fake_count"] == (41 if lazy else 50) for a in result["attackers"])
    if method == "shapley":
        assert len(result["contributions"]) == 8
        assert sum(result["contributions"]) == pytest.approx(result["accuracy"][-1], abs=1e-5)
    else:
        assert result["evaluation"]["valid_auc_rounds"] == 41
        records = result["detection"]
        assert records[0]["roc_auc"] is None
        assert records[1]["active_attackers"] == [1, 2, 3]
        assert sum(records[1]["ground_truth_labels"].values()) == 3
        assert all(record["tp"] + record["fn"] == (0 if r % 6 == 0 else 3)
                   for r, record in enumerate(records))


@pytest.mark.parametrize('family,layerwise', [('moirai', False), ('arima', False), ('moirai', True), ('arima', True)])
def test_real_forecast_classes_keep_separate_state_during_fake_warmup(monkeypatch, family, layerwise):
    evaluator_modules(monkeypatch)
    from module.method import attack_updated_nonlayerwise as plain
    from module.method.attack_updated_nonlayerwise import OnlineMoiraiAttack, AdaptiveARIMAAttack
    from module.method.attack_updated_layerwise import (
        LayerwiseOnlineMoiraiAttackMNISTIMDB, LayerwiseAdaptiveARIMAAttackMNISTIMDB)
    model = type('MNISTMLP', (torch.nn.Linear,), {})(2, 2)
    if family == 'moirai' and not layerwise:
        # This class loads pretrained weights even for round-0 random fallback.
        # Stub only that external dependency; exercise the actual attack/state API.
        monkeypatch.setattr(plain, '_load_moirai_module', lambda path: object())
        monkeypatch.setattr(plain, 'MoiraiForecast', lambda **kwargs:
                            SimpleNamespace(create_predictor=lambda **kw: object()))
    if layerwise:
        attack_class = LayerwiseOnlineMoiraiAttackMNISTIMDB if family == 'moirai' else LayerwiseAdaptiveARIMAAttackMNISTIMDB
        template = attack_class(k=5, random_round=10, model=model)
    elif family == 'moirai':
        template = OnlineMoiraiAttack(k=5, random_round=10)
    else:
        template = AdaptiveARIMAAttack(k=5, random_round=10, model_params=list(model.parameters()))
    attacks = helper().make_attackers(template, [1, 2, 3], seed=42, device='cpu')
    for attack in attacks.values():
        gradients = attack.get_fake_gradient(0, torch.device('cpu'), model)
        assert len(gradients) == len(list(model.parameters()))
        assert all(torch.isfinite(g).all() for g in gradients)
        kwargs = {'round_num': 0} if getattr(attack, 'supports_round_aware_recording', False) else {}
        attack.record_global_gradient([torch.ones_like(p) for p in model.parameters()], **kwargs)
        assert len(attack.global_gradient_history) == 1
        assert len(attack.fake_calls) == 1
    assert len(template.global_gradient_history) == 0
    attacks[1].global_gradient_history.clear()
    assert len(attacks[2].global_gradient_history) == 1
    if family == 'moirai' and not layerwise:
        assert attacks[1].model is not attacks[2].model


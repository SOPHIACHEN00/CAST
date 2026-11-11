"""Opt-in multi-free-rider path; existing attack and detector algorithms are reused."""
from contextlib import contextmanager
from argparse import Namespace
from collections.abc import Callable, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def validate_config(args: Namespace) -> None:
    ids = args.attacker_ids
    if not args.use_attack or len(ids) not in (1, 2, 3):
        raise ValueError('--attacker_ids requires --use_attack and 1, 2 or 3 IDs')
    if len(set(ids)) != len(ids) or any(i < 0 or i >= args.client_num for i in ids):
        raise ValueError('Attacker IDs must be distinct valid client IDs')
    alpha = {'adult': (1000, 20000000), 'mnist': (10, 10000000), 'imdb': (10, 200000)}
    dataset = args.dataset.lower()
    if dataset not in alpha or args.alpha not in alpha[dataset] or args.client_num != 8:
        raise ValueError('This experiment fixes N=8 and first-column or Uniform alpha: '
                         'Adult=1000/20000000, MNIST=10/10000000, IMDB=10/200000')
    if args.num_epoch != 50 or args.num_local_epochs != 1:
        raise ValueError('This experiment fixes 50 rounds and 1 local epoch')
    suffix = '' if dataset == 'adult' else '_mnist_imdb_layerwise'
    if args.attack_method not in {family + suffix for family in ('moirai_online', 'arima_adaptive', 'moirai', 'arima')}:
        raise ValueError('Use the approved offline/online Moirai or offline/adaptive ARIMA variant for this dataset')
    if args.detecting_method not in {'shapley', 'std_dagmm'}:
        raise ValueError('Multi-free-rider detection is limited to exact Shapley and STD-DAGMM')
    if not args.use_last_round_sv or args.qualitative_aggregation_mode != 'passive':
        raise ValueError('Use actual last-round Shapley and passive aggregation')
    if args.use_lazy and (args.lazy_schedule != 'fixed' or args.lazy_honest_gap != 6
                         or args.lazy_honest_phase != 0 or args.lazy_fake_policy != 'attack'):
        raise ValueError('Lazy must repeat 1 real + 5 fake: fixed gap=6, phase=0, fake_policy=attack')
    if args.dump_grad_history:
        raise ValueError('The multi-free-rider path writes JSON intermediates, not per-task gradient CSVs')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')


class IndependentAttacker:
    """Own mutable predictor state and RNG streams; expose only the public attack API.

    Calls run sequentially inside one FL worker. The surrounding training RNG is
    restored even when forecasting raises, so adding an attacker does not consume
    another client's training or attack random stream.
    """

    def __init__(self, template, client_id: int, seed: int, device: torch.device | str):
        self.inner = copy.deepcopy(template)
        self.client_id = int(client_id)
        self.seed = (int(seed) + 9973 * (self.client_id + 1)) % (2**32)
        self.fake_calls = []
        self._python_state = random.Random(self.seed).getstate()
        self._numpy_state = np.random.RandomState(self.seed).get_state()
        self._torch_state = torch.Generator(device='cpu').manual_seed(self.seed).get_state()
        device = torch.device(device)
        self._cuda_devices = [device.index or 0] if device.type == 'cuda' else []
        self._cuda_states = {
            d: torch.Generator(device=f'cuda:{d}').manual_seed(self.seed).get_state()
            for d in self._cuda_devices
        }

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'inner'), name)

    @contextmanager
    def _rng_scope(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=self._cuda_devices):
            random.setstate(self._python_state)
            np.random.set_state(self._numpy_state)
            torch.set_rng_state(self._torch_state)
            for d, state in self._cuda_states.items():
                torch.cuda.set_rng_state(state, device=d)
            try:
                yield
            finally:
                self._python_state = random.getstate()
                self._numpy_state = np.random.get_state()
                self._torch_state = torch.get_rng_state()
                for d in self._cuda_devices:
                    self._cuda_states[d] = torch.cuda.get_rng_state(d)
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    def get_fake_gradient(self, round_num: int, device: torch.device | str, model) -> list[torch.Tensor]:
        with self._rng_scope():
            result = self.inner.get_fake_gradient(round_num, device, model)
        self.fake_calls.append({'round_idx': int(round_num),
                                'source': str(getattr(self.inner, 'last_source', 'unspecified'))})
        return result

    def record_global_gradient(self, gradients: Sequence[torch.Tensor], **kwargs) -> None:
        with self._rng_scope():
            self.inner.record_global_gradient([g.detach().clone() for g in gradients], **kwargs)


def make_attackers(template, attacker_ids: Sequence[int], *, seed: int,
                   device: torch.device | str) -> dict[int, IndependentAttacker]:
    return {i: IndependentAttacker(template, i, seed, device) for i in attacker_ids}


def configure_worker_threads() -> int:
    requested = os.environ.get('FLCE_CPU_THREADS_PER_TASK')
    if requested is not None:
        threads = int(requested)
        if threads < 1:
            raise ValueError('FLCE_CPU_THREADS_PER_TASK must be positive')
        # OMP/MKL environment settings alone can be capped by the local runtime.
        torch.set_num_threads(threads)
    return torch.get_num_threads()


def _json_default(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value)}')


def _sync(device: torch.device | str) -> None:
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _binary_metrics(record: dict) -> dict:
    labels = record['ground_truth_labels']
    predicted = set(record['suspicious_clients'])
    tp = sum(bool(label) and i in predicted for i, label in labels.items())
    fp = sum(not label and i in predicted for i, label in labels.items())
    fn = sum(bool(label) and i not in predicted for i, label in labels.items())
    tn = sum(not label and i not in predicted for i, label in labels.items())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                tpr=tp / (tp + fn) if tp + fn else None,
                fpr=fp / (fp + tn) if fp + tn else None)


def run_experiment(*, args: Namespace, loader, model, attack_template, train: Callable,
                   detect: Callable, attach_truth: Callable, device: torch.device | str,
                   provenance: dict) -> dict:
    """Train once, evaluate actual submissions, atomically publish one JSON result.

    No N-k control trajectory and no per-task CSV are produced. The campaign's
    single merger owns public dataset CSVs. Timing includes CUDA synchronization;
    the legacy attack-call timer is retained separately as host-side diagnostic.
    """
    started = time.perf_counter()
    cpu_threads = configure_worker_threads()
    target = Path(args.results_root) / 'result.json'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite completed task: {target}')
    attackers = make_attackers(attack_template, args.attacker_ids, seed=args.seed, device=device)
    _sync(device)
    fl_started = time.perf_counter()
    aggregation_context = {}
    (trained_model, attack_times, accuracy, history, _, _, activities, _,
     last_state, last_grads, last_weights, _) = train(
        loader, model, None, attackers, True, args.num_epoch, args.num_local_epochs,
        device, collect_history=True, last_round_aggregation_out=aggregation_context,
    )
    _sync(device)
    fl_seconds = time.perf_counter() - fl_started
    expected_fake = [r for r in range(args.num_epoch) if not args.use_lazy or r % 6 != 0]
    audit = []
    for client_id, attack in attackers.items():
        realized = [call['round_idx'] for call in attack.fake_calls]
        if realized != expected_fake:
            raise RuntimeError(f'Client {client_id}: actual fake calls {realized} != {expected_fake}')
        labelled = [a['round_idx'] for a in activities if client_id in a['active_attackers']]
        if labelled != realized:
            raise RuntimeError(f'Client {client_id}: ground truth does not match actual submissions')
        audit.append(dict(client_id=client_id, seed=attack.seed, fake_count=len(realized),
                          real_count=args.num_epoch - len(realized), fake_calls=attack.fake_calls))
    # Each forecast model can be large; evaluation no longer needs any attacker.
    del attack
    attackers.clear()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()

    eval_started = time.perf_counter()
    contributions, detection, evaluation = [], [], {}
    if args.detecting_method == 'shapley':
        from module.method.quantitative_detecting.shapley_value import ShapleyValue

        calculator = ShapleyValue(
            loader, trained_model, {}, ['accuracy'], attacker=None, attacker_id=None,
            use_last_round_sv=True, last_round_state=last_state,
            last_round_client_grads=last_grads, last_round_weights=last_weights,
            **aggregation_context,
        )
        values, _ = calculator.get_contributions()
        contributions = [float(value) for value in values[0]]
        if len(contributions) != args.client_num or not np.isfinite(contributions).all():
            raise RuntimeError('Invalid Shapley result')
        evaluation = dict(utility='accuracy', shapley_scope='actual_last_round_submissions',
                          empty_coalition_utility=0.0, final_round_idx=args.num_epoch - 1)
    else:
        from module.method.qualitative_detecting.std_dagmm import STDDAGMMDetector

        result = attach_truth(detect(history), activities,
                              score_evaluator=STDDAGMMDetector.evaluate_round_scores)
        detection = result['round_records']
        if len(detection) != args.num_epoch:
            raise RuntimeError('STD-DAGMM did not evaluate every round')
        for record in detection:
            if set(record['client_scores']) != set(range(args.client_num)):
                raise RuntimeError('STD-DAGMM is missing client energies')
            if not np.isfinite(list(record['client_scores'].values())).all():
                raise RuntimeError('STD-DAGMM returned non-finite energies')
            record.update(_binary_metrics(record))
        evaluation = dict(result['evaluation'])
        evaluation['expected_valid_auc_rounds'] = len(expected_fake)
        if evaluation['valid_auc_rounds'] != len(expected_fake):
            raise RuntimeError('Unexpected STD-DAGMM AUC coverage')
        totals = {key: sum(record[key] for record in detection) for key in ('tp', 'fp', 'fn', 'tn')}
        evaluation.update(totals)
        evaluation['tpr'] = totals['tp'] / (totals['tp'] + totals['fn'])
        evaluation['fpr'] = totals['fp'] / (totals['fp'] + totals['tn'])
        fake_records = [record for record in detection if record['is_fake_round']]
        fp_fake = sum(record['fp'] for record in fake_records)
        tn_fake = sum(record['tn'] for record in fake_records)
        evaluation['fpr_fake_rounds'] = fp_fake / (fp_fake + tn_fake)
        evaluation['decision_rule'] = 'energy_above_round_median'
    _sync(device)
    eval_seconds = time.perf_counter() - eval_started
    if len(accuracy) != args.num_epoch or not np.isfinite(accuracy).all():
        raise RuntimeError('Invalid accuracy trajectory')
    metadata = {key: getattr(args, key) for key in (
        'dataset', 'alpha', 'client_num', 'attacker_ids', 'attack_method', 'detecting_method',
        'trial_id', 'seed', 'num_epoch', 'num_local_epochs', 'use_hessian', 'lambda_h',
    )}
    metadata.update(attacker_count=len(args.attacker_ids), regime='lazy' if args.use_lazy else 'pure',
                    collusion=False, cpu_threads_per_task=cpu_threads, fake_rounds_per_attacker=len(expected_fake),
                    real_rounds_per_attacker=args.num_epoch - len(expected_fake),
                    lazy_honest_gap=6 if args.use_lazy else None,
                    lazy_honest_phase=0 if args.use_lazy else None,
                    qualitative_aggregation_mode='passive')
    metadata['fl_aggregate'] = aggregation_context['aggregation_config']['method']
    metadata['fl_aggregation'] = aggregation_context['aggregation_config']
    if args.detecting_method == 'shapley':
        evaluation['server_state_rounds_before_last'] = aggregation_context['last_round_server_state']['rounds']
    helper_path = Path(__file__).resolve()
    artifact = dict(
        schema_version=1, metadata=metadata, parameters=vars(args), provenance=provenance,
        helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
        accuracy=[float(value) for value in accuracy], contributions=contributions,
        detection=detection, evaluation=evaluation, activities=activities, attackers=audit,
        shard_sizes=[len(shard) for shard in loader.X_train_parts], shard_weights=last_weights,
        timing=dict(fl_sec=fl_seconds, evaluation_sec=eval_seconds,
                    attack_host_sec=sum(attack_times), total_sec=time.perf_counter() - started),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(artifact, default=_json_default, allow_nan=False, indent=2), encoding='utf-8')
    temporary.replace(target)
    print(f'[MULTI_FREE_RIDER] Completed: {target}', flush=True)
    return artifact

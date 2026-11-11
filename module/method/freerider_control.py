"""Honest-only RQ2 controls on subsets of the original fixed eight-client split."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from module.method.multi_freerider import configure_worker_threads


def validate_control(args):
    ids = args.control_excluded_ids
    if (len(ids) not in (1, 2, 3) or len(set(ids)) != len(ids)
            or any(i not in range(8) for i in ids)):
        raise ValueError('Control requires 1/2/3 distinct excluded original IDs in 0..7')
    if args.use_attack or args.use_lazy or args.attacker_ids is not None:
        raise ValueError('Control cannot enable attacks, lazy training or attacker_ids')
    if (args.client_num != 8 or args.seed != 42 or args.num_epoch != 50
            or args.num_local_epochs != 1 or args.trial_id not in (0, 1, 2)):
        raise ValueError('Control fixes original N=8, seed=42, 50 rounds, 1 local epoch, group=0/1/2')
    alpha = {'adult': (1000, 20000000), 'mnist': (10, 10000000), 'imdb': (10, 200000)}
    if args.dataset not in alpha or args.alpha not in alpha[args.dataset]:
        raise ValueError('Control requires first-column or Uniform alpha: '
                         'Adult=1000/20000000, MNIST=10/10000000, IMDB=10/200000')
    if args.detecting_method != 'shapley' or not args.use_last_round_sv or args.dump_grad_history:
        raise ValueError('Control requires exact last-round Shapley without gradient CSV dumps')
    if args.qualitative_aggregation_mode != 'passive':
        raise ValueError('Control requires passive aggregation')
    configure_worker_threads()


def tensor_hash(value):
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def run_control(*, args, loader, model, train, device, provenance):
    from module.method.quantitative_detecting.shapley_value import ShapleyValue

    target = Path(args.results_root) / 'control_result.json'
    if target.exists():
        raise FileExistsError(target)
    started = time.perf_counter()
    excluded = sorted(args.control_excluded_ids)
    kept = [i for i in range(8) if i not in excluded]
    shard_sizes = [len(x) for x in loader.X_train_parts]
    fingerprints = [dict(client_id=i, size=shard_sizes[i],
                         x_sha256=tensor_hash(loader.X_train_parts[i]),
                         y_sha256=tensor_hash(loader.y_train_parts[i])) for i in range(8)]
    initial_hashes = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    # Preserve preprocessing, original shards and test set. Do not repartition N-k.
    subset = copy.copy(loader)
    subset.X_train_parts = [loader.X_train_parts[i] for i in kept]
    subset.y_train_parts = [loader.y_train_parts[i] for i in kept]
    if hasattr(loader, 'X_val_parts'):
        subset.X_val_parts = [loader.X_val_parts[i] for i in kept]
        subset.y_val_parts = [loader.y_val_parts[i] for i in kept]
        subset.X_train = torch.cat(subset.X_train_parts, dim=0)
        subset.y_train = torch.cat(subset.y_train_parts, dim=0)
    aggregation_context = {}
    result = train(subset, model, None, None, False, 50, 1, device, collect_history=True,
                   last_round_aggregation_out=aggregation_context)
    trained, _, accuracy, _, _, _, activities, _, last_state, last_grads, weights, _ = result
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    if any(a['active_attackers'] for a in activities):
        raise RuntimeError('Unexpected attacker in honest control')
    calculator = ShapleyValue(subset, trained, {}, ['accuracy'], attacker=None, attacker_id=None,
        use_last_round_sv=True, last_round_state=last_state,
        last_round_client_grads=last_grads, last_round_weights=weights, **aggregation_context)
    values, _ = calculator.get_contributions()
    scores = [float(v) for v in values[0]]
    if len(scores) != len(kept) or len(accuracy) != 50 or not all(math.isfinite(float(v)) for v in scores + list(accuracy)):
        raise RuntimeError('Incomplete/nonfinite control result')
    meta = dict(dataset=args.dataset, alpha=args.alpha, original_client_num=8,
        control_client_num=len(kept), excluded_ids=excluded, original_client_ids=kept,
        trial_id=args.trial_id, seed=42, num_epoch=50, num_local_epochs=1,
        detecting_method='shapley', shapley_round=49, use_attack=False, use_lazy=False,
        cpu_threads_per_task=torch.get_num_threads(),
        scope='honest_only_trajectory_actual_last_round_submissions')
    meta['fl_aggregate'] = aggregation_context['aggregation_config']['method']
    meta['fl_aggregation'] = aggregation_context['aggregation_config']
    meta['server_state_rounds_before_last'] = aggregation_context['last_round_server_state']['rounds']
    artifact = dict(schema_version=1, metadata=meta, parameters=vars(args), provenance=provenance,
        original_shards=fingerprints, initial_state_sha256=initial_hashes,
        test_x_sha256=tensor_hash(loader.X_test), test_y_sha256=tensor_hash(loader.y_test),
        accuracy=[float(v) for v in accuracy],
        contributions=[dict(client_id=i, shapley_value=v, shard_size=shard_sizes[i],
                            aggregation_weight=float(w)) for i, v, w in zip(kept, scores, weights)],
        timing=dict(training_sec=training_seconds, total_sec=time.perf_counter()-started))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(f'[RQ2 CONTROL] Completed: {target}', flush=True)

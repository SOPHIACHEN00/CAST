"""Passive PASS-CE with literal Wang et al. scoring and explicit audit policies.

Rows are auditors, columns are submitted models. The detector never receives
attacker identities. Only the simulation observer decides who fabricates reports.
This module does not implement PASS-PPS, client elimination, or aggregation.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PASSConfig:
    alpha: float = 0.95
    beta: float = 1.75
    c0: float | None = None
    report_low: float = -1.0
    report_high: float = 1.0
    seed: int = 42
    audit_batch_size: int = 256
    max_audit_samples: int | None = None

    def validate(self) -> None:
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('PASS alpha must be finite and in [0, 1]')
        if not math.isfinite(self.beta) or self.beta < 1:
            raise ValueError('PASS beta must be finite and at least 1')
        if self.c0 is not None and not math.isfinite(self.c0):
            raise ValueError('PASS c0 must be finite')
        if not (-1 <= self.report_low < self.report_high <= 1):
            raise ValueError('PASS report bounds must satisfy -1 <= low < high <= 1')
        if self.audit_batch_size < 1:
            raise ValueError('PASS audit batch size must be positive')
        if self.max_audit_samples is not None and self.max_audit_samples < 1:
            raise ValueError('PASS audit sample limit must be positive or omitted')
        if not 0 <= self.seed < 2**63:
            raise ValueError('PASS seed must lie in [0, 2**63)')

    @classmethod
    def from_args(cls, args: Any) -> PASSConfig:
        limit = args.pass_max_audit_samples
        if limit < 0:
            raise ValueError('--pass_max_audit_samples must be nonnegative (0 means all)')
        result = cls(
            alpha=args.pass_alpha, beta=args.pass_beta, c0=args.pass_c0,
            report_low=args.pass_report_low, report_high=args.pass_report_high,
            seed=args.pass_seed + int(args.trial_id) * 100003,
            audit_batch_size=args.pass_audit_batch_size,
            max_audit_samples=limit or None,
        )
        result.validate()
        return result


def resolve_forged_auditors(*, use_attack: bool, attacker_id: int | None,
                            attack_method: str, use_lazy: bool) -> tuple[int, ...]:
    """Preserve project regimes; synthetic replacement has real local data."""
    if not use_attack or attacker_id is None or use_lazy or attack_method == 'synthetic_replacement':
        return ()
    return (int(attacker_id),)


def attack_resource_notes(attack_method: str, use_attack: bool, use_lazy: bool) -> list[str]:
    """Describe existing attack-side resource access; never alter that attack."""
    if not use_attack:
        return []
    notes = ['Attack implementation and its existing data access are unchanged.']
    if attack_method == 'dkfr':
        notes.append('DKFR initialization uses pooled private-training feature statistics; not strict zero-private-information.')
    elif attack_method == 'synthetic_replacement':
        notes.append('Synthetic replacement is constructed from real local samples/labels; audits use the original real shard.')
    elif attack_method == 'adam_local_deviation':
        notes.append('The attack has a public-data warm start; its non-lazy audit reports remain random in this experiment.')
    elif attack_method in {'random', 'fraboni_disguised', 'scaled_delta', 'advanced_scaled_delta'}:
        notes.append('Existing pure/fixed-lazy branches may train on the private shard before replacing the submission.')
    if use_lazy:
        notes.append('Lazy audits are truthful in both honest-update and fake-update rounds.')
    return notes


def _finite_list(values: torch.Tensor) -> list[Any]:
    """Export only at the logging boundary; JSON never contains NaN/Infinity."""
    def clean(value: Any) -> Any:
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value if not isinstance(value, float) or math.isfinite(value) else None
    return clean(values.detach().cpu().tolist())


class PASSDetector:
    """Stateful literal recurrence, with one independent state per target."""

    def __init__(self, num_clients: int, config: PASSConfig | None = None,
                 device: torch.device | str = 'cpu') -> None:
        if num_clients < 1:
            raise ValueError('PASS requires at least one participant')
        self.config = config or PASSConfig()
        self.config.validate()
        self.num_clients = int(num_clients)
        self.device = torch.device(device)
        initial = 1.0 / num_clients if self.config.c0 is None else self.config.c0
        self.contributions = torch.full((num_clients,), initial, dtype=torch.float64, device=self.device)
        self.threshold = 1.0 / (self.config.beta * num_clients)

    @torch.no_grad()
    def score_reports(self, reports: torch.Tensor, *, submitted_round: int,
                      decision_round: int) -> dict[str, Any]:
        n = self.num_clients
        matrix = torch.as_tensor(reports, device=self.device, dtype=torch.float64)
        if matrix.shape != (n, n):
            raise ValueError(f'Expected an auditor-by-target {(n, n)} report matrix')
        diagonal = torch.eye(n, device=self.device, dtype=torch.bool)
        report_ok = (torch.isfinite(matrix) & (matrix.abs() <= 1)) | diagonal
        # Mask self-reports, not auditors. One invalid incoming report invalidates
        # the target; do not secretly change the published N-1 denominator.
        means = matrix.masked_fill(diagonal, 0).sum(dim=0) / max(n - 1, 1)
        previous_ok = torch.isfinite(self.contributions)
        denominators = self.config.alpha * self.contributions + (1 - self.config.alpha) * torch.tanh(means)
        next_values = denominators.reciprocal()
        incoming_ok = report_ok.all(dim=0)
        valid = incoming_ok & previous_ok & torch.isfinite(denominators) & (denominators != 0) & torch.isfinite(next_values)
        if n < 2:
            valid.fill_(False)
        self.contributions = torch.where(valid, next_values, torch.full_like(next_values, float('nan')))
        predicted = valid & (self.contributions < self.threshold)
        valid_list = valid.cpu().tolist()
        incoming_list = incoming_ok.cpu().tolist()
        previous_list = previous_ok.cpu().tolist()
        denominator_list = _finite_list(denominators)
        reasons = {}
        for i in range(n):
            if valid_list[i]:
                reasons[i] = None
            elif n < 2:
                reasons[i] = 'no_other_auditor'
            elif not previous_list[i]:
                reasons[i] = 'previous_contribution_invalid'
            elif not incoming_list[i]:
                reasons[i] = 'invalid_incoming_report'
            elif denominator_list[i] == 0:
                reasons[i] = 'zero_denominator'
            else:
                reasons[i] = 'nonfinite_contribution'
        return {
            'round_idx': int(submitted_round), 'submitted_round': int(submitted_round),
            'audit_round': int(decision_round), 'decision_round': int(decision_round),
            'client_scores': dict(enumerate(_finite_list(self.contributions))),
            'mean_divergences': dict(enumerate(_finite_list(means))),
            'denominators': dict(enumerate(denominator_list)),
            'valid_by_client': dict(enumerate(valid_list)), 'invalid_reasons': reasons,
            'suspicious_clients': torch.nonzero(predicted, as_tuple=False).flatten().cpu().tolist(),
            'contribution_threshold': self.threshold,
            'detection_skipped': not any(valid_list),
            'report_matrix': _finite_list(matrix.masked_fill(diagonal, float('nan'))),
        }


class PASSAuditObserver:
    """Audit t-1 at the start of round t without touching live training state."""

    def __init__(self, model: nn.Module, x_parts: Sequence[Any], y_parts: Sequence[Any],
                 *, forged_auditors: Sequence[int] = (), config: PASSConfig | None = None,
                 resource_notes: Sequence[str] = ()) -> None:
        self.config = config or PASSConfig()
        self.config.validate()
        self.num_clients = len(x_parts)
        if len(y_parts) != self.num_clients:
            raise ValueError('PASS X/y client partitions must match')
        self.device = next(model.parameters()).device
        self.detector = PASSDetector(self.num_clients, self.config, self.device)
        self.forged_auditors = frozenset(int(i) for i in forged_auditors)
        if any(i < 0 or i >= self.num_clients for i in self.forged_auditors):
            raise ValueError('PASS forged auditor ID is out of range')
        self.model = copy.deepcopy(model).eval()
        self.resource_notes = list(resource_notes)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)
        self.audit_data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        self.sample_counts = {i: 0 for i in range(self.num_clients)}
        for i in range(self.num_clients):
            if i in self.forged_auditors:
                continue  # Do not even index X/y for a pure auditor.
            x, y = x_parts[i], y_parts[i]
            if len(x) != len(y):
                raise ValueError(f'PASS auditor {i} has mismatched X/y lengths')
            indices = None
            count = len(y)
            limit = self.config.max_audit_samples
            if limit is not None and count > limit:
                sample_rng = torch.Generator(device=x.device)
                sample_rng.manual_seed((self.config.seed + 1000003 * (i + 1)) % (2**63))
                indices = torch.randperm(count, generator=sample_rng, device=x.device)[:limit]
                count = limit
            self.audit_data[i] = (x, y, indices)
            self.sample_counts[i] = count
        self.pending: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.detector_time_sec = 0.0

    def _accuracy(self, auditor: int) -> torch.Tensor:
        x, y, indices = self.audit_data[auditor]
        count = self.sample_counts[auditor]
        correct = torch.zeros((), dtype=torch.float64, device=self.device)
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        if count == 0:
            return correct.fill_(float('nan'))
        for start in range(0, count, self.config.audit_batch_size):
            end = min(start + self.config.audit_batch_size, count)
            if indices is None:
                xb, yb = x[start:end], y[start:end]
            else:
                selected = indices[start:end]
                xb = x.index_select(0, selected)
                yb = y.index_select(0, selected.to(y.device))
            output = self.model(xb.to(self.device))
            finite &= torch.isfinite(output).all()
            prediction = output.argmax(dim=1) if output.ndim > 1 and output.shape[-1] > 1 else (output.reshape(-1) >= 0.5).long()
            target = yb.to(self.device).reshape(-1).long()
            if prediction.numel() != target.numel():
                raise ValueError('PASS model predictions and audit labels have incompatible shapes')
            correct += (prediction.reshape(-1) == target).sum()
        return torch.where(finite, correct / count, torch.full_like(correct, float('nan')))

    @torch.inference_mode()
    def start_round(self, round_index: int, global_model: nn.Module) -> dict[str, Any] | None:
        if self.pending is None:
            return None
        if round_index != self.pending['round_idx'] + 1:
            raise ValueError('PASS audits must follow the submitted round by exactly one round')
        n = self.num_clients
        reports = torch.full((n, n), float('nan'), dtype=torch.float64, device=self.device)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.model.load_state_dict(global_model.state_dict())
            self.model.eval()
            global_accuracies = {i: self._accuracy(i) for i in self.audit_data} if n > 1 else {}
            for target in range(n) if n > 1 else ():
                for parameter, submitted in zip(self.model.parameters(), self.pending['models'][target]):
                    parameter.copy_(submitted)
                for name, buffer in self.model.named_buffers():
                    buffer.copy_(self.pending['buffers'][name])
                for auditor in self.audit_data:
                    if auditor != target:
                        reports[auditor, target] = global_accuracies[auditor] - self._accuracy(auditor)
            for auditor in sorted(self.forged_auditors):
                targets = [j for j in range(n) if j != auditor]
                if targets:
                    reports[auditor, targets] = torch.empty(len(targets), dtype=torch.float64, device=self.device).uniform_(
                        self.config.report_low, self.config.report_high, generator=self.generator)
        record = self.detector.score_reports(reports, submitted_round=self.pending['round_idx'], decision_round=round_index)
        record['audit_sample_counts'] = dict(self.sample_counts) if n > 1 else {i: 0 for i in range(n)}
        record['report_sources'] = {i: 'random_uniform' if i in self.forged_auditors else 'real_local_data' for i in range(n)}
        self.records.append(record)
        self.pending = None
        return record

    @torch.no_grad()
    def capture_submissions(self, round_index: int,
                            base_parameters: Sequence[Sequence[torch.Tensor]],
                            submitted_updates: Sequence[Sequence[torch.Tensor]],
                            base_model: nn.Module) -> None:
        if self.pending is not None:
            raise ValueError('Previous PASS submission has not been audited')
        if len(base_parameters) != self.num_clients or len(submitted_updates) != self.num_clients:
            raise ValueError('PASS requires the actual submission of every participant')
        shapes = [p.shape for p in self.model.parameters()]
        submitted_models = []
        for bases, updates in zip(base_parameters, submitted_updates):
            if len(bases) != len(shapes) or len(updates) != len(shapes):
                raise ValueError('PASS submission parameter counts differ from the model')
            if any(base.shape != shape or update.shape != shape for base, update, shape in zip(bases, updates, shapes)):
                raise ValueError('PASS submission parameter shapes differ from the model')
            submitted_models.append([(base.detach() + update.detach()).to(self.device) for base, update in zip(bases, updates)])
        self.pending = {'round_idx': int(round_index), 'models': submitted_models,
                        'buffers': {name: value.detach().clone() for name, value in base_model.named_buffers()}}

    def result(self, total_rounds: int) -> dict[str, Any]:
        records = list(self.records)
        n = self.num_clients
        if self.pending is not None:
            records.append({
                'round_idx': self.pending['round_idx'], 'submitted_round': self.pending['round_idx'],
                'audit_round': None, 'decision_round': None,
                'client_scores': {i: None for i in range(n)},
                'mean_divergences': {i: None for i in range(n)},
                'denominators': {i: None for i in range(n)},
                'valid_by_client': {i: False for i in range(n)},
                'invalid_reasons': {i: 'not_audited_before_training_end' for i in range(n)},
                'suspicious_clients': [], 'contribution_threshold': self.detector.threshold,
                'detection_skipped': True, 'report_matrix': [[None] * n for _ in range(n)],
                'audit_sample_counts': {i: 0 for i in range(n)},
                'report_sources': {i: 'not_audited' for i in range(n)},
            })
        valid_count = sum(sum(record['valid_by_client'].values()) for record in self.records)
        attempted = len(self.records) * n
        total = total_rounds * n
        last = self.records[-1] if self.records else {}
        return {
            'method': 'pass', 'scope': 'passive_PASS_CE_without_PPS_or_elimination',
            'suspicious_clients': last.get('suspicious_clients', []),
            'final_valid_by_client': last.get('valid_by_client', {i: False for i in range(n)}),
            'final_audited_submission_round': last.get('submitted_round'),
            'suspicious_counts': {i: sum(i in r['suspicious_clients'] for r in self.records) for i in range(n)},
            'round_records': records, 'decision_threshold': self.detector.threshold,
            'effective_rounds': sum(not r['detection_skipped'] for r in self.records),
            'decision_mode': 'passive_previous_round_contribution_below_threshold',
            'qualitative_aggregation_mode': 'passive', 'detector_time_sec': self.detector_time_sec,
            'config': {**asdict(self.config), 'effective_c0': 1.0/n if self.config.c0 is None else self.config.c0,
                       'report_distribution': 'uniform', 'resampling': 'independent_per_audit_round_and_target',
                       'seed_derivation': 'pass_seed + trial_id * 100003', 'self_audit': False,
                       'random_report_policy_origin': 'experiment_assumption_not_specified_by_PASS',
                       'real_audit_data': 'original_local_training_partition',
                       'subset_policy': 'full_partition' if self.config.max_audit_samples is None else 'fixed_random_subset_per_auditor'},
            'resource_notes': self.resource_notes,
            'coverage': {'total_submission_decisions': total, 'attempted_decisions': attempted,
                         'valid_decisions': valid_count, 'invalid_attempted_decisions': attempted-valid_count,
                         'unaudited_decisions': total-attempted,
                         'coverage': valid_count/total if total else None,
                         'valid_fraction_of_attempted': valid_count/attempted if attempted else None},
        }


def _evaluation(records: Sequence[dict[str, Any]], label_key: str) -> dict[str, Any]:
    counts = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'invalid': 0}
    for record in records:
        predicted = set(record['suspicious_clients'])
        labels = record.get(label_key, {})
        for client, valid in record['valid_by_client'].items():
            if not valid:
                counts['invalid'] += 1
                continue
            truth = labels.get(client, labels.get(str(client)))
            if truth is None:
                continue
            detected = client in predicted
            counts['tp' if truth and detected else 'fn' if truth else 'fp' if detected else 'tn'] += 1
    def ratio(top: int, bottom: int) -> float | None:
        return top / bottom if bottom else None
    return {**counts, 'precision': ratio(counts['tp'], counts['tp'] + counts['fp']),
            'recall': ratio(counts['tp'], counts['tp'] + counts['fn']),
            'fpr': ratio(counts['fp'], counts['fp'] + counts['tn']),
            'f1': ratio(2*counts['tp'], 2*counts['tp'] + counts['fp'] + counts['fn'])}


def write_pass_artifacts(directory: str | Path, trial_id: int, result: dict[str, Any],
                         *, suffix: str = '') -> None:
    """Persist complete reports, sampling assumptions, coverage and valid-only metrics."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    records = result['round_records']
    payload['evaluation'] = {
        'active_submission': _evaluation(records, 'ground_truth_labels'),
        'designated_identity': _evaluation(records, 'designated_attacker_labels'),
        'metrics_use_valid_decisions_only': True,
    }
    for name, predicate in (
        ('never_attacker_fpr', lambda row, i: not row.get('designated_attacker_labels', {}).get(i, 0)),
        ('lazy_honest_round_alarm_rate', lambda row, i: row.get('is_honest_round', False) and row.get('designated_attacker_labels', {}).get(i, 0)),
    ):
        flags = [i in row['suspicious_clients'] for row in records for i, valid in row['valid_by_client'].items() if valid and predicate(row, i)]
        payload['evaluation'][name] = sum(flags)/len(flags) if flags else None
    (folder / f'pass_audit_trial{trial_id}{suffix}.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    columns = ['SubmittedRound', 'AuditRound', 'DecisionRound', 'AuditorID', 'TargetID', 'Report', 'ReportSource', 'AuditSampleCount', 'TargetDecisionValid', 'InvalidReason']
    with (folder / f'pass_audit_reports_trial{trial_id}{suffix}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            for auditor, reports in enumerate(row['report_matrix']):
                for target, value in enumerate(reports):
                    if target == auditor:
                        continue
                    writer.writerow(dict(zip(columns, [row['submitted_round'], row['audit_round'], row['decision_round'], auditor, target, value,
                        row['report_sources'][auditor], row['audit_sample_counts'][auditor], int(row['valid_by_client'][target]), row['invalid_reasons'][target]])))
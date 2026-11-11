"""Passive implementation of Nguyen et al.'s log-based free-rider detector.

The client protocol below simulates accuracy reports and complaints.  The
server detector receives only those messages and submitted model parameters;
it never receives the attack schedule or ground-truth attack labels.

This is an observer for the existing FedAvg trajectory.  Unlike the paper's
full protocol, copy hits and detected clients are not removed or reaggregated.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch


ParameterList = Sequence[torch.Tensor]


@dataclass(frozen=True)
class LogBasedFreeRiderConfig:
    """Predeclared choices that the paper leaves underspecified."""

    val_fraction: float = 0.15
    val_seed: int = 42
    report_seed: int = 42
    linkage: str = "average"
    cut_ratio: float = 0.7
    group_order: str = "client_id"
    complaint_policy: str = "silent_fake"
    evaluation_batch_size: int = 256
    shadow_full_audit: bool = False
    passive_includes_copy_reports: bool = True
    cosine_eps: float = 1e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.val_fraction) or not 0.0 < self.val_fraction < 1.0:
            raise ValueError("logfr_val_fraction must be strictly between 0 and 1")
        if not 0 <= self.val_seed < 2**63:
            raise ValueError("logfr_val_seed must lie in [0, 2**63)")
        if not 0 <= self.report_seed < 2**63:
            raise ValueError("effective logfr_report_seed must lie in [0, 2**63)")
        if self.linkage not in {"single", "average", "complete"}:
            raise ValueError("logfr_linkage must be single, average, or complete")
        if not math.isfinite(self.cut_ratio) or not 0.0 <= self.cut_ratio <= 1.0:
            raise ValueError("logfr_cut_ratio must be in [0, 1]")
        if self.group_order not in {"client_id", "medoid_first"}:
            raise ValueError("logfr_group_order must be client_id or medoid_first")
        if self.complaint_policy not in {
            "silent_fake", "truthful_fake", "always_audit"
        }:
            raise ValueError(
                "logfr_complaint_policy must be silent_fake, truthful_fake, "
                "or always_audit"
            )
        if self.evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive")
        if not math.isfinite(self.cosine_eps) or not self.cosine_eps > 0.0:
            raise ValueError("cosine_eps must be positive")

    @classmethod
    def from_args(
        cls, args: Any, *, seed: int, trial_id: int = 0
    ) -> "LogBasedFreeRiderConfig":
        """Read the runner's logfr_* flags without changing its RNG state."""

        report_base_seed = int(getattr(args, "logfr_report_seed", seed))
        return cls(
            val_fraction=float(getattr(args, "logfr_val_fraction", 0.15)),
            val_seed=int(getattr(args, "logfr_val_seed", seed)),
            report_seed=report_base_seed + 100_003 * int(trial_id),
            linkage=str(getattr(args, "logfr_linkage", "average")),
            cut_ratio=float(getattr(args, "logfr_cut_ratio", 0.7)),
            group_order=str(getattr(args, "logfr_group_order", "client_id")),
            complaint_policy=str(
                getattr(args, "logfr_complaint_policy", "silent_fake")
            ),
            evaluation_batch_size=int(
                getattr(args, "batch_size", 256)
            ),
            shadow_full_audit=bool(
                getattr(args, "logfr_shadow_full_audit", False)
            ),
        )


def _snapshot_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _clone_parameter_lists(
    lists: Sequence[ParameterList],
) -> list[list[torch.Tensor]]:
    return [[parameter.detach().clone() for parameter in item] for item in lists]


def _flat(parameters: ParameterList) -> torch.Tensor:
    if not parameters:
        raise ValueError("A submitted model has no parameters")
    return torch.cat(
        [parameter.detach().reshape(-1).to(torch.float32) for parameter in parameters]
    )


def _cosine_distance(
    left: ParameterList, right: ParameterList, *, eps: float
) -> float:
    left_flat, right_flat = _flat(left), _flat(right)
    if left_flat.shape != right_flat.shape:
        raise ValueError("Model parameter shapes differ in cosine distance")
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    left_zero = bool((left_norm <= eps).item())
    right_zero = bool((right_norm <= eps).item())
    if left_zero and right_zero:
        return 0.0
    if left_zero or right_zero:
        return 1.0
    numerator = torch.dot(left_flat, right_flat)
    denominator = left_norm * right_norm
    similarity = numerator / denominator.clamp_min(eps)
    return float((1.0 - similarity.clamp(-1.0, 1.0)).item())


def _pairwise_cosine_distance_matrix(
    submissions: Sequence[ParameterList], *, eps: float
) -> torch.Tensor:
    if not submissions:
        raise ValueError("Cannot cluster an empty submission set")
    flat = torch.stack([_flat(parameters) for parameters in submissions], dim=0)
    raw_norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True)
    zero_norm = raw_norms.reshape(-1) <= eps
    normalized = flat / raw_norms.clamp_min(eps)
    distances = (1.0 - normalized @ normalized.T).clamp(0.0, 2.0)
    distances.masked_fill_(zero_norm[:, None] & zero_norm[None, :], 0.0)
    distances = (distances + distances.T) * 0.5
    distances.fill_diagonal_(0.0)
    return distances


def _cluster_distance(
    distances: torch.Tensor,
    left_members: tuple[int, ...],
    right_members: tuple[int, ...],
    method: str,
) -> float:
    left_idx = torch.as_tensor(left_members, device=distances.device)
    right_idx = torch.as_tensor(right_members, device=distances.device)
    cross = distances.index_select(0, left_idx).index_select(1, right_idx)
    if method == "single":
        value = cross.min()
    elif method == "complete":
        value = cross.max()
    else:
        value = cross.mean()
    return float(value.item())


def _agglomerative_groups(
    distances: torch.Tensor,
    client_ids: Sequence[int],
    *,
    linkage: str,
    cut_ratio: float,
    group_order: str,
) -> tuple[list[list[int]], list[dict[str, Any]], float]:
    """Small-N deterministic linkage with reductions on the tensor's device."""

    count = len(client_ids)
    if count == 0:
        return [], [], 0.0
    if distances.shape != (count, count):
        raise ValueError("Pairwise matrix and client IDs disagree")
    active: dict[int, tuple[int, ...]] = {i: (i,) for i in range(count)}
    next_node = count
    merges: list[dict[str, Any]] = []
    merge_indices: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

    while len(active) > 1:
        nodes = sorted(active)
        best: Optional[tuple[float, tuple[int, ...], int, int]] = None
        for position, left_node in enumerate(nodes):
            for right_node in nodes[position + 1 :]:
                left_members, right_members = active[left_node], active[right_node]
                height = _cluster_distance(
                    distances, left_members, right_members, linkage
                )
                tie_key = tuple(
                    sorted(int(client_ids[index]) for index in left_members + right_members)
                )
                candidate = (height, tie_key, left_node, right_node)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise RuntimeError("Linkage could not select a merge")
        height, _, left_node, right_node = best
        left_members, right_members = active.pop(left_node), active.pop(right_node)
        combined = tuple(sorted(left_members + right_members))
        active[next_node] = combined
        next_node += 1
        merge_indices.append((left_members, right_members, height))
        merges.append(
            {
                "left": sorted(int(client_ids[i]) for i in left_members),
                "right": sorted(int(client_ids[i]) for i in right_members),
                "distance": height,
            }
        )

    max_height = max((merge[2] for merge in merge_indices), default=0.0)
    cut_height = float(cut_ratio * max_height)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_members, right_members, height in merge_indices:
        if height <= cut_height + 1e-12:
            parent[find(left_members[0])] = find(right_members[0])

    partitions: dict[int, list[int]] = {}
    for index in range(count):
        partitions.setdefault(find(index), []).append(index)

    groups: list[list[int]] = []
    for indices in partitions.values():
        if group_order == "medoid_first" and len(indices) > 1:
            def medoid_key(index: int) -> tuple[float, int]:
                others = torch.as_tensor(indices, device=distances.device)
                average = distances[index].index_select(0, others).mean()
                return float(average.item()), int(client_ids[index])

            medoid = min(indices, key=medoid_key)
            ordered = [medoid] + sorted(
                (i for i in indices if i != medoid), key=lambda i: client_ids[i]
            )
        else:
            ordered = sorted(indices, key=lambda i: client_ids[i])
        groups.append([int(client_ids[i]) for i in ordered])
    groups.sort(key=lambda group: min(group))
    return groups, merges, cut_height


class _AccuracyEvaluator:
    """Inference on a private model copy, matching Net._score_accuracy_gpu."""

    def __init__(self, model: torch.nn.Module, batch_size: int) -> None:
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.parameters = list(self.model.parameters())
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

    def evaluate(
        self, parameters: ParameterList, x: torch.Tensor, y: torch.Tensor
    ) -> float:
        if len(parameters) != len(self.parameters):
            raise ValueError("Submitted model has an incompatible parameter count")
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        if len(x) != len(y) or len(y) == 0:
            raise ValueError("Accuracy evaluation requires nonempty aligned X/y")
        with torch.inference_mode():
            for target, source in zip(self.parameters, parameters):
                if target.shape != source.shape:
                    raise ValueError("Submitted model has an incompatible parameter shape")
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            correct = torch.zeros((), device=self.device, dtype=torch.int64)
            for start in range(0, len(y), self.batch_size):
                stop = min(start + self.batch_size, len(y))
                outputs = self.model(x[start:stop].to(self.device))
                labels = y[start:stop].to(self.device).reshape(-1).long()
                if outputs.dim() > 1 and outputs.size(-1) > 1:
                    predictions = outputs.argmax(dim=1)
                else:
                    predictions = (outputs.reshape(-1) >= 0.5).long()
                correct += (predictions == labels).sum()
            return float((correct.float() / len(y)).item())


class LogBasedFreeRiderDetector:
    """Server-only Nguyen investigation; no attack labels enter this class."""

    def __init__(self, config: LogBasedFreeRiderConfig) -> None:
        self.config = config
        self.round_records: list[dict[str, Any]] = []
        self._pending: Optional[dict[str, Any]] = None
        self._previous_global_accuracy: Optional[float] = None
        self._gap_samples: list[float] = []

    def capture_submissions(
        self,
        round_idx: int,
        submitted_parameters: Mapping[int, ParameterList],
        previous_global_parameters: ParameterList,
        reported_accuracies: Mapping[int, float],
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("Previous investigation round was not finished")
        if set(submitted_parameters) != set(reported_accuracies):
            raise ValueError("Every submitted client must provide an accuracy report")
        if not submitted_parameters:
            raise ValueError("Investigation requires at least one client")
        copied = {
            int(client_id): [tensor.detach().clone() for tensor in parameters]
            for client_id, parameters in submitted_parameters.items()
        }
        reference = [tensor.detach().clone() for tensor in previous_global_parameters]
        copy_hits: list[int] = []
        for client_id, parameters in copied.items():
            if len(parameters) != len(reference):
                raise ValueError("Submitted model has incompatible parameter count")
            if all(torch.equal(a, b) for a, b in zip(parameters, reference)):
                copy_hits.append(client_id)
        reports = {int(i): float(value) for i, value in reported_accuracies.items()}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in reports.values()):
            raise ValueError("Accuracy reports must be finite fractions in [0, 1]")
        self._pending = {
            "round_idx": int(round_idx),
            "submitted": copied,
            "reported_accuracies": reports,
            "copy_hits": sorted(copy_hits),
        }

    def finish_round(
        self,
        round_idx: int,
        current_global_parameters: ParameterList,
        global_accuracy: float,
        complaints: Mapping[int, bool],
        server_evaluator: Callable[[ParameterList], float],
    ) -> dict[str, Any]:
        pending = self._pending
        if pending is None or pending["round_idx"] != int(round_idx):
            raise RuntimeError("No matching captured submission round")
        global_accuracy = float(global_accuracy)
        if not math.isfinite(global_accuracy) or not 0.0 <= global_accuracy <= 1.0:
            raise ValueError("Global accuracy must be a finite fraction in [0, 1]")

        submitted: dict[int, list[torch.Tensor]] = pending["submitted"]
        client_ids = sorted(submitted)
        if set(complaints) != set(client_ids):
            raise ValueError("Every client needs an explicit complaint flag")
        complaint_ids = sorted(int(i) for i, flag in complaints.items() if flag)
        previous_accuracy = self._previous_global_accuracy
        prior_gap = (
            float(sum(self._gap_samples) / len(self._gap_samples))
            if self._gap_samples
            else None
        )
        prior_gap_count = len(self._gap_samples)
        first_round = previous_accuracy is None
        forced_audit = self.config.complaint_policy == "always_audit"
        investigation_triggered = first_round or bool(complaint_ids) or forced_audit
        copy_hits = set(pending["copy_hits"])
        audit_reason = (
            "first_round" if first_round
            else "complaint" if complaint_ids
            else "forced" if forced_audit
            else "copy_only" if copy_hits
            else "none"
        )
        candidate_ids = [i for i in client_ids if i not in copy_hits]
        audited_accuracies: dict[int, float] = {}
        shadow_accuracies: dict[int, float] = {}
        decision_status: dict[int, str] = {
            i: "copy_hit" if i in copy_hits else "not_audited"
            for i in client_ids
        }
        groups: list[list[int]] = []
        linkage_merges: list[dict[str, Any]] = []
        cut_height: Optional[float] = None
        pairwise: Optional[dict[str, Any]] = None
        checked_ids: list[int] = []
        skipped_ids: list[int] = []
        suspicious_ids = set(copy_hits)

        if investigation_triggered and candidate_ids:
            matrix = _pairwise_cosine_distance_matrix(
                [submitted[i] for i in candidate_ids], eps=self.config.cosine_eps
            )
            pairwise = {
                "client_ids": candidate_ids,
                "matrix": matrix.detach().cpu().tolist(),
            }
            groups, linkage_merges, cut_height = _agglomerative_groups(
                matrix,
                candidate_ids,
                linkage=self.config.linkage,
                cut_ratio=self.config.cut_ratio,
                group_order=self.config.group_order,
            )
        if first_round:
            # The paper re-evaluates every submission at the first
            # aggregation, but does not define a first-round A_gap.
            for client_id in client_ids:
                accuracy = float(server_evaluator(submitted[client_id]))
                if not math.isfinite(accuracy):
                    raise ValueError("Server audit returned nonfinite accuracy")
                audited_accuracies[client_id] = accuracy
                checked_ids.append(client_id)
                if client_id not in copy_hits:
                    decision_status[client_id] = "uncalibrated_first_round"
        elif investigation_triggered and candidate_ids:
            if prior_gap is None or previous_accuracy is None:
                raise RuntimeError("Prior safe gap missing after the first round")
            maximum_accuracy = max(previous_accuracy, global_accuracy)
            for group in groups:
                for position, client_id in enumerate(group):
                    accuracy = float(server_evaluator(submitted[client_id]))
                    if not math.isfinite(accuracy):
                        raise ValueError("Server audit returned nonfinite accuracy")
                    audited_accuracies[client_id] = accuracy
                    checked_ids.append(client_id)
                    unsafe = (
                        (accuracy < maximum_accuracy and maximum_accuracy - accuracy > prior_gap)
                        or (accuracy < previous_accuracy <= global_accuracy)
                    )
                    if unsafe:
                        decision_status[client_id] = "unsafe"
                        suspicious_ids.add(client_id)
                    else:
                        decision_status[client_id] = "safe"
                        for skipped_id in group[position + 1 :]:
                            decision_status[skipped_id] = "group_skipped"
                            skipped_ids.append(skipped_id)
                        break

        if self.config.shadow_full_audit:
            for client_id in client_ids:
                if client_id in audited_accuracies:
                    shadow_accuracies[client_id] = audited_accuracies[client_id]
                else:
                    accuracy = float(server_evaluator(submitted[client_id]))
                    if not math.isfinite(accuracy):
                        raise ValueError("Server shadow audit returned nonfinite accuracy")
                    shadow_accuracies[client_id] = accuracy

        # The passive run keeps every upload in FedAvg, including direct
        # copies.  Therefore its report also enters the actual-run history.
        if self.config.passive_includes_copy_reports:
            reports_for_history = pending["reported_accuracies"].values()
        else:
            reports_for_history = (
                value for i, value in pending["reported_accuracies"].items()
                if i not in copy_hits
            )
        self._gap_samples.extend(
            abs(global_accuracy - report) for report in reports_for_history
        )
        self._previous_global_accuracy = global_accuracy

        record = {
            "round_idx": int(round_idx),
            "global_accuracy": global_accuracy,
            "previous_global_accuracy": previous_accuracy,
            "decision_threshold": prior_gap,
            "a_gap": prior_gap,
            "safe_gap_prior_count": prior_gap_count,
            "safe_gap_next": float(sum(self._gap_samples) / len(self._gap_samples))
            if self._gap_samples else None,
            "reported_accuracies": dict(pending["reported_accuracies"]),
            "complaint_flags": {int(i): bool(flag) for i, flag in complaints.items()},
            "complaints": {int(i): bool(flag) for i, flag in complaints.items()},
            "complaining_clients": complaint_ids,
            "copy_hits": sorted(copy_hits),
            "copy_flags": {i: i in copy_hits for i in client_ids},
            "investigation_triggered": investigation_triggered,
            "audit_triggered": investigation_triggered,
            "audit_reason": audit_reason,
            "forced_audit": forced_audit,
            "first_round_full_audit": first_round,
            "pairwise_cosine_distances": pairwise,
            "linkage_merges": linkage_merges,
            "clusters": groups,
            "cluster_labels": {
                member: group_index
                for group_index, group in enumerate(groups)
                for member in group
            },
            "cluster_cut_height": cut_height,
            "audited_clients": checked_ids,
            "group_skipped_clients": skipped_ids,
            "audited_accuracies": audited_accuracies,
            "audit_accuracies": audited_accuracies,
            "shadow_full_audit_accuracies": shadow_accuracies,
            "decision_status_by_client": decision_status,
            "invalid_reasons": {
                i: "" if decision_status[i] in {"copy_hit", "safe", "unsafe"}
                else decision_status[i]
                for i in client_ids
            },
            "valid_by_client": {
                i: decision_status[i] in {"copy_hit", "safe", "unsafe"}
                for i in client_ids
            },
            "client_scores": dict(audited_accuracies),
            "suspicious_clients": sorted(suspicious_ids),
            "detection_skipped": not investigation_triggered and not copy_hits,
        }
        self.round_records.append(record)
        self._pending = None
        return record


class LogBasedFreeRiderObserver:
    """Client protocol simulator plus a label-blind server detector."""

    def __init__(
        self,
        model: torch.nn.Module,
        x_val_parts: Sequence[torch.Tensor],
        y_val_parts: Sequence[torch.Tensor],
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        config: LogBasedFreeRiderConfig,
    ) -> None:
        if len(x_val_parts) != len(y_val_parts) or not x_val_parts:
            raise ValueError("Each client needs a local validation partition")
        if any(len(x) != len(y) or len(y) == 0 for x, y in zip(x_val_parts, y_val_parts)):
            raise ValueError("Each local validation partition must be nonempty")
        self.config = config
        self.detector = LogBasedFreeRiderDetector(config)
        # Match the runner's existing global ``model.score`` batch size so
        # server/client accuracy comparisons use the same inference batching.
        score_batch_size = getattr(model, "batch_size", None) or config.evaluation_batch_size
        self.evaluator = _AccuracyEvaluator(model, int(score_batch_size))
        self.x_val_parts = x_val_parts
        self.y_val_parts = y_val_parts
        self.x_test = x_test
        self.y_test = y_test
        self.detector_time_sec = 0.0
        self._submitted: Optional[dict[int, list[torch.Tensor]]] = None
        self._global_before: Optional[list[torch.Tensor]] = None
        self._distance_history: dict[int, list[float]] = {}
        self._last_honest_report: dict[int, float] = {}
        self._fake_clients: set[int] = set()
        self._forced_silent_clients: set[int] = set()
        self._report_generators: dict[int, torch.Generator] = {}
        self._report_sources: dict[int, str] = {}

    def _fake_report(self, client_id: int) -> float:
        generator = self._report_generators.get(client_id)
        if generator is None:
            generator = torch.Generator(device=self.evaluator.device)
            generator.manual_seed(
                (self.config.report_seed + 9_973 * int(client_id)) % (2**63)
            )
            self._report_generators[client_id] = generator
        return float(torch.rand((), device=self.evaluator.device, generator=generator).item())

    def capture_submissions(
        self,
        round_idx: int,
        bases: Sequence[ParameterList],
        updates: Sequence[ParameterList],
        global_model: torch.nn.Module,
        *,
        activity: Mapping[str, Any],
        attacker_id: Optional[int],
        use_attack: bool,
        use_lazy: bool,
        attack_method: str,
    ) -> None:
        if len(bases) != len(updates) or len(bases) != len(self.x_val_parts):
            raise ValueError("Base, update, and validation client counts disagree")
        submitted: dict[int, list[torch.Tensor]] = {}
        for client_id, (base, update) in enumerate(zip(bases, updates)):
            if len(base) != len(update):
                raise ValueError("Base and submitted update shapes disagree")
            submitted[client_id] = [
                old.detach().clone() + delta.detach()
                for old, delta in zip(base, update)
            ]

        fake_clients: set[int] = set()
        forced_silent_clients: set[int] = set()
        reports: dict[int, float] = {}
        is_public_warm_start = activity.get("attack_state") == "public_warm_start"
        is_fake_round = bool(activity.get("is_fake_round", False))
        is_adam_attack = bool(
            use_attack and attack_method.lower() == "adam_local_deviation"
        )
        report_sources: dict[int, str] = {}
        for client_id in submitted:
            is_attacker = bool(use_attack and attacker_id == client_id)
            adam_warm_start = bool(
                is_attacker and is_adam_attack and is_public_warm_start
            )
            # The public-trained update is real; its log-based accuracy report is fake.
            fake = bool(is_attacker and (is_fake_round or adam_warm_start))
            if fake:
                fake_clients.add(client_id)
                if is_adam_attack:
                    forced_silent_clients.add(client_id)
                if use_lazy and not adam_warm_start:
                    if client_id not in self._last_honest_report:
                        raise ValueError(
                            "Lazy fake round precedes every honest accuracy report; "
                            "choose a schedule with an earlier honest round"
                        )
                    report = self._last_honest_report[client_id]
                    report_sources[client_id] = "reused_last_honest"
                else:
                    report = self._fake_report(client_id)
                    report_sources[client_id] = (
                        "random_uniform_0_1_public_warm_start"
                        if adam_warm_start else "random_uniform_0_1"
                    )
            else:
                report = self.evaluator.evaluate(
                    submitted[client_id],
                    self.x_val_parts[client_id],
                    self.y_val_parts[client_id],
                )
                if is_attacker and use_lazy:
                    self._last_honest_report[client_id] = report
                report_sources[client_id] = "truthful_validation"
            reports[client_id] = report

        self._submitted = submitted
        self._global_before = _snapshot_parameters(global_model)
        self._fake_clients = fake_clients
        self._forced_silent_clients = forced_silent_clients
        self._report_sources = report_sources
        self.detector.capture_submissions(
            round_idx,
            submitted,
            self._global_before,
            reports,
        )

    def finish_round(
        self,
        round_idx: int,
        global_model: torch.nn.Module,
        global_accuracy: float,
    ) -> dict[str, Any]:
        if self._submitted is None or self._global_before is None:
            raise RuntimeError("No submissions were captured for this round")
        global_parameters = _snapshot_parameters(global_model)
        complaints: dict[int, bool] = {}
        validation_global_accuracies: dict[int, float] = {}
        client_global_distances: dict[int, float] = {}
        for client_id, submitted in self._submitted.items():
            current_distance = _cosine_distance(
                submitted, global_parameters, eps=self.config.cosine_eps
            )
            distance_history = self._distance_history.setdefault(client_id, [])
            previous_mean_distance = (
                sum(distance_history) / len(distance_history)
                if distance_history else None
            )
            can_complain = (
                client_id not in self._forced_silent_clients
                and (
                    client_id not in self._fake_clients
                    or self.config.complaint_policy == "truthful_fake"
                )
            )
            if can_complain:
                current_accuracy = self.evaluator.evaluate(
                    global_parameters,
                    self.x_val_parts[client_id],
                    self.y_val_parts[client_id],
                )
                validation_global_accuracies[client_id] = current_accuracy
                if round_idx > 0:
                    previous_accuracy = self.evaluator.evaluate(
                        self._global_before,
                        self.x_val_parts[client_id],
                        self.y_val_parts[client_id],
                    )
                    accuracy_complaint = current_accuracy < previous_accuracy
                    distance_complaint = (
                        previous_mean_distance is not None
                        and current_distance > previous_mean_distance
                    )
                    complaints[client_id] = bool(
                        accuracy_complaint or distance_complaint
                    )
                else:
                    complaints[client_id] = False
            else:
                # A silent fake client never accesses its private validation
                # partition, even for an internal diagnostic.
                complaints[client_id] = False
            client_global_distances[client_id] = current_distance
            distance_history.append(current_distance)

        record = self.detector.finish_round(
            round_idx,
            global_parameters,
            global_accuracy,
            complaints,
            lambda submitted: self.evaluator.evaluate(
                submitted, self.x_test, self.y_test
            ),
        )
        # These fields are client messages and diagnostics, attached only
        # after the server detector has made its decision.
        record["client_global_validation_accuracies"] = validation_global_accuracies
        record["client_global_cosine_distances"] = client_global_distances
        record["cosine_distances_to_global"] = client_global_distances
        record["report_sources"] = dict(self._report_sources)
        record["report_fake_clients"] = sorted(self._fake_clients)
        record["report_ground_truth_labels"] = {
            client_id: int(client_id in self._fake_clients)
            for client_id in self._submitted
        }
        record["forced_silent_clients"] = sorted(self._forced_silent_clients)
        self._submitted = None
        self._global_before = None
        self._fake_clients = set()
        self._forced_silent_clients = set()
        self._report_sources = {}
        return record

    def result(self, total_rounds: int) -> dict[str, Any]:
        if self._submitted is not None:
            raise RuntimeError("Cannot report results while a round is pending")
        records = self.detector.round_records
        if len(records) != int(total_rounds):
            raise ValueError("Recorded round count differs from total_rounds")
        counts: dict[int, int] = {}
        for record in records:
            for client_id in record["suspicious_clients"]:
                counts[client_id] = counts.get(client_id, 0) + 1
        coverage = {
            "total_rounds": len(records),
            "investigated_rounds": sum(
                bool(record["investigation_triggered"]) for record in records
            ),
            "audited_submissions": sum(
                len(record["audited_clients"]) for record in records
            ),
            "group_skipped_submissions": sum(
                len(record["group_skipped_clients"]) for record in records
            ),
            "copy_hit_submissions": sum(
                len(record["copy_hits"]) for record in records
            ),
            "decision_valid_submissions": sum(
                sum(record["valid_by_client"].values()) for record in records
            ),
        }
        final_valid_by_client = {
            client_id: any(
                bool(record["valid_by_client"].get(client_id, False))
                for record in records
            )
            for client_id in sorted(
                {i for record in records for i in record["valid_by_client"]}
            )
        }
        return {
            "suspicious_clients": sorted(counts),
            "suspicious_counts": counts,
            "report_ground_truth_semantics": "fabricated_or_stale_accuracy_report",
            "final_valid_by_client": final_valid_by_client,
            "round_records": records,
            "decision_threshold": (
                records[-1]["decision_threshold"] if records else None
            ),
            "effective_rounds": sum(
                any(record["valid_by_client"].values()) for record in records
            ),
            "detector_time_sec": float(self.detector_time_sec),
            "decision_mode": "passive_round_detection",
            "config": {
                **asdict(self.config),
                "effective_evaluation_batch_size": self.evaluator.batch_size,
            },
            "coverage": coverage,
        }


def write_log_based_free_rider_artifacts(
    directory: str | Path,
    trial_id: int,
    result: Mapping[str, Any],
    suffix: str = "",
) -> Path:
    """Write full per-round diagnostics for server-pilot replay and audit."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"log_based_free_rider_diagnostics_trial{trial_id}{suffix}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
    return path

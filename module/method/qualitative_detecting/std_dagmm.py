"""Paper-aligned STD-DAGMM free-rider detection for cadence v21.

Lin, Du, and Liu use STD-DAGMM sample energy as an anomaly score.  The paper's
explicit qualitative rule is to regard samples whose energy is higher than the
majority of the round's samples as suspicious.  This module keeps the v1
network and energy objective, reports the paper-style continuous metrics, and
also exposes that round-local above-median decision without adding an
arbitrary persistence policy.

The previous KMeans/MAD/persistence implementation remains archived in
``std_dagmm_old_v1.py`` and is not used by the canonical detector.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

from .std_dagmm_old_v1 import STDDAGMMConfig
from .std_dagmm_old_v1 import STDDAGMMDetector as _STDDAGMMV1Detector


def _binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Return tie-aware ROC-AUC without introducing another dependency."""
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None

    favorable_pairs = 0.0
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                favorable_pairs += 1.0
            elif positive_score == negative_score:
                favorable_pairs += 0.5
    return favorable_pairs / (len(positives) * len(negatives))


def _binary_average_precision(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float | None:
    """Return threshold-grouped average precision with deterministic ties."""
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    grouped: Dict[float, list[int]] = defaultdict(list)
    for label, score in zip(labels, scores):
        grouped[float(score)].append(int(label))

    true_positives = 0
    false_positives = 0
    average_precision = 0.0
    previous_recall = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        true_positives += sum(group)
        false_positives += len(group) - sum(group)
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        average_precision += precision * (recall - previous_recall)
        previous_recall = recall
    return average_precision


def _descending_average_ranks(scores: Mapping[int, float]) -> Dict[int, float]:
    """Assign one-based descending ranks, averaging exact ties."""
    ranks: Dict[int, float] = {}
    values = list(scores.values())
    for client_id, score in scores.items():
        greater = sum(value > score for value in values)
        equal_others = sum(value == score for value in values) - 1
        ranks[client_id] = 1.0 + greater + 0.5 * equal_others
    return ranks


def _within_round_percentiles(scores: Mapping[int, float]) -> Dict[int, float]:
    """Map each energy to [0, 1], where 1 is most suspicious in its round."""
    values = list(scores.values())
    denominator = max(len(values) - 1, 1)
    percentiles: Dict[int, float] = {}
    for client_id, score in scores.items():
        lower = sum(value < score for value in values)
        equal_others = sum(value == score for value in values) - 1
        percentiles[client_id] = (lower + 0.5 * equal_others) / denominator
    return percentiles


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std_or_none(values: Sequence[float]) -> float | None:
    return float(np.std(values)) if values else None


class STDDAGMMDetector(_STDDAGMMV1Detector):
    """STD-DAGMM energy scorer with paper-aligned threshold-free evaluation."""

    @staticmethod
    def evaluate_round_scores(
        client_scores: Mapping[int, float],
        ground_truth_attackers: Sequence[int] | None = None,
    ) -> Dict[str, object]:
        client_ids = list(client_scores)
        attacker_set = set(ground_truth_attackers or ())
        labels = [int(client_id in attacker_set) for client_id in client_ids]
        scores = [float(client_scores[client_id]) for client_id in client_ids]
        ranks = _descending_average_ranks(client_scores)
        percentiles = _within_round_percentiles(client_scores)

        max_energy = max(scores) if scores else None
        top_clients = sorted(
            client_id
            for client_id, score in client_scores.items()
            if max_energy is not None and score == max_energy
        )
        attacker_ranks = [ranks[client_id] for client_id in client_ids if client_id in attacker_set]
        top1_hit = (
            bool(attacker_set.intersection(top_clients))
            if attacker_set and len(attacker_set) < len(client_ids)
            else None
        )

        return {
            "labels": {client_id: labels[index] for index, client_id in enumerate(client_ids)},
            "energy_ranks": ranks,
            "energy_percentiles": percentiles,
            "top_clients": top_clients,
            "roc_auc": _binary_roc_auc(labels, scores),
            "average_precision": _binary_average_precision(labels, scores),
            "attacker_mean_rank": _mean_or_none(attacker_ranks),
            "top1_hit": top1_hit,
        }

    @staticmethod
    def detect_round_free_riders(
        client_scores: Mapping[int, float],
    ) -> Dict[str, object]:
        """Apply Lin et al.'s explicit above-the-majority energy rule.

        "Higher than the majority" is translated as strictly above the
        within-round median.  No KMeans, MAD cutoff, strict-minority gate, or
        cross-round persistence rule is part of the paper's stated detector.
        """
        client_ids = list(client_scores)
        if not client_ids:
            return {
                "suspicious_clients": [],
                "energy_threshold": None,
                "labels": {},
            }

        energies = np.asarray(
            [float(client_scores[client_id]) for client_id in client_ids],
            dtype=np.float64,
        )
        threshold = float(np.median(energies))
        suspicious = [
            client_id
            for client_id, energy in zip(client_ids, energies)
            if float(energy) > threshold
        ]
        return {
            "suspicious_clients": suspicious,
            "energy_threshold": threshold,
            "labels": {
                client_id: int(client_id in suspicious) for client_id in client_ids
            },
        }

    @staticmethod
    def detect_free_riders_from_history(
        round_client_gradients: List[List[List[torch.Tensor]]],
        config: STDDAGMMConfig | None = None,
        device: torch.device | str = "cpu",
        warmup_rounds: int = 0,
        persistence_ratio: float = 0.5,
        min_suspicious_rounds: int = 2,
        min_rounds_after_warmup: int = 1,
        ground_truth_attackers: Sequence[int] | None = None,
    ) -> Dict[str, object]:
        """Fit one detector per round and apply the paper's round-local rule.

        ``persistence_ratio`` and ``min_suspicious_rounds`` remain in the
        signature so existing v21 callers do not break.  They are intentionally
        ignored because Lin et al.'s detector is round-local and does not define
        a multi-round persistence verdict.
        """
        resolved_config = config or STDDAGMMConfig()
        resolved_config.validate()
        if warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative")
        if min_rounds_after_warmup < 0:
            raise ValueError("min_rounds_after_warmup must be non-negative")

        del persistence_ratio, min_suspicious_rounds

        round_records: list[Dict[str, object]] = []
        suspicious_counts: Dict[int, int] = {}
        client_percentile_history: Dict[int, list[float]] = defaultdict(list)
        client_rank_history: Dict[int, list[float]] = defaultdict(list)
        auc_values: list[float] = []
        average_precision_values: list[float] = []
        attacker_rank_values: list[float] = []
        top1_values: list[float] = []

        for round_index, round_gradients in enumerate(round_client_gradients):
            if round_index < warmup_rounds:
                continue

            score_result = STDDAGMMDetector.fit_score_round(
                client_gradients={
                    client_id: gradient
                    for client_id, gradient in enumerate(round_gradients)
                },
                config=STDDAGMMConfig(
                    **{
                        **asdict(resolved_config),
                        "random_state": resolved_config.random_state + round_index,
                    }
                ),
                device=device,
            )
            evaluation = STDDAGMMDetector.evaluate_round_scores(
                score_result["client_scores"],
                ground_truth_attackers=ground_truth_attackers,
            )
            round_detection = STDDAGMMDetector.detect_round_free_riders(
                score_result["client_scores"]
            )

            for client_id in round_detection["suspicious_clients"]:
                suspicious_counts[client_id] = suspicious_counts.get(client_id, 0) + 1

            for client_id, percentile in evaluation["energy_percentiles"].items():
                client_percentile_history[int(client_id)].append(float(percentile))
            for client_id, rank in evaluation["energy_ranks"].items():
                client_rank_history[int(client_id)].append(float(rank))
            if evaluation["roc_auc"] is not None:
                auc_values.append(float(evaluation["roc_auc"]))
            if evaluation["average_precision"] is not None:
                average_precision_values.append(float(evaluation["average_precision"]))
            if evaluation["attacker_mean_rank"] is not None:
                attacker_rank_values.append(float(evaluation["attacker_mean_rank"]))
            if evaluation["top1_hit"] is not None:
                top1_values.append(float(bool(evaluation["top1_hit"])))

            round_records.append(
                {
                    "round_idx": round_index,
                    "client_scores": score_result["client_scores"],
                    "client_stds": score_result["client_stds"],
                    "ground_truth_labels": evaluation["labels"],
                    "energy_ranks": evaluation["energy_ranks"],
                    "energy_percentiles": evaluation["energy_percentiles"],
                    "top_clients": evaluation["top_clients"],
                    "roc_auc": evaluation["roc_auc"],
                    "average_precision": evaluation["average_precision"],
                    "attacker_mean_rank": evaluation["attacker_mean_rank"],
                    "top1_hit": evaluation["top1_hit"],
                    "suspicious_clients": round_detection["suspicious_clients"],
                    "energy_threshold": round_detection["energy_threshold"],
                    "final_loss": score_result["final_loss"],
                    "input_scale": score_result["input_scale"],
                    "weights": score_result["config"],
                }
            )

        effective_rounds = len(round_records)
        enough_rounds = effective_rounds >= min_rounds_after_warmup
        client_mean_percentiles = {
            client_id: float(np.mean(values))
            for client_id, values in client_percentile_history.items()
        }
        client_mean_ranks = {
            client_id: float(np.mean(values))
            for client_id, values in client_rank_history.items()
        }
        ranked_clients = sorted(
            client_mean_percentiles,
            key=lambda client_id: (-client_mean_percentiles[client_id], client_id),
        )

        last_suspicious = (
            list(round_records[-1]["suspicious_clients"])
            if round_records
            else []
        )
        last_threshold = (
            round_records[-1]["energy_threshold"] if round_records else None
        )

        evaluation_summary = {
            "mean_round_auc": _mean_or_none(auc_values) if enough_rounds else None,
            "std_round_auc": _std_or_none(auc_values) if enough_rounds else None,
            "mean_average_precision": (
                _mean_or_none(average_precision_values) if enough_rounds else None
            ),
            "mean_attacker_rank": (
                _mean_or_none(attacker_rank_values) if enough_rounds else None
            ),
            "top1_hit_rate": _mean_or_none(top1_values) if enough_rounds else None,
            "valid_auc_rounds": len(auc_values),
            "ground_truth_attackers": sorted(set(ground_truth_attackers or ())),
        }

        return {
            "suspicious_clients": last_suspicious,
            "suspicious_counts": dict(suspicious_counts),
            "round_records": round_records,
            "decision_threshold": last_threshold,
            "decision_mode": "energy_above_round_median",
            "effective_rounds": effective_rounds,
            "client_mean_percentiles": client_mean_percentiles,
            "client_mean_ranks": client_mean_ranks,
            "ranked_clients": ranked_clients,
            "evaluation": evaluation_summary,
            "config": asdict(resolved_config),
        }


__all__ = ["STDDAGMMConfig", "STDDAGMMDetector"]

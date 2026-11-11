"""Behavior tests for qualitative passive mode and per-round attack truth."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import tempfile
import time
import typing
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FLCE_PATH = ROOT / "FLCE_updated.py"


class _FLCEFunctionHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = FLCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def compile_functions(self, *names: str) -> dict[str, object]:
        functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }
        missing = set(names) - set(functions)
        self.assertFalse(missing, f"FLCE_updated.py is missing functions: {sorted(missing)}")
        module = ast.Module(body=[functions[name] for name in names], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace: dict[str, object] = {
            "csv": csv,
            "json": json,
            "np": np,
            "Path": Path,
            "time": time,
            "torch": torch,
            "typing": typing,
        }
        exec(compile(module, str(FLCE_PATH), "exec"), namespace)
        return namespace


class QualitativePassiveModeTest(_FLCEFunctionHarness):
    def test_cli_defaults_to_paper_online_and_accepts_passive(self) -> None:
        matching_calls = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if any(
                isinstance(argument, ast.Constant)
                and argument.value == "--qualitative_aggregation_mode"
                for argument in node.args
            ):
                matching_calls.append(node)

        self.assertEqual(len(matching_calls), 1)
        parser = argparse.ArgumentParser()
        expression = ast.Expr(value=matching_calls[0])
        module = ast.Module(body=[expression], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(FLCE_PATH), "exec"), {"parser": parser})

        self.assertEqual(
            parser.parse_args([]).qualitative_aggregation_mode,
            "paper_online",
        )
        self.assertEqual(
            parser.parse_args(
                ["--qualitative_aggregation_mode", "passive"]
            ).qualitative_aggregation_mode,
            "passive",
        )

    def test_passive_wef_detection_keeps_weighted_fedavg(self) -> None:
        namespace = self.compile_functions(
            "_should_use_online_qualitative_aggregation",
            "_aggregate_round_updates",
        )
        should_isolate = namespace["_should_use_online_qualitative_aggregation"]
        aggregate = namespace["_aggregate_round_updates"]

        self.assertFalse(
            should_isolate(True, "qualitative", "wef", "passive")
        )
        self.assertTrue(
            should_isolate(True, "qualitative", "wef", "paper_online")
        )
        self.assertFalse(
            should_isolate(True, "qualitative", "std_dagmm", "paper_online")
        )

        aggregated, client_parameters = aggregate(
            submitted_updates=[
                [torch.tensor([0.0])],
                [torch.tensor([10.0])],
            ],
            weights=torch.tensor([0.25, 0.75]),
            device=torch.device("cpu"),
            current_global_parameters=[torch.tensor([0.0])],
            use_online_qualitative_aggregation=False,
            detecting_method="wef",
            round_client_base_parameters=[
                [torch.tensor([0.0])],
                [torch.tensor([0.0])],
            ],
            suspicious_clients=[1],
        )

        torch.testing.assert_close(aggregated[0], torch.tensor([7.5]))
        self.assertIsNone(client_parameters)


class RoundAttackGroundTruthTest(_FLCEFunctionHarness):
    def test_round_activity_uses_realized_pure_fixed_and_probabilistic_state(self) -> None:
        namespace = self.compile_functions("_resolve_round_attack_activity")
        resolve = namespace["_resolve_round_attack_activity"]
        common = {
            "client_ids": [0, 1, 2, 3],
            "use_attack": True,
            "attacker_id": 1,
            "lazy_honest_gap": 3,
            "lazy_honest_phase": 0,
            "probabilistic_fake_rounds": set(),
            "public_warm_start": False,
        }

        pure = [
            resolve(
                round_index,
                **common,
                use_lazy=False,
                lazy_schedule="fixed",
            )
            for round_index in range(3)
        ]
        self.assertEqual([row["attack_state"] for row in pure], ["fake"] * 3)
        self.assertEqual([row["is_fake_round"] for row in pure], [True] * 3)

        fixed = [
            resolve(
                round_index,
                **common,
                use_lazy=True,
                lazy_schedule="fixed",
            )
            for round_index in range(6)
        ]
        self.assertEqual(
            [row["attack_state"] for row in fixed],
            ["honest", "fake", "fake", "honest", "fake", "fake"],
        )
        self.assertEqual(fixed[0]["ground_truth_labels"], {0: 0, 1: 0, 2: 0, 3: 0})
        self.assertEqual(fixed[1]["ground_truth_labels"], {0: 0, 1: 1, 2: 0, 3: 0})

        probabilistic_common = {
            **common,
            "probabilistic_fake_rounds": {1, 4},
        }
        probabilistic = [
            resolve(
                round_index,
                **probabilistic_common,
                use_lazy=True,
                lazy_schedule="probabilistic",
            )
            for round_index in range(6)
        ]
        self.assertEqual(
            [row["is_fake_round"] for row in probabilistic],
            [False, True, False, False, True, False],
        )

        benign = resolve(
            2,
            **{**common, "use_attack": False, "attacker_id": None},
            use_lazy=True,
            lazy_schedule="fixed",
        )
        self.assertEqual(benign["attack_state"], "benign")
        self.assertEqual(benign["active_attackers"], [])
        self.assertFalse(any(benign["ground_truth_labels"].values()))

        public_warm_start = resolve(
            0,
            **{**common, "public_warm_start": True},
            use_lazy=False,
            lazy_schedule="fixed",
        )
        self.assertEqual(public_warm_start["attack_state"], "public_warm_start")
        self.assertFalse(public_warm_start["is_fake_round"])
        self.assertFalse(public_warm_start["is_honest_round"])

    def test_ground_truth_attachment_recomputes_std_metrics_only_on_fake_rounds(self) -> None:
        namespace = self.compile_functions("_attach_round_attack_ground_truth")
        attach = namespace["_attach_round_attack_ground_truth"]

        def score_evaluator(client_scores, ground_truth_attackers):
            attacker_set = set(ground_truth_attackers)
            labels = {
                client_id: int(client_id in attacker_set)
                for client_id in client_scores
            }
            has_positive_and_negative = 0 < len(attacker_set) < len(client_scores)
            return {
                "labels": labels,
                "energy_ranks": {0: 2.0, 1: 1.0},
                "energy_percentiles": {0: 0.0, 1: 1.0},
                "top_clients": [1],
                "roc_auc": 1.0 if has_positive_and_negative else None,
                "average_precision": 1.0 if has_positive_and_negative else None,
                "attacker_mean_rank": 1.0 if attacker_set else None,
                "top1_hit": True if attacker_set else None,
            }

        activities = [
            {
                "round_idx": 0,
                "attacker_id": 1,
                "attack_state": "honest",
                "is_fake_round": False,
                "is_honest_round": True,
                "active_attackers": [],
                "ground_truth_labels": {0: 0, 1: 0},
                "designated_attacker_labels": {0: 0, 1: 1},
            },
            {
                "round_idx": 1,
                "attacker_id": 1,
                "attack_state": "fake",
                "is_fake_round": True,
                "is_honest_round": False,
                "active_attackers": [1],
                "ground_truth_labels": {0: 0, 1: 1},
                "designated_attacker_labels": {0: 0, 1: 1},
            },
        ]
        result = {
            "round_records": [
                {"round_idx": 0, "client_scores": {0: 0.1, 1: 0.9}},
                {"round_idx": 1, "client_scores": {0: 0.1, 1: 0.9}},
            ],
            "evaluation": {},
        }

        attached = attach(result, activities, score_evaluator=score_evaluator)

        self.assertEqual(
            attached["round_records"][0]["ground_truth_labels"],
            {0: 0, 1: 0},
        )
        self.assertIsNone(attached["round_records"][0]["roc_auc"])
        self.assertEqual(
            attached["round_records"][1]["ground_truth_labels"],
            {0: 0, 1: 1},
        )
        self.assertEqual(attached["round_records"][1]["roc_auc"], 1.0)
        self.assertEqual(attached["evaluation"]["valid_auc_rounds"], 1)
        self.assertEqual(attached["evaluation"]["mean_round_auc"], 1.0)
        self.assertEqual(attached["evaluation"]["top1_hit_rate"], 1.0)
        self.assertEqual(attached["active_attack_rounds"], [1])
        self.assertEqual(attached["honest_attacker_rounds"], [0])

    def test_round_activity_is_written_as_round_client_csv(self) -> None:
        namespace = self.compile_functions("_write_round_attack_activity_csv")
        write_activity = namespace["_write_round_attack_activity_csv"]
        activities = [
            {
                "round_idx": 0,
                "attack_state": "honest",
                "is_fake_round": False,
                "is_honest_round": True,
                "ground_truth_labels": {0: 0, 1: 0},
                "designated_attacker_labels": {0: 0, 1: 1},
            },
            {
                "round_idx": 1,
                "attack_state": "fake",
                "is_fake_round": True,
                "is_honest_round": False,
                "ground_truth_labels": {0: 0, 1: 1},
                "designated_attacker_labels": {0: 0, 1: 1},
            },
        ]
        metadata = {
            "Dataset": "adult",
            "AttackMethod": "moirai_online",
            "DetectingMethod": "std_dagmm",
            "QualitativeAggregationMode": "passive",
            "Alpha": 1000,
            "ClientNum": 2,
            "AttackerID": 1,
            "TrialID": 0,
            "UseAttack": True,
            "UseLazy": True,
            "LazySchedule": "fixed",
            "AttackProbability": 1.0,
            "LazyHonestGap": 3,
            "LazyHonestPhase": 0,
            "LazyFakePolicy": "attack",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "attack_round_activity_trial0.csv"
            write_activity(output_path, activities, metadata)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [row["GroundTruthActiveAttack"] for row in rows],
            ["0", "0", "0", "1"],
        )
        self.assertEqual(
            [row["IsDesignatedAttacker"] for row in rows],
            ["0", "1", "0", "1"],
        )
        self.assertEqual({row["QualitativeAggregationMode"] for row in rows}, {"passive"})
        self.assertEqual({row["AttackState"] for row in rows[:2]}, {"honest"})
        self.assertEqual({row["AttackState"] for row in rows[2:]}, {"fake"})


class QualitativeDetectionArtifactTest(_FLCEFunctionHarness):
    def _metadata(
        self,
        *,
        detecting_method: str = "wef",
        attack_method: str = "moirai_online",
    ) -> dict[str, object]:
        return {
            "TaskID": "adult-wef-trial0",
            "CampaignComplete": 1,
            "Dataset": "adult",
            "AttackMethod": attack_method,
            "AttackRegime": "lazy",
            "Alpha": 1000,
            "ClientNum": 2,
            "AttackerID": 1,
            "TrialID": 0,
            "UseHessian": False,
            "UseLazy": True,
            "LazySchedule": "fixed",
            "LazyHonestGap": 6,
            "LazyHonestPhase": 0,
            "LazyFakePolicy": "attack",
            "DetectingMethod": detecting_method,
            "QualitativeAggregationMode": "passive",
        }

    def test_builder_preserves_truth_rows_and_separates_round_from_client_attack_state(self) -> None:
        """Catches a builder that iterates scores or conflates active truth with round state."""
        namespace = self.compile_functions("_build_qualitative_detection_rows")
        build_rows = namespace["_build_qualitative_detection_rows"]
        result = {
            "suspicious_clients": [1],
            "decision_mode": "passive_round_detection",
            "round_records": [
                {
                    "round_idx": 0,
                    "client_scores": {0: 0.1, 1: 0.2},
                    "reputation_threshold": 0.15,
                    "suspicious_clients": [1],
                    "average_frequencies": {0: 0.1, 1: 0.2},
                    "attack_state": "honest",
                    "is_fake_round": False,
                    "is_honest_round": True,
                    "active_attackers": [],
                    "ground_truth_labels": {0: 0, 1: 0},
                    "designated_attacker_labels": {0: 0, 1: 1},
                    "detection_skipped": False,
                },
                {
                    "round_idx": 1,
                    "client_scores": {0: 0.3, 1: 0.9},
                    "reputation_threshold": 0.5,
                    "suspicious_clients": [1],
                    "average_frequencies": {0: 0.3, 1: 0.9},
                    "attack_state": "fake",
                    "is_fake_round": True,
                    "is_honest_round": False,
                    "active_attackers": [1],
                    "ground_truth_labels": {0: 0, 1: 1},
                    "designated_attacker_labels": {0: 0, 1: 1},
                    "detection_skipped": False,
                },
            ],
        }

        rows = build_rows(result, self._metadata())

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[3]["GroundTruthActiveAttack"], 1)
        self.assertEqual(rows[3]["DesignatedAttackerRoundState"], "fake")
        self.assertEqual(rows[1]["RoundHasActiveAttack"], 0)
        self.assertEqual(rows[3]["WEFPayloadSource"], "counterfeit_from_submitted_update")
        self.assertEqual(rows[2]["WEFPayloadSource"], "observed_local_training")
        self.assertEqual(rows[0]["LambdaH"], "N/A")
        self.assertEqual(
            rows[3]["DetectorDetailsJSON"],
            '{"average_frequency":0.9}',
        )

    def test_s2_skipped_round_writes_all_truth_rows_with_empty_score_and_prediction(self) -> None:
        """Catches an S2 writer that drops a skipped round because it has no scores."""
        namespace = self.compile_functions("_build_qualitative_detection_rows")
        build_rows = namespace["_build_qualitative_detection_rows"]
        result = {
            "suspicious_clients": [],
            "decision_mode": "passive_round_detection",
            "round_records": [
                {
                    "round_idx": 0,
                    "client_scores": {},
                    "gamma_scores": {},
                    "deviation_scores": {},
                    "suspicious_clients": [],
                    "attack_state": "honest",
                    "is_fake_round": False,
                    "is_honest_round": True,
                    "active_attackers": [],
                    "ground_truth_labels": {0: 0, 1: 0},
                    "designated_attacker_labels": {0: 0, 1: 1},
                    "detection_skipped": True,
                }
            ],
        }

        rows = build_rows(result, self._metadata(detecting_method="s2_wef"))

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["ClientID"] for row in rows], [0, 1])
        for row in rows:
            self.assertEqual(row["DetectionSkipped"], 1)
            self.assertEqual(row["DetectorScore"], "")
            self.assertEqual(row["DetectorThreshold"], "")
            self.assertEqual(row["RoundPredictedSuspicious"], "")
            self.assertEqual(row["FinalPredictedSuspicious"], "")
            self.assertEqual(row["WEFPayloadSource"], "observed_local_training")
            self.assertEqual(row["DetectorDetailsJSON"], "{}")

    def test_s2_rows_repeat_round_level_native_decision_diagnostics(self) -> None:
        """Catches an S2 artifact that cannot reconstruct its binary round decision."""
        namespace = self.compile_functions("_build_qualitative_detection_rows")
        build_rows = namespace["_build_qualitative_detection_rows"]
        result = {
            "suspicious_clients": [2],
            "decision_mode": "passive_round_detection",
            "round_records": [
                {
                    "round_idx": 1,
                    "client_scores": {0: 0.1, 1: 0.2, 2: 2.4},
                    "gamma_scores": {0: 0.01, 1: 0.02, 2: 0.8},
                    "deviation_scores": {0: 0.1, 1: 0.2, 2: 0.9},
                    "cluster_labels": {0: 0, 1: 0, 2: 1},
                    "gamma_flags": {0: False, 1: False, 2: True},
                    "deviation_flags": {0: False, 1: False, 2: True},
                    "suspicious_clients": [2],
                    "tentative_suspicious_clients": [2],
                    "cluster_count": 2,
                    "silhouette_score": 0.71,
                    "merge_jump_ratio": 1.9,
                    "gamma_threshold": 0.03,
                    "deviation_threshold": 0.85,
                    "gamma_vote": 1.0,
                    "deviation_vote": 1.0,
                    "simulated_wef_sum": 13.0,
                    "attack_state": "fake",
                    "is_fake_round": True,
                    "is_honest_round": False,
                    "active_attackers": [2],
                    "ground_truth_labels": {0: 0, 1: 0, 2: 1},
                    "designated_attacker_labels": {0: 0, 1: 0, 2: 1},
                    "detection_skipped": False,
                }
            ],
        }

        rows = build_rows(result, self._metadata(detecting_method="s2_wef"))

        self.assertEqual(len(rows), 3)
        for row in rows:
            details = json.loads(row["DetectorDetailsJSON"])
            self.assertEqual(details["cluster_count"], 2)
            self.assertEqual(details["silhouette_score"], 0.71)
            self.assertEqual(details["merge_jump_ratio"], 1.9)
            self.assertEqual(details["tentative_suspicious_clients"], [2])
            self.assertEqual(details["gamma_vote"], 1.0)
            self.assertEqual(details["deviation_vote"], 1.0)
            self.assertEqual(details["simulated_wef_sum"], 13.0)

    def _delta_result(self) -> dict[str, object]:
        return {
            "suspicious_clients": [1],
            "suspicious_counts": {0: 2, 1: 4, 2: 1},
            "decision_threshold": 3,
            "config": {"threshold_multiplier": 1.08},
            "round_records": [
                {
                    "round_idx": 4,
                    "client_scores": {0: -5.0, 1: -2.0, 2: -3.0},
                    "client_means": {0: -0.5, 1: -0.2, 2: -0.3},
                    "mean_energy": -10.0 / 3.0,
                    "energy_threshold": -3.6,
                    "signed_energy_adjusted": False,
                    "suspicious_clients": [0, 1, 2],
                    "attack_state": "fake",
                    "is_fake_round": True,
                    "is_honest_round": False,
                    "active_attackers": [1],
                    "ground_truth_labels": {0: 0, 1: 1, 2: 0},
                    "designated_attacker_labels": {0: 0, 1: 1, 2: 0},
                    "detection_skipped": False,
                }
            ],
        }

    def test_delta_rows_add_descending_score_rank_percentile_and_pathology_flag(self) -> None:
        """Catches Delta output that hides score ordering or a negative-mean native cutoff."""
        namespace = self.compile_functions("_build_qualitative_detection_rows")
        build_rows = namespace["_build_qualitative_detection_rows"]

        rows = build_rows(
            self._delta_result(),
            self._metadata(detecting_method="delta_dagmm"),
        )

        details = {
            row["ClientID"]: json.loads(row["DetectorDetailsJSON"])
            for row in rows
        }
        self.assertEqual(
            {client_id: value["energy_rank"] for client_id, value in details.items()},
            {0: 3.0, 1: 1.0, 2: 2.0},
        )
        self.assertEqual(
            {
                client_id: value["energy_percentile"]
                for client_id, value in details.items()
            },
            {0: 0.0, 1: 1.0, 2: 0.5},
        )
        self.assertEqual(
            [row["DeltaThresholdPathology"] for row in rows],
            [1, 1, 1],
        )

        adjusted_result = self._delta_result()
        adjusted_result["round_records"][0]["signed_energy_adjusted"] = True
        adjusted_rows = build_rows(
            adjusted_result,
            self._metadata(detecting_method="delta_dagmm"),
        )
        self.assertEqual(
            [row["DeltaThresholdPathology"] for row in adjusted_rows],
            [0, 0, 0],
        )

    def test_delta_rows_expose_native_persistence_decision_inputs(self) -> None:
        """Catches a Delta final prediction that cannot be audited from its row data."""
        namespace = self.compile_functions("_build_qualitative_detection_rows")
        build_rows = namespace["_build_qualitative_detection_rows"]

        rows = build_rows(
            self._delta_result(),
            self._metadata(detecting_method="delta_dagmm"),
        )

        self.assertEqual(
            {row["DecisionMode"] for row in rows},
            {
                "native_delta_dagmm:round_energy>1.08*round_mean_energy;"
                "client_detected_round_count>ceil(2R/3)"
            },
        )
        details = {
            row["ClientID"]: json.loads(row["DetectorDetailsJSON"])
            for row in rows
        }
        self.assertEqual(
            {
                client_id: value["client_suspicious_count"]
                for client_id, value in details.items()
            },
            {0: 2, 1: 4, 2: 1},
        )
        self.assertEqual(
            {value["final_persistence_threshold"] for value in details.values()},
            {3},
        )
        self.assertEqual(
            {row["ClientID"]: row["FinalPredictedSuspicious"] for row in rows},
            {0: 0, 1: 1, 2: 0},
        )

    def test_synthetic_replacement_fake_wef_payload_is_observed_local_training(self) -> None:
        """Catches fake synthetic training being mislabeled as a counterfeit WEF payload."""
        namespace = self.compile_functions("_build_qualitative_detection_rows")
        build_rows = namespace["_build_qualitative_detection_rows"]
        result = {
            "suspicious_clients": [1],
            "round_records": [
                {
                    "round_idx": 0,
                    "client_scores": {0: 0.1, 1: 0.9},
                    "reputation_threshold": 0.5,
                    "suspicious_clients": [1],
                    "attack_state": "fake",
                    "is_fake_round": True,
                    "is_honest_round": False,
                    "active_attackers": [1],
                    "ground_truth_labels": {0: 0, 1: 1},
                    "designated_attacker_labels": {0: 0, 1: 1},
                    "detection_skipped": False,
                }
            ],
        }

        rows = build_rows(
            result,
            self._metadata(attack_method="synthetic_replacement"),
        )

        self.assertEqual(rows[1]["WEFPayloadSource"], "observed_local_training")

    def test_writer_uses_approved_uniform_schema(self) -> None:
        """Catches a writer that emits legacy activity columns instead of the joinable artifact."""
        namespace = self.compile_functions(
            "_build_qualitative_detection_rows",
            "_write_qualitative_detection_csv",
        )
        write_csv = namespace["_write_qualitative_detection_csv"]
        result = {
            "suspicious_clients": [],
            "decision_mode": "score_only",
            "round_records": [
                {
                    "round_idx": 0,
                    "client_scores": {0: 0.2, 1: 0.4},
                    "energy_threshold": 0.3,
                    "suspicious_clients": [1],
                    "attack_state": "fake",
                    "is_fake_round": True,
                    "is_honest_round": False,
                    "active_attackers": [1],
                    "ground_truth_labels": {0: 0, 1: 1},
                    "designated_attacker_labels": {0: 0, 1: 1},
                    "detection_skipped": False,
                }
            ],
        }
        expected_schema = [
            "TaskID", "CampaignComplete", "Dataset", "AttackMethod", "AttackRegime",
            "Alpha", "ClientNum", "AttackerID", "TrialID", "UseHessian", "LambdaH",
            "UseLazy", "LazySchedule", "LazyHonestGap", "LazyHonestPhase",
            "LazyFakePolicy", "DetectingMethod", "QualitativeAggregationMode", "Round",
            "ClientID", "IsDesignatedAttacker", "GroundTruthActiveAttack",
            "DesignatedAttackerRoundState", "RoundHasActiveAttack",
            "RoundIsHonestForDesignatedAttacker", "WEFPayloadSource", "DetectionSkipped",
            "DetectorScore", "DetectorThreshold", "DeltaThresholdPathology",
            "RoundPredictedSuspicious",
            "FinalPredictedSuspicious", "DecisionMode", "DetectorDetailsJSON",
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "round_client_detection_trial0.csv"
            write_csv(output_path, result, self._metadata(detecting_method="std_dagmm"))
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(reader.fieldnames, expected_schema)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["DetectorThreshold"], "0.3")


class QualitativeTimingTest(_FLCEFunctionHarness):
    def test_timed_qualitative_call_returns_value_and_nonnegative_elapsed_time(self) -> None:
        """Catches a missing or non-monotonic detector timing wrapper."""
        namespace = self.compile_functions("_timed_qualitative_call")
        timed_call = namespace["_timed_qualitative_call"]

        value, elapsed = timed_call(lambda: "ok", device=None)

        self.assertEqual(value, "ok")
        self.assertGreaterEqual(elapsed, 0.0)

    def _time_metadata(self, detecting_method: str) -> dict[str, object]:
        return {
            "TaskID": "adult-wef-trial0",
            "CampaignComplete": 1,
            "Dataset": "adult",
            "AttackMethod": "moirai_online",
            "AttackRegime": "lazy",
            "Alpha": 1000,
            "ClientNum": 4,
            "AttackerID": 1,
            "TrialID": 0,
            "UseHessian": True,
            "LambdaH": 0.9,
            "UseLazy": True,
            "LazySchedule": "fixed",
            "LazyHonestGap": 6,
            "LazyHonestPhase": 0,
            "LazyFakePolicy": "attack",
            "DetectingMethod": detecting_method,
            "QualitativeAggregationMode": "passive",
        }

    def test_timed_qualitative_call_synchronizes_an_available_cuda_device(self) -> None:
        """Catches CUDA detector timing that omits either boundary synchronization."""
        namespace = self.compile_functions("_timed_qualitative_call")
        timed_call = namespace["_timed_qualitative_call"]

        class FakeCuda:
            def __init__(self) -> None:
                self.synchronized_devices: list[object] = []

            def is_available(self) -> bool:
                return True

            def synchronize(self, device: object) -> None:
                self.synchronized_devices.append(device)

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def device(value: object) -> object:
                return type("FakeDevice", (), {"type": "cuda"})()

        namespace["torch"] = FakeTorch
        value, _elapsed = timed_call(lambda: "ok", device="cuda:7")

        self.assertEqual(value, "ok")
        self.assertEqual(FakeTorch.cuda.synchronized_devices, ["cuda:7", "cuda:7"])

    def test_wef_family_time_row_records_both_detector_totals_and_realized_fake_rounds(self) -> None:
        """Catches WEF timing that skips N-1 detection or counts log entries as fake rounds."""
        namespace = self.compile_functions("_build_qualitative_time_row")
        build_row = namespace["_build_qualitative_time_row"]

        for detecting_method in ("wef", "s2_wef"):
            with self.subTest(detecting_method=detecting_method):
                row = build_row(
                    self._time_metadata(detecting_method),
                    use_attack=True,
                    attack_time_log_with_attacker=[3.0, 4.0, 5.0],
                    round_attack_activity_with=[
                        {"is_fake_round": True},
                        {"is_fake_round": False},
                    ],
                    qualitative_result_with={"detector_time_sec": 1.25},
                    qualitative_result_without={"detector_time_sec": 2.5},
                    fl_with_attacker_wall_sec=10.0,
                    fl_without_attacker_wall_sec=20.0,
                    task_wall_sec=40.0,
                )

                self.assertEqual(row["AttackGenerationWithAttackerTotalSec"], 12.0)
                self.assertEqual(row["FakeRoundsTimed"], 1)
                self.assertEqual(row["DetectorDecisionWithAttackerTotalSec"], 1.25)
                self.assertEqual(row["DetectorDecisionWithoutAttackerTotalSec"], 2.5)
                self.assertEqual(row["DetectorDecisionWithoutAttackerMeasured"], 1)
                self.assertEqual(row["DetectorDecisionWithoutAttackerNotMeasuredReason"], "")
                self.assertEqual(row["FLWithAttackerWallSec"], 10.0)
                self.assertEqual(row["FLWithoutAttackerWallSec"], 20.0)
                self.assertEqual(row["TaskWallSec"], 40.0)

    def test_std_time_row_marks_internal_n_minus_1_detector_as_not_measured(self) -> None:
        """Catches post-hoc STD/Delta timing falsely reporting an internal detector total."""
        namespace = self.compile_functions("_build_qualitative_time_row")
        build_row = namespace["_build_qualitative_time_row"]

        row = build_row(
            self._time_metadata("std_dagmm"),
            use_attack=True,
            attack_time_log_with_attacker=[1.0],
            round_attack_activity_with=[{"is_fake_round": True}],
            qualitative_result_with={"detector_time_sec": 0.5},
            qualitative_result_without={"detector_time_sec": 99.0},
            fl_with_attacker_wall_sec=1.0,
            fl_without_attacker_wall_sec=2.0,
            task_wall_sec=3.0,
        )

        self.assertEqual(row["DetectorDecisionWithoutAttackerTotalSec"], "")
        self.assertEqual(row["DetectorDecisionWithoutAttackerMeasured"], 0)
        self.assertEqual(
            row["DetectorDecisionWithoutAttackerNotMeasuredReason"],
            "posthoc_detector_not_run_for_internal_n_minus_1_control",
        )

    def test_no_attack_time_row_has_only_without_attacker_measurements(self) -> None:
        """Catches a no-attack row that fabricates attacker-side timing values."""
        namespace = self.compile_functions("_build_qualitative_time_row")
        build_row = namespace["_build_qualitative_time_row"]

        row = build_row(
            self._time_metadata("delta_dagmm"),
            use_attack=False,
            attack_time_log_with_attacker=[],
            round_attack_activity_with=[],
            qualitative_result_with=None,
            qualitative_result_without={"detector_time_sec": 0.75},
            fl_with_attacker_wall_sec="",
            fl_without_attacker_wall_sec=6.0,
            task_wall_sec=7.0,
        )

        self.assertEqual(row["AttackGenerationWithAttackerTotalSec"], "")
        self.assertEqual(row["FakeRoundsTimed"], 0)
        self.assertEqual(row["DetectorDecisionWithAttackerTotalSec"], "")
        self.assertEqual(row["DetectorDecisionWithoutAttackerTotalSec"], 0.75)
        self.assertEqual(row["DetectorDecisionWithoutAttackerMeasured"], 1)
        self.assertEqual(row["DetectorDecisionWithoutAttackerNotMeasuredReason"], "")
        self.assertEqual(row["FLWithAttackerWallSec"], "")
        self.assertEqual(row["FLWithoutAttackerWallSec"], 6.0)

    def test_task_wall_timer_starts_immediately_after_cli_parsing(self) -> None:
        """Catches task-wall timing that silently excludes configured setup work."""
        parse_assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets)
        )
        task_wall_assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "task_wall_started_at"
                for target in node.targets
            )
        )

        self.assertEqual(task_wall_assignment.lineno, parse_assignment.lineno + 1)

    def test_qualitative_time_row_is_constructed_after_final_qualitative_artifacts(self) -> None:
        """Catches TaskWallSec being sampled before accuracy or STD summary writes."""
        qualitative_time_assignment = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "qualitative_time_row"
                for target in node.targets
            )
        )
        std_summary_assignment = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "summary_path"
                for target in node.targets
            )
        )

        self.assertGreater(
            qualitative_time_assignment.lineno,
            std_summary_assignment.lineno,
        )


if __name__ == "__main__":
    unittest.main()

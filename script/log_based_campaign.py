"""Run one dataset's LogBased qualitative campaign and publish one CSV.

The PASS campaign supplies the experiment matrix and attack-specific flags.
FLCE output is isolated in a temporary directory for each task. Validated rows
are committed to one SQLite staging file, and the public CSV is published only
after every planned task has completed. Training requires a CUDA server.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading

from passive_pass_campaign import (
    DET_FIELDS,
    PROFILES,
    TIME_COLUMNS,
    Task,
    build_command as build_pass_command,
    check,
    compact,
    csv_rows,
    env_flag,
    env_ints,
    plan_tasks,
    read_json,
)


METHOD = "log_based_free_rider_detection"
EPOCHS = 50
LOGFR_SETTINGS = {
    "val_fraction": 0.15,
    "val_seed": 42,
    "report_seed": 42,
    "linkage": "average",
    "cut_ratio": 0.7,
    "group_order": "client_id",
    "complaint_policy": "silent_fake",
    "shadow_full_audit": False,
}
LOGFR_FLAGS = (
    "--logfr_val_fraction", "0.15",
    "--logfr_val_seed", "42",
    "--logfr_report_seed", "42",
    "--logfr_linkage", "average",
    "--logfr_cut_ratio", "0.7",
    "--logfr_group_order", "client_id",
    "--logfr_complaint_policy", "silent_fake",
)
EXTRA_FIELDS = (
    "CampaignID RunnerTaskID Distribution GroundTruthFakeReport "
    "ReportSource ReportedAccuracy Complaint ExactCopy DecisionValid "
    "DecisionStatus InvalidReason AuditTriggered AuditReason AGap "
    "SafeGapNext ClusterLabel GroupSkipped FirstRoundUncalibrated "
    "ControlDefinition ControlClientID ControlDecisionValid "
    "ControlDecisionStatus ControlRoundPredictedSuspicious "
    "ControlEverPredictedSuspicious ControlGlobalAccuracy "
    "WithAttackerAccuracy WithoutAttackerAccuracy "
    "ClientOriginalSamples ClientTrainSamples ClientValidationSamples "
    "TaskInvestigatedRounds TaskAuditedSubmissions TaskGroupSkippedSubmissions "
    "TaskCopyHitSubmissions TaskValidDecisions ControlCoverageJSON "
    "LogFRValidationFraction LogFRValidationSeed LogFRReportSeedEffective "
    "LogFRLinkage LogFRCutRatio LogFRGroupOrder LogFRComplaintPolicy "
    "LogFRShadowFullAudit DetectorDecisionWithoutAttackerMeasured "
    "DetectorDecisionWithoutAttackerNotMeasuredReason"
).split() + list(TIME_COLUMNS.values())
FIELDNAMES = DET_FIELDS + EXTRA_FIELDS


def task_id(task: Task) -> str:
    regime = "pure" if task.gap is None else f"lazy_gap{task.gap}"
    return (
        f"logfr_{task.dataset}_t{task.trial}_a{task.alpha}_"
        f"n{task.clients}_{regime}_{task.attack}"
    )


def build_command(task: Task, runner: Path, results: Path, python_bin: str) -> list[str]:
    """Keep PASS attack profiles while replacing only its detector arguments."""
    original = build_pass_command(task, runner, results, python_bin, 256)
    command: list[str] = []
    position = 0
    while position < len(original):
        token = original[position]
        if token.startswith("--pass_"):
            check(position + 1 < len(original), f"PASS flag has no value: {token}")
            position += 2
            continue
        if token == "--detecting_method":
            check(position + 1 < len(original), "Missing detector method")
            command.extend((token, METHOD))
            position += 2
            continue
        command.append(token)
        position += 1
    check(not any(token.startswith("--pass_") for token in command), "PASS flag leaked")
    check("--logfr_shadow_full_audit" not in command, "Shadow audit must be off")
    command.extend(LOGFR_FLAGS)
    return command


def _client_value(mapping: dict, client_id: int, *, required: bool = True):
    if not isinstance(mapping, dict):
        raise ValueError("Expected per-client dictionary")
    if str(client_id) in mapping:
        return mapping[str(client_id)]
    if client_id in mapping:
        return mapping[client_id]
    if required:
        raise ValueError(f"Missing client {client_id} from detector record")
    return None


def _validate_metadata(row: dict, task: Task) -> None:
    expected = {
        "Dataset": task.dataset,
        "AttackMethod": task.attack,
        "DetectingMethod": METHOD,
        "Alpha": task.alpha,
        "ClientNum": task.clients,
        "AttackerID": 1,
        "TrialID": task.trial,
    }
    for field, value in expected.items():
        check(field in row, f"{task_id(task)} missing {field}")
        equal = (
            float(row[field]) == float(value)
            if field in {"Alpha", "ClientNum", "AttackerID", "TrialID"}
            else str(row[field]) == str(value)
        )
        check(equal, f"{task_id(task)}: wrong {field}: {row[field]}")
    if "AttackRegime" in row:
        check(row["AttackRegime"] == ("pure" if task.gap is None else "lazy"),
              "Attack regime mismatch")
    if "QualitativeAggregationMode" in row:
        check(row["QualitativeAggregationMode"] == "passive",
              "Aggregation mode mismatch")
    check(str(row["UseLazy"]).lower() in
          (("false", "0") if task.gap is None else ("true", "1")),
          "UseLazy mismatch")
    if task.gap is not None:
        check(int(row["LazyHonestGap"]) == task.gap, "Lazy gap mismatch")
        check(int(row["LazyHonestPhase"]) == 0, "Lazy phase mismatch")
        check(row["LazySchedule"] == "fixed", "Lazy schedule mismatch")
        check(row["LazyFakePolicy"] == "attack", "Lazy policy mismatch")


def _task_rows(results: Path, task: Task, campaign_id: str) -> list[dict]:
    """Validate FLCE artifacts before any row enters the durable stage."""
    detector_dir = results / f"Detecting_qualitative_{METHOD}"
    accuracy_dir = results / f"Accuracy_qualitative_{METHOD}"
    time_dir = results / f"Time_qualitative_{METHOD}"
    trial = task.trial
    detection = csv_rows(detector_dir / f"round_client_detection_trial{trial}.csv")
    accuracy = csv_rows(accuracy_dir / f"global_accuracy_log_trial{trial}.csv")
    timing = csv_rows(time_dir / f"attacking_time_log_trial{trial}.csv")
    diagnostic = read_json(
        detector_dir / f"log_based_free_rider_diagnostics_trial{trial}.json"
    )
    control = read_json(
        detector_dir
        / f"log_based_free_rider_diagnostics_trial{trial}_without_attacker.json"
    )
    check(len(detection) == EPOCHS * task.clients, "Incomplete detection CSV")
    check(len(accuracy) == EPOCHS, "Incomplete accuracy CSV")
    check(len(timing) == 1, "Missing or duplicated timing row")
    for row in detection + accuracy + timing:
        _validate_metadata(row, task)
    check(all("AttackRegime" in row and "QualitativeAggregationMode" in row
              for row in detection), "Detection metadata is incomplete")
    check(
        {(int(row["Round"]), int(row["ClientID"])) for row in detection}
        == {(r, i) for r in range(EPOCHS) for i in range(task.clients)},
        "Missing or duplicated round/client row",
    )
    acc = {int(row["Round"]): row for row in accuracy}
    records = {int(row["round_idx"]): row for row in diagnostic["round_records"]}
    controls = {int(row["round_idx"]): row for row in control["round_records"]}
    expected_rounds = set(range(EPOCHS))
    check(len(acc) == len(records) == len(controls) == EPOCHS, "Duplicate rounds")
    check(set(acc) == set(records) == set(controls) == expected_rounds, "Missing rounds")
    check(diagnostic["experiment"]["original_client_ids"] == list(range(task.clients)),
          "Wrong original client mapping")
    original_control_ids = [i for i in range(task.clients) if i != 1]
    check(control["experiment"]["original_client_ids"] == original_control_ids,
          "Wrong N-1 control mapping")
    check(diagnostic["experiment"]["use_attack"] is True, "Expected attack run")
    check(control["experiment"]["use_attack"] is False, "Expected N-1 control")
    config = diagnostic["config"]
    for name, value in LOGFR_SETTINGS.items():
        expected = value + 100_003 * trial if name == "report_seed" else value
        check(config[name] == expected, f"Wrong LogBased setting: {name}")
    check(control["config"] == config, "Attack/control detector settings differ")
    split = diagnostic["validation_split"]
    check(split["fraction"] == LOGFR_SETTINGS["val_fraction"], "Validation fraction mismatch")
    check(split["seed"] == LOGFR_SETTINGS["val_seed"], "Validation seed mismatch")
    client_counts = {int(x["client_id"]): x for x in split["clients"]}
    check(set(client_counts) == set(range(task.clients)), "Missing validation counts")
    check(control["validation_split"]["scope"] == "parent_run_split",
          "Control does not share parent validation split")
    check(len(control["round_records"]) == EPOCHS, "Missing N-1 control rounds")
    check(int(diagnostic["coverage"]["total_rounds"]) == EPOCHS, "Wrong coverage")
    check(int(control["coverage"]["total_rounds"]) == EPOCHS, "Wrong control coverage")
    check(sum(int(bool(row["valid_by_client"][str(i)]))
              for row in records.values() for i in range(task.clients))
          == int(diagnostic["coverage"]["decision_valid_submissions"]),
          "Decision coverage mismatch")

    final_suspicious = {int(i) for i in diagnostic["suspicious_clients"]}
    final_control_suspicious = {int(i) for i in control["suspicious_clients"]}
    rows: list[dict] = []
    for row in sorted(detection, key=lambda x: (int(x["Round"]), int(x["ClientID"]))):
        r, i = int(row["Round"]), int(row["ClientID"])
        record, control_record = records[r], controls[r]
        check(math.isclose(float(record["global_accuracy"]),
                           float(acc[r]["WithAttackerAccuracy"]),
                           rel_tol=1e-9, abs_tol=1e-9),
              f"Attack accuracy mismatch at round {r}")
        check(math.isclose(float(control_record["global_accuracy"]),
                           float(acc[r]["WithoutAttackerAccuracy"]),
                           rel_tol=1e-9, abs_tol=1e-9),
              f"N-1 control accuracy mismatch at round {r}")
        report_fake = int(_client_value(record["report_ground_truth_labels"], i))
        expected_fake = int(i == 1 and (task.gap is None or r % task.gap != 0))
        check(report_fake == expected_fake, f"Wrong report label at round {r}")
        model_fake = int(i == 1 and expected_fake and
                         not (task.attack == "adam_local_deviation" and r == 0))
        check(int(row["GroundTruthActiveAttack"]) == model_fake,
              f"Wrong model-upload label at round {r}")
        valid = bool(_client_value(record["valid_by_client"], i))
        status = str(_client_value(record["decision_status_by_client"], i))
        suspicious = int(i in record["suspicious_clients"])
        if valid:
            check(row["RoundPredictedSuspicious"] != "" and
                  int(row["RoundPredictedSuspicious"]) == suspicious,
                  "Round prediction differs from diagnostics")
        else:
            check(row["RoundPredictedSuspicious"] == "", "Unaudited verdict is not blank")
        check(int(row["DetectionSkipped"]) == int(not valid), "Skipped flag mismatch")
        final_valid = bool(_client_value(diagnostic["final_valid_by_client"], i))
        if final_valid:
            check(int(row["FinalPredictedSuspicious"]) == int(i in final_suspicious),
                  "Ever-suspicious verdict mismatch")
        else:
            check(row["FinalPredictedSuspicious"] == "", "Invalid final verdict")
        check(bool(_client_value(record["report_sources"], i)), "Missing report source")
        report_source = _client_value(record["report_sources"], i)
        check(bool(_client_value(record["report_ground_truth_labels"], i)) ==
              (i in record["report_fake_clients"]), "Report fake list mismatch")

        control_id = original_control_ids.index(i) if i in original_control_ids else None
        control_valid = (
            bool(_client_value(control_record["valid_by_client"], control_id))
            if control_id is not None else None
        )
        control_status = (
            _client_value(control_record["decision_status_by_client"], control_id)
            if control_id is not None else ""
        )
        control_pred = (
            int(control_id in control_record["suspicious_clients"])
            if control_valid else ""
        )
        counts = client_counts[i]
        merged = {field: row[field] for field in DET_FIELDS}
        merged.update({
            "TaskID": task_id(task),
            "CampaignComplete": 1,
            "RunnerTaskID": row["TaskID"],
            "CampaignID": campaign_id,
            "LazyHonestGap": task.gap if task.gap is not None else "N/A",
            "Distribution": (
                "uniform" if task.alpha == PROFILES[task.dataset]["uniform"]
                else "quantity skew"
            ),
            "GroundTruthFakeReport": report_fake,
            "ReportSource": report_source,
            "ReportedAccuracy": _client_value(record["reported_accuracies"], i),
            "Complaint": int(bool(_client_value(record["complaints"], i))),
            "ExactCopy": int(bool(_client_value(record["copy_flags"], i))),
            "DecisionValid": int(valid),
            "DecisionStatus": status,
            "InvalidReason": _client_value(record["invalid_reasons"], i),
            "AuditTriggered": int(bool(record["audit_triggered"])),
            "AuditReason": record["audit_reason"],
            "AGap": record["a_gap"],
            "SafeGapNext": record["safe_gap_next"],
            "ClusterLabel": _client_value(record["cluster_labels"], i, required=False),
            "GroupSkipped": int(i in record["group_skipped_clients"]),
            "FirstRoundUncalibrated": int(status == "uncalibrated_first_round"),
            "ControlDefinition": "existing_remove_attacker_N_minus_1",
            "ControlClientID": control_id if control_id is not None else "",
            "ControlDecisionValid": int(control_valid) if control_valid is not None else "",
            "ControlDecisionStatus": control_status,
            "ControlRoundPredictedSuspicious": control_pred,
            "ControlEverPredictedSuspicious": (
                int(control_id in final_control_suspicious)
                if control_id is not None and
                bool(_client_value(control["final_valid_by_client"], control_id))
                else ""
            ),
            "ControlGlobalAccuracy": control_record["global_accuracy"],
            "WithAttackerAccuracy": acc[r]["WithAttackerAccuracy"],
            "WithoutAttackerAccuracy": acc[r]["WithoutAttackerAccuracy"],
            "ClientOriginalSamples": counts["original_count"],
            "ClientTrainSamples": counts["train_count"],
            "ClientValidationSamples": counts["validation_count"],
            "TaskInvestigatedRounds": diagnostic["coverage"]["investigated_rounds"],
            "TaskAuditedSubmissions": diagnostic["coverage"]["audited_submissions"],
            "TaskGroupSkippedSubmissions": diagnostic["coverage"]["group_skipped_submissions"],
            "TaskCopyHitSubmissions": diagnostic["coverage"]["copy_hit_submissions"],
            "TaskValidDecisions": diagnostic["coverage"]["decision_valid_submissions"],
            "ControlCoverageJSON": compact(control["coverage"]),
            "LogFRValidationFraction": config["val_fraction"],
            "LogFRValidationSeed": config["val_seed"],
            "LogFRReportSeedEffective": config["report_seed"],
            "LogFRLinkage": config["linkage"],
            "LogFRCutRatio": config["cut_ratio"],
            "LogFRGroupOrder": config["group_order"],
            "LogFRComplaintPolicy": config["complaint_policy"],
            "LogFRShadowFullAudit": int(config["shadow_full_audit"]),
            "DetectorDecisionWithoutAttackerMeasured":
                timing[0]["DetectorDecisionWithoutAttackerMeasured"],
            "DetectorDecisionWithoutAttackerNotMeasuredReason":
                timing[0]["DetectorDecisionWithoutAttackerNotMeasuredReason"],
        })
        merged.update({target: timing[0][source] for source, target in TIME_COLUMNS.items()})
        check(set(merged) == set(FIELDNAMES), "Merged CSV field mismatch")
        rows.append(merged)
    return rows


def _connect(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database, timeout=60)
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _initialize_stage(database: Path, configuration: dict) -> set[str]:
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(database)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS completed (task_id TEXT PRIMARY KEY, row_count INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS rows (task_id TEXT NOT NULL, row_order INTEGER NOT NULL, "
            "payload TEXT NOT NULL, PRIMARY KEY(task_id, row_order))"
        )
        saved = connection.execute(
            "SELECT value FROM metadata WHERE key='configuration'"
        ).fetchone()
        serialized = compact(configuration)
        if saved is None:
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('configuration',?)", (serialized,)
            )
        else:
            check(saved[0] == serialized, "Existing campaign has different scientific settings")
        return {row[0] for row in connection.execute("SELECT task_id FROM completed")}


def _commit_task(database: Path, task: Task, rows: list[dict]) -> None:
    identifier = task_id(task)
    check(len(rows) == EPOCHS * task.clients, "Task has an incomplete row set")
    with closing(_connect(database)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        check(connection.execute(
            "SELECT 1 FROM completed WHERE task_id=?", (identifier,)
        ).fetchone() is None, f"Task already committed: {identifier}")
        connection.executemany(
            "INSERT INTO rows(task_id,row_order,payload) VALUES(?,?,?)",
            ((identifier, index, compact(row)) for index, row in enumerate(rows)),
        )
        connection.execute(
            "INSERT INTO completed(task_id,row_count) VALUES(?,?)", (identifier, len(rows))
        )


def _publish_csv(database: Path, tasks: list[Task], output: Path) -> int:
    check(not output.exists(), f"Final CSV already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(database)) as connection, connection:
        completed = dict(connection.execute("SELECT task_id,row_count FROM completed"))
        expected = {task_id(task): EPOCHS * task.clients for task in tasks}
        check(completed == expected, "Campaign is incomplete or staging has unexpected tasks")
        check(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0] == sum(expected.values()),
              "Staged CSV row count is incomplete")
        temporary: Path | None = None
        row_count = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", dir=output.parent,
                prefix=f".{output.stem}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
                writer.writeheader()
                for task in tasks:
                    identifier = task_id(task)
                    stored = connection.execute(
                        "SELECT row_order,payload FROM rows WHERE task_id=? ORDER BY row_order",
                        (identifier,),
                    ).fetchall()
                    check(len(stored) == expected[identifier], f"Missing rows: {identifier}")
                    for index, payload in stored:
                        check(index < expected[identifier], "Unexpected row index")
                        row = json.loads(payload)
                        check(set(row) == set(FIELDNAMES), "Staged CSV schema mismatch")
                        writer.writerow(row)
                        row_count += 1
                stream.flush()
                os.fsync(stream.fileno())
            check(row_count == sum(expected.values()), "Final row count mismatch")
            # Publish in the same filesystem without overwriting an earlier result.
            os.link(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"MERGED|tasks={len(tasks)}|rows={row_count}|file={output}", flush=True)
    return row_count


class CampaignRunner:
    def __init__(
        self, tasks: list[Task], database: Path, logs: Path, root: Path,
        runner: Path, python_bin: str, campaign_id: str, gpus: list[int],
        workers_per_gpu: int, threads: int,
    ) -> None:
        self.pending: queue.Queue[Task] = queue.Queue()
        for task in tasks:
            self.pending.put(task)
        self.database = database
        self.logs = logs
        self.root = root
        self.runner = runner
        self.python_bin = python_bin
        self.campaign_id = campaign_id
        self.gpus = gpus
        self.workers_per_gpu = workers_per_gpu
        self.threads = threads
        self.stop = threading.Event()
        self.failures: list[str] = []
        self.failure_lock = threading.Lock()

    def _environment(self, gpu: int, folder: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        )
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = str(self.threads)
        for key, name in (
            ("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
            ("XDG_CACHE_HOME", "cache"), ("TORCH_HOME", "torch"),
        ):
            path = folder / name
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
        environment["HF_HOME"] = os.environ.get(
            "SHARED_HF_HOME", os.environ.get("HF_HOME", "/output/huggingface")
        )
        environment["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
        environment["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")
        return environment

    def _run_one(self, task: Task, gpu: int) -> None:
        identifier = task_id(task)
        log_path = self.logs / f"{identifier}.log"
        print(f"START|gpu={gpu}|{identifier}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"{identifier}_") as temporary:
            folder = Path(temporary)
            command = build_command(task, self.runner, folder / "results", self.python_bin)
            with log_path.open("w", encoding="utf-8") as log:
                log.write("COMMAND " + shlex.join(command) + "\n")
                log.flush()
                with subprocess.Popen(
                    command, cwd=self.root, env=self._environment(gpu, folder),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process:
                    while True:
                        try:
                            exit_code = process.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            if self.stop.is_set():
                                os.killpg(process.pid, signal.SIGTERM)
                                try:
                                    exit_code = process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    os.killpg(process.pid, signal.SIGKILL)
                                    exit_code = process.wait()
                                break
            check(exit_code == 0, f"{identifier} exited {exit_code}; see {log_path}")
            rows = _task_rows(folder / "results", task, self.campaign_id)
            _commit_task(self.database, task, rows)
        print(f"END|{identifier}|success|rows={len(rows)}", flush=True)

    def _worker(self, gpu: int) -> None:
        while not self.stop.is_set():
            try:
                task = self.pending.get_nowait()
            except queue.Empty:
                return
            try:
                self._run_one(task, gpu)
            except Exception as error:
                with self.failure_lock:
                    self.failures.append(f"{task_id(task)}: {error}")
                self.stop.set()
                print(f"FAIL|{task_id(task)}|{error}", file=sys.stderr, flush=True)

    def run(self) -> None:
        previous = {}

        def stop(_signum, _frame):
            self.stop.set()

        try:
            for event in (signal.SIGINT, signal.SIGTERM):
                previous[event] = signal.signal(event, stop)
            with ThreadPoolExecutor(max_workers=len(self.gpus) * self.workers_per_gpu) as pool:
                futures = [
                    pool.submit(self._worker, gpu)
                    for gpu in self.gpus
                    for _ in range(self.workers_per_gpu)
                ]
                for future in futures:
                    future.result()
        finally:
            for event, handler in previous.items():
                signal.signal(event, handler)
        check(not self.failures, "Campaign task failed; see flat logs; no final CSV published")
        check(not self.stop.is_set(), "Campaign interrupted; no final CSV published")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=PROFILES)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    dry = env_flag("DRY_RUN", "1")
    merge_only = env_flag("MERGE_ONLY", "0")
    confirm = env_flag("CONFIRM_QUALITATIVE_RUN", "0")
    check(os.environ.get("ATTACKER_ID", "1") == "1", "Only attacker_id=1 is supported")
    check(os.environ.get("NUM_EPOCH", "50") == "50", "Keep NUM_EPOCH=50")
    trials = env_ints("TRIAL_IDS", "0")
    gaps = env_ints("LAZY_GAPS", "3 6 10")
    tasks = plan_tasks(args.dataset, trials, gaps)
    gpus = env_ints("GPU_IDS", PROFILES[args.dataset]["gpu"])
    workers_per_gpu = int(os.environ.get("WORKERS_PER_GPU", "1"))
    threads = int(os.environ.get("CPU_THREADS_PER_WORKER", "8"))
    check(workers_per_gpu > 0, "WORKERS_PER_GPU must be positive")
    check(threads > 0, "CPU_THREADS_PER_WORKER must be positive")
    campaign_id = os.environ.get(
        "CAMPAIGN_ID", f"logfr_{datetime.now():%Y%m%d-%H%M%S}_pid{os.getpid()}"
    )
    check(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", campaign_id) is not None,
          "Invalid CAMPAIGN_ID")
    runner = Path(os.environ.get("SCRIPT", "FLCE_updated.py"))
    if not runner.is_absolute():
        runner = root / runner
    python_bin = os.environ.get("PYTHON_BIN", "python")
    stage = root / ".qualitative_runs" / campaign_id
    database = stage / f"{args.dataset}_log_based.sqlite"
    logs = stage / f"{args.dataset}_log_based_logs"
    output = root / "Results" / campaign_id / f"{args.dataset}_log_based.csv"
    scientific_config = {
        "schema": 1, "dataset": args.dataset, "trials": trials, "gaps": gaps,
        "epochs": EPOCHS, "attacker_id": 1, "method": METHOD,
        "logfr_settings": LOGFR_SETTINGS,
        "profile": PROFILES[args.dataset],
        "tasks": [asdict(task) for task in tasks],
    }
    print(
        f"CONFIG|dataset={args.dataset}|tasks={len(tasks)}|gpus={gpus}|"
        f"workers_per_gpu={workers_per_gpu}|cpu_threads_per_worker={threads}|"
        f"gaps={gaps}|csv={output}", flush=True,
    )
    if dry:
        check(not merge_only, "MERGE_ONLY requires DRY_RUN=0")
        for task in tasks:
            command = build_command(
                task, runner, Path("<temporary-results-root>"), python_bin
            )
            print("TASK|" + compact({"task_id": task_id(task), **asdict(task)}))
            print("command: " + shlex.join(command))
        print(f"DRY_RUN|tasks={len(tasks)}|no experiments launched", flush=True)
        return
    check(confirm or merge_only, "Use DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 to run")
    check(os.name == "posix", "Training campaign requires a Linux GPU server")
    check(runner.is_file(), f"Missing FLCE runner: {runner}")
    check(not output.exists(), f"Final CSV already exists: {output}")
    if merge_only:
        check("CAMPAIGN_ID" in os.environ, "MERGE_ONLY requires CAMPAIGN_ID")
        check(database.is_file(), f"No staging database: {database}")
        completed = _initialize_stage(database, scientific_config)
        check(completed == {task_id(task) for task in tasks}, "Campaign is incomplete")
        _publish_csv(database, tasks, output)
        return

    import fcntl

    lock_root = Path(os.environ.get(
        "QUALITATIVE_GPU_LOCK_ROOT",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / "passive_qualitative_gpu_locks"),
    ))
    lock_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            handle = stack.enter_context((lock_root / f"gpu_{gpu}.lock").open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"GPU {gpu} is locked by another campaign") from error
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               PYTHONDONTWRITEBYTECODE="1")
            subprocess.run(
                [python_bin, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"],
                env=environment, check=True,
            )
        subprocess.run(
            [python_bin, str(runner), "--detecting_method", METHOD,
             "--detecting_type", "qualitative", "--check_detecting_setup"],
            cwd=root, check=True,
        )
        completed = _initialize_stage(database, scientific_config)
        remaining = [task for task in tasks if task_id(task) not in completed]
        logs.mkdir(parents=True, exist_ok=True)
        print(f"RESUME|completed={len(completed)}|remaining={len(remaining)}", flush=True)
        if remaining:
            CampaignRunner(
                remaining, database, logs, root, runner, python_bin,
                campaign_id, gpus, workers_per_gpu, threads,
            ).run()
        _publish_csv(database, tasks, output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, sqlite3.Error,
            subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)

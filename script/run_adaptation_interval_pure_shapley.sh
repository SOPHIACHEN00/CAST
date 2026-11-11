#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# Adaptation-cadence sensitivity: pure attack, exact last-round Shapley, one attacker.
# Default is a read-only preview. Activate flce_attack before the real run.
# Real run: DRY_RUN=0 CONFIRM_INTERVAL_RUN=1 bash script/run_adaptation_interval_pure_shapley.sh
# Retry failed/interrupted tasks: set RESUME=1 and the original CAMPAIGN_ID.
# Public output: Results/<campaign_id>/{adult,mnist,imdb}.csv (42 rows each).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
ENTRY="$PROJECT_ROOT/FLCE_updated.py"
PYTHON_BIN=${PYTHON_BIN:-python}
DRY_RUN=${DRY_RUN:-1}
CONFIRM_INTERVAL_RUN=${CONFIRM_INTERVAL_RUN:-0}
RESUME=${RESUME:-0}
CAMPAIGN_ID=${CAMPAIGN_ID:-adaptation_interval_pure_shapley_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT="$PROJECT_ROOT/.interval_sensitivity_runs/$CAMPAIGN_ID"
PUBLIC_ROOT="$PROJECT_ROOT/Results/$CAMPAIGN_ID"
STAGE_ROOT="$PROJECT_ROOT/Results/.$CAMPAIGN_ID.staging"
TASK_ROOT="$RUN_ROOT/tasks"
QUEUE_PENDING="$RUN_ROOT/queue/pending"
QUEUE_RUNNING="$RUN_ROOT/queue/running"
QUEUE_DONE="$RUN_ROOT/queue/done"
QUEUE_FAILED="$RUN_ROOT/queue/failed"
PLAN_FILE="$RUN_ROOT/tasks.psv"
SUMMARY_FILE="$RUN_ROOT/summary.txt"

GPU_IDS=(0 1 2 3)
WORKERS_PER_GPU=15
CPU_THREADS_PER_WORKER=15
TRIAL_ID=0
ATTACKER_ID=1
ROUND_COUNT=50
LOCAL_EPOCHS=1
WARMUP_ROUND=10
LAMBDA_H=0.9
MOIRAI_ADAPT_INTERVALS=(2 4 8)
ARIMA_RETRAIN_INTERVALS=(8 15 25)
DATASETS=(adult mnist imdb)
TOTAL_TASKS=126

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

alpha_values() {
  case "$1" in
    adult) printf '%s\n' '1000 100000 1000000 20000000' ;;
    mnist) printf '%s\n' '10 1000 100000 10000000' ;;
    imdb)  printf '%s\n' '10 100 1000 200000' ;;
    *) die "unknown dataset: $1" ;;
  esac
}

methods() {
  case "$1" in
    adult) printf '%s\n' 'arima_adaptive moirai_online' ;;
    mnist|imdb) printf '%s\n' \
      'arima_adaptive_mnist_imdb_layerwise moirai_online_mnist_imdb_layerwise' ;;
    *) die "unknown dataset: $1" ;;
  esac
}

emit_tasks() {
  local id=0 dataset alpha0 alpha2 alpha3 alpha_uf method interval client tier
  # Interleave datasets and methods so the first 60 workers are not all Adult.
  for tier in 0 1 2; do
    for dataset in "${DATASETS[@]}"; do
      read -r alpha0 alpha2 alpha3 alpha_uf <<< "$(alpha_values "$dataset")"
      for method in $(methods "$dataset"); do
        case "$method" in
          arima_*) interval=${ARIMA_RETRAIN_INTERVALS[$tier]} ;;
          moirai_*) interval=${MOIRAI_ADAPT_INTERVALS[$tier]} ;;
          *) die "unexpected attack method: $method" ;;
        esac
        # Client-count plot at the first numeric alpha.
        for client in 2 4 6 8; do
          id=$((id + 1))
          printf '%s|%s|%s|%s|%s|%s|%s\n' \
            "$id" "$dataset" "$alpha0" "$client" "$method" "$interval" "$alpha0"
        done
        # Partition plot at client count six. The shared anchor is not repeated.
        for alpha in "$alpha2" "$alpha3" "$alpha_uf"; do
          id=$((id + 1))
          if [[ "$alpha" == "$alpha_uf" ]]; then
            printf '%s|%s|%s|6|%s|%s|UF\n' \
              "$id" "$dataset" "$alpha" "$method" "$interval"
          else
            printf '%s|%s|%s|6|%s|%s|%s\n' \
              "$id" "$dataset" "$alpha" "$method" "$interval" "$alpha"
          fi
        done
      done
    done
  done
}

task_name() { printf 'task_%03d' "$1"; }

build_command() {
  local dataset=$1 alpha=$2 client=$3 method=$4 interval=$5 results=$6
  local arima_var arima_err moirai_lr moirai_err
  CMD=(
    "$PYTHON_BIN" "$ENTRY"
    --dataset "$dataset" --alpha "$alpha" --client_num "$client"
    --attacker_id "$ATTACKER_ID" --attack_method "$method"
    --contribution_method shapley --detecting_type quantitative
    --use_last_round_sv --trial_id "$TRIAL_ID"
    --num_epoch "$ROUND_COUNT" --num_local_epochs "$LOCAL_EPOCHS"
    --random_round "$WARMUP_ROUND" --results_root "$results"
    --use_attack --use_hessian --lambda_h "$LAMBDA_H"
  )
  case "$dataset" in
    adult) arima_var=0.005; arima_err=0.08; moirai_lr=0.003; moirai_err=0.12 ;;
    mnist) arima_var=0.05; arima_err=0.05; moirai_lr=0.003; moirai_err=0.06 ;;
    imdb)  arima_var=0.10; arima_err=0.015; moirai_lr=0.01; moirai_err=0.025 ;;
  esac
  case "$method" in
    arima_adaptive|arima_adaptive_mnist_imdb_layerwise)
      CMD+=(
        --arima_var_window 3 --arima_var_threshold "$arima_var"
        --arima_err_threshold "$arima_err"
        --arima_min_rounds_after_start 4 --arima_retrain_interval "$interval"
        --arima_err_ema_beta 0.9 --arima_trim_history
      )
      ;;
    moirai_online|moirai_online_mnist_imdb_layerwise)
      CMD+=(
        --moirai_online_tuning --moirai_tune_lr "$moirai_lr"
        --moirai_tune_steps 1 --moirai_tune_start_round 10
        --moirai_adapt_interval "$interval" --moirai_residual_decay 0.95
        --moirai_residual_clip 5.0 --moirai_err_ema_beta 0.9
        --moirai_err_threshold "$moirai_err"
        --moirai_min_rounds_after_start 4 --moirai_trim_history
      )
      ;;
    *) die "unexpected attack method: $method" ;;
  esac
  if [[ "$dataset" != adult ]]; then
    CMD+=(
      --layerwise_selection_metric normalized_rms
      --layerwise_selection_fraction 0.5
      --layerwise_pca_threshold 512 --layerwise_pca_components 8
    )
  fi
}

preview() {
  local count=0 id dataset alpha client method interval partition task
  printf 'Moirai adapt intervals: %s; ARIMA retrain intervals: %s; lambda: %s\n' \
    "${MOIRAI_ADAPT_INTERVALS[*]}" "${ARIMA_RETRAIN_INTERVALS[*]}" "$LAMBDA_H"
  printf 'Datasets: %s; GPUs: %s; workers/GPU: %s; CPU threads/task: %s\n' \
    "${DATASETS[*]}" "${GPU_IDS[*]}" "$WORKERS_PER_GPU" "$CPU_THREADS_PER_WORKER"
  while IFS='|' read -r id dataset alpha client method interval partition; do
    count=$((count + 1))
    if ((count <= 12)); then
      task=$(task_name "$id")
      build_command "$dataset" "$alpha" "$client" "$method" "$interval" \
        "$TASK_ROOT/$task/results"
      printf 'TASK %03d dataset=%s alpha=%s client=%s method=%s interval=%s partition=%s\n' \
        "$id" "$dataset" "$alpha" "$client" "$method" "$interval" "$partition"
      printf '  '
      printf '%q ' "${CMD[@]}"
      printf '\n'
    fi
  done < <(emit_tasks)
  [[ "$count" -eq "$TOTAL_TASKS" ]] || die "expected $TOTAL_TASKS tasks, found $count"
  printf 'Dry run: %s tasks; no directory created and no GPU used.\n' "$count"
}

check_prerequisites() {
  [[ -f "$ENTRY" ]] || die "runner missing: $ENTRY"
  [[ -f "$PROJECT_ROOT/module/method/attack_updated_nonlayerwise.py" ]] ||
    die 'ordinary attack module missing'
  [[ -f "$PROJECT_ROOT/module/method/attack_updated_layerwise.py" ]] ||
    die 'layerwise attack module missing'
  command -v "$PYTHON_BIN" >/dev/null || die "Python missing: $PYTHON_BIN"
  command -v flock >/dev/null || die 'flock missing'
  command -v nvidia-smi >/dev/null || die 'nvidia-smi missing'
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader,nounits |
      grep -qx "$gpu" || die "GPU $gpu unavailable"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -c \
      'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1' ||
      die "PyTorch cannot use GPU $gpu"
  done
}

acquire_gpu_locks() {
  mkdir -p "$PROJECT_ROOT/.shapley_runs"
  local gpu lock_fd
  LOCK_FDS=()
  for gpu in "${GPU_IDS[@]}"; do
    exec {lock_fd}>"$PROJECT_ROOT/.shapley_runs/.gpu_${gpu}_campaign.lock"
    flock -n "$lock_fd" || die "GPU $gpu is locked by another campaign"
    LOCK_FDS+=("$lock_fd")
  done
}

prepare_run() {
  [[ ! -e "$RUN_ROOT" && ! -e "$PUBLIC_ROOT" && ! -e "$STAGE_ROOT" ]] ||
    die "campaign path already exists: $CAMPAIGN_ID"
  mkdir -p "$TASK_ROOT" "$QUEUE_PENDING" "$QUEUE_RUNNING" \
    "$QUEUE_DONE" "$QUEUE_FAILED" "$PROJECT_ROOT/Results"
  emit_tasks > "$PLAN_FILE"
  [[ "$(wc -l < "$PLAN_FILE")" -eq "$TOTAL_TASKS" ]] ||
    die 'task enumeration drifted'
  local line id name
  while IFS= read -r line; do
    id=${line%%|*}
    name=$(task_name "$id")
    printf '%s\n' "$line" > "$QUEUE_PENDING/$name.spec"
  done < "$PLAN_FILE"
  {
    printf 'campaign_id=%s\ntrial_id=%s\nattacker_id=%s\n' \
      "$CAMPAIGN_ID" "$TRIAL_ID" "$ATTACKER_ID"
    printf 'moirai_adapt_intervals=%s\narima_retrain_intervals=%s\nlambda_h=%s\n' \
      "${MOIRAI_ADAPT_INTERVALS[*]}" "${ARIMA_RETRAIN_INTERVALS[*]}" "$LAMBDA_H"
    printf 'gpus=%s\nworkers_per_gpu=%s\ncpu_threads_per_task=%s\n' \
      "${GPU_IDS[*]}" "$WORKERS_PER_GPU" "$CPU_THREADS_PER_WORKER"
    printf 'total_tasks=%s\nentry=%s\npython=%s\n' "$TOTAL_TASKS" "$ENTRY" "$PYTHON_BIN"
    printf 'started_at=%s\n' "$(date -Is)"
  } > "$SUMMARY_FILE"
}

resume_run() {
  [[ -d "$RUN_ROOT" && -f "$PLAN_FILE" ]] ||
    die "campaign does not exist for resume: $CAMPAIGN_ID"
  [[ ! -e "$PUBLIC_ROOT" && ! -e "$STAGE_ROOT" ]] ||
    die "published or staged output already exists: $CAMPAIGN_ID"
  cmp -s "$PLAN_FILE" <(emit_tasks) ||
    die 'current experiment plan differs from the saved campaign'
  mkdir -p "$RUN_ROOT/previous_attempts"
  local spec line id name attempt target
  for spec in "$QUEUE_FAILED"/*.spec "$QUEUE_RUNNING"/*.spec; do
    [[ -f "$spec" ]] || continue
    line=$(< "$spec")
    id=${line%%|*}
    name=$(task_name "$id")
    [[ ! -e "$QUEUE_PENDING/$name.spec" ]] ||
      die "duplicate pending task during resume: $name"
    if [[ -d "$TASK_ROOT/$name" ]]; then
      attempt=1
      target="$RUN_ROOT/previous_attempts/$name.attempt_$attempt"
      while [[ -e "$target" ]]; do
        attempt=$((attempt + 1))
        target="$RUN_ROOT/previous_attempts/$name.attempt_$attempt"
      done
      mv -- "$TASK_ROOT/$name" "$target"
    fi
    mv -- "$spec" "$QUEUE_PENDING/$name.spec"
  done
  printf 'resumed_at=%s\n' "$(date -Is)" >> "$SUMMARY_FILE"
}

claim_task() {
  local gpu=$1 slot=$2 pending guard claimed
  for pending in "$QUEUE_PENDING"/*.spec; do
    [[ -f "$pending" ]] || continue
    guard="$QUEUE_RUNNING/.${pending##*/}.claim"
    mkdir "$guard" 2>/dev/null || continue
    claimed="$QUEUE_RUNNING/${pending##*/}.gpu_$gpu.slot_$slot.spec"
    if mv -- "$pending" "$claimed" 2>/dev/null; then
      rmdir "$guard"
      printf '%s\n' "$claimed"
      return 0
    fi
    rmdir "$guard" 2>/dev/null || true
  done
  return 1
}

stop_worker() {
  trap - HUP INT TERM
  if [[ -n "${CHILD_PID:-}" ]]; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  exit 130
}

run_task() {
  local gpu=$1 id=$2 dataset=$3 alpha=$4 client=$5 method=$6 interval=$7 partition=$8
  local name task_dir results log status rc rounds fake hessian cuda contrib accuracy timing
  name=$(task_name "$id")
  task_dir="$TASK_ROOT/$name"
  results="$task_dir/results"
  log="$task_dir/run.log"
  status="$task_dir/status.txt"
  mkdir -p "$results" "$task_dir/tmp" "$task_dir/cache"
  build_command "$dataset" "$alpha" "$client" "$method" "$interval" "$results"
  printf '%q ' "${CMD[@]}" > "$task_dir/command.txt"
  printf '\n' >> "$task_dir/command.txt"
  printf 'gpu=%s\ncpu_threads=%s\n' "$gpu" "$CPU_THREADS_PER_WORKER" > "$task_dir/worker.txt"
  (
    cd "$PROJECT_ROOT"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER"
    export TMPDIR="$task_dir/tmp"
    export HF_HOME="$task_dir/cache/huggingface"
    export TORCH_HOME="$task_dir/cache/torch"
    export XDG_CACHE_HOME="$task_dir/cache/xdg"
    exec "${CMD[@]}"
  ) > "$log" 2>&1 &
  CHILD_PID=$!
  if wait "$CHILD_PID"; then rc=0; else rc=$?; fi
  CHILD_PID=''
  if ((rc != 0)); then
    printf 'failed exit_code=%s\n' "$rc" > "$status"
    return 1
  fi
  rounds=$(grep -c '^\[ROUND ' "$log" || true)
  fake=$(grep -c '^\[ROUND .* honest=False ' "$log" || true)
  hessian=$(grep -c '^\[ROUND .* hessian=on ' "$log" || true)
  cuda=$(grep -c 'FL training is running in pure GPU mode' "$log" || true)
  contrib="$results/Detecting_quantitative_shapley/combined_contributions_trial0.csv"
  accuracy="$results/Accuracy_quantitative_shapley/global_accuracy_log_trial0.csv"
  timing="$results/Time_quantitative_shapley/attacking_time_log_trial0.csv"
  if [[ "$rounds" -ne 50 || "$fake" -ne 50 || "$hessian" -ne 50 ||
        "$cuda" -lt 1 || ! -s "$contrib" || ! -s "$accuracy" || ! -s "$timing" ||
        "$(wc -l < "$contrib")" -ne $((client + 1)) ||
        "$(wc -l < "$accuracy")" -ne 51 ||
        "$(wc -l < "$timing")" -ne 2 ]]; then
    printf 'failed protocol_invalid rounds=%s fake=%s hessian=%s cuda=%s\n' \
      "$rounds" "$fake" "$hessian" "$cuda" > "$status"
    return 1
  fi
  printf 'success gpu=%s\n' "$gpu" > "$status"
}

worker_loop() {
  local gpu=$1 slot=$2 spec line id dataset alpha client method interval partition
  CHILD_PID=''
  trap stop_worker HUP INT TERM
  while spec=$(claim_task "$gpu" "$slot"); do
    line=$(< "$spec")
    IFS='|' read -r id dataset alpha client method interval partition <<< "$line"
    if run_task "$gpu" "$id" "$dataset" "$alpha" "$client" "$method" "$interval" "$partition"; then
      mv -- "$spec" "$QUEUE_DONE/${spec##*/}"
    else
      mv -- "$spec" "$QUEUE_FAILED/${spec##*/}"
    fi
  done
  trap - HUP INT TERM
}

stop_launcher() {
  trap - HUP INT TERM
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${WORKER_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  printf 'interrupted_at=%s\n' "$(date -Is)" >> "$SUMMARY_FILE"
  exit 130
}

run_workers() {
  local gpu slot pid
  WORKER_PIDS=()
  trap stop_launcher HUP INT TERM
  for gpu in "${GPU_IDS[@]}"; do
    for ((slot=1; slot<=WORKERS_PER_GPU; slot++)); do
      worker_loop "$gpu" "$slot" > "$RUN_ROOT/worker_gpu${gpu}_slot${slot}.log" 2>&1 &
      pid=$!
      WORKER_PIDS+=("$pid")
    done
  done
  local worker_failures=0
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || worker_failures=$((worker_failures + 1))
  done
  trap - HUP INT TERM
  local complete=( "$QUEUE_DONE"/*.spec )
  local failed=( "$QUEUE_FAILED"/*.spec )
  printf 'successful_tasks=%s\nfailed_tasks=%s\nworker_failures=%s\n' \
    "${#complete[@]}" "${#failed[@]}" "$worker_failures" >> "$SUMMARY_FILE"
  [[ "${#complete[@]}" -eq "$TOTAL_TASKS" && "${#failed[@]}" -eq 0 &&
     "$worker_failures" -eq 0 ]] || die 'incomplete campaign; task logs and raw CSVs retained'
}

merge_results() {
  # The runner needs isolated task roots for concurrent writes. Publish only
  # three dataset-level CSVs after all 126 task artifacts pass validation.
  "$PYTHON_BIN" - "$RUN_ROOT" "$STAGE_ROOT" "$CAMPAIGN_ID" <<'PY_MERGE'
import csv
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
stage = pathlib.Path(sys.argv[2])
campaign_id = sys.argv[3]
plan = root / "tasks.psv"

fields = [
    "CampaignID", "TaskID", "Dataset", "AttackMethod", "PartitionSetting",
    "Alpha", "ClientNum", "AttackerID", "TrialID", "LambdaH",
    "TunedParameter", "IntervalValue", "ResidualUpdateCount",
    "ResidualDecayCount", "ARIMAFullFitCount", "ARIMAPeriodicRefitCount",
    "ErrorResetCount",
    "UseHessian", "UseLazy", "ShapleyRound",
    "AttackerContribution", "TotalContribution", "Q1AttackerRatio",
    "FinalWithAttackerAccuracy", "FinalWithoutAttackerAccuracy",
    "WithAttackerSec", "WithoutAttackerSec",
    "WithAttackerContributionsJSON", "WithoutAttackerContributionsJSON",
    "WithAttackerAccuracyByRoundJSON", "WithoutAttackerAccuracyByRoundJSON",
    "RunnerSHA256", "AttackModuleSHA256",
]

def fail(message):
    raise ValueError(message)

def read_csv(path):
    if not path.is_file():
        fail(f"missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            fail(f"missing or duplicate header: {path}")
        return list(reader)

def number(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError):
        fail(f"non-numeric {label}: {value!r}")
    if not math.isfinite(result):
        fail(f"non-finite {label}: {value!r}")
    return result

def check_metadata(row, dataset, alpha, client, method):
    expected = {
        "Dataset": dataset,
        "AttackMethod": method,
        "ClientNum": str(client),
        "AttackerID": "1",
        "TrialID": "0",
        "DetectingType": "quantitative",
        "DetectingMethod": "shapley",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            fail(f"{key} mismatch for {dataset}/{method}: {row.get(key)!r} != {value!r}")
    if number(row.get("Alpha"), "Alpha") != float(alpha):
        fail(f"alpha mismatch for {dataset}/{method}")
    for key, expected_bool in (("UseAttack", "true"),
                               ("UseHessian", "true"),
                               ("UseLazy", "false")):
        if str(row.get(key, "")).lower() != expected_bool:
            fail(f"{key} mismatch for {dataset}/{method}")

with plan.open(encoding="utf-8", newline="") as handle:
    specs = [line.rstrip("\r\n").split("|") for line in handle if line.strip()]
if len(specs) != 126 or any(len(spec) != 7 for spec in specs):
    fail(f"invalid plan: expected 126 seven-field rows, got {len(specs)}")

by_dataset = {"adult": [], "mnist": [], "imdb": []}
seen = set()
for spec in specs:
    id_text, dataset, alpha, client_text, method, interval_text, partition = spec
    task_id = f"task_{int(id_text):03d}"
    client = int(client_text)
    interval = int(interval_text)
    key = (dataset, alpha, client, method, interval)
    if key in seen:
        fail(f"duplicate configuration: {key}")
    seen.add(key)
    if dataset not in by_dataset:
        fail(f"unexpected dataset: {dataset}")

    task = root / "tasks" / task_id
    status = (task / "status.txt").read_text(encoding="utf-8").strip()
    if not status.startswith("success "):
        fail(f"task not successful: {task_id}: {status}")
    results = task / "results"
    manifest_paths = list((results / "manifests").glob(
        f"run_manifest_{dataset}_{method}_trial0_*.json"
    ))
    if len(manifest_paths) != 1:
        fail(f"expected one manifest for {task_id}, found {len(manifest_paths)}")
    with manifest_paths[0].open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    params = manifest["attack_parameters"]
    if (params.get("dataset") != dataset or params.get("attack_method") != method
            or int(params.get("client_num")) != client
            or number(params.get("alpha"), "manifest alpha") != float(alpha)
            or number(params.get("lambda_h"), "manifest lambda") != 0.9
            or params.get("use_hessian") is not True
            or params.get("use_lazy") is not False
            or params.get("use_attack") is not True):
        fail(f"manifest parameters mismatch: {task_id}")

    log_lines = (task / "run.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if method.startswith("moirai_"):
        tuned_parameter = "moirai_adapt_interval"
        if (interval not in (2, 4, 8)
                or int(params.get(tuned_parameter)) != interval
                or int(params["effective_moirai_online_adapter"]["adapt_interval"])
                != interval):
            fail(f"Moirai interval mismatch: {task_id}")
        observations = [line for line in log_lines
                        if line.startswith("[Moirai]")
                        and "[OnlineTune][OBSERVE]" in line]
        residual_updates = sum("residual_updated=1" in line
                               for line in observations)
        residual_decays = sum("decay_applied=1" in line
                              for line in observations)
        if residual_decays > residual_updates:
            fail(f"Moirai decay count exceeds update count: {task_id}")
        full_fits = periodic_refits = ""
        resets = sum(line.startswith("[Moirai]")
                     and "RESET completed residual_updates=0" in line
                     for line in log_lines)
    elif method.startswith("arima_"):
        tuned_parameter = "arima_retrain_interval"
        if (interval not in (8, 15, 25)
                or int(params.get(tuned_parameter)) != interval
                or int(params["effective_arima_adaptive"]["retrain_interval"])
                != interval):
            fail(f"ARIMA interval mismatch: {task_id}")
        fits = [line for line in log_lines
                if line.startswith("[ARIMA]")
                and "[ADAPT][FIT] reason=" in line]
        full_fits = len(fits)
        periodic_refits = sum("reason=periodic@" in line for line in fits)
        residual_updates = residual_decays = ""
        resets = sum(line.startswith("[ARIMA]")
                     and "RESET completed" in line for line in log_lines)
    else:
        fail(f"unexpected method: {method}")

    contribution = read_csv(
        results / "Detecting_quantitative_shapley" /
        "combined_contributions_trial0.csv"
    )
    accuracy = read_csv(
        results / "Accuracy_quantitative_shapley" /
        "global_accuracy_log_trial0.csv"
    )
    timing = read_csv(
        results / "Time_quantitative_shapley" /
        "attacking_time_log_trial0.csv"
    )
    if len(contribution) != client or len(accuracy) != 50 or len(timing) != 1:
        fail(f"row counts mismatch: {task_id}")
    for row in contribution + accuracy + timing:
        check_metadata(row, dataset, alpha, client, method)

    contribution.sort(key=lambda row: int(row["ClientID"]))
    if [int(row["ClientID"]) for row in contribution] != list(range(client)):
        fail(f"client IDs mismatch: {task_id}")
    with_values = [number(row["WithAttacker"], "WithAttacker")
                   for row in contribution]
    without_values = [
        None if row["WithoutAttacker"] in ("", "N/A")
        else number(row["WithoutAttacker"], "WithoutAttacker")
        for row in contribution
    ]
    if without_values[1] is not None or any(
        value is None for i, value in enumerate(without_values) if i != 1
    ):
        fail(f"without-attacker values mismatch: {task_id}")
    total = sum(with_values)
    attacker = with_values[1]
    ratio = "" if total == 0 else attacker / total

    accuracy.sort(key=lambda row: int(row["Round"]))
    if [int(row["Round"]) for row in accuracy] != list(range(50)):
        fail(f"round IDs mismatch: {task_id}")
    with_accuracy = [number(row["WithAttackerAccuracy"], "accuracy with")
                     for row in accuracy]
    without_accuracy = [number(row["WithoutAttackerAccuracy"], "accuracy without")
                        for row in accuracy]
    with_time = number(timing[0]["WithAttackerSec"], "WithAttackerSec")
    without_time = number(timing[0]["WithoutAttackerSec"], "WithoutAttackerSec")

    encode = lambda values: json.dumps(values, separators=(",", ":"))
    by_dataset[dataset].append({
        "CampaignID": campaign_id,
        "TaskID": task_id,
        "Dataset": dataset,
        "AttackMethod": method,
        "PartitionSetting": partition,
        "Alpha": alpha,
        "ClientNum": client,
        "AttackerID": 1,
        "TrialID": 0,
        "LambdaH": 0.9,
        "TunedParameter": tuned_parameter,
        "IntervalValue": interval,
        "ResidualUpdateCount": residual_updates,
        "ResidualDecayCount": residual_decays,
        "ARIMAFullFitCount": full_fits,
        "ARIMAPeriodicRefitCount": periodic_refits,
        "ErrorResetCount": resets,
        "UseHessian": True,
        "UseLazy": False,
        "ShapleyRound": 49,
        "AttackerContribution": attacker,
        "TotalContribution": total,
        "Q1AttackerRatio": ratio,
        "FinalWithAttackerAccuracy": with_accuracy[-1],
        "FinalWithoutAttackerAccuracy": without_accuracy[-1],
        "WithAttackerSec": with_time,
        "WithoutAttackerSec": without_time,
        "WithAttackerContributionsJSON": encode(with_values),
        "WithoutAttackerContributionsJSON": encode(without_values),
        "WithAttackerAccuracyByRoundJSON": encode(with_accuracy),
        "WithoutAttackerAccuracyByRoundJSON": encode(without_accuracy),
        "RunnerSHA256": manifest["entry_sha256"],
        "AttackModuleSHA256": manifest["attack_module_sha256"],
    })

if any(len(rows) != 42 for rows in by_dataset.values()):
    fail("dataset output count mismatch")
if stage.exists():
    fail(f"stage already exists: {stage}")
stage.mkdir(parents=True)
for dataset, rows in by_dataset.items():
    with (stage / f"{dataset}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
print("Validated 126 tasks and staged adult.csv, mnist.csv, imdb.csv")
PY_MERGE
  mv -- "$STAGE_ROOT" "$PUBLIC_ROOT"
  # Raw per-task CSVs are transient; all client and round data are retained
  # in JSON columns of the three published dataset CSVs.
  find "$TASK_ROOT" -type f -name '*.csv' -delete
  printf 'published_root=%s\nfinished_at=%s\n' \
    "$PUBLIC_ROOT" "$(date -Is)" >> "$SUMMARY_FILE"
  printf 'Published %s/{adult,mnist,imdb}.csv\n' "$PUBLIC_ROOT"
}

main() {
  [[ "$DRY_RUN" =~ ^[01]$ && "$CONFIRM_INTERVAL_RUN" =~ ^[01]$ &&
     "$RESUME" =~ ^[01]$ ]] ||
    die 'DRY_RUN, CONFIRM_INTERVAL_RUN and RESUME must be 0 or 1'
  [[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ &&
     "$CAMPAIGN_ID" != "." && "$CAMPAIGN_ID" != ".." ]] ||
    die 'CAMPAIGN_ID must be a simple directory name'
  if [[ "$DRY_RUN" == 1 ]]; then
    preview
    return 0
  fi
  [[ "$CONFIRM_INTERVAL_RUN" == 1 ]] ||
    die 'real run requires CONFIRM_INTERVAL_RUN=1 after local review'
  check_prerequisites
  acquire_gpu_locks
  if [[ "$RESUME" == 1 ]]; then resume_run; else prepare_run; fi
  run_workers
  merge_results
}

main "$@"


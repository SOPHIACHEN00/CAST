#!/usr/bin/env bash
set -euo pipefail

# Preview all commands by default. Pass --run to start a local campaign.
# Select any of adult, mnist and imdb as positional arguments.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}
PYTHON_BIN=${PYTHON_BIN:-python}
CAMPAIGN_ID=${CAMPAIGN_ID:-offline_lazy_shapley_$(date -u +%Y%m%dT%H%M%SZ)_pid$$}
RUN_ROOT="$PROJECT_ROOT/Results/$CAMPAIGN_ID"
RAW_ROOT="$RUN_ROOT/raw"
LOG_ROOT="$RUN_ROOT/logs"
ENTRY="$PROJECT_ROOT/FLCE_updated.py"
MERGER="$PROJECT_ROOT/script/merge_forecast_shapley_results.py"

GPU_IDS=(0 1 2 3)
SLOTS_PER_GPU=15
CPU_THREADS=15
CLIENTS=6
ATTACKER_IDS=(0 1 2 3 4 5)
TRIAL_ID=0
ROUNDS=50
# Each record is honest gap, paired fake probability, number of fake rounds.
PAIRS=("2:0.50:25" "3:0.66:33" "6:0.82:41" "10:0.90:45")
THREAD_ENV=(
  "OMP_NUM_THREADS=$CPU_THREADS" "OMP_THREAD_LIMIT=$CPU_THREADS"
  "MKL_NUM_THREADS=$CPU_THREADS" "OPENBLAS_NUM_THREADS=$CPU_THREADS"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS" "NUMEXPR_MAX_THREADS=$CPU_THREADS"
  "BLIS_NUM_THREADS=$CPU_THREADS" "VECLIB_MAXIMUM_THREADS=$CPU_THREADS"
  TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
)
# Set the PyTorch pool explicitly because its default can be smaller than OMP_NUM_THREADS.
TORCH_THREAD_SETUP='
import os
import torch

threads = int(os.environ["OMP_NUM_THREADS"])
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)
if torch.get_num_threads() != threads:
    raise RuntimeError(f"Expected {threads} PyTorch CPU threads")
'
PYTHON_RUNNER="$TORCH_THREAD_SETUP"'
import builtins
import runpy
import sys

sys.argv = sys.argv[1:]
# Match direct script execution for older FLCE entries that set attributes on __builtins__.
runpy.run_path(sys.argv[0], init_globals={"__builtins__": builtins}, run_name="__main__")
'

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

emit_plan() {
  local id=0 dataset alpha arima_var suffix attacker_id pair_index gap probability fake
  local method_index schedule_index family schedule gpu
  local families=(arima moirai) schedules=(fixed probabilistic)
  for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
      adult) alpha=1000; arima_var=0.005; suffix='' ;;
      mnist) alpha=10; arima_var=0.05; suffix=_mnist_imdb_layerwise ;;
      imdb) alpha=10; arima_var=0.10; suffix=_mnist_imdb_layerwise ;;
    esac
    for attacker_id in "${ATTACKER_IDS[@]}"; do
      for pair_index in "${!PAIRS[@]}"; do
        IFS=: read -r gap probability fake <<< "${PAIRS[pair_index]}"
        for method_index in "${!families[@]}"; do
          family=${families[method_index]}
          for schedule_index in "${!schedules[@]}"; do
            schedule=${schedules[schedule_index]}
            # Rotate methods and schedules across GPUs for each attacker and gap.
            gpu=${GPU_IDS[(attacker_id + pair_index + method_index + 2 * schedule_index) % ${#GPU_IDS[@]}]}
            id=$((id + 1))
            printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$id" "$dataset" "$alpha" "$arima_var" "$family" "$family$suffix" \
              "$schedule" "$gap" "$probability" "$fake" "$attacker_id" "$gpu"
          done
        done
      done
    done
  done
}

select_task() {
  IFS=$'\t' read -r TASK_ID DATASET ALPHA ARIMA_VAR FAMILY METHOD SCHEDULE GAP PROBABILITY FAKE ATTACKER_ID GPU <<< "$1"
  DATASET_CAMPAIGN="forecast_${DATASET}_lazy_shapley_${CAMPAIGN_ID}"
  printf -v TASK_LABEL 'task_%06d_%s_lazy_%s_offline_a%s_n%s_%s_g%s_p%s_trial%s' \
    "$TASK_ID" "$DATASET" "$FAMILY" "$ALPHA" "$CLIENTS" "$SCHEDULE" "$GAP" \
    "${PROBABILITY//./p}" "$TRIAL_ID"
  TASK_DIR="$LOG_ROOT/$DATASET_CAMPAIGN/tasks/$TASK_LABEL"
  TASK_RESULTS="$RAW_ROOT/$DATASET_CAMPAIGN/$TASK_LABEL"
  TASK_LOG="$TASK_DIR/run.log"
}

build_command() {
  COMMAND=(
    env "${THREAD_ENV[@]}" CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=$GPU"
    "TMPDIR=$TASK_DIR/tmp" "$PYTHON_BIN" -c "$PYTHON_RUNNER" "$ENTRY"
    --dataset "$DATASET" --alpha "$ALPHA" --client_num "$CLIENTS" --attacker_id "$ATTACKER_ID"
    --attack_method "$METHOD" --use_attack --contribution_method shapley --detecting_type quantitative
    --use_last_round_sv --use_hessian --lambda_h 0.9 --trial_id "$TRIAL_ID"
    --num_epoch "$ROUNDS" --num_local_epochs 1 --random_round 10
    --use_lazy --lazy_schedule "$SCHEDULE" --lazy_honest_gap "$GAP" --lazy_honest_phase 0
    --lazy_fake_policy attack --results_root "$TASK_RESULTS"
  )
  if [[ "$SCHEDULE" == probabilistic ]]; then
    COMMAND+=(--attack_probability "$PROBABILITY")
  fi
  if [[ "$FAMILY" == arima ]]; then
    COMMAND+=(--arima_var_window 3 --arima_var_threshold "$ARIMA_VAR")
  fi
  if [[ "$DATASET" != adult ]]; then
    COMMAND+=(--layerwise_selection_metric normalized_rms --layerwise_selection_fraction 0.5
      --layerwise_pca_threshold 512 --layerwise_pca_components 8)
  fi
}

print_command() {
  printf '(cd %q && ' "$PROJECT_ROOT"
  printf '%q ' "${COMMAND[@]}"
  printf ')\n'
}

check_prerequisites() {
  local gpu
  [[ -r "$ENTRY" && -r "$MERGER" ]] || die 'FLCE runner or forecast CSV merger is missing'
  command -v "$PYTHON_BIN" >/dev/null || die "Python missing: $PYTHON_BIN"
  command -v flock >/dev/null || die 'flock is required for GPU locks'
  command -v nvidia-smi >/dev/null || die 'nvidia-smi is required to check GPUs'
  for gpu in "${GPU_IDS[@]}"; do
    nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader,nounits | grep -qx "$gpu" \
      || die "GPU $gpu is unavailable"
    env "${THREAD_ENV[@]}" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON_BIN" -c "$TORCH_THREAD_SETUP"'
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Expected one visible CUDA device per task")
' \
      || die "PyTorch preflight failed for GPU $gpu or the $CPU_THREADS-thread setting"
  done
}

prepare_run() {
  local record gpu lock_fd
  mkdir -p -- "$PROJECT_ROOT/.shapley_runs" "$PROJECT_ROOT/Results"
  for gpu in "${GPU_IDS[@]}"; do
    # Hold the same per-GPU locks as the existing forecast launchers.
    exec {lock_fd}>"$PROJECT_ROOT/.shapley_runs/.gpu_${gpu}_campaign.lock"
    flock -n "$lock_fd" || die "GPU $gpu is locked by another forecast campaign"
  done
  mkdir -- "$RUN_ROOT" || die "Cannot create a fresh campaign directory: $RUN_ROOT"
  for record in "${PLAN[@]}"; do
    select_task "$record"
    mkdir -p -- "$TASK_DIR/tmp" "$TASK_RESULTS"
    printf '%s\n' "$record" >> "$LOG_ROOT/$DATASET_CAMPAIGN/tasks.tsv"
    {
      printf 'task_id=%s\nvariant=%s_offline\nattack_method=%s\nschedule=%s\ngap=%s\n' \
        "$TASK_ID" "$FAMILY" "$METHOD" "$SCHEDULE" "$GAP"
      printf 'probability=%s\nattacker_id=%s\ntrial=%s\ngpu=%s\ncpu_threads=%s\n' \
        "$PROBABILITY" "$ATTACKER_ID" "$TRIAL_ID" "$GPU" "$CPU_THREADS"
      build_command
      print_command
    } > "$TASK_DIR/manifest.txt"
    printf 'pending\n' > "$TASK_DIR/status.txt"
  done
}

audit_task() {
  local rounds honest fake invalid
  read -r rounds honest fake invalid < <(awk -v schedule="$SCHEDULE" -v gap="$GAP" -v attacker="$ATTACKER_ID" '
    /^\[ROUND [0-9]+\] / {
      round = $2 + 0
      is_honest = ($0 ~ / honest=True /)
      is_fake = ($0 ~ / honest=False /)
      if (round != rounds || $3 != "client=" attacker || $0 !~ / hessian=on / || is_honest == is_fake)
        invalid++
      if (schedule == "fixed" && is_honest != (round % gap == 0))
        invalid++
      rounds++
      honest += is_honest
      fake += is_fake
    }
    END { print rounds+0, honest+0, fake+0, invalid+0 }
  ' "$TASK_LOG")
  if [[ "$rounds" == "$ROUNDS" && "$fake" == "$FAKE" && "$honest" -eq $((ROUNDS - FAKE)) \
        && "$invalid" == 0 ]] && grep -Fq 'FL training is running in pure GPU mode' "$TASK_LOG"; then
    printf 'success protocol_pass rounds=%s honest=%s fake=%s hessian=%s\n' \
      "$rounds" "$honest" "$fake" "$rounds" > "$TASK_DIR/status.txt"
  else
    printf 'failed protocol_invalid rounds=%s honest=%s fake=%s expected_fake=%s invalid=%s\n' \
      "$rounds" "$honest" "$fake" "$FAKE" "$invalid" > "$TASK_DIR/status.txt"
  fi
}

stop_jobs() {
  local running pid
  mapfile -t running < <(jobs -pr)
  if ((${#running[@]})); then
    kill -TERM "${running[@]}" 2>/dev/null || true
    for pid in "${running[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
}

run_gpu_slot() {
  local slot=$1 index pid rc
  shift
  local queue=("$@")
  trap stop_jobs EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  # One task at a time in each slot. Each task index belongs to one slot.
  for ((index=slot; index<${#queue[@]}; index+=SLOTS_PER_GPU)); do
    select_task "${queue[index]}"
    build_command
    (cd -- "$PROJECT_ROOT" && exec "${COMMAND[@]}") > "$TASK_LOG" 2>&1 &
    pid=$!
    printf 'Started %s attacker=%s GPU=%s slot=%s PID=%s\n' \
      "$TASK_LABEL" "$ATTACKER_ID" "$GPU" "$slot" "$pid"
    if wait "$pid"; then
      audit_task
    else
      rc=$?
      printf 'failed exit_code=%s\n' "$rc" > "$TASK_DIR/status.txt"
    fi
  done
}

run_campaign() {
  local gpu record slot pid
  local pids=() gpu_tasks=()
  check_prerequisites
  prepare_run
  trap stop_jobs EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for gpu in "${GPU_IDS[@]}"; do
    gpu_tasks=()
    for record in "${PLAN[@]}"; do
      if [[ ${record##*$'\t'} == "$gpu" ]]; then gpu_tasks+=("$record"); fi
    done
    for ((slot=0; slot<SLOTS_PER_GPU && slot<${#gpu_tasks[@]}; slot++)); do
      run_gpu_slot "$slot" "${gpu_tasks[@]}" &
      pids+=("$!")
    done
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || printf 'Worker %s exited early. Checking task statuses during merge.\n' "$pid" >&2
  done
  "$PYTHON_BIN" "$MERGER" --results-root "$RAW_ROOT" --runs-root "$LOG_ROOT" \
    --output-root "$RUN_ROOT/.csv_staging" --datasets "${DATASETS[@]}" --allow-partial-matrix --strict \
    || die "Campaign incomplete. See logs and merge diagnostics under $RUN_ROOT"
  mv -T -- "$RUN_ROOT/.csv_staging" "$RUN_ROOT/csv"
  printf 'Dataset CSVs published under %s/csv\n' "$RUN_ROOT"
}

main() {
  local mode=preview dataset record gpu
  local -A seen=()
  local gpu_counts=(0 0 0 0)
  if [[ ${1:-} == --run ]]; then mode=run; shift; fi
  DATASETS=(adult mnist imdb)
  if (($#)); then DATASETS=("$@"); fi
  [[ "$PROJECT_ROOT" == /* ]] || die 'PROJECT_ROOT must be an absolute path'
  [[ "$CAMPAIGN_ID" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || die 'Invalid CAMPAIGN_ID'
  for dataset in "${DATASETS[@]}"; do
    case "$dataset" in adult|mnist|imdb) ;; *) die "Unknown dataset: $dataset" ;; esac
    [[ -z ${seen[$dataset]:-} ]] || die "duplicate dataset: $dataset"
    seen[$dataset]=1
  done
  mapfile -t PLAN < <(emit_plan)
  ((${#PLAN[@]} == 96 * ${#DATASETS[@]})) || die 'Unexpected task count'
  for record in "${PLAN[@]}"; do
    gpu=${record##*$'\t'}
    gpu_counts[gpu]=$((gpu_counts[gpu] + 1))
  done
  printf 'mode=%s datasets=%s tasks=%s GPUs=%s slots_per_gpu=%s cpu_threads_per_task=%s\n' \
    "$mode" "${DATASETS[*]}" "${#PLAN[@]}" "${GPU_IDS[*]}" "$SLOTS_PER_GPU" "$CPU_THREADS"
  printf 'attacker_ids=%s\nplanned_tasks_by_gpu=%s\noutput=%s/csv/{%s}/\n' \
    "${ATTACKER_IDS[*]}" "${gpu_counts[*]}" "$RUN_ROOT" "${DATASETS[*]}"
  if [[ "$mode" == run ]]; then
    run_campaign
  else
    for record in "${PLAN[@]}"; do
      select_task "$record"
      build_command
      printf '# %s attacker=%s GPU=%s honest=%s fake=%s\n' \
        "$TASK_LABEL" "$ATTACKER_ID" "$GPU" "$((ROUNDS - FAKE))" "$FAKE"
      print_command
    done
  fi
}

main "$@"

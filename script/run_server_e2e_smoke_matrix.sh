#!/usr/bin/env sh
# Isolated 5-dataset x 4-method end-to-end smoke matrix for the corrected
# Moirai/ARIMA attacks. This is intentionally not the full experiment sweep.

set -eu

PROJECT_ROOT=${PROJECT_ROOT:-/output/gradpred_exp/attack}
script_parent=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
CANDIDATE_ROOT=${CANDIDATE_ROOT:-$script_parent}
PYTHON_BIN=${PYTHON_BIN:-/root/conda_envs/flce_attack/bin/python}
GPU_IDS=${GPU_IDS:-0 1 2 3}
MAX_PARALLEL=${MAX_PARALLEL:-4}
NUM_EPOCH=${NUM_EPOCH:-12}
RANDOM_ROUND=${RANDOM_ROUND:-5}
CLIENT_NUM=${CLIENT_NUM:-2}
ATTACKER_ID=${ATTACKER_ID:-1}
LAMBDA_H=${LAMBDA_H:-0.9}
ARIMA_RETRAIN_INTERVAL=${ARIMA_RETRAIN_INTERVAL:-10}
MOIRAI_TUNE_LR=${MOIRAI_TUNE_LR:-0.001}
MOIRAI_TUNE_STEPS=${MOIRAI_TUNE_STEPS:-1}
MOIRAI_ADAPT_INTERVAL=${MOIRAI_ADAPT_INTERVAL:-10}
MOIRAI_RESIDUAL_DECAY=${MOIRAI_RESIDUAL_DECAY:-0.9}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
CAMPAIGN_ID=${CAMPAIGN_ID:-forecast_e2e_smoke_${timestamp}_pid$$}
RUN_PARENT=${RUN_PARENT:-${PROJECT_ROOT}/.smoke_runs}
RUN_ROOT=${RUN_ROOT:-${RUN_PARENT}/${CAMPAIGN_ID}}
TASKS_ROOT=${TASKS_ROOT:-${RUN_ROOT}/tasks}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_ROOT}/Results/${CAMPAIGN_ID}}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

case "$CAMPAIGN_ID" in
  ''|*[!A-Za-z0-9._-]*) die "CAMPAIGN_ID must be one safe path segment: $CAMPAIGN_ID" ;;
esac
case "$MAX_PARALLEL" in ''|*[!0-9]*) die 'MAX_PARALLEL must be a positive integer' ;; esac
case "$NUM_EPOCH" in ''|*[!0-9]*) die 'NUM_EPOCH must be a positive integer' ;; esac
[ "$MAX_PARALLEL" -gt 0 ] || die 'MAX_PARALLEL must be at least 1'
[ "$NUM_EPOCH" -gt "$RANDOM_ROUND" ] || die 'NUM_EPOCH must exceed RANDOM_ROUND'
[ "$ATTACKER_ID" -lt "$CLIENT_NUM" ] || die 'ATTACKER_ID must be smaller than CLIENT_NUM'

[ -d "$PROJECT_ROOT" ] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[ -d "$CANDIDATE_ROOT" ] || die "CANDIDATE_ROOT does not exist: $CANDIDATE_ROOT"
PROJECT_ROOT=$(CDPATH= cd -- "$PROJECT_ROOT" && pwd -P)
CANDIDATE_ROOT=$(CDPATH= cd -- "$CANDIDATE_ROOT" && pwd -P)
case "$CANDIDATE_ROOT" in
  "$PROJECT_ROOT"/.candidate_runs/*/attack) ;;
  *) die "candidate must be isolated below $PROJECT_ROOT/.candidate_runs" ;;
esac
[ "$RUN_PARENT" = "$PROJECT_ROOT/.smoke_runs" ] || die "RUN_PARENT must be $PROJECT_ROOT/.smoke_runs"
[ "$RUN_ROOT" = "$RUN_PARENT/$CAMPAIGN_ID" ] || die 'RUN_ROOT does not match CAMPAIGN_ID'
[ "$TASKS_ROOT" = "$RUN_ROOT/tasks" ] || die 'TASKS_ROOT must be RUN_ROOT/tasks'
[ "$RESULTS_ROOT" = "$PROJECT_ROOT/Results/$CAMPAIGN_ID" ] || die 'RESULTS_ROOT is outside the isolated Results root'

ENTRY=$CANDIDATE_ROOT/FLCE_updated.py
[ -f "$ENTRY" ] || die "missing candidate entrypoint: $ENTRY"
[ -x "$PYTHON_BIN" ] || die "Python executable is unavailable: $PYTHON_BIN"
[ ! -e "$RUN_ROOT" ] || die "run path already exists: $RUN_ROOT"
[ ! -e "$RESULTS_ROOT" ] || die "Results path already exists: $RESULTS_ROOT"

gpu_count=0
for ignored_gpu in $GPU_IDS; do gpu_count=$((gpu_count + 1)); done
[ "$gpu_count" -gt 0 ] || die 'GPU_IDS is empty'
[ "$MAX_PARALLEL" -le "$gpu_count" ] || die 'MAX_PARALLEL exceeds the number of GPU_IDS'

alpha_for_dataset() {
  case "$1" in
    tictactoe) printf '%s\n' 10 ;;
    adult) printf '%s\n' 1000 ;;
    dota2) printf '%s\n' 10000 ;;
    mnist) printf '%s\n' 10 ;;
    imdb) printf '%s\n' 10 ;;
    *) die "unknown dataset: $1" ;;
  esac
}

methods_for_dataset() {
  case "$1" in
    tictactoe|adult|dota2)
      printf '%s\n' 'arima arima_adaptive moirai moirai_online'
      ;;
    mnist|imdb)
      printf '%s\n' 'arima_mnist_imdb_layerwise arima_adaptive_mnist_imdb_layerwise moirai_mnist_imdb_layerwise moirai_online_mnist_imdb_layerwise'
      ;;
    *) die "unknown dataset: $1" ;;
  esac
}

is_layerwise() {
  case "$1" in *_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

is_adaptive_arima() {
  case "$1" in arima_adaptive|arima_adaptive_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

is_online_moirai() {
  case "$1" in moirai_online|moirai_online_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

telemetry_pattern() {
  case "$1" in
    arima) printf '%s\n' '[ARIMA][OFFLINE][FORECAST]' ;;
    arima_adaptive) printf '%s\n' '[ARIMA][ADAPT][FORECAST]' ;;
    moirai) printf '%s\n' '[Moirai][OFFLINE] round=' ;;
    moirai_online) printf '%s\n' '[Moirai][OnlineTune] round=' ;;
    arima_mnist_imdb_layerwise) printf '%s\n' '[ARIMA][LAYERWISE][OFFLINE][FORECAST]' ;;
    arima_adaptive_mnist_imdb_layerwise) printf '%s\n' '[ARIMA][LAYERWISE][ADAPT][FORECAST]' ;;
    moirai_mnist_imdb_layerwise) printf '%s\n' '[Moirai][LAYERWISE][OFFLINE] round=' ;;
    moirai_online_mnist_imdb_layerwise) printf '%s\n' '[Moirai][LAYERWISE][OnlineTune] round=' ;;
    *) die "unknown method: $1" ;;
  esac
}

observe_pattern() {
  case "$1" in
    arima_adaptive) printf '%s\n' 'alignment=aligned' ;;
    moirai_online) printf '%s\n' 'alignment=aligned' ;;
    arima_adaptive_mnist_imdb_layerwise) printf '%s\n' 'alignment=exact' ;;
    moirai_online_mnist_imdb_layerwise) printf '%s\n' 'alignment=exact' ;;
    *) printf '%s\n' '' ;;
  esac
}

mkdir -p "$RUN_PARENT" "$PROJECT_ROOT/Results"
mkdir "$RUN_ROOT" "$TASKS_ROOT" "$RESULTS_ROOT"

campaign_manifest=$RUN_ROOT/campaign_manifest.txt
{
  printf 'campaign_id=%s\ncreated_utc=%s\n' "$CAMPAIGN_ID" "$timestamp"
  printf 'candidate_root=%s\nresults_root=%s\n' "$CANDIDATE_ROOT" "$RESULTS_ROOT"
  printf 'datasets=tictactoe adult dota2 mnist imdb\n'
  printf 'tasks=20\nnum_epoch=%s\nrandom_round=%s\n' "$NUM_EPOCH" "$RANDOM_ROUND"
  printf 'client_num=%s\nattacker_id=%s\n' "$CLIENT_NUM" "$ATTACKER_ID"
  printf 'lazy_schedule=fixed\nlazy_honest_gap=2\nlazy_honest_phase=0\nlazy_fake_policy=attack\n'
  printf 'use_hessian=1\nlambda_h=%s\n' "$LAMBDA_H"
  sha256sum \
    "$ENTRY" \
    "$CANDIDATE_ROOT/module/method/attack_updated_nonlayerwise.py" \
    "$CANDIDATE_ROOT/module/method/attack_updated_layerwise.py"
} > "$campaign_manifest"

run_task() {
  task_number=$1
  dataset=$2
  attack_method=$3
  alpha=$(alpha_for_dataset "$dataset")
  task_label=$(printf 'task_%02d_%s_%s' "$task_number" "$dataset" "$attack_method")
  task_root=$TASKS_ROOT/$task_label
  task_results=$RESULTS_ROOT/$task_label
  mkdir "$task_root" "$task_results"
  mkdir "$task_root/tmp" "$task_root/cache"

  gpu_position=$(((task_number - 1) % gpu_count + 1))
  gpu_id=$(printf '%s\n' $GPU_IDS | sed -n "${gpu_position}p")

  set -- "$PYTHON_BIN" "$ENTRY" \
    --dataset "$dataset" \
    --alpha "$alpha" \
    --client_num "$CLIENT_NUM" \
    --attacker_id "$ATTACKER_ID" \
    --attack_method "$attack_method" \
    --contribution_method shapley \
    --trial_id 0 \
    --num_epoch "$NUM_EPOCH" \
    --num_local_epochs 1 \
    --random_round "$RANDOM_ROUND" \
    --results_root "$task_results" \
    --use_attack \
    --use_lazy \
    --lazy_schedule fixed \
    --lazy_honest_gap 2 \
    --lazy_honest_phase 0 \
    --lazy_fake_policy attack \
    --use_hessian \
    --lambda_h "$LAMBDA_H"

  if is_layerwise "$attack_method"; then
    set -- "$@" \
      --layerwise_selection_metric normalized_rms \
      --layerwise_selection_fraction 0.5 \
      --layerwise_pca_threshold 512 \
      --layerwise_pca_components 8
  fi
  if is_adaptive_arima "$attack_method"; then
    set -- "$@" \
      --arima_retrain_interval "$ARIMA_RETRAIN_INTERVAL" \
      --arima_err_ema_beta 0.9
  fi
  if is_online_moirai "$attack_method"; then
    set -- "$@" \
      --moirai_tune_lr "$MOIRAI_TUNE_LR" \
      --moirai_tune_steps "$MOIRAI_TUNE_STEPS" \
      --moirai_adapt_interval "$MOIRAI_ADAPT_INTERVAL" \
      --moirai_residual_decay "$MOIRAI_RESIDUAL_DECAY"
  fi

  {
    printf 'task_number=%s\ntask_label=%s\n' "$task_number" "$task_label"
    printf 'dataset=%s\nalpha=%s\nattack_method=%s\ngpu_id=%s\n' "$dataset" "$alpha" "$attack_method" "$gpu_id"
    printf 'results_root=%s\ncommand_argv:\n' "$task_results"
    printf '  %s\n' "$@"
  } > "$task_root/manifest.txt"

  log_file=$task_root/run.log
  started=$(date +%s)
  if (
    cd "$CANDIDATE_ROOT"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TMPDIR=$task_root/tmp \
    HF_HOME=$task_root/cache/huggingface \
    TORCH_HOME=$task_root/cache/torch \
    XDG_CACHE_HOME=$task_root/cache/xdg \
    "$@"
  ) > "$log_file" 2>&1; then
    ended=$(date +%s)
    printf 'process_success runtime_seconds=%s\n' "$((ended - started))" > "$task_root/status.txt"
    return 0
  fi

  exit_code=$?
  ended=$(date +%s)
  printf 'process_failed exit_code=%s runtime_seconds=%s\n' "$exit_code" "$((ended - started))" > "$task_root/status.txt"
  return "$exit_code"
}

task_number=0
batch_count=0
batch_pids=''
process_failures=0

wait_for_batch() {
  for batch_pid in $batch_pids; do
    if ! wait "$batch_pid"; then process_failures=$((process_failures + 1)); fi
  done
  batch_count=0
  batch_pids=''
}

for dataset in tictactoe adult dota2 mnist imdb; do
  for attack_method in $(methods_for_dataset "$dataset"); do
    task_number=$((task_number + 1))
    run_task "$task_number" "$dataset" "$attack_method" &
    batch_pids="$batch_pids $!"
    batch_count=$((batch_count + 1))
    if [ "$batch_count" -ge "$MAX_PARALLEL" ]; then wait_for_batch; fi
  done
done
[ "$batch_count" -eq 0 ] || wait_for_batch
[ "$task_number" -eq 20 ] || die "task enumeration drifted: $task_number"

telemetry_failures=0
task_number=0
summary_file=$RUN_ROOT/telemetry_summary.tsv
printf 'task\tdataset\tmethod\tprocess\trounds\thonest\tfake\thessian\tprediction\tselection\tobserve\tresults_files\n' > "$summary_file"

for dataset in tictactoe adult dota2 mnist imdb; do
  for attack_method in $(methods_for_dataset "$dataset"); do
    task_number=$((task_number + 1))
    task_label=$(printf 'task_%02d_%s_%s' "$task_number" "$dataset" "$attack_method")
    task_root=$TASKS_ROOT/$task_label
    task_results=$RESULTS_ROOT/$task_label
    log_file=$task_root/run.log
    process=fail
    grep -q '^process_success ' "$task_root/status.txt" && process=pass
    rounds=$(grep -c '^\[ROUND ' "$log_file" || true)
    honest=$(grep -c '^\[ROUND .* honest=True ' "$log_file" || true)
    fake=$(grep -c '^\[ROUND .* honest=False ' "$log_file" || true)
    hessian=fail
    [ "$(grep -c '^\[ROUND .* hessian=on ' "$log_file" || true)" -eq "$NUM_EPOCH" ] && hessian=pass
    prediction=fail
    pattern=$(telemetry_pattern "$attack_method")
    grep -Fq "$pattern" "$log_file" && prediction=pass
    selection=na
    if is_layerwise "$attack_method"; then
      selection=fail
      grep -Fq '[LAYERWISE][SELECTION]' "$log_file" && selection=pass
    fi
    observe=na
    if is_adaptive_arima "$attack_method" || is_online_moirai "$attack_method"; then
      observe=fail
      expected_alignment=$(observe_pattern "$attack_method")
      if grep -F '[OBSERVE]' "$log_file" | grep -Fq "$expected_alignment"; then
        observe=pass
      fi
    fi
    results_files=$(find "$task_results" -type f | wc -l | tr -d ' ')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$task_label" "$dataset" "$attack_method" "$process" "$rounds" "$honest" "$fake" \
      "$hessian" "$prediction" "$selection" "$observe" "$results_files" >> "$summary_file"

    if [ "$process" != pass ] || [ "$rounds" -ne "$NUM_EPOCH" ] || \
       [ "$honest" -ne $(((NUM_EPOCH + 1) / 2)) ] || [ "$fake" -ne $((NUM_EPOCH / 2)) ] || \
       [ "$hessian" != pass ] || [ "$prediction" != pass ] || \
       [ "$selection" = fail ] || [ "$observe" = fail ] || [ "$results_files" -lt 4 ]; then
      telemetry_failures=$((telemetry_failures + 1))
    fi
  done
done

printf 'completed_tasks=%s\nprocess_failures=%s\ntelemetry_failures=%s\n' \
  "$task_number" "$process_failures" "$telemetry_failures" >> "$campaign_manifest"
cat "$summary_file"
printf 'campaign_root=%s\nresults_root=%s\nprocess_failures=%s telemetry_failures=%s\n' \
  "$RUN_ROOT" "$RESULTS_ROOT" "$process_failures" "$telemetry_failures"
[ "$process_failures" -eq 0 ] && [ "$telemetry_failures" -eq 0 ]

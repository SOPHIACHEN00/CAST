#!/usr/bin/env sh
# Full corrected Moirai/ARIMA lazy-schedule matrix.
#
# Safety contract:
# - DRY_RUN=1 only validates/counts/renders a small command sample.
# - A real sweep requires CONFIRM_FULL=1 and a candidate produced by
#   deploy_and_smoke_moirai_arima.sh.
# - Every configuration gets a new run root, Results root, log, and manifest.
# - No existing run or Results path is ever reused.

set -eu

PROJECT_ROOT=${PROJECT_ROOT:-/output/gradpred_exp/attack}
script_parent=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
CANDIDATE_ROOT=${CANDIDATE_ROOT:-$script_parent}
PYTHON_BIN=${PYTHON_BIN:-python}
DRY_RUN=${DRY_RUN:-0}
CONFIRM_FULL=${CONFIRM_FULL:-0}
DRY_RUN_PRINT_LIMIT=${DRY_RUN_PRINT_LIMIT:-8}
TRIAL_IDS=${TRIAL_IDS:-0}
GPU_IDS=${GPU_IDS:-0 1 2 3}
MAX_PARALLEL=${MAX_PARALLEL:-4}
ATTACKER_ID=${ATTACKER_ID:-1}
CONTRIBUTION_METHOD=${CONTRIBUTION_METHOD:-shapley}
NUM_EPOCH=${NUM_EPOCH:-50}
RANDOM_ROUND=${RANDOM_ROUND:-10}
USE_HESSIAN=${USE_HESSIAN:-1}
LAMBDA_H=${LAMBDA_H:-0.9}
LAZY_HONEST_PHASE=${LAZY_HONEST_PHASE:-0}
LAZY_FAKE_POLICY=${LAZY_FAKE_POLICY:-attack}
LAYERWISE_SELECTION_METRIC=${LAYERWISE_SELECTION_METRIC:-normalized_rms}
LAYERWISE_SELECTION_FRACTION=${LAYERWISE_SELECTION_FRACTION:-0.5}
LAYERWISE_SELECTION_TOP_N=${LAYERWISE_SELECTION_TOP_N:-}
LAYERWISE_PCA_THRESHOLD=${LAYERWISE_PCA_THRESHOLD:-512}
LAYERWISE_PCA_COMPONENTS=${LAYERWISE_PCA_COMPONENTS:-8}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
CAMPAIGN_ID=${CAMPAIGN_ID:-forecast_full_${timestamp}_pid$$}
CAMPAIGN_PARENT=${CAMPAIGN_PARENT:-${PROJECT_ROOT}/.matrix_runs}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-${CAMPAIGN_PARENT}/${CAMPAIGN_ID}}
EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT:-${CAMPAIGN_ROOT}/experiments}
RESULTS_CAMPAIGN_ROOT=${RESULTS_CAMPAIGN_ROOT:-${PROJECT_ROOT}/Results/${CAMPAIGN_ID}}

DATASETS='tictactoe adult dota2 mnist imdb'
CLIENT_NUMS='2 4 6 8'
GAP_PROB_PAIRS='2:0.50 3:0.66 6:0.82 10:0.90'

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

case "$CAMPAIGN_ID" in
  ''|*[!A-Za-z0-9._-]*) die "CAMPAIGN_ID must be one safe path segment: $CAMPAIGN_ID" ;;
esac
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
case "$CONFIRM_FULL" in 0|1) ;; *) die "CONFIRM_FULL must be 0 or 1" ;; esac
case "$USE_HESSIAN" in 0|1) ;; *) die "USE_HESSIAN must be 0 or 1" ;; esac
case "$MAX_PARALLEL" in ''|*[!0-9]*) die "MAX_PARALLEL must be a positive integer" ;; esac
[ "$MAX_PARALLEL" -gt 0 ] || die "MAX_PARALLEL must be at least 1"

# These values define the declared 50-round honest/fake protocol. Do not allow
# an accidentally different campaign to be labelled as this full matrix.
[ "$NUM_EPOCH" -eq 50 ] || die "full matrix requires NUM_EPOCH=50"
[ "$USE_HESSIAN" -eq 1 ] || die "full matrix requires USE_HESSIAN=1"
[ "$LAZY_HONEST_PHASE" -eq 0 ] || die "full matrix requires LAZY_HONEST_PHASE=0"
[ "$LAZY_FAKE_POLICY" = attack ] || die "full matrix requires LAZY_FAKE_POLICY=attack"

[ -d "$PROJECT_ROOT" ] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[ -d "$CANDIDATE_ROOT" ] || die "CANDIDATE_ROOT does not exist: $CANDIDATE_ROOT"
PROJECT_ROOT=$(CDPATH= cd -- "$PROJECT_ROOT" && pwd -P)
CANDIDATE_ROOT=$(CDPATH= cd -- "$CANDIDATE_ROOT" && pwd -P)
case "$CANDIDATE_ROOT" in
  "$PROJECT_ROOT"/.candidate_runs/*/attack) ;;
  *) die "candidate root must be an isolated deploy below $PROJECT_ROOT/.candidate_runs" ;;
esac
[ "$CAMPAIGN_PARENT" = "$PROJECT_ROOT/.matrix_runs" ] ||
  die "CAMPAIGN_PARENT must be $PROJECT_ROOT/.matrix_runs"
[ "$CAMPAIGN_ROOT" = "$CAMPAIGN_PARENT/$CAMPAIGN_ID" ] ||
  die "CAMPAIGN_ROOT must be $CAMPAIGN_PARENT/$CAMPAIGN_ID"
[ "$EXPERIMENTS_ROOT" = "$CAMPAIGN_ROOT/experiments" ] ||
  die "EXPERIMENTS_ROOT must be $CAMPAIGN_ROOT/experiments"
[ "$RESULTS_CAMPAIGN_ROOT" = "$PROJECT_ROOT/Results/$CAMPAIGN_ID" ] ||
  die "RESULTS_CAMPAIGN_ROOT must be $PROJECT_ROOT/Results/$CAMPAIGN_ID"
ENTRY=$CANDIDATE_ROOT/FLCE_updated.py
SCRIPT_PATH=$CANDIDATE_ROOT/script/run_full_moirai_arima_matrix.sh
[ -f "$ENTRY" ] || die "candidate runner is missing: $ENTRY"
[ -f "$SCRIPT_PATH" ] || die "candidate full-matrix launcher is missing: $SCRIPT_PATH"
[ -f "$CANDIDATE_ROOT/module/method/attack_updated_nonlayerwise.py" ] ||
  die 'candidate non-layerwise attack module is missing'
[ -f "$CANDIDATE_ROOT/module/method/attack_updated_layerwise.py" ] ||
  die 'candidate layerwise attack module is missing'

alpha_values() {
  case "$1" in
    tictactoe) printf '%s\n' '10 100 1000 20000' ;;
    adult) printf '%s\n' '1000 100000 1000000 20000000' ;;
    dota2) printf '%s\n' '10000 1000000 50000000 200000000' ;;
    mnist) printf '%s\n' '10 1000 100000 10000000' ;;
    imdb) printf '%s\n' '10 100 1000 200000' ;;
    *) die "unknown dataset: $1" ;;
  esac
}

attack_methods() {
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

is_layerwise_method() {
  case "$1" in *_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

is_adaptive_arima_method() {
  case "$1" in arima_adaptive|arima_adaptive_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

is_online_moirai_method() {
  case "$1" in moirai_online|moirai_online_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

word_count() {
  count=0
  for ignored_word in $1; do count=$((count + 1)); done
  printf '%s\n' "$count"
}

trial_count=$(word_count "$TRIAL_IDS")
[ "$trial_count" -gt 0 ] || die 'TRIAL_IDS is empty'
gpu_count=0
active_gpu_ids=''
seen_gpu_ids=' '
for configured_gpu in $GPU_IDS; do
  case "$configured_gpu" in
    ''|*[!0-9]*) die "GPU_IDS contains a non-negative-integer violation: $configured_gpu" ;;
  esac
  case "$configured_gpu" in
    0|[1-9]*) ;;
    *) die "GPU_IDS must use canonical decimal ids without leading zeros: $configured_gpu" ;;
  esac
  case "$seen_gpu_ids" in
    *" $configured_gpu "*) die "GPU_IDS contains duplicate GPU id: $configured_gpu" ;;
  esac
  seen_gpu_ids="$seen_gpu_ids$configured_gpu "
  gpu_count=$((gpu_count + 1))
  if [ "$gpu_count" -le "$MAX_PARALLEL" ]; then
    active_gpu_ids="$active_gpu_ids $configured_gpu"
  fi
done
active_gpu_ids=${active_gpu_ids# }
[ "$gpu_count" -gt 0 ] || die 'GPU_IDS is empty'
[ "$MAX_PARALLEL" -le "$gpu_count" ] ||
  die "MAX_PARALLEL=$MAX_PARALLEL exceeds configured GPU count=$gpu_count"

# Count independently from the closed-form expectation so accidental matrix
# drift fails before any real directory or process is created.
matrix_per_rep=0
for dataset in $DATASETS; do
  for alpha in $(alpha_values "$dataset"); do
    for client_num in $CLIENT_NUMS; do
      for attack_method in $(attack_methods "$dataset"); do
        for pair in $GAP_PROB_PAIRS; do
          matrix_per_rep=$((matrix_per_rep + 2))
        done
      done
    done
  done
done
[ "$matrix_per_rep" -eq 2560 ] ||
  die "matrix drift: expected 2560 configs/rep, counted $matrix_per_rep"
total_tasks=$((matrix_per_rep * trial_count))

printf '%s\n' '=== FULL MOIRAI/ARIMA MATRIX PLAN ==='
printf 'candidate_root       : %s\n' "$CANDIDATE_ROOT"
printf 'campaign_root        : %s\n' "$CAMPAIGN_ROOT"
printf 'results_campaign_root: %s\n' "$RESULTS_CAMPAIGN_ROOT"
printf 'trial_ids            : %s\n' "$TRIAL_IDS"
printf 'configs_per_rep       : %s (1280 fixed + 1280 probabilistic)\n' "$matrix_per_rep"
printf 'total_tasks           : %s\n' "$total_tasks"
printf 'fixed/prob pairs      : %s\n' "$GAP_PROB_PAIRS"
printf 'client counts         : %s\n' "$CLIENT_NUMS"
printf 'use_hessian/lambda_h  : %s / %s\n' "$USE_HESSIAN" "$LAMBDA_H"
printf 'gpu_ids/max_parallel  : %s / %s\n' "$GPU_IDS" "$MAX_PARALLEL"
printf 'active_gpu_slots      : %s (one serial worker per GPU)\n' "$active_gpu_ids"

print_command() {
  printf 'command:'
  for command_arg do printf ' <%s>' "$command_arg"; done
  printf '\n'
}

# Build and either print or execute one task without serializing argv through
# a shell string. Positional parameters remain the exact command argv.
handle_task() {
  mode=$1
  task_number=$2
  assigned_gpu_id=$3
  shift 3
  trial_id=$1
  scenario=$2
  gap=$3
  probability=$4
  dataset=$5
  alpha=$6
  client_num=$7
  attack_method=$8

  task_label=$(printf 'task_%06d_%s_%s_a%s_n%s_%s_g%s_t%s' \
    "$task_number" "$dataset" "$attack_method" "$alpha" "$client_num" \
    "$scenario" "$gap" "$trial_id")
  task_root=$EXPERIMENTS_ROOT/$task_label
  task_results=$RESULTS_CAMPAIGN_ROOT/$task_label

  set -- "$PYTHON_BIN" "$ENTRY" \
    --dataset "$dataset" \
    --alpha "$alpha" \
    --client_num "$client_num" \
    --attacker_id "$ATTACKER_ID" \
    --attack_method "$attack_method" \
    --contribution_method "$CONTRIBUTION_METHOD" \
    --trial_id "$trial_id" \
    --num_epoch "$NUM_EPOCH" \
    --random_round "$RANDOM_ROUND" \
    --results_root "$task_results" \
    --use_attack \
    --use_lazy \
    --lazy_schedule "$scenario" \
    --lazy_honest_gap "$gap" \
    --lazy_honest_phase "$LAZY_HONEST_PHASE" \
    --lazy_fake_policy "$LAZY_FAKE_POLICY"

  if [ "$scenario" = probabilistic ]; then
    set -- "$@" --attack_probability "$probability"
  fi
  if [ "$USE_HESSIAN" = 1 ]; then
    set -- "$@" --use_hessian --lambda_h "$LAMBDA_H"
  fi
  if is_layerwise_method "$attack_method"; then
    set -- "$@" \
      --layerwise_selection_metric "$LAYERWISE_SELECTION_METRIC" \
      --layerwise_selection_fraction "$LAYERWISE_SELECTION_FRACTION" \
      --layerwise_pca_threshold "$LAYERWISE_PCA_THRESHOLD" \
      --layerwise_pca_components "$LAYERWISE_PCA_COMPONENTS"
    if [ -n "$LAYERWISE_SELECTION_TOP_N" ]; then
      set -- "$@" --layerwise_selection_top_n "$LAYERWISE_SELECTION_TOP_N"
    fi
  fi
  if is_adaptive_arima_method "$attack_method"; then
    set -- "$@" \
      --arima_retrain_interval "${ARIMA_RETRAIN_INTERVAL:-10}" \
      --arima_err_ema_beta "${ARIMA_ERR_EMA_BETA:-0.9}"
  fi
  if is_online_moirai_method "$attack_method"; then
    set -- "$@" \
      --moirai_tune_lr "${MOIRAI_TUNE_LR:-0.001}" \
      --moirai_tune_steps "${MOIRAI_TUNE_STEPS:-1}" \
      --moirai_adapt_interval "${MOIRAI_ADAPT_INTERVAL:-10}" \
      --moirai_residual_decay "${MOIRAI_RESIDUAL_DECAY:-0.9}"
  fi

  if [ "$mode" = print ]; then
    print_command "$@"
    return 0
  fi

  [ ! -e "$task_root" ] || die "run-root collision: $task_root"
  [ ! -e "$task_results" ] || die "Results collision: $task_results"
  mkdir "$task_root"
  mkdir "$task_root/logs" "$task_root/tmp" "$task_root/cache"
  mkdir "$task_results"

  case "$assigned_gpu_id" in
    ''|*[!0-9]*) die "task $task_number has invalid assigned GPU id: $assigned_gpu_id" ;;
  esac
  manifest=$task_root/manifest.txt
  {
    printf 'task_number=%s\ntask_label=%s\n' "$task_number" "$task_label"
    printf 'dataset=%s\nalpha=%s\nclient_num=%s\n' "$dataset" "$alpha" "$client_num"
    printf 'attacker_id=%s\nattack_method=%s\ntrial_id=%s\n' "$ATTACKER_ID" "$attack_method" "$trial_id"
    printf 'scenario=%s\ngap=%s\npaired_probability=%s\n' "$scenario" "$gap" "$probability"
    printf 'gpu_id=%s\nrun_root=%s\nresults_root=%s\n' "$assigned_gpu_id" "$task_root" "$task_results"
    printf 'use_hessian=%s\nlambda_h=%s\n' "$USE_HESSIAN" "$LAMBDA_H"
    printf '%s\n' 'command_argv:'
    printf '  %s\n' "$@"
  } > "$manifest"

  log_file=$task_root/logs/run.log
  if (
    cd "$CANDIDATE_ROOT"
    CUDA_VISIBLE_DEVICES=$assigned_gpu_id \
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
    printf '%s\n' success > "$task_root/status.txt"
    return 0
  else
    exit_code=$?
    printf 'failed exit_code=%s\n' "$exit_code" > "$task_root/status.txt"
    return "$exit_code"
  fi
}

# Dynamic shared queue with one serial worker per active GPU.  Each worker
# claims another task only after its current task completes, eliminating the
# old global batch barrier while preserving the one-process-per-GPU invariant.
enqueue_task() {
  queued_task_number=$1
  shift
  queued_spec=$(printf '%s/task_%06d.spec' "$SCHEDULER_PENDING" "$queued_task_number")
  [ ! -e "$queued_spec" ] || die "scheduler spec collision: $queued_spec"
  {
    printf '%s\n' "$queued_task_number"
    printf '%s\n' "$@"
  } > "$queued_spec"
}

claim_next_task() {
  claim_gpu_id=$1
  for pending_spec in "$SCHEDULER_PENDING"/*.spec; do
    [ -f "$pending_spec" ] || return 1
    pending_name=${pending_spec##*/}
    claimed_spec=$SCHEDULER_RUNNING/${pending_name%.spec}.gpu_${claim_gpu_id}.spec
    if mv "$pending_spec" "$claimed_spec" 2>/dev/null; then
      printf '%s\n' "$claimed_spec"
      return 0
    fi
  done
  return 1
}

run_gpu_worker() {
  worker_gpu_id=$1
  worker_log=$SCHEDULER_WORKERS/gpu_${worker_gpu_id}.log
  worker_status=$SCHEDULER_WORKERS/gpu_${worker_gpu_id}.status
  worker_attempted=0
  worker_failed=0
  printf 'gpu=%s state=started\n' "$worker_gpu_id" > "$worker_log"

  while claimed_spec=$(claim_next_task "$worker_gpu_id"); do
    if ! {
      IFS= read -r claimed_task_number
      IFS= read -r claimed_trial_id
      IFS= read -r claimed_scenario
      IFS= read -r claimed_gap
      IFS= read -r claimed_probability
      IFS= read -r claimed_dataset
      IFS= read -r claimed_alpha
      IFS= read -r claimed_client_num
      IFS= read -r claimed_attack_method
    } < "$claimed_spec"; then
      printf 'gpu=%s state=invalid_spec path=%s\n' "$worker_gpu_id" "$claimed_spec" >> "$worker_log"
      return 1
    fi

    worker_attempted=$((worker_attempted + 1))
    printf 'gpu=%s task=%s state=started spec=%s\n' \
      "$worker_gpu_id" "$claimed_task_number" "$claimed_spec" >> "$worker_log"
    task_exit_code=0
    if handle_task run "$claimed_task_number" "$worker_gpu_id" \
      "$claimed_trial_id" "$claimed_scenario" "$claimed_gap" \
      "$claimed_probability" "$claimed_dataset" "$claimed_alpha" \
      "$claimed_client_num" "$claimed_attack_method"; then
      task_state=success
    else
      task_exit_code=$?
      task_state=failed
      worker_failed=$((worker_failed + 1))
      failure_marker=$(printf '%s/task_%06d.failed' "$SCHEDULER_FAILURES" "$claimed_task_number")
      printf 'task=%s gpu=%s exit_code=%s spec=%s\n' \
        "$claimed_task_number" "$worker_gpu_id" "$task_exit_code" "$claimed_spec" \
        > "$failure_marker"
    fi
    completed_name=${claimed_spec##*/}
    mv "$claimed_spec" "$SCHEDULER_COMPLETED/$completed_name"
    printf 'gpu=%s task=%s state=%s exit_code=%s\n' \
      "$worker_gpu_id" "$claimed_task_number" "$task_state" "$task_exit_code" >> "$worker_log"
  done

  printf 'gpu=%s attempted=%s failed=%s state=finished\n' \
    "$worker_gpu_id" "$worker_attempted" "$worker_failed" > "$worker_status"
  printf 'gpu=%s state=finished attempted=%s failed=%s\n' \
    "$worker_gpu_id" "$worker_attempted" "$worker_failed" >> "$worker_log"
}

count_scheduler_files() {
  count_directory=$1
  count_suffix=$2
  counted_files=0
  for counted_file in "$count_directory"/*"$count_suffix"; do
    [ -f "$counted_file" ] || continue
    counted_files=$((counted_files + 1))
  done
  printf '%s\n' "$counted_files"
}

if [ "$DRY_RUN" = 1 ]; then
  printed=0
  task_number=0
  for trial_id in $TRIAL_IDS; do
    for dataset in $DATASETS; do
      for alpha in $(alpha_values "$dataset"); do
        for client_num in $CLIENT_NUMS; do
          for attack_method in $(attack_methods "$dataset"); do
            for pair in $GAP_PROB_PAIRS; do
              gap=${pair%%:*}
              probability=${pair#*:}
              for scenario in fixed probabilistic; do
                task_number=$((task_number + 1))
                if [ "$printed" -lt "$DRY_RUN_PRINT_LIMIT" ]; then
                  handle_task print "$task_number" 0 "$trial_id" "$scenario" "$gap" "$probability" \
                    "$dataset" "$alpha" "$client_num" "$attack_method"
                  printed=$((printed + 1))
                fi
              done
            done
          done
        done
      done
    done
  done
  [ "$task_number" -eq "$total_tasks" ] || die 'DRY_RUN task enumeration drifted'
  printf 'DRY_RUN rendered %s representative commands; enumerated %s total tasks.\n' \
    "$printed" "$task_number"
  printf '%s\n' 'No directory was created and no experiment was started.'
  exit 0
fi

[ "$CONFIRM_FULL" = 1 ] || {
  printf '%s\n' 'Full sweep is locked. Re-run with CONFIRM_FULL=1 after smoke passes.' >&2
  exit 2
}
[ -f "$CANDIDATE_ROOT/.candidate_deployment_manifest.txt" ] ||
  die 'real sweep requires a candidate created by deploy_and_smoke_moirai_arima.sh'
[ -f "$CANDIDATE_ROOT/.smoke_passed_manifest.txt" ] ||
  die 'real sweep requires a successful end-to-end smoke manifest'
[ -f "$CANDIDATE_ROOT/.smoke_passed.sha256" ] ||
  die 'real sweep requires smoke-passed candidate checksums'
(
  cd "$CANDIDATE_ROOT"
  sha256sum -c .smoke_passed.sha256
) || die 'candidate files changed after the successful smoke'
[ "$CANDIDATE_ROOT" != "$PROJECT_ROOT" ] ||
  die 'refusing to run the full sweep from the immutable base project root'
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python executable not found: $PYTHON_BIN"
[ ! -e "$CAMPAIGN_ROOT" ] || die "campaign path already exists: $CAMPAIGN_ROOT"
[ ! -e "$RESULTS_CAMPAIGN_ROOT" ] || die "Results path already exists: $RESULTS_CAMPAIGN_ROOT"

mkdir -p "$CAMPAIGN_PARENT" "$PROJECT_ROOT/Results"
mkdir "$CAMPAIGN_ROOT"
mkdir "$EXPERIMENTS_ROOT"
mkdir "$RESULTS_CAMPAIGN_ROOT"
SCHEDULER_ROOT=$CAMPAIGN_ROOT/scheduler
SCHEDULER_PENDING=$SCHEDULER_ROOT/pending
SCHEDULER_RUNNING=$SCHEDULER_ROOT/running
SCHEDULER_COMPLETED=$SCHEDULER_ROOT/completed
SCHEDULER_FAILURES=$SCHEDULER_ROOT/failures
SCHEDULER_WORKERS=$SCHEDULER_ROOT/workers
mkdir "$SCHEDULER_ROOT" "$SCHEDULER_PENDING" "$SCHEDULER_RUNNING" \
  "$SCHEDULER_COMPLETED" "$SCHEDULER_FAILURES" "$SCHEDULER_WORKERS"

campaign_manifest=$CAMPAIGN_ROOT/campaign_manifest.txt
{
  printf 'campaign_id=%s\ncreated_utc=%s\n' "$CAMPAIGN_ID" "$timestamp"
  printf 'candidate_root=%s\nresults_root=%s\n' "$CANDIDATE_ROOT" "$RESULTS_CAMPAIGN_ROOT"
  printf 'trial_ids=%s\nconfigs_per_rep=%s\ntotal_tasks=%s\n' "$TRIAL_IDS" "$matrix_per_rep" "$total_tasks"
  printf 'dataset_alphas=tictactoe:[10 100 1000 20000]; adult:[1000 100000 1000000 20000000]; dota2:[10000 1000000 50000000 200000000]; mnist:[10 1000 100000 10000000]; imdb:[10 100 1000 200000]\n'
  printf 'client_nums=%s\ngap_probability_pairs=%s\n' "$CLIENT_NUMS" "$GAP_PROB_PAIRS"
  printf 'use_hessian=%s\nlambda_h=%s\n' "$USE_HESSIAN" "$LAMBDA_H"
  printf 'scheduler=dynamic_shared_queue\nactive_gpu_ids=%s\nmax_parallel=%s\n' \
    "$active_gpu_ids" "$MAX_PARALLEL"
  sha256sum \
    "$ENTRY" \
    "$CANDIDATE_ROOT/module/method/attack_updated_nonlayerwise.py" \
    "$CANDIDATE_ROOT/module/method/attack_updated_layerwise.py" \
    "$SCRIPT_PATH"
} > "$campaign_manifest"

task_number=0
for trial_id in $TRIAL_IDS; do
  for dataset in $DATASETS; do
    for alpha in $(alpha_values "$dataset"); do
      for client_num in $CLIENT_NUMS; do
        [ "$ATTACKER_ID" -lt "$client_num" ] ||
          die "ATTACKER_ID=$ATTACKER_ID is invalid for client_num=$client_num"
        for attack_method in $(attack_methods "$dataset"); do
          for pair in $GAP_PROB_PAIRS; do
            gap=${pair%%:*}
            probability=${pair#*:}
            for scenario in fixed probabilistic; do
              task_number=$((task_number + 1))
              enqueue_task "$task_number" "$trial_id" "$scenario" "$gap" "$probability" \
                "$dataset" "$alpha" "$client_num" "$attack_method"
            done
          done
        done
      done
    done
  done
done

[ "$task_number" -eq "$total_tasks" ] || die 'executed task enumeration drifted'
queued_tasks=$(count_scheduler_files "$SCHEDULER_PENDING" .spec)
[ "$queued_tasks" -eq "$total_tasks" ] || die 'scheduler queue enumeration drifted'

worker_pids=''
for active_gpu_id in $active_gpu_ids; do
  run_gpu_worker "$active_gpu_id" &
  worker_pids="$worker_pids $!"
done
scheduler_worker_failures=0
for worker_pid in $worker_pids; do
  if ! wait "$worker_pid"; then
    scheduler_worker_failures=$((scheduler_worker_failures + 1))
  fi
done

attempted_tasks=$(count_scheduler_files "$SCHEDULER_COMPLETED" .spec)
task_failures=$(count_scheduler_files "$SCHEDULER_FAILURES" .failed)
unfinished_tasks=$((total_tasks - attempted_tasks))
[ "$unfinished_tasks" -ge 0 ] || die 'scheduler completed more tasks than were enumerated'
successful_tasks=$((attempted_tasks - task_failures))
failures=$((task_failures + unfinished_tasks))
printf 'completed_tasks=%s\nsuccessful_tasks=%s\nfailed_tasks=%s\n' \
  "$attempted_tasks" "$successful_tasks" "$task_failures" >> "$campaign_manifest"
printf 'unfinished_tasks=%s\nscheduler_worker_failures=%s\n' \
  "$unfinished_tasks" "$scheduler_worker_failures" >> "$campaign_manifest"
printf 'campaign_root=%s\nresults_root=%s\ntasks=%s successful=%s failures=%s unfinished=%s worker_failures=%s\n' \
  "$CAMPAIGN_ROOT" "$RESULTS_CAMPAIGN_ROOT" "$total_tasks" "$successful_tasks" \
  "$failures" "$unfinished_tasks" "$scheduler_worker_failures"
[ "$failures" -eq 0 ] && [ "$scheduler_worker_failures" -eq 0 ]

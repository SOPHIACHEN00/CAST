#!/usr/bin/env sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
fixture_root=$(mktemp -d "${TMPDIR:-/tmp}/flce-gpu-scheduler.XXXXXX")
trap 'rm -rf "$fixture_root"' EXIT HUP INT TERM

project_root=$fixture_root/project
candidate_root=$project_root/.candidate_runs/mock/attack
mkdir -p "$candidate_root/module/method" "$candidate_root/script"

cp "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  "$candidate_root/script/run_moirai_arima_tuning_pilot.sh"
cp "$repo_root/script/run_full_moirai_arima_matrix.sh" \
  "$candidate_root/script/run_full_moirai_arima_matrix.sh"

# The launcher invokes PYTHON_BIN ENTRY argv... .  Using `sh` as PYTHON_BIN
# turns this isolated fixture into a deterministic task-duration/failure mock.
cat > "$candidate_root/FLCE_updated.py" <<'MOCK_RUNNER'
#!/usr/bin/env sh
set -eu

: "${MOCK_EVENT_LOG:?MOCK_EVENT_LOG is required}"
results_root=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --results_root)
      results_root=$2
      shift 2
      ;;
    *) shift ;;
  esac
done
[ -n "$results_root" ]
task_label=${results_root##*/}
gpu_id=${CUDA_VISIBLE_DEVICES:-unset}
printf 'start %s gpu=%s\n' "$task_label" "$gpu_id" >> "$MOCK_EVENT_LOG"

case "$task_label" in
  task_000001_*) sleep 5 ;;
  *) sleep 1 ;;
esac

round=0
while [ "$round" -lt 50 ]; do
  if [ $((round % 2)) -eq 0 ]; then
    honest=True
  else
    honest=False
  fi
  printf '[ROUND %s] honest=%s hessian=on mock=1\n' "$round" "$honest"
  round=$((round + 1))
done
printf 'end %s gpu=%s\n' "$task_label" "$gpu_id" >> "$MOCK_EVENT_LOG"

case "$task_label" in
  task_000002_*) exit 7 ;;
esac
MOCK_RUNNER

: > "$candidate_root/module/method/attack_updated_nonlayerwise.py"
: > "$candidate_root/module/method/attack_updated_layerwise.py"
: > "$candidate_root/.candidate_deployment_manifest.txt"
: > "$candidate_root/.smoke_passed_manifest.txt"
(
  cd "$candidate_root"
  sha256sum \
    FLCE_updated.py \
    module/method/attack_updated_nonlayerwise.py \
    module/method/attack_updated_layerwise.py \
    script/run_moirai_arima_tuning_pilot.sh \
    script/run_full_moirai_arima_matrix.sh \
    > .smoke_passed.sha256
)

pilot_output=$(
  PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  DRY_RUN_PRINT_LIMIT=0 \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh"
)
printf '%s\n' "$pilot_output" | grep -F 'enumerated 36 tasks.' >/dev/null

arima_output=$(
  PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  DRY_RUN_PRINT_LIMIT=0 \
  PILOT_FAMILIES=arima \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh"
)
printf '%s\n' "$arima_output" | grep -F 'enumerated 16 tasks.' >/dev/null

moirai_output=$(
  PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  DRY_RUN_PRINT_LIMIT=0 \
  PILOT_FAMILIES=moirai \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh"
)
printf '%s\n' "$moirai_output" | grep -F 'enumerated 20 tasks.' >/dev/null

arima_refine_output=$(
  PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  DRY_RUN_PRINT_LIMIT=4 \
  PILOT_STAGE=screen \
  PILOT_PROFILE_SET=arima_refine \
  PILOT_FAMILIES=arima \
  PILOT_DATASETS='mnist imdb' \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh"
)
printf '%s\n' "$arima_refine_output" | grep -F 'enumerated 4 tasks.' >/dev/null
[ "$(printf '%s\n' "$arima_refine_output" | grep -c 'variant=arima_adaptive')" -eq 4 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -c 'variant=arima_offline')" -eq 0 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -c 'profile=refine_i10')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -c 'profile=refine_i15')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--attack_method> <arima_adaptive_mnist_imdb_layerwise>')" -eq 4 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_var_window> <3>')" -eq 4 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_var_threshold> <0.05>')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_var_threshold> <0.10>')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_err_threshold> <0.05>')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_err_threshold> <0.015>')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_retrain_interval> <10>')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_retrain_interval> <15>')" -eq 2 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_min_rounds_after_start> <4>')" -eq 4 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_err_ema_beta> <0.9>')" -eq 4 ]
[ "$(printf '%s\n' "$arima_refine_output" | grep -F -c '<--arima_trim_history>')" -eq 4 ]

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  PILOT_PROFILE_SET=arima_refine \
  PILOT_FAMILIES=arima \
  PILOT_DATASETS=tictactoe \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'arima_refine unexpectedly accepted a non-layerwise dataset' >&2
  exit 1
fi

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  PILOT_STAGE=robustness \
  PILOT_PROFILE_SET=arima_refine \
  PILOT_FAMILIES=arima \
  PILOT_DATASETS=mnist \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'arima_refine unexpectedly accepted PILOT_STAGE=robustness' >&2
  exit 1
fi

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  PILOT_PROFILE_SET=arima_refine \
  PILOT_FAMILIES=moirai \
  PILOT_DATASETS=mnist \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'arima_refine unexpectedly accepted a non-ARIMA family' >&2
  exit 1
fi

confirm7_output=$(
  PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  DRY_RUN_PRINT_LIMIT=7 \
  PILOT_STAGE=screen \
  PILOT_PROFILE_SET=confirm7 \
  PILOT_FAMILIES='arima moirai' \
  PILOT_DATASETS='mnist imdb' \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh"
)
printf '%s\n' "$confirm7_output" | grep -F 'enumerated 7 tasks.' >/dev/null
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'dataset=mnist')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'dataset=imdb')" -eq 5 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'variant=arima_adaptive')" -eq 4 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'variant=moirai_online')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'variant=arima_offline')" -eq 0 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'variant=moirai_offline')" -eq 0 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--attack_method> <arima_adaptive_mnist_imdb_layerwise>')" -eq 4 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--attack_method> <moirai_online_mnist_imdb_layerwise>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'profile=confirm_i10')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'profile=confirm_i15')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'profile=confirm_lr01')" -eq 1 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'profile=confirm_lr02')" -eq 1 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -c 'profile=confirm_lr03')" -eq 1 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--arima_var_threshold> <0.05>')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--arima_var_threshold> <0.10>')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--arima_err_threshold> <0.05>')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--arima_err_threshold> <0.015>')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--arima_retrain_interval> <10>')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--arima_retrain_interval> <15>')" -eq 2 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_tune_lr> <0.01>')" -eq 1 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_tune_lr> <0.02>')" -eq 1 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_tune_lr> <0.03>')" -eq 1 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_tune_steps> <1>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_tune_start_round> <10>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_adapt_interval> <4>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_residual_decay> <0.95>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_residual_clip> <5.0>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_err_ema_beta> <0.9>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_err_threshold> <0.025>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_min_rounds_after_start> <4>')" -eq 3 ]
[ "$(printf '%s\n' "$confirm7_output" | grep -F -c '<--moirai_trim_history>')" -eq 3 ]

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  PILOT_STAGE=robustness \
  PILOT_PROFILE_SET=confirm7 \
  PILOT_FAMILIES='arima moirai' \
  PILOT_DATASETS='mnist imdb' \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'confirm7 unexpectedly accepted PILOT_STAGE=robustness' >&2
  exit 1
fi

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  PILOT_PROFILE_SET=confirm7 \
  PILOT_FAMILIES='arima moirai' \
  PILOT_DATASETS=imdb \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'confirm7 unexpectedly accepted an incomplete dataset set' >&2
  exit 1
fi

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  PILOT_PROFILE_SET=confirm7 \
  PILOT_FAMILIES=arima \
  PILOT_DATASETS='mnist imdb' \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'confirm7 unexpectedly accepted an incomplete family set' >&2
  exit 1
fi

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  PILOT_PROFILE_SET=confirm7 \
  PILOT_FAMILIES='arima moirai' \
  PILOT_DATASETS='mnist imdb' \
  TRIAL_IDS='0 1' \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'confirm7 unexpectedly accepted a multiplicative trial override' >&2
  exit 1
fi

robustness_output=$(
  PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  DRY_RUN_PRINT_LIMIT=0 \
  PILOT_STAGE=robustness \
  PILOT_PROFILE_SET=responsive \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh"
)
printf '%s\n' "$robustness_output" | grep -F 'enumerated 640 tasks.' >/dev/null

full_output=$(
  PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  DRY_RUN_PRINT_LIMIT=0 \
  sh "$repo_root/script/run_full_moirai_arima_matrix.sh"
)
printf '%s\n' "$full_output" | grep -F 'enumerated 2560 total tasks.' >/dev/null

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  GPU_IDS='0 0' \
  MAX_PARALLEL=2 \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'duplicate GPU_IDS unexpectedly passed the one-worker-per-GPU gate' >&2
  exit 1
fi

if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  DRY_RUN=1 \
  GPU_IDS='0 00' \
  MAX_PARALLEL=2 \
  sh "$repo_root/script/run_full_moirai_arima_matrix.sh" \
  >/dev/null 2>&1; then
  printf '%s\n' 'leading-zero GPU alias unexpectedly passed the one-worker-per-GPU gate' >&2
  exit 1
fi

# Exercise the full-matrix task handler directly so a failed child process
# cannot regress to exit_code=0 before the scheduler records its marker.
full_handler=$fixture_root/full_handle_task.sh
sed -n '/^handle_task() {/,/^}$/p' \
  "$repo_root/script/run_full_moirai_arima_matrix.sh" > "$full_handler"
full_unit_root=$fixture_root/full_handler_unit
EXPERIMENTS_ROOT=$full_unit_root/experiments
RESULTS_CAMPAIGN_ROOT=$full_unit_root/results
mkdir -p "$EXPERIMENTS_ROOT" "$RESULTS_CAMPAIGN_ROOT"
PYTHON_BIN=sh
ENTRY=$candidate_root/FLCE_updated.py
CANDIDATE_ROOT=$candidate_root
ATTACKER_ID=1
CONTRIBUTION_METHOD=shapley
NUM_EPOCH=50
RANDOM_ROUND=10
LAZY_HONEST_PHASE=0
LAZY_FAKE_POLICY=attack
USE_HESSIAN=1
LAMBDA_H=0.9
LAYERWISE_SELECTION_METRIC=normalized_rms
LAYERWISE_SELECTION_FRACTION=0.5
LAYERWISE_SELECTION_TOP_N=
LAYERWISE_PCA_THRESHOLD=512
LAYERWISE_PCA_COMPONENTS=8
MOCK_EVENT_LOG=$fixture_root/full_handler_events.log
export MOCK_EVENT_LOG
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
is_layerwise_method() { return 1; }
is_adaptive_arima_method() { return 1; }
is_online_moirai_method() { return 1; }
. "$full_handler"
full_handler_exit=0
handle_task run 2 0 0 fixed 2 0.50 tictactoe 10 4 arima || full_handler_exit=$?
[ "$full_handler_exit" -eq 7 ]
grep -Fx 'failed exit_code=7' \
  "$EXPERIMENTS_ROOT/task_000002_tictactoe_arima_a10_n4_fixed_g2_t0/status.txt" >/dev/null

event_log=$fixture_root/events.log
campaign_output=$fixture_root/campaign.out
if PROJECT_ROOT="$project_root" \
  CANDIDATE_ROOT="$candidate_root" \
  PYTHON_BIN=sh \
  DRY_RUN=0 \
  CONFIRM_TUNING=1 \
  CAMPAIGN_ID=scheduler_mock \
  PILOT_STAGE=screen \
  PILOT_PROFILE_SET=screen \
  PILOT_FAMILIES=moirai \
  PILOT_DATASETS=tictactoe \
  PILOT_CLIENT_NUMS=4 \
  PILOT_GAP_PROB_PAIRS=2:0.50 \
  PILOT_SCENARIOS=fixed \
  TRIAL_IDS=0 \
  GPU_IDS='0 1' \
  MAX_PARALLEL=2 \
  MOCK_EVENT_LOG="$event_log" \
  sh "$repo_root/script/run_moirai_arima_tuning_pilot.sh" \
  > "$campaign_output" 2>&1; then
  printf '%s\n' 'mock campaign unexpectedly succeeded despite one injected failure' >&2
  exit 1
fi

campaign_root=$project_root/.tuning_runs/scheduler_mock
campaign_manifest=$campaign_root/campaign_manifest.txt
[ -f "$campaign_manifest" ]
grep -Fx 'completed_tasks=4' "$campaign_manifest" >/dev/null
grep -Fx 'successful_tasks=3' "$campaign_manifest" >/dev/null
grep -Fx 'failed_or_protocol_invalid_tasks=1' "$campaign_manifest" >/dev/null
grep -Fx 'unfinished_tasks=0' "$campaign_manifest" >/dev/null
grep -Fx 'scheduler_worker_failures=0' "$campaign_manifest" >/dev/null
grep -F 'failures=1 unfinished=0 worker_failures=0' "$campaign_output" >/dev/null

completed_count=0
for completed_spec in "$campaign_root/scheduler/completed"/*.spec; do
  [ -f "$completed_spec" ] || continue
  completed_count=$((completed_count + 1))
done
[ "$completed_count" -eq 4 ]

failure_count=0
for failure_marker in "$campaign_root/scheduler/failures"/*.failed; do
  [ -f "$failure_marker" ] || continue
  failure_count=$((failure_count + 1))
done
[ "$failure_count" -eq 1 ]

# A per-GPU worker must finish one task before starting another on that GPU.
awk '
  $1 == "start" {
    split($3, field, "=")
    gpu = field[2]
    if (active[gpu]) exit 1
    active[gpu] = 1
  }
  $1 == "end" {
    split($3, field, "=")
    active[field[2]] = 0
  }
' "$event_log"

# Task 1 sleeps five seconds.  Tasks 2-4 each sleep one second.  In a global
# two-task barrier, task 3 cannot even start until task 1 ends.  A dynamic slot
# queue lets the other GPU start task 3 before the slow task finishes.
awk '
  /^start task_000003_/ { fast_start = NR }
  /^end task_000001_/ { slow_end = NR }
  END { exit !(fast_start > 0 && slow_end > 0 && fast_start < slow_end) }
' "$event_log"

printf '%s\n' 'gpu_slot_scheduler_mock=PASS dry_run_screen=36 arima=16 moirai=20 arima_refine=4 confirm7=7 robustness=640 full=2560 attempted=4 failed=1 full_handler_exit=7'

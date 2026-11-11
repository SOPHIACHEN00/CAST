#!/usr/bin/env sh
# End-to-end mock of one self-contained single-GPU, 15-worker Bash launcher.
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/shapley-launcher-test.XXXXXX")
interrupt_launcher_pid=''
interrupt_watchdog_pid=''
interrupt_pid_dir=''
interrupt_worker_pid_file=''

terminate_test_processes() {
  if [ -n "$interrupt_watchdog_pid" ]; then
    kill -TERM "$interrupt_watchdog_pid" 2>/dev/null || true
    wait "$interrupt_watchdog_pid" 2>/dev/null || true
    interrupt_watchdog_pid=''
  fi

  tracked_pids=$interrupt_launcher_pid
  if [ -n "$interrupt_worker_pid_file" ] && [ -f "$interrupt_worker_pid_file" ]; then
    for tracked_pid in $(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$interrupt_worker_pid_file"); do
      case " $tracked_pids " in
        *" $tracked_pid "*) ;;
        *) tracked_pids="$tracked_pids $tracked_pid" ;;
      esac
    done
  fi
  if [ -n "$interrupt_pid_dir" ] && [ -d "$interrupt_pid_dir" ]; then
    for tracked_pid_file in "$interrupt_pid_dir"/*.pid; do
      [ -f "$tracked_pid_file" ] || continue
      for tracked_pid in $(sed -n '1p' "$tracked_pid_file"); do
        case "$tracked_pid" in
          ''|*[!0-9]*) ;;
          *)
            case " $tracked_pids " in
              *" $tracked_pid "*) ;;
              *) tracked_pids="$tracked_pids $tracked_pid" ;;
            esac
            ;;
        esac
      done
    done
  fi

  for tracked_pid in $tracked_pids; do
    kill -TERM "$tracked_pid" 2>/dev/null || true
  done
  cleanup_step=0
  while [ "$cleanup_step" -lt 3 ]; do
    any_tracked_alive=0
    for tracked_pid in $tracked_pids; do
      if kill -0 "$tracked_pid" 2>/dev/null; then
        any_tracked_alive=1
      fi
    done
    [ "$any_tracked_alive" -eq 0 ] && break
    sleep 1
    cleanup_step=$((cleanup_step + 1))
  done
  for tracked_pid in $tracked_pids; do
    if kill -0 "$tracked_pid" 2>/dev/null; then
      kill -KILL "$tracked_pid" 2>/dev/null || true
    fi
  done
  if [ -n "$interrupt_launcher_pid" ]; then
    wait "$interrupt_launcher_pid" 2>/dev/null || true
    interrupt_launcher_pid=''
  fi
}

cleanup() {
  cleanup_status=$?
  trap - 0 HUP INT TERM
  terminate_test_processes
  case "$temporary_root" in
    "${TMPDIR:-/tmp}"/shapley-launcher-test.*) rm -rf -- "$temporary_root" ;;
    *) printf 'Refusing unsafe cleanup target: %s\n' "$temporary_root" >&2 ;;
  esac
  exit "$cleanup_status"
}
trap cleanup 0
trap 'exit 130' HUP INT TERM

project_root=$temporary_root/project
candidate_root=$project_root/.candidate_runs/mock/attack
mock_bin=$temporary_root/bin
mkdir -p "$candidate_root/module/method" "$mock_bin"

printf '%s\n' '# mock candidate entry' > "$candidate_root/FLCE_updated.py"
printf '%s\n' '# mock ordinary attacks' > "$candidate_root/module/method/attack_updated_nonlayerwise.py"
printf '%s\n' '# mock layerwise attacks' > "$candidate_root/module/method/attack_updated_layerwise.py"
printf '%s\n' 'deployment=mock' > "$candidate_root/.candidate_deployment_manifest.txt"
printf '%s\n' 'smoke=passed' > "$candidate_root/.smoke_passed_manifest.txt"
mkdir "$candidate_root/script"
cp "$repo_root/script/run_tictactoe_forecast_pure_shapley.sh" \
  "$candidate_root/script/run_tictactoe_forecast_pure_shapley.sh"
(
  cd "$candidate_root"
  sha256sum FLCE_updated.py \
    module/method/attack_updated_nonlayerwise.py \
    module/method/attack_updated_layerwise.py \
    script/run_tictactoe_forecast_pure_shapley.sh > .smoke_passed.sha256
)

cat > "$mock_bin/python" <<'MOCK_PYTHON'
#!/usr/bin/env sh
set -eu
if [ "${1:-}" = -c ]; then
  printf '%s\n' 'Mock NVIDIA GeForce RTX 3090'
  exit 0
fi

[ "${OMP_NUM_THREADS:-}" = 15 ] || {
  printf 'Expected OMP_NUM_THREADS=15, got %s\n' "${OMP_NUM_THREADS:-unset}" >&2
  exit 21
}
[ "${MKL_NUM_THREADS:-}" = 15 ] || {
  printf 'Expected MKL_NUM_THREADS=15, got %s\n' "${MKL_NUM_THREADS:-unset}" >&2
  exit 22
}
[ "${OPENBLAS_NUM_THREADS:-}" = 15 ] || {
  printf 'Expected OPENBLAS_NUM_THREADS=15, got %s\n' "${OPENBLAS_NUM_THREADS:-unset}" >&2
  exit 23
}
[ "${NUMEXPR_NUM_THREADS:-}" = 15 ] || {
  printf 'Expected NUMEXPR_NUM_THREADS=15, got %s\n' "${NUMEXPR_NUM_THREADS:-unset}" >&2
  exit 24
}

attack_method=''
dataset=''
client_num=''
trial_id=''
results_root=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --attack_method) attack_method=$2; shift 2 ;;
    --dataset) dataset=$2; shift 2 ;;
    --client_num) client_num=$2; shift 2 ;;
    --trial_id) trial_id=$2; shift 2 ;;
    --results_root) results_root=$2; shift 2 ;;
    *) shift ;;
  esac
done

if [ "${MOCK_HANG_ALL:-0}" = 1 ]; then
  : "${MOCK_PID_DIR:?MOCK_PID_DIR is required for hanging-child test}"
  mkdir -p "$MOCK_PID_DIR"
  printf '%s\n' "$$" > "$MOCK_PID_DIR/child_$$.pid"
  trap '' HUP INT TERM
  while :; do sleep 1; done
fi

if [ -n "${MOCK_FAIL_METHOD:-}" ] && [ "$attack_method" = "$MOCK_FAIL_METHOD" ]; then
  exit 7
fi

printf '%s\n' 'detect available GPU: Mock NVIDIA GeForce RTX 3090, FL training is running in pure GPU mode.'
round=0
while [ "$round" -lt 50 ]; do
  printf '[ROUND %s] attacker=1 honest=False hessian=on mock=1\n' "$round"
  round=$((round + 1))
done

if [ -n "${MOCK_PROTOCOL_FAIL_METHOD:-}" ] && [ "$attack_method" = "$MOCK_PROTOCOL_FAIL_METHOD" ]; then
  exit 0
fi

detecting_dir=$results_root/Detecting_quantitative_shapley
time_dir=$results_root/Time_quantitative_shapley
accuracy_dir=$results_root/Accuracy_quantitative_shapley
manifest_dir=$results_root/manifests
mkdir -p "$detecting_dir" "$time_dir" "$accuracy_dir" "$manifest_dir"

contribution_file=$detecting_dir/combined_contributions_trial${trial_id}.csv
printf '%s\n' header > "$contribution_file"
client_index=0
while [ "$client_index" -lt "$client_num" ]; do
  printf '%s\n' row >> "$contribution_file"
  client_index=$((client_index + 1))
done

printf '%s\n%s\n' header row > "$time_dir/attacking_time_log_trial${trial_id}.csv"
accuracy_file=$accuracy_dir/global_accuracy_log_trial${trial_id}.csv
printf '%s\n' header > "$accuracy_file"
round=0
while [ "$round" -lt 50 ]; do
  printf '%s\n' row >> "$accuracy_file"
  round=$((round + 1))
done
printf '%s\n' '{}' > \
  "$manifest_dir/run_manifest_${dataset}_${attack_method}_trial${trial_id}_mock.json"
MOCK_PYTHON

cat > "$mock_bin/nvidia-smi" <<'MOCK_NVIDIA_SMI'
#!/usr/bin/env sh
set -eu
gpu_id=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    -i) gpu_id=$2; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\n' "$gpu_id"
MOCK_NVIDIA_SMI

cat > "$mock_bin/flock" <<'MOCK_FLOCK'
#!/usr/bin/env sh
exit 0
MOCK_FLOCK
chmod +x "$mock_bin/python" "$mock_bin/nvidia-smi" "$mock_bin/flock"

set +e
PATH="$mock_bin:$PATH" \
PROJECT_ROOT="$project_root" \
CANDIDATE_ROOT="$candidate_root" \
PYTHON_BIN="$mock_bin/python" \
CAMPAIGN_ID=mock_tictactoe_pure \
DRY_RUN=0 \
CONFIRM_SHAPLEY_RUN=1 \
MOCK_FAIL_METHOD=arima_adaptive \
MOCK_PROTOCOL_FAIL_METHOD=moirai \
  bash "$candidate_root/script/run_tictactoe_forecast_pure_shapley.sh" \
  > "$temporary_root/launcher.out" 2> "$temporary_root/launcher.err"
launcher_status=$?
set -e

[ "$launcher_status" -ne 0 ] || {
  printf '%s\n' 'Expected injected task failures to make the launcher nonzero.' >&2
  exit 1
}

campaign_root=$project_root/.shapley_runs/mock_tictactoe_pure
campaign_manifest=$campaign_root/launcher_summary.txt
[ -f "$campaign_manifest" ]
require_campaign_line() {
  required_campaign_line=$1
  if ! grep -qx "$required_campaign_line" "$campaign_manifest"; then
    printf 'Missing campaign assertion: %s\n' "$required_campaign_line" >&2
    printf '%s\n' '--- campaign manifest ---' >&2
    sed -n '1,240p' "$campaign_manifest" >&2
    printf '%s\n' '--- launcher stderr ---' >&2
    sed -n '1,240p' "$temporary_root/launcher.err" >&2
    return 1
  fi
}
require_campaign_line 'total_tasks=64'
require_campaign_line 'gpus=1'
require_campaign_line 'workers_per_gpu=15'
require_campaign_line 'total_gpu_workers=15'
require_campaign_line 'completed_tasks=64'
require_campaign_line 'successful_tasks=32'
require_campaign_line 'failed_tasks=32'
require_campaign_line 'unfinished_tasks=0'
require_campaign_line 'worker_failures=0'

completed_count=$(find "$campaign_root/queue/completed" -type f -name '*.spec' | wc -l | tr -d ' ')
failure_count=$(find "$campaign_root/failures" -type f -name '*.failed' | wc -l | tr -d ' ')
status_count=$(find "$campaign_root/tasks" -type f -name status.txt | wc -l | tr -d ' ')
protocol_pass_count=$(grep -rl '^success protocol_pass ' "$campaign_root/tasks"/*/status.txt | wc -l | tr -d ' ')
child_exit_count=$(grep -rl '^failed exit_code=7 ' "$campaign_root/tasks"/*/status.txt | wc -l | tr -d ' ')
[ "$completed_count" -eq 64 ]
[ "$failure_count" -eq 32 ]
[ "$status_count" -eq 64 ]
[ "$protocol_pass_count" -eq 32 ]
[ "$child_exit_count" -eq 16 ]
protocol_invalid_count=$(grep -rl '^failed protocol_invalid exit_code=3 ' \
  "$campaign_root/tasks"/*/status.txt | wc -l | tr -d ' ')
[ "$protocol_invalid_count" -eq 16 ]

worker_pid_file=$campaign_root/workers/worker_pids.txt
worker_pid_count=$(wc -l < "$worker_pid_file" | tr -d ' ')
[ "$worker_pid_count" -eq 15 ]

attempted_sum=0
worker_slot=1
while [ "$worker_slot" -le 15 ]; do
  worker_status=$campaign_root/workers/gpu_1_worker_${worker_slot}.status
  worker_log=$campaign_root/workers/gpu_1_worker_${worker_slot}.log
  [ -f "$worker_status" ]
  [ -f "$worker_log" ]
  grep -q "^gpu=1 worker_slot=${worker_slot} attempted=[0-9][0-9]* failed=[0-9][0-9]* state=finished$" \
    "$worker_status"
  grep -q "^gpu=1 worker_slot=${worker_slot} pid=[0-9][0-9]*$" "$worker_pid_file"
  worker_attempted=$(sed -n 's/.*attempted=\([0-9][0-9]*\).*/\1/p' "$worker_status")
  [ -n "$worker_attempted" ]
  attempted_sum=$((attempted_sum + worker_attempted))
  worker_slot=$((worker_slot + 1))
done
[ "$attempted_sum" -eq 64 ]

interrupt_pid_dir=$temporary_root/interrupt_pids
PATH="$mock_bin:$PATH" \
PROJECT_ROOT="$project_root" \
CANDIDATE_ROOT="$candidate_root" \
PYTHON_BIN="$mock_bin/python" \
CAMPAIGN_ID=mock_tictactoe_interrupt \
DRY_RUN=0 \
CONFIRM_SHAPLEY_RUN=1 \
MOCK_HANG_ALL=1 \
MOCK_PID_DIR="$interrupt_pid_dir" \
  bash "$candidate_root/script/run_tictactoe_forecast_pure_shapley.sh" \
  > "$temporary_root/interrupt.out" 2> "$temporary_root/interrupt.err" &
interrupt_launcher_pid=$!
interrupt_worker_pid_file=$project_root/.shapley_runs/mock_tictactoe_interrupt/workers/worker_pids.txt

poll_step=0
interrupt_child_count=0
while [ "$poll_step" -lt 20 ]; do
  if [ -d "$interrupt_pid_dir" ]; then
    interrupt_child_count=$(find "$interrupt_pid_dir" -type f -name '*.pid' | wc -l | tr -d ' ')
  fi
  [ "$interrupt_child_count" -ge 15 ] && break
  sleep 1
  poll_step=$((poll_step + 1))
done
[ "$interrupt_child_count" -ge 15 ] || {
  terminate_test_processes
  printf '%s\n' 'Interrupt mock did not start fifteen child tasks in time.' >&2
  exit 1
}
interrupt_worker_count=$(wc -l < "$interrupt_worker_pid_file" | tr -d ' ')
[ "$interrupt_worker_count" -eq 15 ]

kill -TERM "$interrupt_launcher_pid"
interrupt_timeout_marker=$temporary_root/interrupt_wait_timed_out
(
  sleep 20
  if kill -0 "$interrupt_launcher_pid" 2>/dev/null; then
    printf '%s\n' timeout > "$interrupt_timeout_marker"
    kill -KILL "$interrupt_launcher_pid" 2>/dev/null || true
  fi
) &
interrupt_watchdog_pid=$!
set +e
wait "$interrupt_launcher_pid"
interrupt_status=$?
set -e
kill -TERM "$interrupt_watchdog_pid" 2>/dev/null || true
wait "$interrupt_watchdog_pid" 2>/dev/null || true
interrupt_watchdog_pid=''
interrupt_launcher_pid=''
[ ! -f "$interrupt_timeout_marker" ] || {
  terminate_test_processes
  printf '%s\n' 'Interrupted launcher exceeded the test watchdog timeout.' >&2
  exit 1
}
[ "$interrupt_status" -eq 130 ] || {
  terminate_test_processes
  printf 'Expected interrupted launcher exit 130, got %s.\n' "$interrupt_status" >&2
  exit 1
}

for interrupt_pid_file in "$interrupt_pid_dir"/*.pid; do
  interrupt_child_pid=$(sed -n '1p' "$interrupt_pid_file")
  if kill -0 "$interrupt_child_pid" 2>/dev/null; then
    printf 'Interrupted child is still alive: %s\n' "$interrupt_child_pid" >&2
    exit 1
  fi
done
interrupt_pid_dir=''
interrupt_worker_pid_file=''
grep -qx 'campaign_state=interrupted' \
  "$project_root/.shapley_runs/mock_tictactoe_interrupt/launcher_summary.txt"

printf '%s\n' \
  'shapley_forecast_scheduler_mock=PASS tasks=64 success=32 child_failures=16 protocol_failures=16 gpu_workers=15 interrupt_cleanup=PASS'

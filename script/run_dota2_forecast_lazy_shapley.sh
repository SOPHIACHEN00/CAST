#!/usr/bin/env bash
set -Eeuo pipefail

# Dota2 lazy attack: Shapley-only, ordinary ARIMA/Moirai offline + online/adaptive.
# Preview: bash script/run_dota2_forecast_lazy_shapley.sh
# Run:     DRY_RUN=0 CONFIRM_SHAPLEY_RUN=1 nohup bash script/run_dota2_forecast_lazy_shapley.sh > dota2_lazy_launcher.log 2>&1 &

# ----------------------------- visible experiment configuration -----------------------------
SCRIPT="${SCRIPT:-FLCE_updated.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_REQUESTED="${GPU_IDS:-3}"
unset GPU_IDS
GPU_IDS=(3)
WORKERS_PER_GPU="${WORKERS_PER_GPU:-15}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-15}"

DATASET="dota2"
MODE="lazy"
ALPHAS=(10000 1000000 50000000 200000000)
CLIENT_NUMS=(2 4 6 8)
METHOD_RECORDS=(
  "arima_offline:arima"
  "arima_adaptive:arima_adaptive"
  "moirai_offline:moirai"
  "moirai_online:moirai_online"
)
SCHEDULE_RECORDS=(
  "fixed|2|0.50|25|25"
  "probabilistic|2|0.50|25|25"
  "fixed|3|0.66|17|33"
  "probabilistic|3|0.66|17|33"
  "fixed|6|0.82|9|41"
  "probabilistic|6|0.82|9|41"
  "fixed|10|0.90|5|45"
  "probabilistic|10|0.90|5|45"
)

ATTACKER_ID="${ATTACKER_ID:-1}"
TRIAL_IDS_RAW="${TRIAL_IDS:-0}"
read -r -a TRIAL_IDS <<< "${TRIAL_IDS_RAW}"
CONTRIBUTION_METHOD="${CONTRIBUTION_METHOD:-shapley}"
DETECTING_TYPE="quantitative"
NUM_EPOCH="${NUM_EPOCH:-50}"
NUM_LOCAL_EPOCHS=1
RANDOM_ROUND=10
USE_HESSIAN="${USE_HESSIAN:-1}"
LAMBDA_H=0.9

# ARIMA: common offline/adaptive readiness + adaptive-only reset/refit parameters.
ARIMA_VAR_WINDOW=3
ARIMA_VAR_THRESHOLD=0.005
ARIMA_ERR_THRESHOLD=0.18
ARIMA_MIN_ROUNDS=4
ARIMA_RETRAIN_INTERVAL=15
ARIMA_EMA_BETA=0.9

# Online Moirai only. Offline Moirai receives none of these adapter parameters.
MOIRAI_TUNE_LR=0.001
MOIRAI_TUNE_STEPS=1
MOIRAI_TUNE_START_ROUND=10
MOIRAI_ADAPT_INTERVAL=4
MOIRAI_RESIDUAL_DECAY=0.95
MOIRAI_RESIDUAL_CLIP=5.0
MOIRAI_EMA_BETA=0.9
MOIRAI_ERR_THRESHOLD=0.25
MOIRAI_MIN_ROUNDS=4

DRY_RUN="${DRY_RUN:-1}"
DRY_RUN_PRINT_LIMIT="${DRY_RUN_PRINT_LIMIT:-12}"
CONFIRM_SHAPLEY_RUN="${CONFIRM_SHAPLEY_RUN:-0}"

# ----------------------------- paths and campaign identity -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  BASE_PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ "${CANDIDATE_ROOT}" == */.candidate_runs/*/attack ]]; then
  BASE_PROJECT_ROOT="${CANDIDATE_ROOT%%/.candidate_runs/*}"
else
  BASE_PROJECT_ROOT="${CANDIDATE_ROOT}"
fi
SCRIPT_SOURCE="${SCRIPT}"
[[ "${SCRIPT_SOURCE}" = /* ]] || SCRIPT_SOURCE="${CANDIDATE_ROOT}/${SCRIPT_SOURCE}"
TS="${TS:-$(date +'%Y%m%d-%H%M%S')}"
CAMPAIGN_ID="${CAMPAIGN_ID:-forecast_${DATASET}_${MODE}_shapley_${TS}_pid$$}"
CAMPAIGN_ROOT="${BASE_PROJECT_ROOT}/.shapley_runs/${CAMPAIGN_ID}"
RESULTS_ROOT="${BASE_PROJECT_ROOT}/Results/${CAMPAIGN_ID}"
TASK_FILE="${CAMPAIGN_ROOT}/tasks.tsv"
QUEUE_ROOT="${CAMPAIGN_ROOT}/queue"
QUEUE_PENDING="${QUEUE_ROOT}/pending"
QUEUE_RUNNING="${QUEUE_ROOT}/running"
QUEUE_COMPLETED="${QUEUE_ROOT}/completed"
TASK_ROOT="${CAMPAIGN_ROOT}/tasks"
FAILURE_ROOT="${CAMPAIGN_ROOT}/failures"
WORKER_ROOT="${CAMPAIGN_ROOT}/workers"
SUMMARY_FILE="${CAMPAIGN_ROOT}/launcher_summary.txt"

shopt -s nullglob

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

validate_configuration() {
  [[ "${GPU_IDS_REQUESTED}" == "3" ]] || die 'GPU_IDS must remain: 3'
  [[ "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || die 'WORKERS_PER_GPU must be a positive integer'
  [[ "${CPU_THREADS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]] || die 'CPU_THREADS_PER_WORKER must be a positive integer'
  [[ "${CONTRIBUTION_METHOD}" == "shapley" ]] || die 'CONTRIBUTION_METHOD must remain shapley'
  [[ "${NUM_EPOCH}" == "50" ]] || die 'NUM_EPOCH must remain 50'
  [[ "${USE_HESSIAN}" == "1" ]] || die 'USE_HESSIAN must remain 1'
  [[ "${ATTACKER_ID}" == "1" ]] || die 'ATTACKER_ID must remain 1'
  [[ "${DRY_RUN}" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
  [[ "${CONFIRM_SHAPLEY_RUN}" =~ ^[01]$ ]] || die 'CONFIRM_SHAPLEY_RUN must be 0 or 1'
  [[ "${DRY_RUN_PRINT_LIMIT}" =~ ^[0-9]+$ ]] || die 'DRY_RUN_PRINT_LIMIT must be non-negative'
  ((${#TRIAL_IDS[@]} > 0)) || die 'TRIAL_IDS is empty'
  local trial seen=" "
  for trial in "${TRIAL_IDS[@]}"; do
    [[ "${trial}" =~ ^(0|[1-9][0-9]*)$ ]] || die "non-canonical trial id: ${trial}"
    [[ "${seen}" != *" ${trial} "* ]] || die "duplicate trial id: ${trial}"
    seen+="${trial} "
  done
}

emit_tasks() {
  local task_id=0 trial alpha client method_record variant method schedule_record
  local schedule gap probability expected_honest expected_fake
  for trial in "${TRIAL_IDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for client in "${CLIENT_NUMS[@]}"; do
        for schedule_record in "${SCHEDULE_RECORDS[@]}"; do
          IFS='|' read -r schedule gap probability expected_honest expected_fake <<< "${schedule_record}"
          for method_record in "${METHOD_RECORDS[@]}"; do
            IFS=':' read -r variant method <<< "${method_record}"
            task_id=$((task_id + 1))
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "${task_id}" "${alpha}" "${client}" "${variant}" "${method}" \
              "${schedule}" "${gap}" "${probability}" "${expected_honest}" \
              "${expected_fake}" "${trial}"
          done
        done
      done
    done
  done
}

build_command() {
  local task_id="$1" alpha="$2" client="$3" variant="$4" method="$5"
  local schedule="$6" gap="$7" probability="$8" trial="$9"
  local probability_label="${probability//./p}"
  TASK_LABEL=$(printf 'task_%06d_%s_%s_%s_a%s_n%s_%s_g%s_p%s_trial%s' \
    "${task_id}" "${DATASET}" "${MODE}" "${variant}" "${alpha}" "${client}" \
    "${schedule}" "${gap}" "${probability_label}" "${trial}")
  TASK_DIR="${TASK_ROOT}/${TASK_LABEL}"
  TASK_RESULTS="${RESULTS_ROOT}/${TASK_LABEL}"
  CMD=(
    "${PYTHON_BIN}" "${SCRIPT_SOURCE}"
    --dataset "${DATASET}"
    --alpha "${alpha}"
    --client_num "${client}"
    --attacker_id "${ATTACKER_ID}"
    --attack_method "${method}"
    --contribution_method "${CONTRIBUTION_METHOD}"
    --detecting_type "${DETECTING_TYPE}"
    --use_last_round_sv
    --trial_id "${trial}"
    --num_epoch "${NUM_EPOCH}"
    --num_local_epochs "${NUM_LOCAL_EPOCHS}"
    --random_round "${RANDOM_ROUND}"
    --results_root "${TASK_RESULTS}"
    --use_attack
    --use_hessian
    --lambda_h "${LAMBDA_H}"
  )

  if [[ "${MODE}" == "lazy" ]]; then
    CMD+=(
      --use_lazy
      --lazy_schedule "${schedule}"
      --lazy_honest_gap "${gap}"
      --lazy_honest_phase 0
      --lazy_fake_policy attack
    )
    [[ "${schedule}" == "probabilistic" ]] && CMD+=(--attack_probability "${probability}")
  fi

  case "${variant}" in
    arima_offline|arima_adaptive)
      CMD+=(--arima_var_window "${ARIMA_VAR_WINDOW}" --arima_var_threshold "${ARIMA_VAR_THRESHOLD}")
      ;;
  esac
  if [[ "${variant}" == "arima_adaptive" ]]; then
    CMD+=(
      --arima_err_threshold "${ARIMA_ERR_THRESHOLD}"
      --arima_min_rounds_after_start "${ARIMA_MIN_ROUNDS}"
      --arima_retrain_interval "${ARIMA_RETRAIN_INTERVAL}"
      --arima_err_ema_beta "${ARIMA_EMA_BETA}"
      --arima_trim_history
    )
  fi
  if [[ "${variant}" == "moirai_online" ]]; then
    CMD+=(
      --moirai_online_tuning
      --moirai_tune_lr "${MOIRAI_TUNE_LR}"
      --moirai_tune_steps "${MOIRAI_TUNE_STEPS}"
      --moirai_tune_start_round "${MOIRAI_TUNE_START_ROUND}"
      --moirai_adapt_interval "${MOIRAI_ADAPT_INTERVAL}"
      --moirai_residual_decay "${MOIRAI_RESIDUAL_DECAY}"
      --moirai_residual_clip "${MOIRAI_RESIDUAL_CLIP}"
      --moirai_err_ema_beta "${MOIRAI_EMA_BETA}"
      --moirai_err_threshold "${MOIRAI_ERR_THRESHOLD}"
      --moirai_min_rounds_after_start "${MOIRAI_MIN_ROUNDS}"
      --moirai_trim_history
    )
  fi
}

print_command() {
  printf 'command:'
  local argument
  for argument in "${CMD[@]}"; do
    printf ' <%s>' "${argument}"
  done
  printf '\n'
}

print_plan() {
  local printed=0 enumerated=0 task_id alpha client variant method schedule gap probability
  local expected_honest expected_fake trial
  while IFS=$'\t' read -r task_id alpha client variant method schedule gap probability \
    expected_honest expected_fake trial; do
    enumerated=$((enumerated + 1))
    if ((printed < DRY_RUN_PRINT_LIMIT)); then
      printf 'TASK|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "${task_id}" "${DATASET}" "${MODE}" "${alpha}" "${client}" "${variant}" \
        "${method}" "${schedule}" "${gap}" "${probability}" "${trial}"
      build_command "${task_id}" "${alpha}" "${client}" "${variant}" "${method}" \
        "${schedule}" "${gap}" "${probability}" "${trial}"
      print_command
      printed=$((printed + 1))
    fi
  done < <(emit_tasks)
  [[ "${enumerated}" -eq "${TOTAL_TASKS}" ]] || die 'dry-run task enumeration drifted'
  printf 'DRY_RUN rendered %s commands and enumerated %s tasks.\n' "${printed}" "${enumerated}"
  printf '%s\n' 'No directory was created and no experiment was started.'
}

ensure_prerequisites() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python executable not found: ${PYTHON_BIN}"
  command -v flock >/dev/null 2>&1 || die 'flock is required for the dynamic queue'
  command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is required'
  [[ -f "${SCRIPT_SOURCE}" ]] || die "runner not found: ${SCRIPT_SOURCE}"
  [[ "${SCRIPT_SOURCE}" == "${CANDIDATE_ROOT}/FLCE_updated.py" ]] ||
    die 'SCRIPT must resolve to the candidate FLCE_updated.py'
  [[ -f "${CANDIDATE_ROOT}/module/method/attack_updated_nonlayerwise.py" ]] ||
    die 'candidate ordinary attack module is missing'
}

probe_gpus() {
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | grep -qx "${gpu}" \
      || die "GPU ${gpu} is unavailable"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -c \
      'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1' \
      >/dev/null || die "PyTorch cannot use isolated GPU ${gpu}"
  done
}

claim_next_spec() {
  local gpu="$1" worker_slot="$2" pending_spec pending_name claim_guard claimed_spec
  for pending_spec in "${QUEUE_PENDING}"/*.spec; do
    [[ -f "${pending_spec}" ]] || continue
    pending_name="${pending_spec##*/}"
    claim_guard="${QUEUE_RUNNING}/.${pending_name}.claim"
    mkdir "${claim_guard}" 2>/dev/null || continue
    claimed_spec="${QUEUE_RUNNING}/${pending_name%.spec}.gpu_${gpu}.worker_${worker_slot}.spec"
    if mv "${pending_spec}" "${claimed_spec}" 2>/dev/null; then
      rmdir "${claim_guard}" 2>/dev/null || true
      printf '%s\n' "${claimed_spec}"
      return 0
    fi
    rmdir "${claim_guard}" 2>/dev/null || true
  done
  return 1
}

job_is_owned() {
  local target_pid="${1:-}" job_pid
  [[ "${target_pid}" =~ ^[0-9]+$ ]] || return 1
  while IFS= read -r job_pid; do
    [[ "${job_pid}" == "${target_pid}" ]] && return 0
  done < <(jobs -pr 2>/dev/null || true)
  return 1
}

terminate_pid() {
  local pid="${1:-}" step=0
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  job_is_owned "${pid}" || return 0
  kill -TERM "${pid}" 2>/dev/null || true
  while job_is_owned "${pid}" && ((step < 5)); do
    sleep 1
    step=$((step + 1))
  done
  job_is_owned "${pid}" && kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

run_one() {
  local gpu="$1" task_id="$2" alpha="$3" client="$4" variant="$5" method="$6"
  local schedule="$7" gap="$8" probability="$9" expected_honest="${10}"
  local expected_fake="${11}" trial="${12}"
  build_command "${task_id}" "${alpha}" "${client}" "${variant}" "${method}" \
    "${schedule}" "${gap}" "${probability}" "${trial}"
  [[ ! -e "${TASK_DIR}" && ! -e "${TASK_RESULTS}" ]] || die "task path collision: ${TASK_LABEL}"
  mkdir -p "${TASK_DIR}/logs" "${TASK_DIR}/tmp" "${TASK_DIR}/cache" "${TASK_RESULTS}"
  local log_file="${TASK_DIR}/logs/run.log" status_file="${TASK_DIR}/status.txt"
  {
    printf '============================================================\n'
    printf '[START] %s\n' "$(date '+%F %T')"
    printf '[TASK] id=%s gpu=%s dataset=%s mode=%s alpha=%s clients=%s method=%s schedule=%s gap=%s probability=%s trial=%s\n' \
      "${task_id}" "${gpu}" "${DATASET}" "${MODE}" "${alpha}" "${client}" "${method}" \
      "${schedule}" "${gap}" "${probability}" "${trial}"
    printf '[RESULTS] %s\n' "${TASK_RESULTS}"
    print_command
    printf '============================================================\n'
  } > "${log_file}"
  {
    printf 'task_id=%s\ndataset=%s\nmode=%s\nalpha=%s\nclient_num=%s\n' \
      "${task_id}" "${DATASET}" "${MODE}" "${alpha}" "${client}"
    printf 'variant=%s\nattack_method=%s\nschedule=%s\ngap=%s\nprobability=%s\ntrial=%s\ngpu=%s\n' \
      "${variant}" "${method}" "${schedule}" "${gap}" "${probability}" "${trial}" "${gpu}"
    printf 'results_root=%s\ncommand_argv:\n' "${TASK_RESULTS}"
    printf '  %s\n' "${CMD[@]}"
  } > "${TASK_DIR}/manifest.txt"

  local started_at ended_at task_exit=0
  started_at=$(date +%s)
  (
    cd "${CANDIDATE_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export TMPDIR="${TASK_DIR}/tmp" HF_HOME="${TASK_DIR}/cache/huggingface"
    export TORCH_HOME="${TASK_DIR}/cache/torch" XDG_CACHE_HOME="${TASK_DIR}/cache/xdg"
    exec "${CMD[@]}"
  ) >> "${log_file}" 2>&1 &
  CURRENT_TASK_PID=$!
  wait "${CURRENT_TASK_PID}" || task_exit=$?
  CURRENT_TASK_PID=''
  ended_at=$(date +%s)
  if ((task_exit != 0)); then
    printf 'failed exit_code=%s runtime_seconds=%s\n' "${task_exit}" "$((ended_at - started_at))" > "${status_file}"
    : > "${FAILURE_ROOT}/${TASK_LABEL}.failed"
    return "${task_exit}"
  fi

  local observed_rounds observed_honest observed_fake observed_hessian observed_cuda
  observed_rounds=$(grep -c '^\[ROUND ' "${log_file}" || true)
  observed_honest=$(grep -c '^\[ROUND .* honest=True ' "${log_file}" || true)
  observed_fake=$(grep -c '^\[ROUND .* honest=False ' "${log_file}" || true)
  observed_hessian=$(grep -c '^\[ROUND .* hessian=on ' "${log_file}" || true)
  observed_cuda=$(grep -c 'FL training is running in pure GPU mode' "${log_file}" || true)
  local contribution_file="${TASK_RESULTS}/Detecting_quantitative_shapley/combined_contributions_trial${trial}.csv"
  local time_file="${TASK_RESULTS}/Time_quantitative_shapley/attacking_time_log_trial${trial}.csv"
  local accuracy_file="${TASK_RESULTS}/Accuracy_quantitative_shapley/global_accuracy_log_trial${trial}.csv"
  local contribution_rows=0 time_rows=0 accuracy_rows=0
  [[ -s "${contribution_file}" ]] && contribution_rows=$(awk 'END {print NR+0}' "${contribution_file}")
  [[ -s "${time_file}" ]] && time_rows=$(awk 'END {print NR+0}' "${time_file}")
  [[ -s "${accuracy_file}" ]] && accuracy_rows=$(awk 'END {print NR+0}' "${accuracy_file}")
  local provenance=("${TASK_RESULTS}"/manifests/run_manifest_${DATASET}_${method}_trial${trial}_*.json)
  if [[ "${observed_rounds}" -eq 50 && "${observed_honest}" -eq "${expected_honest}" \
        && "${observed_fake}" -eq "${expected_fake}" && "${observed_hessian}" -eq 50 \
        && "${observed_cuda}" -ge 1 && "${contribution_rows}" -eq $((client + 1)) \
        && "${time_rows}" -eq 2 && "${accuracy_rows}" -eq 51 && "${#provenance[@]}" -eq 1 ]]; then
    printf 'success protocol_pass runtime_seconds=%s rounds=%s honest=%s fake=%s hessian=%s\n' \
      "$((ended_at - started_at))" "${observed_rounds}" "${observed_honest}" \
      "${observed_fake}" "${observed_hessian}" > "${status_file}"
    return 0
  fi
  printf 'failed protocol_invalid exit_code=3 rounds=%s honest=%s fake=%s hessian=%s cuda=%s contribution_rows=%s time_rows=%s accuracy_rows=%s provenance=%s\n' \
    "${observed_rounds}" "${observed_honest}" "${observed_fake}" "${observed_hessian}" \
    "${observed_cuda}" "${contribution_rows}" "${time_rows}" "${accuracy_rows}" "${#provenance[@]}" > "${status_file}"
  : > "${FAILURE_ROOT}/${TASK_LABEL}.failed"
  return 3
}

interrupt_worker() {
  trap '' HUP INT TERM
  local pid cleanup_pids=()
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    [[ " ${cleanup_pids[*]:-} " == *" ${pid} "* ]] || cleanup_pids+=("${pid}")
  done < <(jobs -pr 2>/dev/null || true)
  for pid in "${cleanup_pids[@]:-}"; do
    terminate_pid "${pid}"
  done
  exit 130
}

worker_loop() {
  local gpu="$1" worker_slot="$2" claimed_spec task_line task_id alpha client variant method schedule gap probability
  local expected_honest expected_fake trial attempted=0 failed=0
  CURRENT_TASK_PID=''
  trap 'interrupt_worker' HUP INT TERM
  while claimed_spec=$(claim_next_spec "${gpu}" "${worker_slot}"); do
    task_line=$(<"${claimed_spec}")
    IFS=$'\t' read -r task_id alpha client variant method schedule gap probability \
      expected_honest expected_fake trial <<< "${task_line}"
    attempted=$((attempted + 1))
    run_one "${gpu}" "${task_id}" "${alpha}" "${client}" "${variant}" "${method}" \
      "${schedule}" "${gap}" "${probability}" "${expected_honest}" "${expected_fake}" "${trial}" \
      || failed=$((failed + 1))
    mv "${claimed_spec}" "${QUEUE_COMPLETED}/${claimed_spec##*/}"
  done
  printf 'gpu=%s worker_slot=%s attempted=%s failed=%s state=finished\n' \
    "${gpu}" "${worker_slot}" "${attempted}" "${failed}" \
    > "${WORKER_ROOT}/gpu_${gpu}_worker_${worker_slot}.status"
  trap - HUP INT TERM
}

interrupt_launcher() {
  trap '' HUP INT TERM
  local pid step=0 any_alive cleanup_pids=()
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    [[ " ${cleanup_pids[*]:-} " == *" ${pid} "* ]] || cleanup_pids+=("${pid}")
  done < <(jobs -pr 2>/dev/null || true)
  for pid in "${cleanup_pids[@]:-}"; do
    job_is_owned "${pid}" && kill -TERM "${pid}" 2>/dev/null || true
  done
  while ((step < 8)); do
    any_alive=0
    for pid in "${cleanup_pids[@]:-}"; do
      job_is_owned "${pid}" && any_alive=1
    done
    ((any_alive == 0)) && break
    sleep 1
    step=$((step + 1))
  done
  for pid in "${cleanup_pids[@]:-}"; do
    job_is_owned "${pid}" && kill -KILL "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
  printf 'campaign_state=interrupted\ninterrupted_at=%s\n' "$(date '+%F %T')" >> "${SUMMARY_FILE}"
  exit 130
}

launch_workers() {
  local gpu worker_slot pid worker_log worker_failures=0
  WORKER_PIDS=()
  : > "${WORKER_ROOT}/worker_pids.txt"
  trap 'interrupt_launcher' HUP INT TERM
  for gpu in "${GPU_IDS[@]}"; do
    for ((worker_slot = 1; worker_slot <= WORKERS_PER_GPU; worker_slot++)); do
      worker_log="${WORKER_ROOT}/gpu_${gpu}_worker_${worker_slot}.log"
      worker_loop "${gpu}" "${worker_slot}" > "${worker_log}" 2>&1 &
      pid=$!
      WORKER_PIDS+=("${pid}")
      printf 'gpu=%s worker_slot=%s pid=%s\n' \
        "${gpu}" "${worker_slot}" "${pid}" >> "${WORKER_ROOT}/worker_pids.txt"
      printf '[LAUNCHED] gpu=%s worker_slot=%s pid=%s worker_log=%s\n' \
        "${gpu}" "${worker_slot}" "${pid}" "${worker_log}"
    done
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" || worker_failures=$((worker_failures + 1))
  done
  trap - HUP INT TERM

  local status_files=("${TASK_ROOT}"/*/status.txt)
  local failure_files=("${FAILURE_ROOT}"/*.failed)
  local completed=${#status_files[@]} failures=${#failure_files[@]}
  local successes=$((completed - failures)) unfinished=$((TOTAL_TASKS - completed))
  {
    printf 'completed_tasks=%s\nsuccessful_tasks=%s\nfailed_tasks=%s\n' \
      "${completed}" "${successes}" "${failures}"
    printf 'unfinished_tasks=%s\nworker_failures=%s\nfinished_at=%s\n' \
      "${unfinished}" "${worker_failures}" "$(date '+%F %T')"
  } >> "${SUMMARY_FILE}"
  printf '[DONE] completed=%s success=%s failed=%s unfinished=%s worker_failures=%s\n' \
    "${completed}" "${successes}" "${failures}" "${unfinished}" "${worker_failures}"
  ((failures == 0 && unfinished == 0 && worker_failures == 0))
}

prepare_real_run() {
  ensure_prerequisites
  mkdir -p "${BASE_PROJECT_ROOT}/.shapley_runs" "${BASE_PROJECT_ROOT}/Results"
  exec 8>"${BASE_PROJECT_ROOT}/.shapley_runs/.gpu_${GPU_IDS[0]}_campaign.lock"
  flock -n 8 || die "another Shapley campaign is already using GPU ${GPU_IDS[0]}"
  probe_gpus
  [[ ! -e "${CAMPAIGN_ROOT}" ]] || die "campaign already exists: ${CAMPAIGN_ROOT}"
  [[ ! -e "${RESULTS_ROOT}" ]] || die "Results already exists: ${RESULTS_ROOT}"
  mkdir -p "${CAMPAIGN_ROOT}" "${RESULTS_ROOT}" "${TASK_ROOT}" "${FAILURE_ROOT}" \
    "${WORKER_ROOT}" "${QUEUE_ROOT}" "${QUEUE_PENDING}" "${QUEUE_RUNNING}" "${QUEUE_COMPLETED}"
  emit_tasks > "${TASK_FILE}"
  local task_record queued_task_id queued_spec queued_spec_count=0
  while IFS= read -r task_record; do
    queued_task_id="${task_record%%$'\t'*}"
    printf -v queued_spec 'task_%06d.spec' "${queued_task_id}"
    printf '%s\n' "${task_record}" > "${QUEUE_PENDING}/${queued_spec}"
    queued_spec_count=$((queued_spec_count + 1))
  done < "${TASK_FILE}"
  [[ "${queued_spec_count}" -eq "${TOTAL_TASKS}" ]] || die 'real task queue enumeration drifted'
  {
    printf 'dataset=%s\nmode=%s\nalphas=%s\nclients=%s\nmethods=%s\n' \
      "${DATASET}" "${MODE}" "${ALPHAS[*]}" "${CLIENT_NUMS[*]}" "${METHOD_RECORDS[*]}"
    printf 'trial_ids=%s\nrounds=%s\nrandom_round=%s\nuse_hessian=%s\nlambda_h=%s\n' \
      "${TRIAL_IDS[*]}" "${NUM_EPOCH}" "${RANDOM_ROUND}" "${USE_HESSIAN}" "${LAMBDA_H}"
    printf 'gpus=%s\nworkers_per_gpu=%s\ncpu_threads_per_worker=%s\ntotal_gpu_workers=%s\n' \
      "${GPU_IDS[*]}" "${WORKERS_PER_GPU}" "${CPU_THREADS_PER_WORKER}" \
      "$(( ${#GPU_IDS[@]} * WORKERS_PER_GPU ))"
    printf 'total_tasks=%s\nresults_root=%s\nstarted_at=%s\n' \
      "${TOTAL_TASKS}" "${RESULTS_ROOT}" "$(date '+%F %T')"
    printf '%s\n' 'hessian_semantics=fixed_curvature_proxy_not_true_hvp'
  } > "${SUMMARY_FILE}"
}

main() {
  validate_configuration
  local trial_count=${#TRIAL_IDS[@]}
  TOTAL_TASKS=$((512 * trial_count))
  printf '%s\n' '=== FORECAST SHAPLEY EXPERIMENT ==='
  printf 'dataset/mode         : %s / %s\n' "${DATASET}" "${MODE}"
  printf 'alphas               : %s\n' "${ALPHAS[*]}"
  printf 'clients              : %s\n' "${CLIENT_NUMS[*]}"
  printf 'methods              : arima arima_adaptive moirai moirai_online\n'
  printf 'gpu assignment       : %s\n' "${GPU_IDS[*]}"
  printf 'GPU workers          : %s per GPU (%s total, dynamic queue)\n' \
    "${WORKERS_PER_GPU}" "$(( ${#GPU_IDS[@]} * WORKERS_PER_GPU ))"
  printf 'CPU threads/worker   : %s\n' "${CPU_THREADS_PER_WORKER}"
  printf 'total_tasks          : %s (fixed=%s probabilistic=%s)\n' \
    "${TOTAL_TASKS}" "$((256 * trial_count))" "$((256 * trial_count))"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_plan
    return 0
  fi
  [[ "${CONFIRM_SHAPLEY_RUN}" == "1" ]] || {
    printf '%s\n' 'Run is locked. Set DRY_RUN=0 CONFIRM_SHAPLEY_RUN=1 after review.' >&2
    return 2
  }
  prepare_real_run
  launch_workers
}

main "$@"

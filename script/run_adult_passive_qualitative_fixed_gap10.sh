#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT="${SCRIPT:-FLCE_updated.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MERGER_PYTHON_BIN="${MERGER_PYTHON_BIN:-python}"
FLOCK_BIN="${FLOCK_BIN:-flock}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
ATTACKER_ID="${ATTACKER_ID:-1}"
TRIAL_IDS_RAW="${TRIAL_IDS:-0}"
NUM_EPOCH="${NUM_EPOCH:-50}"
DRY_RUN="${DRY_RUN:-1}"
CONFIRM_QUALITATIVE_RUN="${CONFIRM_QUALITATIVE_RUN:-0}"
MERGE_ONLY="${MERGE_ONLY:-0}"

# Visible dataset/resource/attack configuration.
DATASET=adult
ALPHAS=(1000 100000 1000000 20000000)
PURE_ATTACKS=(moirai_online adam_local_deviation)
LAZY_ATTACKS=(arima_adaptive synthetic_replacement)
DEFAULT_GPU_IDS=2
DEFAULT_WORKERS=15
DEFAULT_THREADS=15
ARIMA_VAR_THRESHOLD=0.005
ARIMA_ERR_THRESHOLD=0.08
MOIRAI_LR=0.003
MOIRAI_ERR_THRESHOLD=0.12
DKFR_VALUES=''

GPU_IDS_RAW="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-${DEFAULT_WORKERS}}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-${DEFAULT_THREADS}}"
read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"
read -r -a TRIAL_IDS <<< "${TRIAL_IDS_RAW}"
CLIENT_NUMS=(2 4 6 8)
DETECTORS=(std_dagmm delta_dagmm wef s2_wef)

# BEGIN IDENTICAL CORE
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

validate_list() {
  local label="$1"; shift
  (($# > 0)) || die "${label} is empty"
  local value seen=' '
  for value in "$@"; do
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || die "non-canonical ${label}: ${value}"
    [[ "${seen}" != *" ${value} "* ]] || die "duplicate ${label}: ${value}"
    seen+="${value} "
  done
}

validate_configuration() {
  [[ "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || die 'WORKERS_PER_GPU must be a positive integer'
  [[ "${CPU_THREADS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]] || die 'CPU_THREADS_PER_WORKER must be a positive integer'
  [[ "${ATTACKER_ID}" == 1 ]] || die 'ATTACKER_ID must remain 1'
  [[ "${NUM_EPOCH}" == 50 ]] || die 'NUM_EPOCH must remain 50'
  [[ "${DRY_RUN}" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
  [[ "${CONFIRM_QUALITATIVE_RUN}" =~ ^[01]$ ]] || die 'CONFIRM_QUALITATIVE_RUN must be 0 or 1'
  [[ "${MERGE_ONLY}" =~ ^[01]$ ]] || die 'MERGE_ONLY must be 0 or 1'
  validate_list GPU_IDS "${GPU_IDS[@]}"
  validate_list TRIAL_IDS "${TRIAL_IDS[@]}"
  if [[ "${MERGE_ONLY}" == 1 ]]; then
    [[ -n "${CAMPAIGN_ID:-}" ]] || die 'MERGE_ONLY=1 requires CAMPAIGN_ID'
    [[ "${DRY_RUN}" == 0 ]] || die 'MERGE_ONLY=1 requires DRY_RUN=0'
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
SCRIPT_SOURCE="${SCRIPT}"
[[ "${SCRIPT_SOURCE}" = /* ]] || SCRIPT_SOURCE="${PROJECT_ROOT}/${SCRIPT_SOURCE}"
CAMPAIGN_ID="${CAMPAIGN_ID:-passive_qualitative_fixed_gap10_${DATASET}_$(date +'%Y%m%d-%H%M%S')_pid$$}"
CAMPAIGN_ROOT="${PROJECT_ROOT}/.qualitative_runs/${CAMPAIGN_ID}"
PUBLIC_ROOT="${PROJECT_ROOT}/Results/${CAMPAIGN_ID}"
PLANNED_FILE="${CAMPAIGN_ROOT}/planned.tsv"
TASK_ROOT="${CAMPAIGN_ROOT}/tasks"
QUEUE_PENDING="${CAMPAIGN_ROOT}/queue/pending"
QUEUE_RUNNING="${CAMPAIGN_ROOT}/queue/running"
QUEUE_DONE="${CAMPAIGN_ROOT}/queue/done"
FAILURE_ROOT="${CAMPAIGN_ROOT}/failures"
WORKER_ROOT="${CAMPAIGN_ROOT}/workers"
SUMMARY_FILE="${CAMPAIGN_ROOT}/launcher_summary.txt"
LOCK_ROOT="${QUALITATIVE_GPU_LOCK_ROOT:-${TMPDIR:-/tmp}/passive_qualitative_gpu_locks}"
shopt -s nullglob

emit_plan() {
  local sequence=0 trial alpha client regime attack detector task_id status reason
  local attacks=()
  for trial in "${TRIAL_IDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for client in "${CLIENT_NUMS[@]}"; do
        for regime in pure lazy; do
          [[ "${regime}" == pure ]] && attacks=("${PURE_ATTACKS[@]}") || attacks=("${LAZY_ATTACKS[@]}")
          for attack in "${attacks[@]}"; do
            for detector in "${DETECTORS[@]}"; do
              sequence=$((sequence + 1))
              task_id=$(printf 'q%06d_%s_t%s_a%s_n%s_%s_%s_%s' \
                "${sequence}" "${DATASET}" "${trial}" "${alpha}" "${client}" \
                "${regime}" "${attack}" "${detector}")
              status=planned; reason=''
              if [[ "${client}" == 2 && ( "${detector}" == wef || "${detector}" == s2_wef ) ]]; then
                status=N/A; reason=wef_family_requires_at_least_3_clients
              fi
              printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${task_id}" "${DATASET}" "${trial}" "${alpha}" "${client}" \
                "${regime}" "${attack}" "${detector}" "${status}" "${reason}"
            done
          done
        done
      done
    done
  done
}

build_command() {
  local task_id="$1" trial="$2" alpha="$3" client="$4" regime="$5" attack="$6" detector="$7"
  TASK_DIR="${TASK_ROOT}/${task_id}"
  TASK_RESULTS="${TASK_DIR}/results"
  local random_round=10
  [[ "${attack}" != advanced_scaled_delta ]] || random_round=2
  CMD=("${PYTHON_BIN}" "${SCRIPT_SOURCE}"
    --dataset "${DATASET}" --alpha "${alpha}" --client_num "${client}"
    --attacker_id "${ATTACKER_ID}" --attack_method "${attack}"
    --detecting_method "${detector}" --detecting_type qualitative
    --qualitative_aggregation_mode passive --trial_id "${trial}"
    --num_epoch "${NUM_EPOCH}" --num_local_epochs 1 --random_round "${random_round}"
    --results_root "${TASK_RESULTS}" --use_attack)
  case "${attack}" in
    arima_adaptive|arima_adaptive_mnist_imdb_layerwise)
      CMD+=(--arima_var_window 3 --arima_var_threshold "${ARIMA_VAR_THRESHOLD}"
        --arima_err_threshold "${ARIMA_ERR_THRESHOLD}" --arima_min_rounds_after_start 4
        --arima_retrain_interval 15 --arima_err_ema_beta 0.9 --arima_trim_history) ;;
    moirai_online|moirai_online_mnist_imdb_layerwise)
      CMD+=(--moirai_online_tuning --moirai_tune_lr "${MOIRAI_LR}"
        --moirai_tune_steps 1 --moirai_tune_start_round 10 --moirai_adapt_interval 4
        --moirai_residual_decay 0.95 --moirai_residual_clip 5 --moirai_err_ema_beta 0.9
        --moirai_err_threshold "${MOIRAI_ERR_THRESHOLD}"
        --moirai_min_rounds_after_start 4 --moirai_trim_history) ;;
    advanced_scaled_delta)
      CMD+=(--scaled_delta_c_const 1.0 --scaled_delta_noise_fraction 1.0
        --scaled_delta_max_noise_ratio 5.0 --scaled_delta_max_scale 10.0) ;;
    adam_local_deviation)
      CMD+=(--wang_adam_lr 8e-3 --wang_adam_beta1 0.9 --wang_adam_beta2 0.999
        --wang_adam_eps 1e-8 --adam_public_fraction 0.05 --adam_public_seed 20260726) ;;
    dkfr)
      local d=(); read -r -a d <<< "${DKFR_VALUES}"
      ((${#d[@]} == 18)) || die "missing DKFR profile for ${DATASET}"
      CMD+=(--dkfr_synth_batch_size "${d[0]}" --dkfr_generator_steps "${d[1]}"
        --dkfr_kd_steps "${d[2]}" --dkfr_lr_g "${d[3]}" --dkfr_lr_student "${d[4]}"
        --dkfr_temperature "${d[5]}" --dkfr_alpha_feat "${d[6]}" --dkfr_alpha_kl "${d[7]}"
        --dkfr_alpha_cls "${d[8]}" --dkfr_hidden_dim "${d[9]}" --dkfr_lambda_kl_g "${d[10]}"
        --dkfr_kl_cap "${d[11]}" --dkfr_student_kl_cap "${d[12]}"
        --dkfr_student_loss_cap "${d[13]}" --dkfr_teacher_conf_min "${d[14]}"
        --dkfr_grad_clip "${d[15]}" --dkfr_max_update_norm "${d[16]}"
        --dkfr_guard_reset_patience "${d[17]}")
      [[ "${DATASET}" != mnist ]] || CMD+=(--dkfr_mnist_tv 0.1
        --dkfr_mnist_pixel_reg 0.2 --dkfr_mnist_logit_reg 1e-4) ;;
    synthetic_replacement)
      CMD+=(--synthetic_mixup_alpha 0.4 --synthetic_noise_std 0.02
        --synthetic_tabular_noise_std 0.05 --synthetic_tabular_flip_prob 0.02
        --synthetic_seed 0) ;;
  esac
  case "${attack}" in
    arima_adaptive_mnist_imdb_layerwise|moirai_online_mnist_imdb_layerwise)
      CMD+=(--layerwise_selection_metric normalized_rms --layerwise_selection_fraction 0.5
        --layerwise_pca_threshold 512 --layerwise_pca_components 8) ;;
  esac
  case "${attack}" in
    arima_adaptive|arima_adaptive_mnist_imdb_layerwise|moirai_online|moirai_online_mnist_imdb_layerwise)
      CMD+=(--use_hessian --lambda_h 0.9) ;;
  esac
  [[ "${regime}" != lazy ]] || CMD+=(--use_lazy --lazy_schedule fixed
    --lazy_honest_gap 10 --lazy_honest_phase 0 --lazy_fake_policy attack)
}

task_use_hessian() {
  case "$1" in
    arima_adaptive|arima_adaptive_mnist_imdb_layerwise|moirai_online|moirai_online_mnist_imdb_layerwise) printf '1\n' ;;
    *) printf '0\n' ;;
  esac
}
task_lambda_h() { [[ "$(task_use_hessian "$1")" == 1 ]] && printf '0.9\n' || printf 'N/A\n'; }
task_disclosure() {
  case "$1" in
    advanced_scaled_delta) printf 'adapted_baseline:zhu_advanced_scaled_delta\n' ;;
    adam_local_deviation) printf 'adapted_baseline:wang_public_warm_start_local_deviation\n' ;;
    dkfr) printf 'adapted_baseline:fu_dkfr_lite\n' ;;
    synthetic_replacement) printf 'adapted_baseline:lee_synthetic_replacement\n' ;;
    arima_adaptive|arima_adaptive_mnist_imdb_layerwise) printf 'gradpred:adaptive_arima_frobenius_identity_curvature_proxy\n' ;;
    moirai_online|moirai_online_mnist_imdb_layerwise) printf 'gradpred:online_moirai_frobenius_identity_curvature_proxy\n' ;;
  esac
}

print_command() {
  printf 'command:'
  local argument
  for argument in "${CMD[@]}"; do printf ' <%s>' "${argument}"; done
  printf '\n'
}

print_dry_run() {
  printf 'CONFIG|%s|gpu=%s|workers=%s|threads=%s\n' "${DATASET}" "${GPU_IDS_RAW}" \
    "${WORKERS_PER_GPU}" "${CPU_THREADS_PER_WORKER}"
  local task_id dataset trial alpha client regime attack detector status reason
  local launched=0 unavailable=0
  while IFS=$'\t' read -r task_id dataset trial alpha client regime attack detector status reason; do
    if [[ "${status}" == N/A ]]; then
      printf 'N/A|%s|%s|%s|%s|%s|%s|%s|%s|N/A|%s\n' "${task_id}" "${dataset}" \
        "${trial}" "${alpha}" "${client}" "${regime}" "${attack}" "${detector}" "${reason}"
      unavailable=$((unavailable + 1))
    else
      printf 'TASK|%s|%s|%s|%s|%s|%s|%s|%s|planned\n' "${task_id}" "${dataset}" \
        "${trial}" "${alpha}" "${client}" "${regime}" "${attack}" "${detector}"
      build_command "${task_id}" "${trial}" "${alpha}" "${client}" "${regime}" "${attack}" "${detector}"
      print_command; launched=$((launched + 1))
    fi
  done < <(emit_plan)
  [[ "${launched}" -eq $((224 * ${#TRIAL_IDS[@]})) ]] || die "task enumeration drift"
  [[ "${unavailable}" -eq $((32 * ${#TRIAL_IDS[@]})) ]] || die "N/A enumeration drift"
  printf 'DRY_RUN|dataset=%s|launched=%s|na=%s\n' "${DATASET}" "${launched}" "${unavailable}"
}

prepare_campaign() {
  [[ ! -e "${CAMPAIGN_ROOT}" ]] || die "campaign already exists: ${CAMPAIGN_ROOT}"
  [[ ! -e "${PUBLIC_ROOT}" ]] || die "public result already exists: ${PUBLIC_ROOT}"
  mkdir -p "${TASK_ROOT}" "${QUEUE_PENDING}" "${QUEUE_RUNNING}" "${QUEUE_DONE}" \
    "${FAILURE_ROOT}" "${WORKER_ROOT}"
  { printf 'TaskID\tDataset\tTrialID\tAlpha\tClientNum\tAttackRegime\tAttackMethod\tDetectingMethod\tPlannedStatus\tReason\n'; emit_plan; } > "${PLANNED_FILE}"
  {
    printf 'Dataset\tCPUThreadsPerWorker\n'
    printf '%s\t%s\n' "${DATASET}" "${CPU_THREADS_PER_WORKER}"
  } > "${CAMPAIGN_ROOT}/campaign.tsv"
  local task_id dataset trial alpha client regime attack detector status reason
  while IFS=$'\t' read -r task_id dataset trial alpha client regime attack detector status reason; do
    [[ "${status}" != planned ]] || printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${trial}" "${alpha}" "${client}" "${regime}" "${attack}" "${detector}" \
      > "${QUEUE_PENDING}/${task_id}.spec"
  done < <(emit_plan)
  printf 'campaign_id=%s\ndataset=%s\nplanned_rows=%s\nlaunched_tasks=%s\nna_rows=%s\n' \
    "${CAMPAIGN_ID}" "${DATASET}" "$((256 * ${#TRIAL_IDS[@]}))" \
    "$((224 * ${#TRIAL_IDS[@]}))" "$((32 * ${#TRIAL_IDS[@]}))" > "${SUMMARY_FILE}"
}

GPU_LOCK_FDS=()
acquire_gpu_locks() {
  mkdir -p "${LOCK_ROOT}"
  local gpu fd
  for gpu in "${GPU_IDS[@]}"; do
    exec {fd}> "${LOCK_ROOT}/gpu_${gpu}.lock"
    "${FLOCK_BIN}" -n "${fd}" || die "GPU ${gpu} is locked"
    GPU_LOCK_FDS+=("${fd}")
  done
}

ensure_prerequisites() {
  command -v "${PYTHON_BIN}" >/dev/null || die "runner not found: ${PYTHON_BIN}"
  command -v "${MERGER_PYTHON_BIN}" >/dev/null || die "merger Python not found"
  command -v "${FLOCK_BIN}" >/dev/null || die "flock not found"
  command -v "${NVIDIA_SMI_BIN}" >/dev/null || die "nvidia-smi not found"
  command -v readlink >/dev/null || die "readlink not found"
  [[ -f "${SCRIPT_SOURCE}" ]] || die "FLCE entry not found: ${SCRIPT_SOURCE}"
  local attack_module_real attack_project_root moirai_model_dir
  attack_module_real="$(readlink -f "${PROJECT_ROOT}/module/method/attack_updated_nonlayerwise.py")"
  [[ -n "${attack_module_real}" ]] || die "cannot resolve attack module"
  attack_project_root="$(dirname "$(dirname "$(dirname "${attack_module_real}")")")"
  moirai_model_dir="${attack_project_root}/moirai-1.1-R-small"
  [[ -f "${moirai_model_dir}/config.json" ]] || die "local Moirai config not found: ${moirai_model_dir}/config.json"
  [[ -f "${moirai_model_dir}/model.safetensors" || -f "${moirai_model_dir}/pytorch_model.bin" ]] || die "local Moirai weights not found under: ${moirai_model_dir}"
}

probe_gpus() {
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    "${NVIDIA_SMI_BIN}" -i "${gpu}" --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
      | grep -qx "${gpu}" || die "GPU ${gpu} is unavailable"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -c \
      'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1' \
      >/dev/null || die "PyTorch cannot use isolated GPU ${gpu}"
  done
}

claim_next_spec() {
  local gpu="$1" slot="$2" pending base guard claimed
  for pending in "${QUEUE_PENDING}"/*.spec; do
    [[ -f "${pending}" ]] || continue
    base="${pending##*/}"; guard="${QUEUE_RUNNING}/.${base}.claim"
    mkdir "${guard}" 2>/dev/null || continue
    claimed="${QUEUE_RUNNING}/${base%.spec}.gpu_${gpu}.worker_${slot}.spec"
    if mv "${pending}" "${claimed}" 2>/dev/null; then
      rmdir "${guard}" 2>/dev/null || true; printf '%s\n' "${claimed}"; return 0
    fi
    rmdir "${guard}" 2>/dev/null || true
  done
  return 1
}

job_is_owned() {
  local wanted="${1:-}" pid
  [[ "${wanted}" =~ ^[0-9]+$ ]] || return 1
  while IFS= read -r pid; do [[ "${pid}" != "${wanted}" ]] || return 0; done < <(jobs -pr 2>/dev/null || true)
  return 1
}

terminate_owned_jobs() {
  local grace="${1:-5}" pid step=0 any_owned
  local owned=()
  while IFS= read -r pid; do
    [[ "${pid}" =~ ^[0-9]+$ ]] && owned+=("${pid}")
  done < <(jobs -pr 2>/dev/null || true)
  for pid in "${owned[@]:-}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  while ((step < grace)); do
    any_owned=0
    for pid in "${owned[@]:-}"; do job_is_owned "${pid}" && any_owned=1; done
    ((any_owned == 0)) && break
    sleep 1
    step=$((step + 1))
  done
  for pid in "${owned[@]:-}"; do
    job_is_owned "${pid}" && kill -KILL "${pid}" 2>/dev/null || true
  done
  for pid in "${owned[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}

run_one() {
  local gpu="$1" task_id="$2" trial="$3" alpha="$4" client="$5" regime="$6" attack="$7" detector="$8"
  build_command "${task_id}" "${trial}" "${alpha}" "${client}" "${regime}" "${attack}" "${detector}"
  mkdir -p "${TASK_DIR}/logs" "${TASK_DIR}/tmp" "${TASK_DIR}/cache" "${TASK_RESULTS}"
  local use_hessian lambda_h disclosure command_argv
  use_hessian=$(task_use_hessian "${attack}"); lambda_h=$(task_lambda_h "${attack}")
  disclosure=$(task_disclosure "${attack}"); command_argv=$(printf '%q ' "${CMD[@]}")
  { printf 'TaskID\tDataset\tTrialID\tAlpha\tClientNum\tAttackRegime\tAttackMethod\tDetectingMethod\tGPU\tCPUThreadsPerWorker\tUseHessian\tLambdaH\tQualitativeAggregationMode\tAccuracyControlDefinition\tBaselineAdaptationDisclosure\tCommandArgv\tResultsRoot\tCommandFile\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${task_id}" "${DATASET}" "${trial}" \
      "${alpha}" "${client}" "${regime}" "${attack}" "${detector}" "${gpu}" \
      "${CPU_THREADS_PER_WORKER}" "${use_hessian}" "${lambda_h}" passive internal_n_minus_1_remove_designated_attacker_and_shard "${disclosure}" \
      "${command_argv}" "${TASK_RESULTS}" "${TASK_DIR}/command.argv"; } > "${TASK_DIR}/manifest.tsv"
  printf '%q ' "${CMD[@]}" > "${TASK_DIR}/command.argv"; printf '\n' >> "${TASK_DIR}/command.argv"
  local code=0
  ( cd "${PROJECT_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER}" MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER}" NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="${TASK_DIR}/tmp" HF_HOME="${TASK_DIR}/cache/huggingface"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export TORCH_HOME="${TASK_DIR}/cache/torch" XDG_CACHE_HOME="${TASK_DIR}/cache/xdg"
    exec "${CMD[@]}" ) > "${TASK_DIR}/logs/run.log" 2>&1 &
  CURRENT_TASK_PID=$!; wait "${CURRENT_TASK_PID}" || code=$?; CURRENT_TASK_PID=''
  if ((code == 0)); then printf 'success\t0\t\n' > "${TASK_DIR}/status.tsv"; return 0; fi
  printf 'failed\t%s\tchild_exit_%s\n' "${code}" "${code}" > "${TASK_DIR}/status.tsv"
  : > "${FAILURE_ROOT}/${task_id}.failed"; return "${code}"
}

interrupt_worker() { trap '' HUP INT TERM; terminate_owned_jobs 2; exit 130; }
worker_loop() {
  local gpu="$1" slot="$2" claimed line task_id trial alpha client regime attack detector
  local attempted=0 failed=0
  CURRENT_TASK_PID=''; trap interrupt_worker HUP INT TERM
  while claimed=$(claim_next_spec "${gpu}" "${slot}"); do
    line=$(<"${claimed}"); IFS=$'\t' read -r task_id trial alpha client regime attack detector <<< "${line}"
    attempted=$((attempted + 1))
    run_one "${gpu}" "${task_id}" "${trial}" "${alpha}" "${client}" "${regime}" "${attack}" "${detector}" || failed=$((failed + 1))
    mv "${claimed}" "${QUEUE_DONE}/${claimed##*/}"
  done
  printf 'gpu=%s worker=%s attempted=%s failed=%s state=finished\n' "${gpu}" "${slot}" \
    "${attempted}" "${failed}" > "${WORKER_ROOT}/gpu_${gpu}_worker_${slot}.status"
}

WORKER_PIDS=()
interrupt_launcher() {
  trap '' HUP INT TERM
  terminate_owned_jobs 5
  printf 'campaign_state=interrupted\n' >> "${SUMMARY_FILE}" 2>/dev/null || true
  exit 130
}
start_workers() {
  local gpu slot pid
  for gpu in "${GPU_IDS[@]}"; do
    for ((slot=1; slot<=WORKERS_PER_GPU; slot++)); do
      worker_loop "${gpu}" "${slot}" > "${WORKER_ROOT}/gpu_${gpu}_worker_${slot}.log" 2>&1 &
      pid=$!; WORKER_PIDS+=("${pid}")
      printf 'gpu=%s worker=%s pid=%s\n' "${gpu}" "${slot}" "${pid}" >> "${WORKER_ROOT}/worker_pids.txt"
    done
  done
}
wait_for_workers() {
  local pid failures=0
  for pid in "${WORKER_PIDS[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
  printf 'worker_failures=%s\n' "${failures}" >> "${SUMMARY_FILE}"; return "${failures}"
}

# Single-process merger; workers never write a shared CSV.
run_merger() {
  set +e
  "${MERGER_PYTHON_BIN}" - "${CAMPAIGN_ROOT}" "${PUBLIC_ROOT}" "${DATASET}" "${NUM_EPOCH}" <<'PY_MERGER'
import csv, math, os, pathlib, shlex, shutil, sys
c, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dataset, epochs = sys.argv[3], int(sys.argv[4])
stage = out.parent / ("." + out.name + ".staging")
PLAN=["TaskID","Dataset","TrialID","Alpha","ClientNum","AttackRegime","AttackMethod","DetectingMethod","PlannedStatus","Reason"]
CAMPAIGN=["Dataset","CPUThreadsPerWorker"]
MAN=["TaskID","Dataset","TrialID","Alpha","ClientNum","AttackRegime","AttackMethod","DetectingMethod","GPU","CPUThreadsPerWorker","UseHessian","LambdaH","QualitativeAggregationMode","AccuracyControlDefinition","BaselineAdaptationDisclosure","CommandArgv","ResultsRoot","CommandFile"]
ACC=["Dataset","AttackMethod","Alpha","ClientNum","AttackerID","TrialID","UseAttack","DetectingType","DetectingMethod","UseHessian","UseLazy","LazySchedule","AttackProbability","LazyHonestGap","LazyHonestPhase","LazyFakePolicy","Round","WithAttackerAccuracy","WithoutAttackerAccuracy"]
DET=["TaskID","CampaignComplete","Dataset","AttackMethod","AttackRegime","Alpha","ClientNum","AttackerID","TrialID","UseHessian","LambdaH","UseLazy","LazySchedule","LazyHonestGap","LazyHonestPhase","LazyFakePolicy","DetectingMethod","QualitativeAggregationMode","Round","ClientID","IsDesignatedAttacker","GroundTruthActiveAttack","DesignatedAttackerRoundState","RoundHasActiveAttack","RoundIsHonestForDesignatedAttacker","WEFPayloadSource","DetectionSkipped","DetectorScore","DetectorThreshold","DeltaThresholdPathology","RoundPredictedSuspicious","FinalPredictedSuspicious","DecisionMode","DetectorDetailsJSON"]
TIME=["TaskID","CampaignComplete","Dataset","AttackMethod","AttackRegime","Alpha","ClientNum","AttackerID","TrialID","UseHessian","LambdaH","UseLazy","LazySchedule","LazyHonestGap","LazyHonestPhase","LazyFakePolicy","DetectingMethod","QualitativeAggregationMode","AttackGenerationWithAttackerTotalSec","FakeRoundsTimed","DetectorDecisionWithAttackerTotalSec","DetectorDecisionWithoutAttackerTotalSec","DetectorDecisionWithoutAttackerMeasured","DetectorDecisionWithoutAttackerNotMeasuredReason","FLWithAttackerWallSec","FLWithoutAttackerWallSec","TaskWallSec"]
PUBLIC_ACC=["TaskID","CampaignComplete","AttackRegime","LambdaH","QualitativeAggregationMode","AccuracyControlDefinition"]+ACC
AUD=["TaskID","CampaignComplete","Dataset","TrialID","Alpha","ClientNum","AttackRegime","AttackMethod","DetectingMethod","GPU","CPUThreadsPerWorker","UseHessian","LambdaH","QualitativeAggregationMode","AccuracyControlDefinition","BaselineAdaptationDisclosure","CommandArgv","Status","Reason"]
HESSIAN={"arima_adaptive","arima_adaptive_mnist_imdb_layerwise","moirai_online","moirai_online_mnist_imdb_layerwise"}
DISCLOSURE={
 "advanced_scaled_delta":"adapted_baseline:zhu_advanced_scaled_delta",
 "adam_local_deviation":"adapted_baseline:wang_public_warm_start_local_deviation",
 "dkfr":"adapted_baseline:fu_dkfr_lite",
 "synthetic_replacement":"adapted_baseline:lee_synthetic_replacement",
 "arima_adaptive":"gradpred:adaptive_arima_frobenius_identity_curvature_proxy",
 "arima_adaptive_mnist_imdb_layerwise":"gradpred:adaptive_arima_frobenius_identity_curvature_proxy",
 "moirai_online":"gradpred:online_moirai_frobenius_identity_curvature_proxy",
 "moirai_online_mnist_imdb_layerwise":"gradpred:online_moirai_frobenius_identity_curvature_proxy",
}
PROFILES={
 "tictactoe":(["10","100","1000","20000"],["moirai_online","advanced_scaled_delta"],["moirai_online","synthetic_replacement"]),
 "adult":(["1000","100000","1000000","20000000"],["moirai_online","adam_local_deviation"],["arima_adaptive","synthetic_replacement"]),
 "dota2":(["10000","1000000","50000000","200000000"],["arima_adaptive","dkfr"],["synthetic_replacement","moirai_online"]),
 "mnist":(["10","1000","100000","10000000"],["moirai_online_mnist_imdb_layerwise","dkfr"],["synthetic_replacement","moirai_online_mnist_imdb_layerwise"]),
 "imdb":(["10","100","1000","200000"],["dkfr","moirai_online_mnist_imdb_layerwise"],["synthetic_replacement","arima_adaptive_mnist_imdb_layerwise"]),
}
CONTROL="internal_n_minus_1_remove_designated_attacker_and_shard"
ARIMA_PROFILE={"tictactoe":["0.005","0.016"],"adult":["0.005","0.08"],"dota2":["0.005","0.18"],"mnist":["0.05","0.05"],"imdb":["0.10","0.015"]}
MOIRAI_PROFILE={"tictactoe":["0.05","0.025"],"adult":["0.003","0.12"],"dota2":["0.001","0.25"],"mnist":["0.003","0.06"],"imdb":["0.01","0.025"]}
DKFR_PROFILE={
 "dota2":"32 2 2 5e-4 5e-4 2 0.1 1 1 128 1 0 0 0 0 5 0 3".split(),
 "mnist":"32 2 2 1e-5 5e-6 12 0 0.05 1 128 0 1 5 8 0.15 0.2 0.1 2".split(),
 "imdb":"32 2 2 5e-5 5e-5 4 0 0.1 1 128 0 1 2 4 0 0.5 0.1 2".split(),
}

def read(path, header):
    if not path.is_file(): raise ValueError("missing artifact: "+str(path))
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f, delimiter="\t" if path.suffix==".tsv" else ",")
        if r.fieldnames != header: raise ValueError("malformed header: "+str(path))
        return list(r)
def num(a,b):
    try: return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12)
    except (TypeError,ValueError): return str(a)==str(b)
def truth(v): return str(v).lower() in {"1","true","yes"}
def expected_hessian(p): return p["AttackMethod"] in HESSIAN
def binary(v,name):
    text=str(v).lower()
    if text not in {"0","1","false","true"}: raise ValueError(name+" is not binary")
    return text in {"1","true"}
def expected_command(p,task,executable,script_source,results_root):
    attack=p["AttackMethod"]
    args=[executable,script_source,
      "--dataset",p["Dataset"],"--alpha",p["Alpha"],"--client_num",p["ClientNum"],
      "--attacker_id","1","--attack_method",attack,"--detecting_method",p["DetectingMethod"],
      "--detecting_type","qualitative","--qualitative_aggregation_mode","passive",
      "--trial_id",p["TrialID"],"--num_epoch",str(epochs),"--num_local_epochs","1",
      "--random_round","2" if attack=="advanced_scaled_delta" else "10",
      "--results_root",results_root,"--use_attack"]
    if attack.startswith("arima_adaptive"):
        var,err=ARIMA_PROFILE[p["Dataset"]]
        args += ["--arima_var_window","3","--arima_var_threshold",var,"--arima_err_threshold",err,
          "--arima_min_rounds_after_start","4","--arima_retrain_interval","15",
          "--arima_err_ema_beta","0.9","--arima_trim_history"]
    elif attack.startswith("moirai_online"):
        lr,err=MOIRAI_PROFILE[p["Dataset"]]
        args += ["--moirai_online_tuning","--moirai_tune_lr",lr,"--moirai_tune_steps","1",
          "--moirai_tune_start_round","10","--moirai_adapt_interval","4","--moirai_residual_decay","0.95",
          "--moirai_residual_clip","5","--moirai_err_ema_beta","0.9","--moirai_err_threshold",err,
          "--moirai_min_rounds_after_start","4","--moirai_trim_history"]
    elif attack=="advanced_scaled_delta":
        args += ["--scaled_delta_c_const","1.0","--scaled_delta_noise_fraction","1.0",
          "--scaled_delta_max_noise_ratio","5.0","--scaled_delta_max_scale","10.0"]
    elif attack=="adam_local_deviation":
        args += ["--wang_adam_lr","8e-3","--wang_adam_beta1","0.9","--wang_adam_beta2","0.999",
          "--wang_adam_eps","1e-8","--adam_public_fraction","0.05","--adam_public_seed","20260726"]
    elif attack=="dkfr":
        names=["--dkfr_synth_batch_size","--dkfr_generator_steps","--dkfr_kd_steps","--dkfr_lr_g",
          "--dkfr_lr_student","--dkfr_temperature","--dkfr_alpha_feat","--dkfr_alpha_kl","--dkfr_alpha_cls",
          "--dkfr_hidden_dim","--dkfr_lambda_kl_g","--dkfr_kl_cap","--dkfr_student_kl_cap",
          "--dkfr_student_loss_cap","--dkfr_teacher_conf_min","--dkfr_grad_clip",
          "--dkfr_max_update_norm","--dkfr_guard_reset_patience"]
        for name,value in zip(names,DKFR_PROFILE[p["Dataset"]]): args += [name,value]
        if p["Dataset"]=="mnist":
            args += ["--dkfr_mnist_tv","0.1","--dkfr_mnist_pixel_reg","0.2","--dkfr_mnist_logit_reg","1e-4"]
    elif attack=="synthetic_replacement":
        args += ["--synthetic_mixup_alpha","0.4","--synthetic_noise_std","0.02",
          "--synthetic_tabular_noise_std","0.05","--synthetic_tabular_flip_prob","0.02","--synthetic_seed","0"]
    if attack.endswith("_mnist_imdb_layerwise"):
        args += ["--layerwise_selection_metric","normalized_rms","--layerwise_selection_fraction","0.5",
          "--layerwise_pca_threshold","512","--layerwise_pca_components","8"]
    if expected_hessian(p): args += ["--use_hessian","--lambda_h","0.9"]
    if p["AttackRegime"]=="lazy":
        args += ["--use_lazy","--lazy_schedule","fixed","--lazy_honest_gap","10",
          "--lazy_honest_phase","0","--lazy_fake_policy","attack"]
    return args
def check_truth(row,p):
    round_id,client_id=int(row["Round"]),int(row["ClientID"])
    designated=client_id==1
    if p["AttackRegime"]=="lazy":
        honest=round_id%10==0; active=not honest; state="honest" if honest else "fake"
    elif p["Dataset"]=="adult" and p["AttackMethod"]=="adam_local_deviation" and round_id==0:
        honest=False; active=False; state="public_warm_start"
    else:
        honest=False; active=True; state="fake"
    expected={"IsDesignatedAttacker":designated,"GroundTruthActiveAttack":designated and active,
      "RoundHasActiveAttack":active,"RoundIsHonestForDesignatedAttacker":honest}
    for name,value in expected.items():
        if binary(row.get(name),name)!=value: raise ValueError(name+" mismatch")
    if row.get("DesignatedAttackerRoundState")!=state: raise ValueError("attack state mismatch")
    wef=p["DetectingMethod"] in {"wef","s2_wef"}
    source="counterfeit_from_submitted_update" if wef and designated and active and p["AttackMethod"]!="synthetic_replacement" else "observed_local_training" if wef else "N/A"
    if row.get("WEFPayloadSource")!=source: raise ValueError("WEFPayloadSource mismatch")
    skipped=p["DetectingMethod"]=="s2_wef" and round_id==0
    if binary(row.get("DetectionSkipped"),"DetectionSkipped")!=skipped: raise ValueError("DetectionSkipped mismatch")
    if skipped and any(row.get(name)!="" for name in ("DetectorScore","DetectorThreshold","RoundPredictedSuspicious","FinalPredictedSuspicious")):
        raise ValueError("S2 skipped-round outputs must be empty")
    binary(row.get("DeltaThresholdPathology"),"DeltaThresholdPathology")
def check(row,p,regime):
    fields=["Dataset","AttackMethod","ClientNum","TrialID","DetectingMethod"]
    if regime: fields.append("AttackRegime")
    for k in fields:
        if str(row.get(k,"")) != p[k]: raise ValueError(k+" mismatch for "+p["TaskID"])
    if str(row.get("AttackerID"))!="1" or not num(row.get("Alpha"),p["Alpha"]):
        raise ValueError("configuration mismatch for "+p["TaskID"])
    if "UseAttack" in row and not truth(row.get("UseAttack")):
        raise ValueError("UseAttack mismatch for "+p["TaskID"])
    if "DetectingType" in row and row.get("DetectingType")!="qualitative":
        raise ValueError("DetectingType mismatch for "+p["TaskID"])
    if binary(row.get("UseHessian"),"UseHessian") != expected_hessian(p):
        raise ValueError("UseHessian mismatch for "+p["TaskID"])
    if "LambdaH" in row and str(row.get("LambdaH")) != ("0.9" if expected_hessian(p) else "N/A"):
        raise ValueError("LambdaH mismatch for "+p["TaskID"])
    lazy=p["AttackRegime"]=="lazy"
    if binary(row.get("UseLazy"),"UseLazy") != lazy: raise ValueError("UseLazy mismatch")
    if lazy:
        for k,v in {"LazySchedule":"fixed","LazyHonestGap":"10","LazyHonestPhase":"0","LazyFakePolicy":"attack"}.items():
            if str(row.get(k)) != v: raise ValueError(k+" mismatch")
    if "GroundTruthActiveAttack" in row: check_truth(row,p)
def check_manifest(m,p,cpu_threads,task):
    for k in PLAN[:8]:
        if m[k]!=p[k]: raise ValueError("manifest mismatch "+k)
    if not m["GPU"].isdigit() or m["CPUThreadsPerWorker"]!=cpu_threads:
        raise ValueError("manifest resource mismatch")
    use_hessian="1" if expected_hessian(p) else "0"
    if m["UseHessian"]!=use_hessian or m["LambdaH"]!=("0.9" if expected_hessian(p) else "N/A"):
        raise ValueError("manifest Hessian mismatch")
    if m["QualitativeAggregationMode"]!="passive" or m["AccuracyControlDefinition"]!=CONTROL or m["BaselineAdaptationDisclosure"]!=DISCLOSURE[p["AttackMethod"]] or not m["CommandArgv"]:
        raise ValueError("manifest disclosure mismatch")
    results_root=pathlib.PurePosixPath(m["ResultsRoot"].replace("\\","/"))
    command_file=pathlib.PurePosixPath(m["CommandFile"].replace("\\","/"))
    if results_root.name!="results" or results_root.parent.name!=p["TaskID"] or command_file!=results_root.parent/"command.argv":
        raise ValueError("manifest private path mismatch")
    persisted_command=task/"command.argv"
    if not persisted_command.is_file() or persisted_command.read_text(encoding="utf-8").rstrip("\r\n")!=m["CommandArgv"]:
        raise ValueError("manifest command file mismatch")
    argv=shlex.split(m["CommandArgv"],posix=True)
    if len(argv)<2: raise ValueError("manifest argv mismatch")
    script_source=pathlib.PurePosixPath(argv[1].replace("\\","/"))
    if script_source!=results_root.parents[4]/"FLCE_updated.py": raise ValueError("manifest script path mismatch")
    if argv!=expected_command(p,task,argv[0],argv[1],m["ResultsRoot"]): raise ValueError("manifest argv mismatch")
def write(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=header,extrasaction="raise")
        w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())

try:
    if out.exists(): raise ValueError("public result already exists")
    if stage.exists(): shutil.rmtree(stage)
    plans=read(c/"planned.tsv",PLAN)
    if not plans: raise ValueError("empty campaign plan")
    campaign_rows=read(c/"campaign.tsv",CAMPAIGN)
    if len(campaign_rows)!=1 or campaign_rows[0]["Dataset"]!=dataset:
        raise ValueError("campaign configuration mismatch")
    cpu_threads=campaign_rows[0]["CPUThreadsPerWorker"]
    if not cpu_threads.isdigit() or int(cpu_threads)<1: raise ValueError("invalid campaign CPU threads")
    tids=[p["TaskID"] for p in plans]
    keys=[tuple(p[k] for k in PLAN[1:8]) for p in plans]
    trials={p["TrialID"] for p in plans}
    if not trials: raise ValueError("empty campaign trials")
    if len(tids)!=len(set(tids)) or len(keys)!=len(set(keys)): raise ValueError("duplicate logical task key")
    if any(p["Dataset"]!=dataset for p in plans): raise ValueError("dataset mismatch")
    alphas,pure_attacks,lazy_attacks=PROFILES[dataset]
    expected={}
    for trial in trials:
        for alpha in alphas:
            for clients in ("2","4","6","8"):
                for regime,attacks in (("pure",pure_attacks),("lazy",lazy_attacks)):
                    for attack in attacks:
                        for detector in ("std_dagmm","delta_dagmm","wef","s2_wef"):
                            key=(dataset,trial,alpha,clients,regime,attack,detector)
                            is_na=clients=="2" and detector in {"wef","s2_wef"}
                            expected[key]=("N/A","wef_family_requires_at_least_3_clients") if is_na else ("planned","")
    observed={tuple(p[k] for k in PLAN[1:8]):(p["PlannedStatus"],p["Reason"]) for p in plans}
    if observed!=expected: raise ValueError("plan is not the exact canonical matrix")
    if len(plans)!=256*len(trials): raise ValueError("wrong planned row count")
    if sum(p["PlannedStatus"]=="planned" for p in plans)!=224*len(trials): raise ValueError("wrong task count")
    if sum(p["PlannedStatus"]=="N/A" for p in plans)!=32*len(trials): raise ValueError("wrong N/A count")
    states={}
    for p in plans:
        tid=p["TaskID"]
        if p["PlannedStatus"]=="N/A": states[tid]=("N/A",p["Reason"]); continue
        status=c/"tasks"/tid/"status.tsv"
        if not status.is_file(): states[tid]=("missing","status_missing"); continue
        parts=status.read_text(encoding="utf-8").strip().split("\t")
        states[tid]=(parts[0],parts[2] if len(parts)>2 else "")
    # A child exit code of zero is provisional until its private artifacts pass protocol validation.
    for p in plans:
        tid=p["TaskID"]
        if states[tid][0]!="success": continue
        try:
            task=c/"tasks"/tid
            mans=read(task/"manifest.tsv",MAN)
            if len(mans)!=1: raise ValueError("manifest row count")
            m=mans[0]
            check_manifest(m,p,cpu_threads,task)
            root=task/"results"; method=p["DetectingMethod"]; trial=p["TrialID"]
            ars=read(root/("Accuracy_qualitative_"+method)/("global_accuracy_log_trial"+trial+".csv"),ACC)
            drs=read(root/("Detecting_qualitative_"+method)/("round_client_detection_trial"+trial+".csv"),DET)
            trs=read(root/("Time_qualitative_"+method)/("attacking_time_log_trial"+trial+".csv"),TIME)
            expected_raw_task_id="{}_{}_{}_alpha{}_clients{}_attacker1_trial{}".format(
              p["Dataset"],p["AttackMethod"],p["DetectingMethod"],str(float(p["Alpha"])),p["ClientNum"],p["TrialID"])
            raw_task_ids={r["TaskID"] for r in drs+trs}
            if raw_task_ids!={expected_raw_task_id}: raise ValueError("raw TaskID mismatch")
            clients=int(p["ClientNum"])
            if len(ars)!=epochs or len(drs)!=epochs*clients or len(trs)!=1: raise ValueError("wrong artifact row count")
            if {int(r["Round"]) for r in ars}!=set(range(epochs)): raise ValueError("wrong accuracy rounds")
            want={(x,y) for x in range(epochs) for y in range(clients)}
            if {(int(r["Round"]),int(r["ClientID"])) for r in drs}!=want: raise ValueError("wrong detection keys")
            for r in ars: check(r,p,False)
            for r in drs:
                check(r,p,True)
                if r["QualitativeAggregationMode"]!="passive": raise ValueError("non-passive detection row")
            for r in trs:
                check(r,p,True)
                if r["QualitativeAggregationMode"]!="passive": raise ValueError("non-passive time row")
        except Exception as error:
            states[tid]=("failed","protocol_invalid:"+str(error))
    complete=all(states[p["TaskID"]][0]=="success" for p in plans if p["PlannedStatus"]=="planned")
    flag="1" if complete else "0"
    acc=[]; det=[]; timing=[]; ak=set(); dk=set(); tk=set()
    for p in plans:
        tid=p["TaskID"]
        if states[tid][0]!="success": continue
        task=c/"tasks"/tid
        mans=read(task/"manifest.tsv",MAN)
        if len(mans)!=1: raise ValueError("manifest row count")
        m=mans[0]
        check_manifest(m,p,cpu_threads,task)
        root=task/"results"; method=p["DetectingMethod"]; trial=p["TrialID"]
        ars=read(root/("Accuracy_qualitative_"+method)/("global_accuracy_log_trial"+trial+".csv"),ACC)
        drs=read(root/("Detecting_qualitative_"+method)/("round_client_detection_trial"+trial+".csv"),DET)
        trs=read(root/("Time_qualitative_"+method)/("attacking_time_log_trial"+trial+".csv"),TIME)
        expected_raw_task_id="{}_{}_{}_alpha{}_clients{}_attacker1_trial{}".format(
          p["Dataset"],p["AttackMethod"],p["DetectingMethod"],str(float(p["Alpha"])),p["ClientNum"],p["TrialID"])
        raw_task_ids={r["TaskID"] for r in drs+trs}
        if raw_task_ids!={expected_raw_task_id}: raise ValueError("raw TaskID mismatch")
        clients=int(p["ClientNum"])
        if len(ars)!=epochs or len(drs)!=epochs*clients or len(trs)!=1: raise ValueError("wrong artifact row count")
        if {int(r["Round"]) for r in ars}!=set(range(epochs)): raise ValueError("wrong accuracy rounds")
        want={(x,y) for x in range(epochs) for y in range(clients)}
        if {(int(r["Round"]),int(r["ClientID"])) for r in drs}!=want: raise ValueError("wrong detection keys")
        for r in ars:
            check(r,p,False); key=(tid,int(r["Round"]))
            if key in ak: raise ValueError("duplicate accuracy key")
            ak.add(key); acc.append({"TaskID":tid,"CampaignComplete":flag,
              "AttackRegime":p["AttackRegime"],
              "LambdaH":"0.9" if expected_hessian(p) else "N/A",
              "QualitativeAggregationMode":"passive","AccuracyControlDefinition":CONTROL,**r})
        for r in drs:
            check(r,p,True)
            if r["QualitativeAggregationMode"]!="passive": raise ValueError("non-passive row")
            key=(tid,int(r["Round"]),int(r["ClientID"]))
            if key in dk: raise ValueError("duplicate detection key")
            dk.add(key); r["TaskID"]=tid; r["CampaignComplete"]=flag; det.append(r)
        for r in trs:
            check(r,p,True)
            if r["QualitativeAggregationMode"]!="passive" or tid in tk: raise ValueError("bad time row")
            tk.add(tid); r["TaskID"]=tid; r["CampaignComplete"]=flag; timing.append(r)
    audit=[]
    for p in plans:
        tid=p["TaskID"]; manifest={}
        manifest_path=c/"tasks"/tid/"manifest.tsv"
        if manifest_path.is_file():
            try:
                manifest_rows=read(manifest_path,MAN)
                if len(manifest_rows)==1: manifest=manifest_rows[0]
            except Exception:
                manifest={}
        audit.append({"TaskID":tid,"CampaignComplete":flag,"Dataset":p["Dataset"],
          "TrialID":p["TrialID"],"Alpha":p["Alpha"],"ClientNum":p["ClientNum"],
          "AttackRegime":p["AttackRegime"],"AttackMethod":p["AttackMethod"],
          "DetectingMethod":p["DetectingMethod"],"GPU":manifest.get("GPU","N/A"),
          "CPUThreadsPerWorker":cpu_threads,
          "UseHessian":"1" if expected_hessian(p) else "0",
          "LambdaH":"0.9" if expected_hessian(p) else "N/A",
          "QualitativeAggregationMode":"passive",
          "AccuracyControlDefinition":CONTROL,
          "BaselineAdaptationDisclosure":manifest.get("BaselineAdaptationDisclosure",DISCLOSURE[p["AttackMethod"]]),
          "CommandArgv":manifest.get("CommandArgv","N/A"),
          "Status":states[tid][0],"Reason":states[tid][1]})
    acc.sort(key=lambda r:(r["TaskID"],int(r["Round"])))
    det.sort(key=lambda r:(r["TaskID"],int(r["Round"]),int(r["ClientID"])))
    timing.sort(key=lambda r:r["TaskID"]); audit.sort(key=lambda r:r["TaskID"])
    stage.mkdir(parents=True)
    write(stage/"accuracy"/"accuracy.csv",PUBLIC_ACC,acc)
    write(stage/"time"/"time.csv",TIME,timing)
    write(stage/"detection"/"detection.csv",DET,det)
    write(stage/"audit"/"tasks.csv",AUD,audit)
    out.parent.mkdir(parents=True,exist_ok=True); os.replace(stage,out)
    print("MERGED|complete=%s|accuracy=%d|time=%d|detection=%d|audit=%d" %
      (flag,len(acc),len(timing),len(det),len(audit)))
    sys.exit(0 if complete else 4)
except Exception as e:
    if stage.exists(): shutil.rmtree(stage)
    print("MERGE_ERROR: "+str(e),file=sys.stderr); sys.exit(3)
PY_MERGER
  local status=$?
  set -e
  return "${status}"
}

main() {
  validate_configuration
  if [[ "${DRY_RUN}" == 1 ]]; then print_dry_run; return 0; fi
  if [[ "${MERGE_ONLY}" == 1 ]]; then
    [[ -d "${CAMPAIGN_ROOT}" ]] || die "campaign not found: ${CAMPAIGN_ROOT}"
    run_merger; return $?
  fi
  if [[ "${CONFIRM_QUALITATIVE_RUN}" != 1 ]]; then
    printf '%s\n' 'Run is locked. Set DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1.' >&2
    return 2
  fi
  ensure_prerequisites; acquire_gpu_locks; probe_gpus; prepare_campaign
  trap interrupt_launcher HUP INT TERM
  start_workers
  local worker_status=0 merge_status=0 failure_count
  wait_for_workers || worker_status=$?
  trap - HUP INT TERM
  failure_count=$(find "${FAILURE_ROOT}" -type f -name '*.failed' | wc -l | tr -d ' ')
  printf 'failed_tasks=%s\n' "${failure_count}" >> "${SUMMARY_FILE}"
  run_merger || merge_status=$?
  ((worker_status==0 && failure_count==0 && merge_status==0))
}
main "$@"
# END IDENTICAL CORE
# CORE EOF

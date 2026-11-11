#!/usr/bin/env bash
# Standalone launcher: all campaign, worker, GPU, and merge logic is in this file.
# No shared launcher include is required.
set -Eeuo pipefail

SAMPLING_DATASET=adult
SAMPLING_METHOD=sampling_least_core
DEFAULT_GPU_IDS=2
DEFAULT_WORKERS=10
DEFAULT_THREADS=15

: "${SAMPLING_DATASET:?standalone launcher must set SAMPLING_DATASET}"
: "${SAMPLING_METHOD:?standalone launcher must set SAMPLING_METHOD}"

SCRIPT="${SCRIPT:-FLCE_updated.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MERGER_PYTHON_BIN="${MERGER_PYTHON_BIN:-python}"
FLOCK_BIN="${FLOCK_BIN:-flock}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
TRIAL_IDS_RAW="${TRIAL_IDS:-0}"
NUM_EPOCH="${NUM_EPOCH:-50}"
NUM_LOCAL_EPOCHS="${NUM_LOCAL_EPOCHS:-1}"
RANDOM_ROUND_OVERRIDE="${RANDOM_ROUND:-}"
USE_LAST_ROUND_SV="${USE_LAST_ROUND_SV:-1}"
DRY_RUN="${DRY_RUN:-1}"
DRY_RUN_PRINT_LIMIT="${DRY_RUN_PRINT_LIMIT:-5}"
CONFIRM_SAMPLING_RUN="${CONFIRM_SAMPLING_RUN:-0}"
MERGE_ONLY="${MERGE_ONLY:-0}"
LAZY_HONEST_PHASE="${LAZY_HONEST_PHASE:-0}"
LAZY_FAKE_POLICY="${LAZY_FAKE_POLICY:-attack}"

# Sampling controls.  Empty budget/seed values intentionally mean "let the
# runner resolve its current automatic budget/deterministic trial seed".
SAMPLING_SHAPLEY_NUM_PERMUTATIONS="${SAMPLING_SHAPLEY_NUM_PERMUTATIONS:-}"
SAMPLING_SHAPLEY_SEED="${SAMPLING_SHAPLEY_SEED:-}"
SAMPLING_SHAPLEY_TOLERANCE="${SAMPLING_SHAPLEY_TOLERANCE:-1e-6}"
SAMPLING_LEAST_CORE_NUM_COALITIONS="${SAMPLING_LEAST_CORE_NUM_COALITIONS:-}"
SAMPLING_LEAST_CORE_SEED="${SAMPLING_LEAST_CORE_SEED:-}"

DATASET="${SAMPLING_DATASET}"

# Fixed protocol shared with the qualitative small experiments.
CLIENT_NUMS=(2 4 6 8)
LAZY_GAPS=(3 6 10)

# The standalone header supplies these three resource defaults.  They remain
# environment-overridable through GPU_IDS, WORKERS_PER_GPU, and
# CPU_THREADS_PER_WORKER, exactly as in the qualitative launchers.
DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-}"
DEFAULT_WORKERS="${DEFAULT_WORKERS:-}"
DEFAULT_THREADS="${DEFAULT_THREADS:-}"

case "${DATASET}" in
  tictactoe)
    ALPHAS=(10 100 1000 20000)
    PURE_ATTACKS=(moirai_online advanced_scaled_delta)
    LAZY_ATTACKS=(moirai_online synthetic_replacement)
    DATASET_DEFAULT_GPU_IDS=1
    DATASET_DEFAULT_WORKERS=15
    DATASET_DEFAULT_THREADS=15
    ARIMA_VAR_THRESHOLD=0.005
    ARIMA_ERR_THRESHOLD=0.016
    MOIRAI_LR=0.05
    MOIRAI_ERR_THRESHOLD=0.025
    DKFR_VALUES=''
    ;;
  adult)
    ALPHAS=(1000 100000 1000000 20000000)
    PURE_ATTACKS=(moirai_online adam_local_deviation)
    LAZY_ATTACKS=(arima_adaptive synthetic_replacement)
    DATASET_DEFAULT_GPU_IDS=2
    DATASET_DEFAULT_WORKERS=15
    DATASET_DEFAULT_THREADS=15
    ARIMA_VAR_THRESHOLD=0.005
    ARIMA_ERR_THRESHOLD=0.08
    MOIRAI_LR=0.003
    MOIRAI_ERR_THRESHOLD=0.12
    DKFR_VALUES=''
    ;;
  dota2)
    ALPHAS=(10000 1000000 50000000 200000000)
    PURE_ATTACKS=(arima_adaptive dkfr)
    LAZY_ATTACKS=(synthetic_replacement moirai_online)
    DATASET_DEFAULT_GPU_IDS=3
    DATASET_DEFAULT_WORKERS=15
    DATASET_DEFAULT_THREADS=15
    ARIMA_VAR_THRESHOLD=0.005
    ARIMA_ERR_THRESHOLD=0.18
    MOIRAI_LR=0.001
    MOIRAI_ERR_THRESHOLD=0.25
    DKFR_VALUES='32 2 2 5e-4 5e-4 2 0.1 1 1 128 1 0 0 0 0 5 0 3'
    ;;
  mnist)
    ALPHAS=(10 1000 100000 10000000)
    PURE_ATTACKS=(moirai_online_mnist_imdb_layerwise dkfr)
    LAZY_ATTACKS=(synthetic_replacement moirai_online_mnist_imdb_layerwise)
    DATASET_DEFAULT_GPU_IDS=0
    DATASET_DEFAULT_WORKERS=12
    DATASET_DEFAULT_THREADS=10
    ARIMA_VAR_THRESHOLD=0.05
    ARIMA_ERR_THRESHOLD=0.05
    MOIRAI_LR=0.003
    MOIRAI_ERR_THRESHOLD=0.06
    DKFR_VALUES='32 2 2 1e-5 5e-6 12 0 0.05 1 128 0 1 5 8 0.15 0.2 0.1 2'
    ;;
  imdb)
    ALPHAS=(10 100 1000 200000)
    PURE_ATTACKS=(dkfr moirai_online_mnist_imdb_layerwise)
    LAZY_ATTACKS=(synthetic_replacement arima_adaptive_mnist_imdb_layerwise)
    DATASET_DEFAULT_GPU_IDS=1
    DATASET_DEFAULT_WORKERS=12
    DATASET_DEFAULT_THREADS=10
    ARIMA_VAR_THRESHOLD=0.10
    ARIMA_ERR_THRESHOLD=0.015
    MOIRAI_LR=0.01
    MOIRAI_ERR_THRESHOLD=0.025
    DKFR_VALUES='32 2 2 5e-5 5e-5 4 0 0.1 1 128 0 1 2 4 0 0.5 0.1 2'
    ;;
  *)
    printf 'ERROR: unsupported dataset: %s\n' "${DATASET}" >&2
    exit 1
    ;;
esac

DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-${DATASET_DEFAULT_GPU_IDS}}"
DEFAULT_WORKERS="${DEFAULT_WORKERS:-${DATASET_DEFAULT_WORKERS}}"
DEFAULT_THREADS="${DEFAULT_THREADS:-${DATASET_DEFAULT_THREADS}}"
GPU_IDS_RAW="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-${DEFAULT_WORKERS}}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-${DEFAULT_THREADS}}"
read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"
read -r -a TRIAL_IDS <<< "${TRIAL_IDS_RAW}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
SCRIPT_SOURCE="${SCRIPT}"
[[ "${SCRIPT_SOURCE}" = /* ]] || SCRIPT_SOURCE="${PROJECT_ROOT}/${SCRIPT_SOURCE}"
MERGER_SCRIPT="${SCRIPT_DIR}/merge_quantitative_sampling_results.py"

CAMPAIGN_ID_INPUT="${CAMPAIGN_ID:-}"
CAMPAIGN_ID="${CAMPAIGN_ID:-sampling_${SAMPLING_METHOD}_${DATASET}_$(date +'%Y%m%d-%H%M%S')_pid$$}"
CAMPAIGN_PARENT="${CAMPAIGN_PARENT:-${PROJECT_ROOT}/.sampling_quantitative_runs}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${CAMPAIGN_PARENT}/${CAMPAIGN_ID}}"
PUBLIC_ROOT="${PUBLIC_ROOT:-${PROJECT_ROOT}/Results/${CAMPAIGN_ID}}"
PLANNED_FILE="${CAMPAIGN_ROOT}/planned.tsv"
TASK_ROOT="${CAMPAIGN_ROOT}/tasks"
QUEUE_PENDING="${CAMPAIGN_ROOT}/queue/pending"
QUEUE_RUNNING="${CAMPAIGN_ROOT}/queue/running"
QUEUE_DONE="${CAMPAIGN_ROOT}/queue/done"
FAILURE_ROOT="${CAMPAIGN_ROOT}/failures"
WORKER_ROOT="${CAMPAIGN_ROOT}/workers"
SUMMARY_FILE="${CAMPAIGN_ROOT}/launcher_summary.txt"
LOCK_ROOT="${SAMPLING_GPU_LOCK_ROOT:-${TMPDIR:-/tmp}/sampling_quantitative_gpu_locks}"
shopt -s nullglob

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

validate_float() {
  local label="$1" value="$2"
  [[ "${value}" =~ ^[-+]?[0-9]*([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
    die "${label} must be numeric: ${value}"
}

validate_optional_integer() {
  local label="$1" value="$2"
  [[ -z "${value}" || "${value}" =~ ^-?[0-9]+$ ]] || die "${label} must be an integer: ${value}"
}

validate_configuration() {
  [[ "${SAMPLING_METHOD}" == sampling_shapley || "${SAMPLING_METHOD}" == sampling_least_core ]] ||
    die "unsupported sampling method: ${SAMPLING_METHOD}"
  [[ "${WORKERS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || die 'WORKERS_PER_GPU must be a positive integer'
  [[ "${CPU_THREADS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]] || die 'CPU_THREADS_PER_WORKER must be a positive integer'
  [[ "${NUM_EPOCH}" =~ ^[1-9][0-9]*$ ]] || die 'NUM_EPOCH must be a positive integer'
  [[ "${NUM_LOCAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || die 'NUM_LOCAL_EPOCHS must be a positive integer'
  [[ "${DRY_RUN_PRINT_LIMIT}" =~ ^[0-9]+$ ]] || die 'DRY_RUN_PRINT_LIMIT must be a non-negative integer'
  [[ "${LAZY_HONEST_PHASE}" =~ ^[0-9]+$ ]] || die 'LAZY_HONEST_PHASE must be a non-negative integer'
  [[ "${DRY_RUN}" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
  [[ "${CONFIRM_SAMPLING_RUN}" =~ ^[01]$ ]] || die 'CONFIRM_SAMPLING_RUN must be 0 or 1'
  [[ "${MERGE_ONLY}" =~ ^[01]$ ]] || die 'MERGE_ONLY must be 0 or 1'
  [[ "${USE_LAST_ROUND_SV}" =~ ^[01]$ ]] || die 'USE_LAST_ROUND_SV must be 0 or 1'
  [[ "${LAZY_FAKE_POLICY}" == attack ]] || die 'LAZY_FAKE_POLICY must be attack for this protocol'
  [[ -z "${RANDOM_ROUND_OVERRIDE}" || "${RANDOM_ROUND_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] ||
    die 'RANDOM_ROUND must be a positive integer when provided'
  [[ "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe CAMPAIGN_ID: ${CAMPAIGN_ID}"
  [[ "${PUBLIC_ROOT}" != "${PROJECT_ROOT}" ]] || die 'PUBLIC_ROOT cannot be PROJECT_ROOT'
  validate_list GPU_IDS "${GPU_IDS[@]}"
  validate_list TRIAL_IDS "${TRIAL_IDS[@]}"
  validate_optional_integer sampling_shapley_num_permutations "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS}"
  validate_optional_integer sampling_shapley_seed "${SAMPLING_SHAPLEY_SEED}"
  validate_optional_integer sampling_least_core_num_coalitions "${SAMPLING_LEAST_CORE_NUM_COALITIONS}"
  validate_optional_integer sampling_least_core_seed "${SAMPLING_LEAST_CORE_SEED}"
  validate_float sampling_shapley_tolerance "${SAMPLING_SHAPLEY_TOLERANCE}"
  if [[ "${SAMPLING_METHOD}" == sampling_shapley && -n "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS}" ]]; then
    [[ "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS}" =~ ^[1-9][0-9]*$ ]] ||
      die 'SAMPLING_SHAPLEY_NUM_PERMUTATIONS must be positive'
  fi
  if [[ "${SAMPLING_METHOD}" == sampling_least_core && -n "${SAMPLING_LEAST_CORE_NUM_COALITIONS}" ]]; then
    [[ "${SAMPLING_LEAST_CORE_NUM_COALITIONS}" =~ ^[2-9][0-9]*$ ]] ||
      die 'SAMPLING_LEAST_CORE_NUM_COALITIONS must be at least 2'
    # The full matrix includes client_num=2.  Its N-minus-one world has one
    # client and therefore only two unique coalitions; one global CLI budget
    # cannot be larger without making part of the campaign invalid.  Leave the
    # variable empty for the per-world automatic budget used by the runner.
    [[ "${SAMPLING_LEAST_CORE_NUM_COALITIONS}" -le 2 ]] ||
      die 'SAMPLING_LEAST_CORE_NUM_COALITIONS must be <= 2 for the full client 2/4/6/8 matrix; omit it to use per-world auto budgets'
  fi
  if [[ "${MERGE_ONLY}" == 1 ]]; then
    [[ -n "${CAMPAIGN_ID_INPUT}" ]] || die 'MERGE_ONLY=1 requires CAMPAIGN_ID'
    [[ "${DRY_RUN}" == 0 ]] || die 'MERGE_ONLY=1 requires DRY_RUN=0'
    [[ -d "${CAMPAIGN_ROOT}" ]] || die "campaign not found: ${CAMPAIGN_ROOT}"
  fi
}

append_sampling_options() {
  case "${SAMPLING_METHOD}" in
    sampling_shapley)
      CMD+=(--sampling_shapley_tolerance "${SAMPLING_SHAPLEY_TOLERANCE}")
      [[ -z "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS}" ]] ||
        CMD+=(--sampling_shapley_num_permutations "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS}")
      [[ -z "${SAMPLING_SHAPLEY_SEED}" ]] ||
        CMD+=(--sampling_shapley_seed "${SAMPLING_SHAPLEY_SEED}")
      ;;
    sampling_least_core)
      [[ -z "${SAMPLING_LEAST_CORE_NUM_COALITIONS}" ]] ||
        CMD+=(--sampling_least_core_num_coalitions "${SAMPLING_LEAST_CORE_NUM_COALITIONS}")
      [[ -z "${SAMPLING_LEAST_CORE_SEED}" ]] ||
        CMD+=(--sampling_least_core_seed "${SAMPLING_LEAST_CORE_SEED}")
      ;;
  esac
}

append_attack_options() {
  local attack="$1"
  case "${attack}" in
    arima_adaptive|arima_adaptive_mnist_imdb_layerwise)
      CMD+=(--arima_var_window 3 --arima_var_threshold "${ARIMA_VAR_THRESHOLD}"
        --arima_err_threshold "${ARIMA_ERR_THRESHOLD}" --arima_min_rounds_after_start 4
        --arima_retrain_interval 15 --arima_err_ema_beta 0.9 --arima_trim_history)
      ;;
    arima|arima_mnist_imdb_layerwise)
      CMD+=(--arima_var_window 3 --arima_var_threshold "${ARIMA_VAR_THRESHOLD}"
        --arima_err_threshold "${ARIMA_ERR_THRESHOLD}" --arima_trim_history)
      ;;
    moirai_online|moirai_online_mnist_imdb_layerwise)
      CMD+=(--moirai_online_tuning --moirai_tune_lr "${MOIRAI_LR}"
        --moirai_tune_steps 1 --moirai_tune_start_round 10 --moirai_adapt_interval 4
        --moirai_residual_decay 0.95 --moirai_residual_clip 5 --moirai_err_ema_beta 0.9
        --moirai_err_threshold "${MOIRAI_ERR_THRESHOLD}"
        --moirai_min_rounds_after_start 4 --moirai_trim_history)
      ;;
    advanced_scaled_delta)
      CMD+=(--scaled_delta_c_const 1.0 --scaled_delta_noise_fraction 1.0
        --scaled_delta_max_noise_ratio 5.0 --scaled_delta_max_scale 10.0)
      ;;
    adam_local_deviation)
      CMD+=(--wang_adam_lr 8e-3 --wang_adam_beta1 0.9 --wang_adam_beta2 0.999
        --wang_adam_eps 1e-8 --adam_public_fraction 0.05 --adam_public_seed 20260726)
      ;;
    dkfr)
      local d=()
      read -r -a d <<< "${DKFR_VALUES}"
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
        --dkfr_mnist_pixel_reg 0.2 --dkfr_mnist_logit_reg 1e-4)
      ;;
    synthetic_replacement)
      CMD+=(--synthetic_mixup_alpha 0.4 --synthetic_noise_std 0.02
        --synthetic_tabular_noise_std 0.05 --synthetic_tabular_flip_prob 0.02
        --synthetic_seed 0)
      ;;
  esac
  case "${attack}" in
    *_mnist_imdb_layerwise)
      CMD+=(--layerwise_selection_metric normalized_rms --layerwise_selection_fraction 0.5
        --layerwise_pca_threshold 512 --layerwise_pca_components 8)
      ;;
  esac
  # This is the same forecast matrix rule used by the qualitative scripts:
  # prediction-style Moirai/ARIMA tasks use the Hessian/curvature proxy;
  # non-forecast attacks do not receive it.
  case "${attack}" in
    arima|arima_adaptive|moirai|moirai_online|*_mnist_imdb_layerwise)
      case "${attack}" in
        arima*|moirai*) CMD+=(--use_hessian --lambda_h 0.9) ;;
      esac
      ;;
  esac
}

task_use_hessian() {
  case "$1" in
    arima|arima_adaptive|moirai|moirai_online|arima_mnist_imdb_layerwise|arima_adaptive_mnist_imdb_layerwise|moirai_mnist_imdb_layerwise|moirai_online_mnist_imdb_layerwise)
      printf '1\n' ;;
    *) printf '0\n' ;;
  esac
}

task_lambda_h() {
  [[ "$(task_use_hessian "$1")" == 1 ]] && printf '0.9\n' || printf 'N/A\n'
}

task_requested_budget() {
  if [[ "${SAMPLING_METHOD}" == sampling_shapley ]]; then
    [[ -n "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS}" ]] && printf '%s\n' "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS}" || printf 'auto\n'
  else
    [[ -n "${SAMPLING_LEAST_CORE_NUM_COALITIONS}" ]] && printf '%s\n' "${SAMPLING_LEAST_CORE_NUM_COALITIONS}" || printf 'auto\n'
  fi
}

task_tolerance() {
  [[ "${SAMPLING_METHOD}" == sampling_shapley ]] && printf '%s\n' "${SAMPLING_SHAPLEY_TOLERANCE}" || printf 'N/A\n'
}

task_disclosure() {
  case "$1" in
    advanced_scaled_delta) printf 'adapted_baseline:zhu_advanced_scaled_delta\n' ;;
    adam_local_deviation) printf 'adapted_baseline:wang_public_warm_start_local_deviation\n' ;;
    dkfr) printf 'adapted_baseline:fu_dkfr_lite\n' ;;
    synthetic_replacement) printf 'adapted_baseline:lee_synthetic_replacement\n' ;;
    arima|arima_adaptive|arima_mnist_imdb_layerwise|arima_adaptive_mnist_imdb_layerwise)
      printf 'gradpred:arima_frobenius_identity_curvature_proxy\n' ;;
    moirai|moirai_online|moirai_mnist_imdb_layerwise|moirai_online_mnist_imdb_layerwise)
      printf 'gradpred:moirai_frobenius_identity_curvature_proxy\n' ;;
    *) printf 'N/A\n' ;;
  esac
}

emit_plan() {
  local sequence=0 trial alpha client attacker_id regime attack gap task_id
  for trial in "${TRIAL_IDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for client in "${CLIENT_NUMS[@]}"; do
        # for attacker_id: every client is designated once for each attack.
        for ((attacker_id=0; attacker_id<client; attacker_id++)); do
          for attack in "${PURE_ATTACKS[@]}"; do
            sequence=$((sequence + 1))
            task_id=$(printf 'q%06d_%s_t%s_a%s_n%s_att%s_pure_%s' \
              "${sequence}" "${DATASET}" "${trial}" "${alpha}" "${client}" "${attacker_id}" "${attack}")
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tplanned\t\n' \
              "${task_id}" "${DATASET}" "${trial}" "${alpha}" "${client}" "${attacker_id}" \
              pure "${attack}" N/A
          done
          for gap in "${LAZY_GAPS[@]}"; do
            for attack in "${LAZY_ATTACKS[@]}"; do
              sequence=$((sequence + 1))
              task_id=$(printf 'q%06d_%s_t%s_a%s_n%s_att%s_lazy_g%s_%s' \
                "${sequence}" "${DATASET}" "${trial}" "${alpha}" "${client}" "${attacker_id}" "${gap}" "${attack}")
              printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tplanned\t\n' \
                "${task_id}" "${DATASET}" "${trial}" "${alpha}" "${client}" "${attacker_id}" \
                lazy "${attack}" "${gap}"
            done
          done
        done
      done
    done
  done
}

build_command() {
  local task_id="$1" trial="$2" alpha="$3" client="$4" attacker_id="$5" regime="$6" attack="$7" gap="$8"
  TASK_DIR="${TASK_ROOT}/${task_id}"
  TASK_RESULTS="${TASK_DIR}/results"
  local random_round=10
  [[ "${attack}" != advanced_scaled_delta ]] || random_round=2
  [[ -z "${RANDOM_ROUND_OVERRIDE}" ]] || random_round="${RANDOM_ROUND_OVERRIDE}"
  CMD=("${PYTHON_BIN}" "${SCRIPT_SOURCE}"
    --dataset "${DATASET}" --alpha "${alpha}" --client_num "${client}"
    --attacker_id "${attacker_id}" --attack_method "${attack}"
    --detecting_method "${SAMPLING_METHOD}" --detecting_type quantitative
    --trial_id "${trial}" --num_epoch "${NUM_EPOCH}" --num_local_epochs "${NUM_LOCAL_EPOCHS}"
    --random_round "${random_round}" --results_root "${TASK_RESULTS}" --use_attack)
  append_sampling_options
  if [[ "${USE_LAST_ROUND_SV}" == 1 ]]; then
    CMD+=(--use_last_round_sv)
  else
    CMD+=(--no-use_last_round_sv)
  fi
  append_attack_options "${attack}"
  if [[ "${regime}" == lazy ]]; then
    CMD+=(--use_lazy --lazy_schedule fixed --lazy_honest_gap "${gap}"
      --lazy_honest_phase "${LAZY_HONEST_PHASE}" --lazy_fake_policy "${LAZY_FAKE_POLICY}")
  fi
}

print_command() {
  printf 'command:'
  local argument
  for argument in "${CMD[@]}"; do printf ' <%s>' "${argument}"; done
  printf '\n'
}

expected_tasks_per_trial() {
  # 4 alphas x (2+4+6+8) attackers x (2 pure + 2 lazy x 3 gaps).
  printf '%s\n' "$(( ${#ALPHAS[@]} * 20 * (${#PURE_ATTACKS[@]} + ${#LAZY_ATTACKS[@]} * ${#LAZY_GAPS[@]}) ))"
}

print_dry_run() {
  local expected_total=$(( $(expected_tasks_per_trial) * ${#TRIAL_IDS[@]} ))
  printf 'CONFIG|dataset=%s|method=%s|gpu=%s|workers=%s|threads=%s|epochs=%s\n' \
    "${DATASET}" "${SAMPLING_METHOD}" "${GPU_IDS_RAW}" "${WORKERS_PER_GPU}" \
    "${CPU_THREADS_PER_WORKER}" "${NUM_EPOCH}"
  printf 'SAMPLING|permutations=%s|coalitions=%s|seed_shapley=%s|seed_least_core=%s|tolerance=%s\n' \
    "${SAMPLING_SHAPLEY_NUM_PERMUTATIONS:-auto}" "${SAMPLING_LEAST_CORE_NUM_COALITIONS:-auto}" \
    "${SAMPLING_SHAPLEY_SEED:-trial-derived}" "${SAMPLING_LEAST_CORE_SEED:-trial-derived}" \
    "${SAMPLING_SHAPLEY_TOLERANCE}"
  local task_id dataset trial alpha client attacker_id regime attack gap status reason
  local enumerated=0 printed=0
  while IFS=$'\t' read -r task_id dataset trial alpha client attacker_id regime attack gap status reason; do
    enumerated=$((enumerated + 1))
    # Build every command even when only a small sample is printed.  This
    # validates all dataset-specific attack profiles during a dry run.
    build_command "${task_id}" "${trial}" "${alpha}" "${client}" "${attacker_id}" "${regime}" "${attack}" "${gap}"
    if ((printed < DRY_RUN_PRINT_LIMIT)); then
      printf 'TASK|%s|%s|%s|%s|%s|%s|%s|%s|%s|planned\n' "${task_id}" "${dataset}" \
        "${trial}" "${alpha}" "${client}" "${attacker_id}" "${regime}" "${attack}" "${gap}"
      print_command
      printed=$((printed + 1))
    fi
  done < <(emit_plan)
  [[ "${enumerated}" -eq "${expected_total}" ]] || die "task enumeration drift: ${enumerated} != ${expected_total}"
  printf 'DRY_RUN|dataset=%s|method=%s|tasks=%s|printed=%s|lazy_gaps=%s\n' \
    "${DATASET}" "${SAMPLING_METHOD}" "${enumerated}" "${printed}" "${LAZY_GAPS[*]}"
}

prepare_campaign() {
  [[ ! -e "${CAMPAIGN_ROOT}" ]] || die "campaign already exists: ${CAMPAIGN_ROOT}"
  [[ ! -e "${PUBLIC_ROOT}" ]] || die "public result already exists: ${PUBLIC_ROOT}"
  mkdir -p "${TASK_ROOT}" "${QUEUE_PENDING}" "${QUEUE_RUNNING}" "${QUEUE_DONE}" \
    "${FAILURE_ROOT}" "${WORKER_ROOT}"
  { printf 'TaskID\tDataset\tTrialID\tAlpha\tClientNum\tAttackerID\tAttackRegime\tAttackMethod\tLazyGap\tPlannedStatus\tReason\n'; emit_plan; } > "${PLANNED_FILE}"
  {
    printf 'Dataset\tSamplingMethod\tCPUThreadsPerWorker\tNumEpoch\tNumLocalEpochs\tUseLastRoundSV\tTolerance\tRequestedSamplingBudget\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${DATASET}" "${SAMPLING_METHOD}" \
      "${CPU_THREADS_PER_WORKER}" "${NUM_EPOCH}" "${NUM_LOCAL_EPOCHS}" "${USE_LAST_ROUND_SV}" \
      "$(task_tolerance)" "$(task_requested_budget)"
  } > "${CAMPAIGN_ROOT}/campaign.tsv"
  local task_id dataset trial alpha client attacker_id regime attack gap status reason
  while IFS=$'\t' read -r task_id dataset trial alpha client attacker_id regime attack gap status reason; do
    [[ "${status}" == planned ]] || continue
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${trial}" "${alpha}" "${client}" "${attacker_id}" "${regime}" "${attack}" "${gap}" \
      > "${QUEUE_PENDING}/${task_id}.spec"
  done < <(emit_plan)
  local expected_total=$(( $(expected_tasks_per_trial) * ${#TRIAL_IDS[@]} ))
  printf 'campaign_id=%s\ndataset=%s\nsampling_method=%s\nplanned_tasks=%s\n' \
    "${CAMPAIGN_ID}" "${DATASET}" "${SAMPLING_METHOD}" "${expected_total}" > "${SUMMARY_FILE}"
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
  command -v "${MERGER_PYTHON_BIN}" >/dev/null || die "merger Python not found: ${MERGER_PYTHON_BIN}"
  command -v "${FLOCK_BIN}" >/dev/null || die "flock not found: ${FLOCK_BIN}"
  command -v "${NVIDIA_SMI_BIN}" >/dev/null || die "nvidia-smi not found: ${NVIDIA_SMI_BIN}"
  [[ -f "${SCRIPT_SOURCE}" ]] || die "FLCE entry not found: ${SCRIPT_SOURCE}"
  [[ -f "${MERGER_SCRIPT}" ]] || die "merger not found: ${MERGER_SCRIPT}"
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
      rmdir "${guard}" 2>/dev/null || true
      printf '%s\n' "${claimed}"
      return 0
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
  for pid in "${owned[@]:-}"; do job_is_owned "${pid}" && kill -KILL "${pid}" 2>/dev/null || true; done
  for pid in "${owned[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}

run_one() {
  local gpu="$1" task_id="$2" trial="$3" alpha="$4" client="$5" attacker_id="$6" regime="$7" attack="$8" gap="$9"
  build_command "${task_id}" "${trial}" "${alpha}" "${client}" "${attacker_id}" "${regime}" "${attack}" "${gap}"
  mkdir -p "${TASK_DIR}/logs" "${TASK_DIR}/tmp" "${TASK_DIR}/cache" "${TASK_RESULTS}"
  local use_hessian lambda_h command_argv requested_budget tolerance
  use_hessian=$(task_use_hessian "${attack}")
  lambda_h=$(task_lambda_h "${attack}")
  command_argv=$(printf '%q ' "${CMD[@]}")
  requested_budget=$(task_requested_budget)
  tolerance=$(task_tolerance)
  {
    printf 'TaskID\tDataset\tTrialID\tAlpha\tClientNum\tAttackerID\tAttackRegime\tAttackMethod\tLazyGap\tGPU\tCPUThreadsPerWorker\tSamplingMethod\tUseLastRoundSV\tTolerance\tRequestedSamplingBudget\tUseHessian\tLambdaH\tCommandArgv\tResultsRoot\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${DATASET}" "${trial}" "${alpha}" "${client}" "${attacker_id}" \
      "${regime}" "${attack}" "${gap}" "${gpu}" "${CPU_THREADS_PER_WORKER}" \
      "${SAMPLING_METHOD}" "${USE_LAST_ROUND_SV}" "${tolerance}" "${requested_budget}" \
      "${use_hessian}" "${lambda_h}" "${command_argv}" "${TASK_RESULTS}"
  } > "${TASK_DIR}/manifest.tsv"
  printf '%q ' "${CMD[@]}" > "${TASK_DIR}/command.argv"
  printf '\n' >> "${TASK_DIR}/command.argv"
  local code=0
  (
    cd "${PROJECT_ROOT}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER}" MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER}" NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER}"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export TMPDIR="${TASK_DIR}/tmp" HF_HOME="${TASK_DIR}/cache/huggingface"
    export TORCH_HOME="${TASK_DIR}/cache/torch" XDG_CACHE_HOME="${TASK_DIR}/cache/xdg"
    exec "${CMD[@]}"
  ) > "${TASK_DIR}/logs/run.log" 2>&1 &
  CURRENT_TASK_PID=$!
  wait "${CURRENT_TASK_PID}" || code=$?
  CURRENT_TASK_PID=''
  if ((code == 0)); then
    printf 'success\t0\t\n' > "${TASK_DIR}/status.tsv"
    return 0
  fi
  printf 'failed\t%s\tchild_exit_%s\n' "${code}" "${code}" > "${TASK_DIR}/status.tsv"
  : > "${FAILURE_ROOT}/${task_id}.failed"
  return "${code}"
}

interrupt_worker() { trap '' HUP INT TERM; terminate_owned_jobs 2; exit 130; }

worker_loop() {
  local gpu="$1" slot="$2" claimed line task_id trial alpha client attacker_id regime attack gap
  local attempted=0 failed=0
  CURRENT_TASK_PID=''
  trap interrupt_worker HUP INT TERM
  while claimed=$(claim_next_spec "${gpu}" "${slot}"); do
    line=$(<"${claimed}")
    IFS=$'\t' read -r task_id trial alpha client attacker_id regime attack gap <<< "${line}"
    attempted=$((attempted + 1))
    run_one "${gpu}" "${task_id}" "${trial}" "${alpha}" "${client}" "${attacker_id}" "${regime}" "${attack}" "${gap}" || failed=$((failed + 1))
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
      pid=$!
      WORKER_PIDS+=("${pid}")
      printf 'gpu=%s worker=%s pid=%s\n' "${gpu}" "${slot}" "${pid}" >> "${WORKER_ROOT}/worker_pids.txt"
    done
  done
}

wait_for_workers() {
  local pid failures=0
  for pid in "${WORKER_PIDS[@]}"; do wait "${pid}" || failures=$((failures + 1)); done
  printf 'worker_failures=%s\n' "${failures}" >> "${SUMMARY_FILE}"
  return "${failures}"
}

run_merger() {
  "${MERGER_PYTHON_BIN}" "${MERGER_SCRIPT}" "${CAMPAIGN_ROOT}" "${PUBLIC_ROOT}" \
    "${DATASET}" "${SAMPLING_METHOD}" "${NUM_EPOCH}"
}

main() {
  validate_configuration
  if [[ "${DRY_RUN}" == 1 ]]; then
    print_dry_run
    return 0
  fi
  if [[ "${MERGE_ONLY}" == 1 ]]; then
    run_merger
    return $?
  fi
  if [[ "${CONFIRM_SAMPLING_RUN}" != 1 ]]; then
    printf '%s\n' 'Run is locked. Set DRY_RUN=0 CONFIRM_SAMPLING_RUN=1 for the full 50-round sweep.' >&2
    return 2
  fi
  ensure_prerequisites
  acquire_gpu_locks
  probe_gpus
  prepare_campaign
  trap interrupt_launcher HUP INT TERM
  start_workers
  local worker_status=0 merge_status=0 failure_count
  wait_for_workers || worker_status=$?
  trap - HUP INT TERM
  failure_count=$(find "${FAILURE_ROOT}" -type f -name '*.failed' | wc -l | tr -d ' ')
  printf 'failed_tasks=%s\n' "${failure_count}" >> "${SUMMARY_FILE}"
  run_merger || merge_status=$?
  ((worker_status == 0 && failure_count == 0 && merge_status == 0))
}

main "$@"

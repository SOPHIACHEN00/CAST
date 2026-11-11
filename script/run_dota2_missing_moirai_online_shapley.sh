#!/usr/bin/env bash
set -Eeuo pipefail

# Re-run only the three missing Dota2 tasks found in the merged-results audit.
# The three Python processes run sequentially on GPU 3 and append into one
# shared Results root, producing one CSV per metric rather than one directory
# per task.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${PROJECT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRYPOINT="${CANDIDATE_ROOT}/FLCE_updated.py"

GPU_ID="${GPU_ID:-3}"
CPU_THREADS="${CPU_THREADS:-10}"
DRY_RUN="${DRY_RUN:-1}"

ALPHAS=(1000000 50000000 200000000)
CLIENT_NUM=8
ATTACKER_ID=1
TRIAL_ID=0

RUN_TAG="${RUN_TAG:-$(date +'%Y%m%d-%H%M%S')_pid$$}"
CAMPAIGN_ID="dota2_missing3_moirai_online_${RUN_TAG}"
RUN_ROOT="${PROJECT_ROOT}/.shapley_patch_runs/${CAMPAIGN_ID}"
RESULTS_ROOT="${PROJECT_ROOT}/Results/${CAMPAIGN_ID}"
EXPORT_ROOT="${PROJECT_ROOT}/patch_exports/${CAMPAIGN_ID}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

build_command() {
  local alpha="$1"
  CMD=(
    "${PYTHON_BIN}" "${ENTRYPOINT}"
    --dataset dota2
    --alpha "${alpha}"
    --client_num "${CLIENT_NUM}"
    --attacker_id "${ATTACKER_ID}"
    --attack_method moirai_online
    --contribution_method shapley
    --detecting_type quantitative
    --use_last_round_sv
    --trial_id "${TRIAL_ID}"
    --num_epoch 50
    --num_local_epochs 1
    --random_round 10
    --results_root "${RESULTS_ROOT}"
    --use_attack
    --use_hessian
    --lambda_h 0.9
    --use_lazy
    --lazy_schedule probabilistic
    --attack_probability 0.90
    --lazy_honest_gap 10
    --lazy_honest_phase 0
    --lazy_fake_policy attack
    --moirai_online_tuning
    --moirai_tune_lr 0.001
    --moirai_tune_steps 1
    --moirai_tune_start_round 10
    --moirai_adapt_interval 4
    --moirai_residual_decay 0.95
    --moirai_residual_clip 5.0
    --moirai_err_ema_beta 0.9
    --moirai_err_threshold 0.25
    --moirai_min_rounds_after_start 4
    --moirai_trim_history
  )
}

print_command() {
  local argument
  printf 'command:'
  for argument in "${CMD[@]}"; do
    printf ' %q' "${argument}"
  done
  printf '\n'
}

validate_outputs() {
  local accuracy_csv contribution_csv time_csv
  local accuracy_lines contribution_lines time_lines

  accuracy_csv="${RESULTS_ROOT}/Accuracy_quantitative_shapley/global_accuracy_log_trial0.csv"
  contribution_csv="${RESULTS_ROOT}/Detecting_quantitative_shapley/combined_contributions_trial0.csv"
  time_csv="${RESULTS_ROOT}/Time_quantitative_shapley/attacking_time_log_trial0.csv"

  [[ -f "${accuracy_csv}" ]] || die "missing ${accuracy_csv}"
  [[ -f "${contribution_csv}" ]] || die "missing ${contribution_csv}"
  [[ -f "${time_csv}" ]] || die "missing ${time_csv}"

  accuracy_lines=$(wc -l < "${accuracy_csv}")
  contribution_lines=$(wc -l < "${contribution_csv}")
  time_lines=$(wc -l < "${time_csv}")

  [[ "${accuracy_lines}" -eq 151 ]] || die "accuracy CSV expected 151 lines, found ${accuracy_lines}"
  [[ "${contribution_lines}" -eq 25 ]] || die "contribution CSV expected 25 lines, found ${contribution_lines}"
  [[ "${time_lines}" -eq 4 ]] || die "time CSV expected 4 lines, found ${time_lines}"

  mkdir -p "${EXPORT_ROOT}"
  cp "${accuracy_csv}" "${EXPORT_ROOT}/dota2_missing3_accuracy.csv"
  cp "${contribution_csv}" "${EXPORT_ROOT}/dota2_missing3_contribution.csv"
  cp "${time_csv}" "${EXPORT_ROOT}/dota2_missing3_time.csv"
  if [[ -d "${RESULTS_ROOT}/manifests" ]]; then
    cp -R "${RESULTS_ROOT}/manifests" "${EXPORT_ROOT}/manifests"
  fi

  {
    printf 'dataset=dota2\n'
    printf 'method=moirai_online\n'
    printf 'mode=lazy\n'
    printf 'schedule=probabilistic\n'
    printf 'probability=0.90\n'
    printf 'gap=10\n'
    printf 'client_num=8\n'
    printf 'alphas=%s\n' "${ALPHAS[*]}"
    printf 'trial_id=0\n'
    printf 'rounds=50\n'
    printf 'local_epochs=1\n'
    printf 'random_round=10\n'
    printf 'hessian_proxy=on\n'
    printf 'lambda_h=0.9\n'
    printf 'moirai_tune_lr=0.001\n'
    printf 'moirai_tune_steps=1\n'
    printf 'moirai_tune_start_round=10\n'
    printf 'moirai_adapt_interval=4\n'
    printf 'moirai_residual_decay=0.95\n'
    printf 'moirai_residual_clip=5.0\n'
    printf 'moirai_err_ema_beta=0.9\n'
    printf 'moirai_err_threshold=0.25\n'
    printf 'moirai_min_rounds_after_start=4\n'
    printf 'moirai_trim_history=true\n'
    printf 'accuracy_rows=150\n'
    printf 'contribution_rows=24\n'
    printf 'time_rows=3\n'
    printf 'source_results=%s\n' "${RESULTS_ROOT}"
  } > "${EXPORT_ROOT}/summary.txt"

  tar -czf "${EXPORT_ROOT}.tar.gz" \
    -C "$(dirname "${EXPORT_ROOT}")" "$(basename "${EXPORT_ROOT}")"

  printf '\nCompleted export:\n%s\n' "${EXPORT_ROOT}"
  printf 'CSV files:\n'
  printf '  %s\n' "${EXPORT_ROOT}/dota2_missing3_accuracy.csv"
  printf '  %s\n' "${EXPORT_ROOT}/dota2_missing3_contribution.csv"
  printf '  %s\n' "${EXPORT_ROOT}/dota2_missing3_time.csv"
  printf 'Download bundle:\n  %s\n' "${EXPORT_ROOT}.tar.gz"
}

main() {
  local alpha task_tmp task_log failures=0
  local attack_module_real attack_project_root moirai_model_dir

  [[ -f "${ENTRYPOINT}" ]] || die "entrypoint not found: ${ENTRYPOINT}"
  [[ -f "${CANDIDATE_ROOT}/module/method/attack_updated_nonlayerwise.py" ]] ||
    die "ordinary attack module not found under candidate: ${CANDIDATE_ROOT}"
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python not found: ${PYTHON_BIN}"
  [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer"
  [[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || die "CPU_THREADS must be positive"

  attack_module_real="$(readlink -f "${CANDIDATE_ROOT}/module/method/attack_updated_nonlayerwise.py")"
  attack_project_root="$(dirname "$(dirname "$(dirname "${attack_module_real}")")")"
  moirai_model_dir="${attack_project_root}/moirai-1.1-R-small"
  [[ -f "${moirai_model_dir}/config.json" ]] ||
    die "local Moirai config not found: ${moirai_model_dir}/config.json"
  [[ -f "${moirai_model_dir}/model.safetensors" ]] ||
    die "local Moirai weights not found: ${moirai_model_dir}/model.safetensors"

  printf '%s\n' '=== Dota2 missing-task patch ==='
  printf 'alphas       : %s\n' "${ALPHAS[*]}"
  printf 'configuration: client=8, moirai_online, probabilistic, gap=10, p=0.90, trial=0\n'
  printf 'GPU / CPU    : physical GPU %s, one sequential process, %s CPU threads\n' \
    "${GPU_ID}" "${CPU_THREADS}"
  printf 'candidate    : %s\n' "${CANDIDATE_ROOT}"
  printf 'attack module: %s\n' "${attack_module_real}"
  printf 'Moirai model : %s\n' "${moirai_model_dir}"
  printf 'results root : %s\n' "${RESULTS_ROOT}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    for alpha in "${ALPHAS[@]}"; do
      build_command "${alpha}"
      printf '\nalpha=%s\n' "${alpha}"
      print_command
    done
    printf '\nDRY_RUN only. Run with DRY_RUN=0 after checking the commands.\n'
    return 0
  fi

  [[ "${DRY_RUN}" == "0" ]] || die "DRY_RUN must be 0 or 1"

  mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/tmp" \
    "${RUN_ROOT}/cache/hf" "${RUN_ROOT}/cache/torch" \
    "${RUN_ROOT}/cache/xdg" "${RESULTS_ROOT}"

  exec 9>"${PROJECT_ROOT}/.gpu_${GPU_ID}_campaign.lock"
  flock -n 9 || die "GPU ${GPU_ID} is already reserved by another campaign"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'

  for alpha in "${ALPHAS[@]}"; do
    build_command "${alpha}"
    task_tmp="${RUN_ROOT}/tmp/alpha_${alpha}"
    task_log="${RUN_ROOT}/logs/alpha_${alpha}.log"
    mkdir -p "${task_tmp}"

    printf '\n[START] alpha=%s log=%s\n' "${alpha}" "${task_log}"
    if (
      cd "${CANDIDATE_ROOT}"
      env \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        OMP_NUM_THREADS="${CPU_THREADS}" \
        MKL_NUM_THREADS="${CPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
        NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
        PYTHONDONTWRITEBYTECODE=1 \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TMPDIR="${task_tmp}" \
        HF_HOME="${RUN_ROOT}/cache/hf" \
        TORCH_HOME="${RUN_ROOT}/cache/torch" \
        XDG_CACHE_HOME="${RUN_ROOT}/cache/xdg" \
        "${CMD[@]}"
    ) > "${task_log}" 2>&1; then
      printf '[PASS] alpha=%s\n' "${alpha}"
    else
      printf '[FAIL] alpha=%s; inspect %s\n' "${alpha}" "${task_log}" >&2
      failures=$((failures + 1))
    fi
  done

  [[ "${failures}" -eq 0 ]] || die "${failures} of 3 patch tasks failed"
  validate_outputs
}

main "$@"

#!/usr/bin/env bash
# One-run diagnostic launcher for the corrected Moirai/ARIMA implementation.
#
# Example:
#   ALPHA=1000 CLIENT_NUM=4 ATTACKER_ID=1 \
#   bash script/run_corrected_forecast_attack.sh adult moirai_online
#
# This launcher intentionally avoids historical FL_Attack*.py aliases and
# run_env symlinks. Each invocation gets a new results directory by default.
# Its online-Moirai defaults (lr=0.1, no decay, very high reset threshold)
# deliberately make adaptation observable in an offline-vs-online A/B check;
# set the MOIRAI_* variables explicitly for the final paper protocol.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: ALPHA=<value> $0 <dataset> <attack_method>" >&2
  exit 2
fi

dataset="$1"
attack_method="$2"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
entry="${ENTRY:-${project_root}/FLCE_updated.py}"
python_bin="${PYTHON_BIN:-python}"

case "$attack_method" in
  arima|arima_adaptive|moirai|moirai_online|\
  arima_mnist_imdb_layerwise|arima_adaptive_mnist_imdb_layerwise|\
  moirai_mnist_imdb_layerwise|moirai_online_mnist_imdb_layerwise)
    ;;
  *)
    echo "ERROR: unsupported forecast attack method: $attack_method" >&2
    exit 2
    ;;
esac

alpha="${ALPHA:?Set ALPHA for the selected dataset/distribution}"
client_num="${CLIENT_NUM:-4}"
attacker_id="${ATTACKER_ID:-1}"
trial_id="${TRIAL_ID:-0}"
detecting_method="${DETECTING_METHOD:-shapley}"
random_round="${RANDOM_ROUND:-10}"
timestamp="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
results_root="${RESULTS_ROOT:-${project_root}/Results_corrected/${timestamp}_pid$$_${dataset}_${attack_method}_trial${trial_id}}"

cmd=(
  "$python_bin" "$entry"
  --dataset "$dataset"
  --alpha "$alpha"
  --client_num "$client_num"
  --attacker_id "$attacker_id"
  --attack_method "$attack_method"
  --contribution_method "$detecting_method"
  --random_round "$random_round"
  --trial_id "$trial_id"
  --results_root "$results_root"
  --use_attack
)

if [[ "${USE_HESSIAN:-0}" == "1" ]]; then
  cmd+=(--use_hessian --lambda_h "${LAMBDA_H:-0.9}")
fi

if [[ "$attack_method" == "arima_adaptive" || "$attack_method" == "arima_adaptive_mnist_imdb_layerwise" ]]; then
  cmd+=(
    --arima_retrain_interval "${ARIMA_RETRAIN_INTERVAL:-10}"
    --arima_err_ema_beta "${ARIMA_ERR_EMA_BETA:-0.9}"
  )
  if [[ -n "${ARIMA_ERR_THRESHOLD:-}" ]]; then
    cmd+=(--arima_err_threshold "$ARIMA_ERR_THRESHOLD")
  fi
  if [[ -n "${ARIMA_MIN_ROUNDS:-}" ]]; then
    cmd+=(--arima_min_rounds_after_start "$ARIMA_MIN_ROUNDS")
  fi
  if [[ "${ARIMA_TRIM_HISTORY:-1}" == "1" ]]; then
    cmd+=(--arima_trim_history)
  else
    cmd+=(--no-arima_trim_history)
  fi
fi

if [[ "$attack_method" == "moirai_online" || "$attack_method" == "moirai_online_mnist_imdb_layerwise" ]]; then
  cmd+=(
    --moirai_tune_lr "${MOIRAI_TUNE_LR:-0.1}"
    --moirai_tune_steps "${MOIRAI_TUNE_STEPS:-1}"
    --moirai_adapt_interval "${MOIRAI_ADAPT_INTERVAL:-0}"
    --moirai_residual_decay "${MOIRAI_RESIDUAL_DECAY:-1.0}"
    --moirai_residual_clip "${MOIRAI_RESIDUAL_CLIP:-5.0}"
    --moirai_err_ema_beta "${MOIRAI_ERR_EMA_BETA:-0.9}"
    --moirai_err_threshold "${MOIRAI_ERR_THRESHOLD:-1000000000}"
  )
  if [[ -n "${MOIRAI_MIN_ROUNDS:-}" ]]; then
    cmd+=(--moirai_min_rounds_after_start "$MOIRAI_MIN_ROUNDS")
  fi
  if [[ "${MOIRAI_TRIM_HISTORY:-1}" == "1" ]]; then
    cmd+=(--moirai_trim_history)
  else
    cmd+=(--no-moirai_trim_history)
  fi
fi

if [[ "${USE_LAZY:-0}" == "1" ]]; then
  cmd+=(
    --use_lazy
    --lazy_honest_gap "${LAZY_HONEST_GAP:-3}"
    --lazy_honest_phase "${LAZY_HONEST_PHASE:-0}"
    --lazy_fake_policy "${LAZY_FAKE_POLICY:-attack}"
  )
fi

echo "=== CORRECTED FORECAST ATTACK RUN ==="
echo "project_root : $project_root"
echo "entry        : $entry"
sha256sum -- "$entry"
sha256sum -- \
  "$project_root/module/method/attack_updated_nonlayerwise.py" \
  "$project_root/module/method/attack_updated_layerwise.py"
echo "results_root : $results_root"
printf 'command      :'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; command not executed."
  exit 0
fi

cd "$project_root"
exec "${cmd[@]}"

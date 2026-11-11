#!/usr/bin/env bash
# Default: preview only. Review before setting DRY_RUN=0 on the server.
# Full matrix: bash script/run_multi_freerider.sh
# One dataset: bash script/run_multi_freerider.sh adult
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Four 3090s, 15 concurrent tasks per GPU, 15 CPU threads per task.
read -r -a GPU_ARRAY <<< "${GPU_IDS:-0 1 2 3}"
TASKS_PER_GPU="${TASKS_PER_GPU:-15}"
CPU_THREADS_PER_TASK="${CPU_THREADS_PER_TASK:-15}"
read -r -a TRIAL_ARRAY <<< "${TRIAL_IDS:-0 1 2}"
SEED="${SEED:-42}"
ATTACKER_SELECTION_SEED="${ATTACKER_SELECTION_SEED:-2026}"
PARTITION_SETTING="${PARTITION_SETTING:-first}"
read -r -a DETECTOR_ARRAY <<< "${DETECTORS:-shapley std_dagmm}"
read -r -a ATTACK_ARRAY <<< "${ATTACKS:-moirai_online arima_adaptive}"
RESULT_SUFFIX=""
if [[ "${PARTITION_SETTING}" == uniform ]]; then RESULT_SUFFIX="_uniform"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/Results_multi_freerider${RESULT_SUFFIX}}"
DRY_RUN="${DRY_RUN:-1}"
PRINT_LIMIT="${PRINT_LIMIT:-6}"

if (($#)); then
  DATASET_ARRAY=("$@")
else
  DATASET_ARRAY=(adult mnist imdb)
fi
case "${DRY_RUN}" in
  1) MODE=--dry-run ;;
  0) MODE=--run ;;
  *) printf 'DRY_RUN must be 1 (preview) or 0 (run).\n' >&2; exit 2 ;;
esac

# Shared queue across datasets; each GPU has exactly TASKS_PER_GPU worker slots.
# N=8, k=1/2/3. Default first-column alpha: Adult=1000, MNIST/IMDB=10.
# PARTITION_SETTING=uniform: Adult=20000000, MNIST=10000000, IMDB=200000.
# Three groups: default 216 tasks; DETECTORS=shapley selects 108 (36/dataset).
# Data/training/detector seed is always 42; TRIAL_IDS select attacker groups only.
# ATTACKER_SELECTION_SEED changes IDs only; assignments are paired across methods/regimes.
# Pure: all 50 rounds fake. Lazy: fixed 1 real + 5 fake, phase 0 => 9 real / 41 fake.
# Exact last-round Shapley + STD-DAGMM; independent attack instances, no collusion.
# Public CSVs: <campaign>/merged/<dataset>/{accuracy,contributions,detection,summary,time,audit}.csv
# Also summary_mean_std.csv: across-group mean/sample SD (ddof=1), with completion counts.
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/multi_freerider_campaign.py" \
  "${MODE}" --python "${PYTHON_BIN}" \
  --datasets "${DATASET_ARRAY[@]}" --gpus "${GPU_ARRAY[@]}" \
  --partition "${PARTITION_SETTING}" --detectors "${DETECTOR_ARRAY[@]}" \
  --attacks "${ATTACK_ARRAY[@]}" \
  --tasks-per-gpu "${TASKS_PER_GPU}" --cpu-threads "${CPU_THREADS_PER_TASK}" \
  --trials "${TRIAL_ARRAY[@]}" --seed "${SEED}" \
  --attacker-selection-seed "${ATTACKER_SELECTION_SEED}" \
  --output-root "${OUTPUT_ROOT}" --print-limit "${PRINT_LIMIT}"

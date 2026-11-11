#!/usr/bin/env bash
# Preview by default. Approval is required before a separate server launch.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3}"
TASKS_PER_GPU="${TASKS_PER_GPU:-15}"
CPU_THREADS_PER_TASK="${CPU_THREADS_PER_TASK:-15}"
PARTITION_SETTING="${PARTITION_SETTING:-first}"
RESULT_SUFFIX=""
if [[ "${PARTITION_SETTING}" == uniform ]]; then RESULT_SUFFIX="_uniform"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../Results_freerider_controls${RESULT_SUFFIX}}"
case "${DRY_RUN:-1}" in
  1) MODE=--dry-run ;;
  0) MODE=--run ;;
  *) printf 'DRY_RUN must be 1 or 0.\n' >&2; exit 2 ;;
esac
if (($#)); then DATASETS=("$@"); else DATASETS=(adult mnist imdb); fi
# 27 controls; no attack model, no lazy schedule, no reuse, no attacked reruns.
# 8 original shards -> remove 1/2/3 IDs -> train remaining 7/6/5 clients.
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/freerider_control_campaign.py" \
  "${MODE}" --python "${PYTHON_BIN}" --datasets "${DATASETS[@]}" \
  --partition "${PARTITION_SETTING}" \
  --gpus "${GPUS[@]}" --tasks-per-gpu "${TASKS_PER_GPU}" \
  --cpu-threads "${CPU_THREADS_PER_TASK}" --output-root "${OUTPUT_ROOT}"

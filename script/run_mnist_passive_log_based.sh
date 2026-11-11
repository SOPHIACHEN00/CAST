#!/usr/bin/env bash
set -Eeuo pipefail
# LogBasedFreeRiderDetector campaign for MNIST. Defaults to DRY_RUN=1.
# Optional: GPU_IDS="0 1" WORKERS_PER_GPU=2 CPU_THREADS_PER_WORKER=4.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
exec "${MERGER_PYTHON_BIN:-${PYTHON_BIN:-python}}" "${SCRIPT_DIR}/log_based_campaign.py" --project-root "${PROJECT_ROOT}" --dataset mnist "$@"

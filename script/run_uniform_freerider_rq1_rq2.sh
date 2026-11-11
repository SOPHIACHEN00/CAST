#!/usr/bin/env bash
# RQ1/RQ2 only: 108 attacked Shapley tasks + 27 honest controls. Preview by default.
# All datasets: bash script/run_uniform_freerider_rq1_rq2.sh
# One dataset:  bash script/run_uniform_freerider_rq1_rq2.sh adult
# Actual server execution requires explicitly setting DRY_RUN=0 after review.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export DRY_RUN="${DRY_RUN:-1}"
export PRINT_LIMIT="${PRINT_LIMIT:-108}"

# Freeze the reviewed resource and experiment settings for this dedicated entry.
export GPU_IDS="0 1 2 3"
export TASKS_PER_GPU=15
export CPU_THREADS_PER_TASK=15
export TRIAL_IDS="0 1 2"
export SEED=42
export ATTACKER_SELECTION_SEED=2026
export PARTITION_SETTING=uniform
export DETECTORS=shapley
UNIFORM_OUTPUT_ROOT="${UNIFORM_OUTPUT_ROOT:-${PROJECT_ROOT}/Results_uniform_freerider_rq12}"
if (($#)); then DATASETS=("$@"); else DATASETS=(adult mnist imdb); fi

# Reuse the existing task queue and GPU locks. Each task has its own JSON/status/log;
# only coordinators write public CSVs, grouped by dataset. Never launch the two
# phases simultaneously: the 15 slots/GPU limit is shared, not 15 per campaign.
# Pure: all 50 fake. Lazy: fixed 1 real + 5 fake (9 real + 41 fake in 50 rounds).
printf 'Uniform RQ1/RQ2: 36 attack + 9 control tasks per dataset; 135 for all three.\n'
printf 'Phase 1: exact Shapley attacks, shared queue across four GPUs.\n'
OUTPUT_ROOT="${UNIFORM_OUTPUT_ROOT}/attacks" bash "${SCRIPT_DIR}/run_multi_freerider.sh" "${DATASETS[@]}"
printf 'Phase 2: honest controls on the same original eight Uniform shards.\n'
OUTPUT_ROOT="${UNIFORM_OUTPUT_ROOT}/controls" bash "${SCRIPT_DIR}/run_freerider_controls.sh" "${DATASETS[@]}"

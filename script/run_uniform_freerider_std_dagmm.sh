#!/usr/bin/env bash
# Uniform STD-DAGMM only: 108 tasks, 36 per dataset. Preview by default.
# All datasets: bash script/run_uniform_freerider_std_dagmm.sh
# One dataset:  bash script/run_uniform_freerider_std_dagmm.sh adult
# After approval, add this entry to a copy of the completed Uniform deployment.
# Actual execution requires explicitly setting DRY_RUN=0 on the server.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export DRY_RUN="${DRY_RUN:-1}"
export PRINT_LIMIT="${PRINT_LIMIT:-108}"

# Freeze the approved resource and attacker-selection configuration.
export GPU_IDS="0 1 2 3"
export TASKS_PER_GPU=15
export CPU_THREADS_PER_TASK=15
export TRIAL_IDS="0 1 2"
export SEED=42
export ATTACKER_SELECTION_SEED=2026
export PARTITION_SETTING=uniform
export DETECTORS=std_dagmm
export OUTPUT_ROOT="${UNIFORM_STD_DAGMM_OUTPUT_ROOT:-${PROJECT_ROOT}/Results_uniform_freerider_std_dagmm}"

if (($#)); then DATASETS=("$@"); else DATASETS=(adult mnist imdb); fi

# N=8; k=1/2/3; three original non-colluding attacker-identity groups.
# Adult alpha=20000000; MNIST alpha=10000000; IMDB alpha=200000.
# Pure: all 50 rounds fake. Lazy: 1 real / 5 fake, 9 real + 41 fake in 50 rounds.
# Existing shared queue + GPU locks: at most 60 tasks over GPUs 0/1/2/3.
# Each worker writes JSON/status/log; only the coordinator writes dataset CSVs.
printf 'Uniform STD-DAGMM: 36 tasks per dataset; 108 for Adult + MNIST + IMDB.\n'
printf 'Four GPUs, 15 tasks/GPU, 15 CPU threads/task; three groups, seed=42.\n'
printf 'Output root: %s\n' "${OUTPUT_ROOT}"
exec bash "${SCRIPT_DIR}/run_multi_freerider.sh" "${DATASETS[@]}"

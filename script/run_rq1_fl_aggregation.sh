#!/usr/bin/env bash
# RQ1: original single-attacker path; 108 tasks across all three datasets.
# Default is a read-only command preview. No SSH/upload is performed here.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/Results_rq1_fl_aggregation}"

if (($#)); then DATASETS=("$@"); else DATASETS=(adult mnist imdb); fi
for dataset in "${DATASETS[@]}"; do
  case "${dataset}" in
    adult|mnist|imdb) ;;
    *) printf 'Unsupported dataset: %s\n' "${dataset}" >&2; exit 2 ;;
  esac
done
case "${DRY_RUN}" in
  1) MODE=--dry-run ;;
  0) MODE=--run ;;
  *) printf 'DRY_RUN must be 1 or 0.\n' >&2; exit 2 ;;
esac

# Each worker additionally enforces torch.set_num_threads(15).
export FLCE_CPU_THREADS_PER_TASK=15
export OMP_NUM_THREADS=15 MKL_NUM_THREADS=15 OPENBLAS_NUM_THREADS=15
export NUMEXPR_NUM_THREADS=15 NUMEXPR_MAX_THREADS=15 BLIS_NUM_THREADS=15
export VECLIB_MAXIMUM_THREADS=15 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# First-column alphas: adult=1000, mnist=10, imdb=10.
# Adult uses ordinary attacks; MNIST/IMDB use their layerwise counterparts.
# Attacker sweep means SEPARATE tasks, each with singular --attacker_id.
# Pure warmup remains fake; no --use_lazy and no private attacker training.
# One queue across four GPUs; only its coordinator writes dataset-level CSVs.
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/rq1_fl_aggregation_campaign.py" \
  "${MODE}" --python "${PYTHON_BIN}" \
  --datasets "${DATASETS[@]}" --partition first \
  --client-num 6 --attacker-sweep 0 1 2 3 4 5 \
  --fl-methods fedavg fedadam weighted_median \
  --attack-families arima_adaptive moirai_online \
  --regime pure --detecting-method shapley --use-last-round-sv \
  --qualitative-aggregation-mode passive \
  --num-epoch 50 --num-local-epochs 1 --random-round 10 \
  --use-hessian --lambda-h 0.9 \
  --server-lr 0.01 --server-beta1 0.9 --server-beta2 0.99 --server-tau 0.001 \
  --seed 42 --trial-id 0 \
  --gpus 0 1 2 3 --tasks-per-gpu 15 --cpu-threads 15 \
  --output-root "${OUTPUT_ROOT}" --print-all-commands

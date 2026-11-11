#!/usr/bin/env bash
set -Eeuo pipefail
# Review draft. Running this file starts the full LogBased campaign.

ROOT=/output/gradpred_exp/attack
cd "$ROOT"
source "$ROOT/flce_attack"
export PYTHON_BIN="$ROOT/bin/python"
export MERGER_PYTHON_BIN="$PYTHON_BIN"

RUN_ID="${CAMPAIGN_ID:-logfr_full_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# GPU 0: Dota2 starts only after TicTacToe succeeds.
nohup bash -c '
    set -Eeuo pipefail
    cd "$1"
    export PYTHON_BIN="$2" MERGER_PYTHON_BIN="$2"
    export CAMPAIGN_ID="$3" DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1
    GPU_IDS=0 WORKERS_PER_GPU=30 CPU_THREADS_PER_WORKER=15 \
        bash script/run_tictactoe_passive_log_based.sh
    GPU_IDS=0 WORKERS_PER_GPU=15 CPU_THREADS_PER_WORKER=15 \
        bash script/run_dota2_passive_log_based.sh
' bash "$ROOT" "$PYTHON_BIN" "$RUN_ID" \
    > "$LOG_DIR/gpu0_tictactoe_then_dota2.log" 2>&1 < /dev/null &
PID_GPU0=$!

nohup env PYTHON_BIN="$PYTHON_BIN" MERGER_PYTHON_BIN="$PYTHON_BIN" \
    CAMPAIGN_ID="$RUN_ID" DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 \
    GPU_IDS=1 WORKERS_PER_GPU=15 CPU_THREADS_PER_WORKER=15 \
    bash script/run_adult_passive_log_based.sh \
    > "$LOG_DIR/gpu1_adult.log" 2>&1 < /dev/null &
PID_ADULT=$!

nohup env PYTHON_BIN="$PYTHON_BIN" MERGER_PYTHON_BIN="$PYTHON_BIN" \
    CAMPAIGN_ID="$RUN_ID" DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 \
    GPU_IDS=2 WORKERS_PER_GPU=20 CPU_THREADS_PER_WORKER=15 \
    bash script/run_mnist_passive_log_based.sh \
    > "$LOG_DIR/gpu2_mnist.log" 2>&1 < /dev/null &
PID_MNIST=$!

nohup env PYTHON_BIN="$PYTHON_BIN" MERGER_PYTHON_BIN="$PYTHON_BIN" \
    CAMPAIGN_ID="$RUN_ID" DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 \
    GPU_IDS=3 WORKERS_PER_GPU=20 CPU_THREADS_PER_WORKER=15 \
    bash script/run_imdb_passive_log_based.sh \
    > "$LOG_DIR/gpu3_imdb.log" 2>&1 < /dev/null &
PID_IMDB=$!

printf 'Campaign: %s\nGPU0 chain PID: %s\nAdult PID: %s\nMNIST PID: %s\nIMDB PID: %s\nLogs: %s\n' \
    "$RUN_ID" "$PID_GPU0" "$PID_ADULT" "$PID_MNIST" "$PID_IMDB" "$LOG_DIR"

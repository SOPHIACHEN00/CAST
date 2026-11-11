#!/usr/bin/env sh
# Deploy a corrected candidate snapshot into a NEW isolated server tree, then
# run short Moirai/ARIMA smoke jobs. This script never edits the base project.
#
# Required on the server:
#   CANDIDATE_SOURCE=/path/to/preuploaded/relative-layout/snapshot \
#   sh script/deploy_and_smoke_moirai_arima.sh
#
# Safe preview (no mkdir/copy/process launch):
#   PROJECT_ROOT="$PWD" CANDIDATE_SOURCE="$PWD" DRY_RUN=1 \
#   sh script/deploy_and_smoke_moirai_arima.sh

set -eu

PROJECT_ROOT=${PROJECT_ROOT:-/output/gradpred_exp/attack}
: "${CANDIDATE_SOURCE:?Set CANDIDATE_SOURCE to the pre-uploaded candidate snapshot}"
PYTHON_BIN=${PYTHON_BIN:-/root/conda_envs/flce_attack/bin/python}
DRY_RUN=${DRY_RUN:-0}
RUN_DEPENDENCY_PROBES=${RUN_DEPENDENCY_PROBES:-1}
DEPLOY_ONLY=${DEPLOY_ONLY:-0}
SMOKE_STAGE=${SMOKE_STAGE:-gate}
SMOKE_NUM_EPOCH=${SMOKE_NUM_EPOCH:-12}
SMOKE_RANDOM_ROUND=${SMOKE_RANDOM_ROUND:-5}
SMOKE_CLIENT_NUM=${SMOKE_CLIENT_NUM:-2}
SMOKE_ATTACKER_ID=${SMOKE_ATTACKER_ID:-1}
SMOKE_TRIAL_ID=${SMOKE_TRIAL_ID:-0}
SMOKE_GPU_IDS=${SMOKE_GPU_IDS:-0}
SMOKE_SCHEDULE=${SMOKE_SCHEDULE:-fixed}
SMOKE_GAP=${SMOKE_GAP:-2}
SMOKE_PROBABILITY=${SMOKE_PROBABILITY:-0.50}
CONTRIBUTION_METHOD=${CONTRIBUTION_METHOD:-shapley}
USE_HESSIAN=${USE_HESSIAN:-1}
LAMBDA_H=${LAMBDA_H:-0.9}
LAYERWISE_SELECTION_METRIC=${LAYERWISE_SELECTION_METRIC:-normalized_rms}
LAYERWISE_SELECTION_FRACTION=${LAYERWISE_SELECTION_FRACTION:-0.5}
LAYERWISE_SELECTION_TOP_N=${LAYERWISE_SELECTION_TOP_N:-}
LAYERWISE_PCA_THRESHOLD=${LAYERWISE_PCA_THRESHOLD:-512}
LAYERWISE_PCA_COMPONENTS=${LAYERWISE_PCA_COMPONENTS:-8}
# Optional read-only source for a previously verified local Moirai-small
# checkpoint.  It is copied/reflinked into the new candidate; no symlink back
# to the old candidate is created.
MOIRAI_ASSET_SOURCE=${MOIRAI_ASSET_SOURCE:-}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
DEPLOY_ID=${DEPLOY_ID:-forecast_candidate_${timestamp}_pid$$}
CANDIDATE_PARENT=${CANDIDATE_PARENT:-${PROJECT_ROOT}/.candidate_runs}
DEPLOY_ROOT=${DEPLOY_ROOT:-${CANDIDATE_PARENT}/${DEPLOY_ID}}
CANDIDATE_ROOT=${CANDIDATE_ROOT:-${DEPLOY_ROOT}/attack}
SMOKE_RUNS_ROOT=${SMOKE_RUNS_ROOT:-${DEPLOY_ROOT}/smoke_runs}
SMOKE_RESULTS_ROOT=${SMOKE_RESULTS_ROOT:-${PROJECT_ROOT}/Results/${DEPLOY_ID}_smoke}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

case "$DEPLOY_ID" in
  ''|*[!A-Za-z0-9._-]*) die "DEPLOY_ID must be one safe path segment: $DEPLOY_ID" ;;
esac

case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
case "$USE_HESSIAN" in 0|1) ;; *) die "USE_HESSIAN must be 0 or 1" ;; esac
case "$DEPLOY_ONLY" in 0|1) ;; *) die "DEPLOY_ONLY must be 0 or 1" ;; esac
case "$SMOKE_STAGE" in gate|all) ;; *) die "SMOKE_STAGE must be gate or all" ;; esac
case "$SMOKE_SCHEDULE" in fixed|probabilistic) ;;
  *) die "SMOKE_SCHEDULE must be fixed or probabilistic" ;;
esac

[ -d "$PROJECT_ROOT" ] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[ -d "$CANDIDATE_SOURCE" ] || die "CANDIDATE_SOURCE does not exist: $CANDIDATE_SOURCE"

PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd -P)
CANDIDATE_SOURCE=$(cd "$CANDIDATE_SOURCE" && pwd -P)
if [ -n "$MOIRAI_ASSET_SOURCE" ]; then
  [ -d "$MOIRAI_ASSET_SOURCE" ] ||
    die "MOIRAI_ASSET_SOURCE is not a directory: $MOIRAI_ASSET_SOURCE"
  MOIRAI_ASSET_SOURCE=$(cd "$MOIRAI_ASSET_SOURCE" && pwd -P)
  case "$MOIRAI_ASSET_SOURCE" in
    "$PROJECT_ROOT"/.candidate_runs/*/attack/moirai-1.1-R-small) ;;
    *) die 'MOIRAI_ASSET_SOURCE must be a verified prior candidate moirai-1.1-R-small directory' ;;
  esac
  [ -f "$MOIRAI_ASSET_SOURCE/config.json" ] ||
    die 'MOIRAI_ASSET_SOURCE lacks config.json'
  [ -f "$MOIRAI_ASSET_SOURCE/model.safetensors" ] ||
    die 'MOIRAI_ASSET_SOURCE lacks model.safetensors'
fi

# All mutable deployment paths are deliberately pinned below the project.
# Environment overrides may select the project and deploy id, but may not
# redirect writes to an unrelated server path.
[ "$CANDIDATE_PARENT" = "$PROJECT_ROOT/.candidate_runs" ] ||
  die "CANDIDATE_PARENT must be $PROJECT_ROOT/.candidate_runs"
[ "$DEPLOY_ROOT" = "$CANDIDATE_PARENT/$DEPLOY_ID" ] ||
  die "DEPLOY_ROOT must be $CANDIDATE_PARENT/$DEPLOY_ID"
[ "$CANDIDATE_ROOT" = "$DEPLOY_ROOT/attack" ] ||
  die "CANDIDATE_ROOT must be $DEPLOY_ROOT/attack"
[ "$SMOKE_RUNS_ROOT" = "$DEPLOY_ROOT/smoke_runs" ] ||
  die "SMOKE_RUNS_ROOT must be $DEPLOY_ROOT/smoke_runs"
[ "$SMOKE_RESULTS_ROOT" = "$PROJECT_ROOT/Results/${DEPLOY_ID}_smoke" ] ||
  die "SMOKE_RESULTS_ROOT must be $PROJECT_ROOT/Results/${DEPLOY_ID}_smoke"

required_overlay_files='FLCE_updated.py
module/method/attack_updated_nonlayerwise.py
module/method/attack_updated_layerwise.py
tests/test_moirai_arima_correctness.py
tests/server_smoke_moirai_arima.py
tests/test_gpu_slot_scheduler.sh
tests/test_shapley_forecast_launchers.py
tests/test_shapley_forecast_scheduler.sh
script/deploy_and_smoke_moirai_arima.sh
script/run_full_moirai_arima_matrix.sh
script/run_moirai_arima_tuning_pilot.sh
script/analyze_moirai_arima_tuning.py
script/run_tictactoe_forecast_pure_shapley.sh
script/run_tictactoe_forecast_lazy_shapley.sh
script/run_adult_forecast_pure_shapley.sh
script/run_adult_forecast_lazy_shapley.sh
script/run_dota2_forecast_pure_shapley.sh
script/run_dota2_forecast_lazy_shapley.sh'
printf '%s\n' "$required_overlay_files" | while IFS= read -r relative_path; do
  [ -f "$CANDIDATE_SOURCE/$relative_path" ] ||
    die "candidate snapshot is missing relative path: $relative_path"
done

[ ! -e "$DEPLOY_ROOT" ] || die "refusing to reuse existing deploy path: $DEPLOY_ROOT"
[ ! -e "$SMOKE_RESULTS_ROOT" ] || die "refusing to reuse existing Results path: $SMOKE_RESULTS_ROOT"

datasets_for_stage() {
  if [ "$SMOKE_STAGE" = all ]; then
    printf '%s\n' 'tictactoe adult dota2 mnist imdb'
  else
    # First gate one ordinary and one layerwise route. Use SMOKE_STAGE=all
    # only after this gate passes.
    printf '%s\n' 'tictactoe mnist'
  fi
}

alpha_for_dataset() {
  case "$1" in
    tictactoe) printf '%s\n' 10 ;;
    adult) printf '%s\n' 1000 ;;
    dota2) printf '%s\n' 10000 ;;
    mnist) printf '%s\n' 10 ;;
    imdb) printf '%s\n' 10 ;;
    *) die "unknown dataset: $1" ;;
  esac
}

methods_for_dataset() {
  case "$1" in
    tictactoe|adult|dota2)
      printf '%s\n' 'arima arima_adaptive moirai moirai_online'
      ;;
    mnist|imdb)
      printf '%s\n' 'arima_mnist_imdb_layerwise arima_adaptive_mnist_imdb_layerwise moirai_mnist_imdb_layerwise moirai_online_mnist_imdb_layerwise'
      ;;
    *) die "unknown dataset: $1" ;;
  esac
}

is_layerwise_method() {
  case "$1" in *_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

is_adaptive_arima_method() {
  case "$1" in arima_adaptive|arima_adaptive_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

is_online_moirai_method() {
  case "$1" in moirai_online|moirai_online_mnist_imdb_layerwise) return 0 ;; *) return 1 ;; esac
}

print_command() {
  printf 'command:'
  for command_arg do
    printf ' <%s>' "$command_arg"
  done
  printf '\n'
}

handle_smoke_task() {
  mode=$1
  dataset=$2
  attack_method=$3
  alpha=$4
  result_root=$5
  task_root=$6
  gpu_id=$7
  set -- "$PYTHON_BIN" "$CANDIDATE_ROOT/FLCE_updated.py" \
    --dataset "$dataset" \
    --alpha "$alpha" \
    --client_num "$SMOKE_CLIENT_NUM" \
    --attacker_id "$SMOKE_ATTACKER_ID" \
    --attack_method "$attack_method" \
    --contribution_method "$CONTRIBUTION_METHOD" \
    --trial_id "$SMOKE_TRIAL_ID" \
    --num_epoch "$SMOKE_NUM_EPOCH" \
    --random_round "$SMOKE_RANDOM_ROUND" \
    --results_root "$result_root" \
    --use_attack \
    --use_lazy \
    --lazy_schedule "$SMOKE_SCHEDULE" \
    --lazy_honest_gap "$SMOKE_GAP" \
    --lazy_honest_phase 0 \
    --lazy_fake_policy attack

  if [ "$SMOKE_SCHEDULE" = probabilistic ]; then
    set -- "$@" --attack_probability "$SMOKE_PROBABILITY"
  fi
  if [ "$USE_HESSIAN" = 1 ]; then
    set -- "$@" --use_hessian --lambda_h "$LAMBDA_H"
  fi
  if is_layerwise_method "$attack_method"; then
    set -- "$@" \
      --layerwise_selection_metric "$LAYERWISE_SELECTION_METRIC" \
      --layerwise_selection_fraction "$LAYERWISE_SELECTION_FRACTION" \
      --layerwise_pca_threshold "$LAYERWISE_PCA_THRESHOLD" \
      --layerwise_pca_components "$LAYERWISE_PCA_COMPONENTS"
    if [ -n "$LAYERWISE_SELECTION_TOP_N" ]; then
      set -- "$@" --layerwise_selection_top_n "$LAYERWISE_SELECTION_TOP_N"
    fi
  fi
  if is_adaptive_arima_method "$attack_method"; then
    set -- "$@" --arima_retrain_interval "${ARIMA_RETRAIN_INTERVAL:-10}"
  fi
  if is_online_moirai_method "$attack_method"; then
    set -- "$@" \
      --moirai_tune_lr "${MOIRAI_TUNE_LR:-0.1}" \
      --moirai_tune_steps "${MOIRAI_TUNE_STEPS:-1}" \
      --moirai_adapt_interval "${MOIRAI_ADAPT_INTERVAL:-0}"
  fi

  if [ "$mode" = print ]; then
    print_command "$@"
    return 0
  fi

  manifest=$task_root/manifest.txt
  {
    printf 'task_id=%s\n' "$(basename "$task_root")"
    printf 'dataset=%s\nalpha=%s\nattack_method=%s\n' "$dataset" "$alpha" "$attack_method"
    printf 'client_num=%s\nattacker_id=%s\ntrial_id=%s\n' "$SMOKE_CLIENT_NUM" "$SMOKE_ATTACKER_ID" "$SMOKE_TRIAL_ID"
    printf 'schedule=%s\ngap=%s\nprobability=%s\n' "$SMOKE_SCHEDULE" "$SMOKE_GAP" "$SMOKE_PROBABILITY"
    printf 'gpu_id=%s\nrun_root=%s\nresults_root=%s\n' "$gpu_id" "$task_root" "$result_root"
    printf 'use_hessian=%s\nlambda_h=%s\n' "$USE_HESSIAN" "$LAMBDA_H"
    printf '%s\n' 'command_argv:'
    printf '  %s\n' "$@"
  } > "$manifest"

  log_file=$task_root/logs/run.log
  if (
    cd "$CANDIDATE_ROOT"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=$task_root/tmp \
    HF_HOME=$task_root/cache/huggingface \
    TORCH_HOME=$task_root/cache/torch \
    XDG_CACHE_HOME=$task_root/cache/xdg \
    "$@"
  ) > "$log_file" 2>&1; then
    printf '%s\n' success > "$task_root/status.txt"
    return 0
  fi
  exit_code=$?
  printf 'failed exit_code=%s\n' "$exit_code" > "$task_root/status.txt"
  return "$exit_code"
}

smoke_task_count=0
for dataset in $(datasets_for_stage); do
  for attack_method in $(methods_for_dataset "$dataset"); do
    smoke_task_count=$((smoke_task_count + 1))
  done
done

printf '%s\n' '=== CANDIDATE DEPLOY + SMOKE PLAN ==='
printf 'project_root       : %s\n' "$PROJECT_ROOT"
printf 'candidate_source   : %s\n' "$CANDIDATE_SOURCE"
printf 'deploy_root        : %s\n' "$DEPLOY_ROOT"
printf 'candidate_root     : %s\n' "$CANDIDATE_ROOT"
printf 'smoke_results_root : %s\n' "$SMOKE_RESULTS_ROOT"
printf 'smoke_stage/tasks  : %s / %s\n' "$SMOKE_STAGE" "$smoke_task_count"
printf 'short rounds/warmup: %s / %s\n' "$SMOKE_NUM_EPOCH" "$SMOKE_RANDOM_ROUND"
printf 'hessian/lambda_h   : %s / %s\n' "$USE_HESSIAN" "$LAMBDA_H"
printf 'moirai_asset_source: %s\n' "${MOIRAI_ASSET_SOURCE:-base-project lookup}"

if [ "$DRY_RUN" = 1 ]; then
  for dataset in $(datasets_for_stage); do
    alpha=$(alpha_for_dataset "$dataset")
    for attack_method in $(methods_for_dataset "$dataset"); do
      task_id=smoke_${dataset}_${attack_method}_trial${SMOKE_TRIAL_ID}
      handle_smoke_task print "$dataset" "$attack_method" "$alpha" \
        "$SMOKE_RESULTS_ROOT/$task_id" "$DEPLOY_ROOT/dry_run/$task_id" 0
    done
  done
  printf '%s\n' 'DRY_RUN=1; no directory was created and no command was executed.'
  exit 0
fi

command -v rsync >/dev/null 2>&1 || die 'rsync is required for isolated deployment'
command -v cp >/dev/null 2>&1 || die 'cp is required for isolated asset copies'
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python executable not found: $PYTHON_BIN"

mkdir -p "$CANDIDATE_PARENT" "$PROJECT_ROOT/Results"
mkdir "$DEPLOY_ROOT"
mkdir "$CANDIDATE_ROOT"

# Copy the code/config portion of the immutable base into a new destination.
# Large datasets, Uni2TS, and local model weights are copied below. GNU cp uses
# copy-on-write reflinks when the filesystem supports them and falls back to a
# real copy. This keeps the base tree protected even if runtime code writes.
rsync -a \
  --exclude='/Results/***' \
  --exclude='/.candidate_runs/***' \
  --exclude='/.matrix_runs/***' \
  --exclude='/candidate_uploads/***' \
  --exclude='/run_env*/***' \
  --exclude='/runs/***' \
  --exclude='/data/***' \
  --exclude='/uni2ts/***' \
  --exclude='/moirai-*/***' \
  --exclude='/distilbert*/***' \
  --exclude='/aclImdb/***' \
  "$PROJECT_ROOT/" "$CANDIDATE_ROOT/"

# Overlay the pre-uploaded corrected snapshot only inside the new candidate.
rsync -a \
  --exclude='/Results/***' \
  --exclude='/.candidate_runs/***' \
  --exclude='/.matrix_runs/***' \
  --exclude='/data/***' \
  --exclude='/uni2ts/***' \
  --exclude='/moirai-*/***' \
  --exclude='/distilbert*/***' \
  --exclude='/aclImdb/***' \
  "$CANDIDATE_SOURCE/" "$CANDIDATE_ROOT/"

copy_candidate_asset() {
  source_path=$1
  [ -e "$source_path" ] || return 0
  asset_name=$(basename "$source_path")
  destination=$CANDIDATE_ROOT/$asset_name
  [ ! -e "$destination" ] || die "candidate asset destination already exists: $destination"
  cp -a --reflink=auto "$source_path" "$destination"
}

# Preserve the original relative lookup paths without writable symlinks back
# into the existing project. Unmatched globs are ignored by the helper.
copy_candidate_asset "$PROJECT_ROOT/data"
copy_candidate_asset "$PROJECT_ROOT/uni2ts"
for immutable_asset in \
  "$PROJECT_ROOT"/moirai-1.1-R-* \
  "$PROJECT_ROOT"/distilbert* \
  "$PROJECT_ROOT"/aclImdb; do
  copy_candidate_asset "$immutable_asset"
done

if [ -n "$MOIRAI_ASSET_SOURCE" ]; then
  moirai_destination=$CANDIDATE_ROOT/moirai-1.1-R-small
  [ ! -e "$moirai_destination" ] ||
    die "Moirai destination already exists; refusing ambiguous asset override: $moirai_destination"
  cp -a --reflink=auto "$MOIRAI_ASSET_SOURCE" "$moirai_destination"
fi

[ -f "$CANDIDATE_ROOT/moirai-1.1-R-small/config.json" ] ||
  die 'candidate lacks local Moirai-small config.json'
[ -f "$CANDIDATE_ROOT/moirai-1.1-R-small/model.safetensors" ] ||
  die 'candidate lacks local Moirai-small model.safetensors'

for relative_path in $required_overlay_files; do
  [ -f "$CANDIDATE_ROOT/$relative_path" ] ||
    die "deployed candidate is missing: $relative_path"
done
[ -d "$CANDIDATE_ROOT/module" ] || die 'deployed candidate lacks module/'
[ -d "$CANDIDATE_ROOT/data" ] || die 'deployed candidate lacks data/'
[ -d "$CANDIDATE_ROOT/uni2ts" ] || die 'deployed candidate lacks uni2ts/'

mkdir "$SMOKE_RUNS_ROOT"
mkdir "$SMOKE_RESULTS_ROOT"

deployment_manifest=$CANDIDATE_ROOT/.candidate_deployment_manifest.txt
{
  printf 'deploy_id=%s\n' "$DEPLOY_ID"
  printf 'created_utc=%s\n' "$timestamp"
  printf 'project_root=%s\n' "$PROJECT_ROOT"
  printf 'candidate_source=%s\n' "$CANDIDATE_SOURCE"
  printf 'candidate_root=%s\n' "$CANDIDATE_ROOT"
  printf 'smoke_results_root=%s\n' "$SMOKE_RESULTS_ROOT"
  printf 'moirai_asset_source=%s\n' "${MOIRAI_ASSET_SOURCE:-$PROJECT_ROOT/moirai-1.1-R-small}"
  printf 'use_hessian=%s\n' "$USE_HESSIAN"
  printf 'lambda_h=%s\n' "$LAMBDA_H"
  sha256sum \
    "$CANDIDATE_ROOT/FLCE_updated.py" \
    "$CANDIDATE_ROOT/module/method/attack_updated_nonlayerwise.py" \
    "$CANDIDATE_ROOT/module/method/attack_updated_layerwise.py" \
    "$CANDIDATE_ROOT/tests/test_moirai_arima_correctness.py" \
    "$CANDIDATE_ROOT/tests/server_smoke_moirai_arima.py" \
    "$CANDIDATE_ROOT/tests/test_gpu_slot_scheduler.sh" \
    "$CANDIDATE_ROOT/tests/test_shapley_forecast_launchers.py" \
    "$CANDIDATE_ROOT/tests/test_shapley_forecast_scheduler.sh" \
    "$CANDIDATE_ROOT/script/deploy_and_smoke_moirai_arima.sh" \
    "$CANDIDATE_ROOT/script/run_full_moirai_arima_matrix.sh" \
    "$CANDIDATE_ROOT/script/run_moirai_arima_tuning_pilot.sh" \
    "$CANDIDATE_ROOT/script/analyze_moirai_arima_tuning.py" \
    "$CANDIDATE_ROOT/script/run_tictactoe_forecast_pure_shapley.sh" \
    "$CANDIDATE_ROOT/script/run_tictactoe_forecast_lazy_shapley.sh" \
    "$CANDIDATE_ROOT/script/run_adult_forecast_pure_shapley.sh" \
    "$CANDIDATE_ROOT/script/run_adult_forecast_lazy_shapley.sh" \
    "$CANDIDATE_ROOT/script/run_dota2_forecast_pure_shapley.sh" \
    "$CANDIDATE_ROOT/script/run_dota2_forecast_lazy_shapley.sh" \
    "$CANDIDATE_ROOT/moirai-1.1-R-small/config.json" \
    "$CANDIDATE_ROOT/moirai-1.1-R-small/model.safetensors"
} > "$deployment_manifest"

if [ "$RUN_DEPENDENCY_PROBES" = 1 ]; then
  mkdir "$DEPLOY_ROOT/probes"
  mkdir "$DEPLOY_ROOT/probes/cache" "$DEPLOY_ROOT/probes/tmp"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L > "$DEPLOY_ROOT/probes/nvidia-smi.txt"
  else
    die 'nvidia-smi is not available'
  fi
  (
    cd "$CANDIDATE_ROOT"
    export PYTHONDONTWRITEBYTECODE=1
    export TMPDIR=$DEPLOY_ROOT/probes/tmp
    export HF_HOME=$DEPLOY_ROOT/probes/cache/huggingface
    export TORCH_HOME=$DEPLOY_ROOT/probes/cache/torch
    export XDG_CACHE_HOME=$DEPLOY_ROOT/probes/cache/xdg
    export PYTHONPATH=$CANDIDATE_ROOT/uni2ts/src${PYTHONPATH:+:$PYTHONPATH}
    "$PYTHON_BIN" -c \
      'import torch, numpy, pandas, sklearn, pmdarima; from uni2ts.model.moirai import MoiraiForecast, MoiraiModule; print("imports_ok", torch.__version__)'
    "$PYTHON_BIN" -c \
      'import numpy as np; from pmdarima import auto_arima; x=np.array([0.,1.,0.,1.,0.,1.,0.,1.]); m=auto_arima(x, seasonal=False, suppress_warnings=True, error_action="ignore"); print("pmdarima_predict", m.predict(n_periods=1))'
  ) > "$DEPLOY_ROOT/probes/python_dependencies.log" 2>&1
fi

# The six production launchers are part of the deployed trust boundary.  Run
# their exhaustive argv contract plus a mock real-run scheduler/failure test
# before a candidate may receive either a deploy-only handoff or smoke marker.
launcher_contract_log=$DEPLOY_ROOT/shapley_launcher_contracts.log
if ! (
  cd "$CANDIDATE_ROOT"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m unittest \
    tests.test_shapley_forecast_launchers
  sh tests/test_shapley_forecast_scheduler.sh
) > "$launcher_contract_log" 2>&1; then
  die "Shapley launcher contracts failed; inspect $launcher_contract_log"
fi
printf 'shapley_launcher_contracts=passed\nlauncher_contract_log=%s\n' \
  "$launcher_contract_log" >> "$deployment_manifest"

if [ "$DEPLOY_ONLY" = 1 ]; then
  printf 'candidate_root=%s\n' "$CANDIDATE_ROOT"
  printf 'smoke_results_root=%s\n' "$SMOKE_RESULTS_ROOT"
  printf '%s\n' 'DEPLOY_ONLY=1; candidate deployed and probes completed; no end-to-end smoke task was launched.'
  exit 0
fi

gpu_count=0
for unused_gpu in $SMOKE_GPU_IDS; do gpu_count=$((gpu_count + 1)); done
[ "$gpu_count" -gt 0 ] || die 'SMOKE_GPU_IDS is empty'

failures=0
task_index=0
for dataset in $(datasets_for_stage); do
  alpha=$(alpha_for_dataset "$dataset")
  for attack_method in $(methods_for_dataset "$dataset"); do
    task_id=smoke_${dataset}_${attack_method}_trial${SMOKE_TRIAL_ID}
    task_root=$SMOKE_RUNS_ROOT/$task_id
    task_results=$SMOKE_RESULTS_ROOT/$task_id
    [ ! -e "$task_root" ] || die "smoke run root collision: $task_root"
    [ ! -e "$task_results" ] || die "smoke results collision: $task_results"
    mkdir "$task_root"
    mkdir "$task_root/logs" "$task_root/tmp" "$task_root/cache"
    mkdir "$task_results"

    gpu_position=$((task_index % gpu_count + 1))
    gpu_id=$(printf '%s\n' $SMOKE_GPU_IDS | sed -n "${gpu_position}p")
    task_index=$((task_index + 1))
    if ! handle_smoke_task run "$dataset" "$attack_method" "$alpha" \
      "$task_results" "$task_root" "$gpu_id"; then
      failures=$((failures + 1))
    fi
  done
done

printf 'candidate_root=%s\n' "$CANDIDATE_ROOT"
printf 'smoke_results_root=%s\n' "$SMOKE_RESULTS_ROOT"
printf 'smoke_tasks=%s failures=%s\n' "$smoke_task_count" "$failures"
[ "$failures" -eq 0 ] || exit 1

# Only a completely successful end-to-end smoke creates the gate consumed by
# the full-matrix launcher. Store relative paths so the checksum file is
# verifiable from this exact candidate root.
(
  cd "$CANDIDATE_ROOT"
  sha256sum \
    FLCE_updated.py \
    module/method/attack_updated_nonlayerwise.py \
    module/method/attack_updated_layerwise.py \
    tests/test_moirai_arima_correctness.py \
    tests/server_smoke_moirai_arima.py \
    tests/test_gpu_slot_scheduler.sh \
    tests/test_shapley_forecast_launchers.py \
    tests/test_shapley_forecast_scheduler.sh \
    script/deploy_and_smoke_moirai_arima.sh \
    script/run_full_moirai_arima_matrix.sh \
    script/run_moirai_arima_tuning_pilot.sh \
    script/analyze_moirai_arima_tuning.py \
    script/run_tictactoe_forecast_pure_shapley.sh \
    script/run_tictactoe_forecast_lazy_shapley.sh \
    script/run_adult_forecast_pure_shapley.sh \
    script/run_adult_forecast_lazy_shapley.sh \
    script/run_dota2_forecast_pure_shapley.sh \
    script/run_dota2_forecast_lazy_shapley.sh \
    moirai-1.1-R-small/config.json \
    moirai-1.1-R-small/model.safetensors
) > "$CANDIDATE_ROOT/.smoke_passed.sha256"
{
  printf 'deploy_id=%s\n' "$DEPLOY_ID"
  printf 'passed_utc=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'smoke_stage=%s\nsmoke_tasks=%s\nfailures=0\n' "$SMOKE_STAGE" "$smoke_task_count"
  printf 'smoke_results_root=%s\n' "$SMOKE_RESULTS_ROOT"
} > "$CANDIDATE_ROOT/.smoke_passed_manifest.txt"

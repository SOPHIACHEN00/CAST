#!/usr/bin/env sh
# Staged, isolated Moirai/ARIMA parameter-tuning launcher.
#
# This is deliberately a PILOT, not the paper's full matrix.  It preserves the
# full protocol universes below, but selects an explicit, validated subset for
# parameter screening.  After a profile wins, use the robustness stage and
# only then lock the winner into run_full_moirai_arima_matrix.sh.
#
# Safe default (enumerates/prints only; creates nothing):
#   CANDIDATE_ROOT=/output/gradpred_exp/attack/.candidate_runs/<id>/attack \
#     sh script/run_moirai_arima_tuning_pilot.sh
#
# Real screen (requires a newly deployed, smoke-passed candidate):
#   DRY_RUN=0 CONFIRM_TUNING=1 \
#   CANDIDATE_ROOT=/output/gradpred_exp/attack/.candidate_runs/<id>/attack \
#     sh script/run_moirai_arima_tuning_pilot.sh
#
# Robustness check after inspecting the screen results:
#   PILOT_STAGE=robustness PILOT_PROFILE_SET=responsive \
#   DRY_RUN=0 CONFIRM_TUNING=1 CANDIDATE_ROOT=... \
#     sh script/run_moirai_arima_tuning_pilot.sh

set -eu

PROJECT_ROOT=${PROJECT_ROOT:-/output/gradpred_exp/attack}
script_parent=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
CANDIDATE_ROOT=${CANDIDATE_ROOT:-$script_parent}
PYTHON_BIN=${PYTHON_BIN:-/root/conda_envs/flce_attack/bin/python}

# Safety gates.  A bare invocation is always read-only.
DRY_RUN=${DRY_RUN:-1}
CONFIRM_TUNING=${CONFIRM_TUNING:-0}
DRY_RUN_PRINT_LIMIT=${DRY_RUN_PRINT_LIMIT:-12}
PILOT_STAGE=${PILOT_STAGE:-screen}

# These are experimental-protocol locks, not tuning knobs.
NUM_EPOCH=${NUM_EPOCH:-50}
NUM_LOCAL_EPOCHS=${NUM_LOCAL_EPOCHS:-1}
RANDOM_ROUND=${RANDOM_ROUND:-10}
USE_HESSIAN=${USE_HESSIAN:-1}
LAMBDA_H=${LAMBDA_H:-0.9}
LAZY_HONEST_PHASE=${LAZY_HONEST_PHASE:-0}
LAZY_FAKE_POLICY=${LAZY_FAKE_POLICY:-attack}
CONTRIBUTION_METHOD=${CONTRIBUTION_METHOD:-shapley}
USE_LAST_ROUND_SV=${USE_LAST_ROUND_SV:-1}
ATTACKER_ID=${ATTACKER_ID:-1}

# Layerwise architecture is held fixed while forecast parameters are screened.
LAYERWISE_SELECTION_METRIC=${LAYERWISE_SELECTION_METRIC:-normalized_rms}
LAYERWISE_SELECTION_FRACTION=${LAYERWISE_SELECTION_FRACTION:-0.5}
LAYERWISE_SELECTION_TOP_N=${LAYERWISE_SELECTION_TOP_N:-}
LAYERWISE_PCA_THRESHOLD=${LAYERWISE_PCA_THRESHOLD:-512}
LAYERWISE_PCA_COMPONENTS=${LAYERWISE_PCA_COMPONENTS:-8}

GPU_IDS=${GPU_IDS:-0 1 2 3}
MAX_PARALLEL=${MAX_PARALLEL:-4}

# Canonical protocol universes.  Pilot selections are checked against these
# exact values, so the launcher cannot silently invent a new alpha, client
# count, gap, or paired probability.
CANONICAL_DATASETS='tictactoe adult dota2 mnist imdb'
CANONICAL_CLIENT_NUMS='2 4 6 8'
CANONICAL_GAP_PROB_PAIRS='2:0.50 3:0.66 6:0.82 10:0.90'
CANONICAL_SCENARIOS='fixed probabilistic'

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

case "$PILOT_STAGE" in
  screen)
    # Minimal timing/parameter gate: one representative alpha, client=4, and
    # fixed gap=2.  ARIMA compares baseline/responsive; inexpensive Moirai also
    # includes the stronger stable profile.  Expand schedules only explicitly.
    PILOT_PROFILE_SET=${PILOT_PROFILE_SET:-screen}
    if [ "$PILOT_PROFILE_SET" = confirm7 ]; then
      PILOT_FAMILIES=${PILOT_FAMILIES:-'arima moirai'}
      PILOT_DATASETS=${PILOT_DATASETS:-'mnist imdb'}
    else
      PILOT_FAMILIES=${PILOT_FAMILIES:-'arima moirai'}
      PILOT_DATASETS=${PILOT_DATASETS:-$CANONICAL_DATASETS}
    fi
    PILOT_CLIENT_NUMS=${PILOT_CLIENT_NUMS:-4}
    PILOT_GAP_PROB_PAIRS=${PILOT_GAP_PROB_PAIRS:-'2:0.50'}
    PILOT_SCENARIOS=${PILOT_SCENARIOS:-fixed}
    TRIAL_IDS=${TRIAL_IDS:-0}
    PILOT_ALPHAS_TICTACTOE=${PILOT_ALPHAS_TICTACTOE:-1000}
    PILOT_ALPHAS_ADULT=${PILOT_ALPHAS_ADULT:-1000000}
    PILOT_ALPHAS_DOTA2=${PILOT_ALPHAS_DOTA2:-50000000}
    PILOT_ALPHAS_MNIST=${PILOT_ALPHAS_MNIST:-100000}
    PILOT_ALPHAS_IMDB=${PILOT_ALPHAS_IMDB:-1000}
    ;;
  robustness)
    # Edge alphas/clients and all declared schedules.  One trial is the safe
    # runtime gate; set TRIAL_IDS='0 1 2' explicitly after it passes.  Exactly
    # one screened profile must be named; the script never chooses a winner on
    # the researcher's behalf.
    : "${PILOT_PROFILE_SET:?For PILOT_STAGE=robustness set PILOT_PROFILE_SET=responsive or stable}"
    case "$PILOT_PROFILE_SET" in
      responsive|stable) ;;
      *) die 'robustness requires PILOT_PROFILE_SET=responsive or stable' ;;
    esac
    PILOT_FAMILIES=${PILOT_FAMILIES:-'arima moirai'}
    PILOT_DATASETS=${PILOT_DATASETS:-$CANONICAL_DATASETS}
    PILOT_CLIENT_NUMS=${PILOT_CLIENT_NUMS:-'2 8'}
    PILOT_GAP_PROB_PAIRS=${PILOT_GAP_PROB_PAIRS:-$CANONICAL_GAP_PROB_PAIRS}
    PILOT_SCENARIOS=${PILOT_SCENARIOS:-$CANONICAL_SCENARIOS}
    TRIAL_IDS=${TRIAL_IDS:-0}
    PILOT_ALPHAS_TICTACTOE=${PILOT_ALPHAS_TICTACTOE:-'10 20000'}
    PILOT_ALPHAS_ADULT=${PILOT_ALPHAS_ADULT:-'1000 20000000'}
    PILOT_ALPHAS_DOTA2=${PILOT_ALPHAS_DOTA2:-'10000 200000000'}
    PILOT_ALPHAS_MNIST=${PILOT_ALPHAS_MNIST:-'10 10000000'}
    PILOT_ALPHAS_IMDB=${PILOT_ALPHAS_IMDB:-'10 200000'}
    ;;
  *) die 'PILOT_STAGE must be screen or robustness' ;;
esac

case "$PILOT_PROFILE_SET" in
  screen|responsive|stable|arima_refine|confirm7) ;;
  *) die 'PILOT_PROFILE_SET must be screen, responsive, stable, arima_refine, or confirm7' ;;
esac
if [ "$PILOT_STAGE" = robustness ] && [ "$PILOT_PROFILE_SET" = screen ]; then
  die 'robustness cannot run the full screening profile set'
fi
if [ "$PILOT_PROFILE_SET" = arima_refine ]; then
  [ "$PILOT_STAGE" = screen ] || die 'arima_refine requires PILOT_STAGE=screen'
  [ "$PILOT_FAMILIES" = arima ] || die 'arima_refine requires PILOT_FAMILIES=arima'
  for refine_dataset in $PILOT_DATASETS; do
    case "$refine_dataset" in
      mnist|imdb) ;;
      *) die 'arima_refine datasets must be a non-empty subset of: mnist imdb' ;;
    esac
  done
fi
if [ "$PILOT_PROFILE_SET" = confirm7 ]; then
  [ "$PILOT_STAGE" = screen ] || die 'confirm7 requires PILOT_STAGE=screen'
  [ "$PILOT_FAMILIES" = 'arima moirai' ] ||
    die 'confirm7 requires PILOT_FAMILIES="arima moirai"'
  [ "$PILOT_DATASETS" = 'mnist imdb' ] ||
    die 'confirm7 requires PILOT_DATASETS="mnist imdb"'
  [ "$PILOT_CLIENT_NUMS" = 4 ] || die 'confirm7 requires PILOT_CLIENT_NUMS=4'
  [ "$PILOT_GAP_PROB_PAIRS" = '2:0.50' ] ||
    die 'confirm7 requires PILOT_GAP_PROB_PAIRS=2:0.50'
  [ "$PILOT_SCENARIOS" = fixed ] || die 'confirm7 requires PILOT_SCENARIOS=fixed'
  [ "$TRIAL_IDS" = 0 ] || die 'confirm7 requires TRIAL_IDS=0'
  [ "$PILOT_ALPHAS_MNIST" = 100000 ] ||
    die 'confirm7 requires PILOT_ALPHAS_MNIST=100000'
  [ "$PILOT_ALPHAS_IMDB" = 1000 ] ||
    die 'confirm7 requires PILOT_ALPHAS_IMDB=1000'
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
CAMPAIGN_ID=${CAMPAIGN_ID:-forecast_tuning_${PILOT_STAGE}_${PILOT_PROFILE_SET}_${timestamp}_pid$$}
CAMPAIGN_PARENT=${CAMPAIGN_PARENT:-${PROJECT_ROOT}/.tuning_runs}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-${CAMPAIGN_PARENT}/${CAMPAIGN_ID}}
TASKS_ROOT=${TASKS_ROOT:-${CAMPAIGN_ROOT}/tasks}
RESULTS_CAMPAIGN_ROOT=${RESULTS_CAMPAIGN_ROOT:-${PROJECT_ROOT}/Results/${CAMPAIGN_ID}}

case "$CAMPAIGN_ID" in
  ''|*[!A-Za-z0-9._-]*) die "CAMPAIGN_ID must be one safe path segment: $CAMPAIGN_ID" ;;
esac
case "$DRY_RUN" in 0|1) ;; *) die 'DRY_RUN must be 0 or 1' ;; esac
case "$CONFIRM_TUNING" in 0|1) ;; *) die 'CONFIRM_TUNING must be 0 or 1' ;; esac
case "$USE_HESSIAN" in 0|1) ;; *) die 'USE_HESSIAN must be 0 or 1' ;; esac
case "$USE_LAST_ROUND_SV" in 0|1) ;; *) die 'USE_LAST_ROUND_SV must be 0 or 1' ;; esac
case "$MAX_PARALLEL" in ''|*[!0-9]*) die 'MAX_PARALLEL must be a positive integer' ;; esac
case "$DRY_RUN_PRINT_LIMIT" in ''|*[!0-9]*) die 'DRY_RUN_PRINT_LIMIT must be a non-negative integer' ;; esac
[ "$MAX_PARALLEL" -gt 0 ] || die 'MAX_PARALLEL must be at least 1'

# Refuse any drift in settings that the user explicitly locked.
[ "$NUM_EPOCH" -eq 50 ] || die 'tuning protocol requires NUM_EPOCH=50'
[ "$NUM_LOCAL_EPOCHS" -eq 1 ] || die 'tuning protocol requires NUM_LOCAL_EPOCHS=1'
[ "$RANDOM_ROUND" -eq 10 ] || die 'tuning protocol requires RANDOM_ROUND=10'
[ "$USE_HESSIAN" -eq 1 ] || die 'tuning protocol requires USE_HESSIAN=1'
[ "$LAMBDA_H" = 0.9 ] || die 'tuning protocol requires LAMBDA_H=0.9'
[ "$LAZY_HONEST_PHASE" -eq 0 ] || die 'tuning protocol requires LAZY_HONEST_PHASE=0'
[ "$LAZY_FAKE_POLICY" = attack ] || die 'tuning protocol requires LAZY_FAKE_POLICY=attack'
[ "$CONTRIBUTION_METHOD" = shapley ] || die 'tuning protocol requires CONTRIBUTION_METHOD=shapley'
[ "$USE_LAST_ROUND_SV" -eq 1 ] || die 'tuning protocol requires USE_LAST_ROUND_SV=1'

[ -d "$PROJECT_ROOT" ] || die "PROJECT_ROOT does not exist: $PROJECT_ROOT"
[ -d "$CANDIDATE_ROOT" ] || die "CANDIDATE_ROOT does not exist: $CANDIDATE_ROOT"
PROJECT_ROOT=$(CDPATH= cd -- "$PROJECT_ROOT" && pwd -P)
CANDIDATE_ROOT=$(CDPATH= cd -- "$CANDIDATE_ROOT" && pwd -P)
case "$CANDIDATE_ROOT" in
  "$PROJECT_ROOT"/.candidate_runs/*/attack) ;;
  *) die "candidate must be isolated below $PROJECT_ROOT/.candidate_runs" ;;
esac
[ "$CAMPAIGN_PARENT" = "$PROJECT_ROOT/.tuning_runs" ] ||
  die "CAMPAIGN_PARENT must be $PROJECT_ROOT/.tuning_runs"
[ "$CAMPAIGN_ROOT" = "$CAMPAIGN_PARENT/$CAMPAIGN_ID" ] ||
  die 'CAMPAIGN_ROOT must be CAMPAIGN_PARENT/CAMPAIGN_ID'
[ "$TASKS_ROOT" = "$CAMPAIGN_ROOT/tasks" ] || die 'TASKS_ROOT must be CAMPAIGN_ROOT/tasks'
[ "$RESULTS_CAMPAIGN_ROOT" = "$PROJECT_ROOT/Results/$CAMPAIGN_ID" ] ||
  die 'RESULTS_CAMPAIGN_ROOT must be PROJECT_ROOT/Results/CAMPAIGN_ID'

ENTRY=$CANDIDATE_ROOT/FLCE_updated.py
NONLAYER_MODULE=$CANDIDATE_ROOT/module/method/attack_updated_nonlayerwise.py
LAYER_MODULE=$CANDIDATE_ROOT/module/method/attack_updated_layerwise.py
SCRIPT_PATH=$CANDIDATE_ROOT/script/run_moirai_arima_tuning_pilot.sh
[ -f "$ENTRY" ] || die "candidate runner is missing: $ENTRY"
[ -f "$NONLAYER_MODULE" ] || die 'candidate non-layerwise module is missing'
[ -f "$LAYER_MODULE" ] || die 'candidate layerwise module is missing'
[ -f "$SCRIPT_PATH" ] || die 'candidate tuning launcher is missing'

canonical_alpha_values() {
  case "$1" in
    tictactoe) printf '%s\n' '10 100 1000 20000' ;;
    adult) printf '%s\n' '1000 100000 1000000 20000000' ;;
    dota2) printf '%s\n' '10000 1000000 50000000 200000000' ;;
    mnist) printf '%s\n' '10 1000 100000 10000000' ;;
    imdb) printf '%s\n' '10 100 1000 200000' ;;
    *) die "unknown dataset: $1" ;;
  esac
}

selected_alpha_values() {
  case "$1" in
    tictactoe) printf '%s\n' "$PILOT_ALPHAS_TICTACTOE" ;;
    adult) printf '%s\n' "$PILOT_ALPHAS_ADULT" ;;
    dota2) printf '%s\n' "$PILOT_ALPHAS_DOTA2" ;;
    mnist) printf '%s\n' "$PILOT_ALPHAS_MNIST" ;;
    imdb) printf '%s\n' "$PILOT_ALPHAS_IMDB" ;;
    *) die "unknown dataset: $1" ;;
  esac
}

contains_word() {
  wanted=$1
  shift
  for candidate_word do
    [ "$candidate_word" = "$wanted" ] && return 0
  done
  return 1
}

validate_unique_words() {
  value_list=$1
  label=$2
  seen_words=' '
  for value_word in $value_list; do
    case "$seen_words" in
      *" $value_word "*) die "$label contains duplicate value: $value_word" ;;
    esac
    seen_words="$seen_words$value_word "
  done
}

validate_unique_words "$PILOT_DATASETS" PILOT_DATASETS
validate_unique_words "$PILOT_FAMILIES" PILOT_FAMILIES
validate_unique_words "$PILOT_CLIENT_NUMS" PILOT_CLIENT_NUMS
validate_unique_words "$PILOT_GAP_PROB_PAIRS" PILOT_GAP_PROB_PAIRS
validate_unique_words "$PILOT_SCENARIOS" PILOT_SCENARIOS
validate_unique_words "$TRIAL_IDS" TRIAL_IDS

[ -n "$PILOT_DATASETS" ] || die 'PILOT_DATASETS is empty'
[ -n "$PILOT_FAMILIES" ] || die 'PILOT_FAMILIES is empty'
[ -n "$PILOT_CLIENT_NUMS" ] || die 'PILOT_CLIENT_NUMS is empty'
[ -n "$PILOT_GAP_PROB_PAIRS" ] || die 'PILOT_GAP_PROB_PAIRS is empty'
[ -n "$PILOT_SCENARIOS" ] || die 'PILOT_SCENARIOS is empty'
[ -n "$TRIAL_IDS" ] || die 'TRIAL_IDS is empty'

for dataset in $PILOT_DATASETS; do
  contains_word "$dataset" $CANONICAL_DATASETS || die "non-canonical dataset selected: $dataset"
  selected_alphas=$(selected_alpha_values "$dataset")
  [ -n "$selected_alphas" ] || die "selected alpha list is empty for $dataset"
  validate_unique_words "$selected_alphas" "selected alphas for $dataset"
  canonical_alphas=$(canonical_alpha_values "$dataset")
  for alpha in $selected_alphas; do
    contains_word "$alpha" $canonical_alphas ||
      die "alpha=$alpha is outside the canonical $dataset alpha set"
  done
done
for pilot_family in $PILOT_FAMILIES; do
  case "$pilot_family" in
    arima|moirai) ;;
    *) die "PILOT_FAMILIES contains unknown family: $pilot_family" ;;
  esac
done
for client_num in $PILOT_CLIENT_NUMS; do
  contains_word "$client_num" $CANONICAL_CLIENT_NUMS ||
    die "client_num=$client_num is outside the canonical client set"
  [ "$ATTACKER_ID" -lt "$client_num" ] ||
    die "ATTACKER_ID=$ATTACKER_ID is invalid for client_num=$client_num"
done
for pair in $PILOT_GAP_PROB_PAIRS; do
  contains_word "$pair" $CANONICAL_GAP_PROB_PAIRS ||
    die "gap/probability pair is not canonical: $pair"
done
for scenario in $PILOT_SCENARIOS; do
  contains_word "$scenario" $CANONICAL_SCENARIOS || die "unknown scenario: $scenario"
done
for trial_id in $TRIAL_IDS; do
  case "$trial_id" in ''|*[!0-9]*) die "trial id must be a non-negative integer: $trial_id" ;; esac
done

gpu_count=0
active_gpu_ids=''
seen_gpu_ids=' '
for configured_gpu in $GPU_IDS; do
  case "$configured_gpu" in
    ''|*[!0-9]*) die "GPU_IDS contains a non-negative-integer violation: $configured_gpu" ;;
  esac
  case "$configured_gpu" in
    0|[1-9]*) ;;
    *) die "GPU_IDS must use canonical decimal ids without leading zeros: $configured_gpu" ;;
  esac
  case "$seen_gpu_ids" in
    *" $configured_gpu "*) die "GPU_IDS contains duplicate GPU id: $configured_gpu" ;;
  esac
  seen_gpu_ids="$seen_gpu_ids$configured_gpu "
  gpu_count=$((gpu_count + 1))
  if [ "$gpu_count" -le "$MAX_PARALLEL" ]; then
    active_gpu_ids="$active_gpu_ids $configured_gpu"
  fi
done
active_gpu_ids=${active_gpu_ids# }
[ "$gpu_count" -gt 0 ] || die 'GPU_IDS is empty'
[ "$MAX_PARALLEL" -le "$gpu_count" ] || die 'MAX_PARALLEL exceeds configured GPU_IDS'

is_layerwise_dataset() {
  case "$1" in mnist|imdb) return 0 ;; *) return 1 ;; esac
}

method_for_variant() {
  dataset_name=$1
  variant_kind=$2
  if is_layerwise_dataset "$dataset_name"; then
    case "$variant_kind" in
      arima_offline) printf '%s\n' arima_mnist_imdb_layerwise ;;
      arima_adaptive) printf '%s\n' arima_adaptive_mnist_imdb_layerwise ;;
      moirai_offline) printf '%s\n' moirai_mnist_imdb_layerwise ;;
      moirai_online) printf '%s\n' moirai_online_mnist_imdb_layerwise ;;
      *) die "unknown variant kind: $variant_kind" ;;
    esac
  else
    case "$variant_kind" in
      arima_offline) printf '%s\n' arima ;;
      arima_adaptive) printf '%s\n' arima_adaptive ;;
      moirai_offline) printf '%s\n' moirai ;;
      moirai_online) printf '%s\n' moirai_online ;;
      *) die "unknown variant kind: $variant_kind" ;;
    esac
  fi
}

# ARIMA record fields:
# name|var_window|var_threshold|err_threshold|min_after_start|retrain_interval|ema_beta|trim_history
arima_profile_records() {
  dataset_name=$1
  case "$dataset_name" in
    tictactoe)
      baseline='baseline|3|0.005|0.005|2|10|0.9|1'
      responsive='responsive|3|0.005|0.016|4|15|0.9|1'
      stable='stable|3|0.005|0.024|5|10|0.9|1'
      ;;
    adult)
      baseline='baseline|3|0.005|0.005|2|10|0.9|1'
      responsive='responsive|3|0.005|0.08|4|15|0.9|1'
      stable='stable|3|0.005|0.12|5|10|0.9|1'
      ;;
    dota2)
      baseline='baseline|3|0.005|0.005|2|10|0.9|1'
      responsive='responsive|3|0.005|0.18|4|15|0.9|1'
      stable='stable|3|0.005|0.25|5|10|0.9|1'
      ;;
    mnist)
      baseline='baseline|3|0.05|0.035|4|10|0.9|1'
      responsive='responsive|3|0.05|0.05|4|5|0.9|1'
      stable='stable|3|0.05|0.08|5|10|0.9|1'
      ;;
    imdb)
      baseline='baseline|3|0.05|0.015|4|10|0.9|1'
      responsive='responsive|3|0.10|0.015|4|5|0.9|1'
      stable='stable|3|0.10|0.025|5|10|0.9|1'
      ;;
    *) die "unknown dataset: $dataset_name" ;;
  esac
  case "$PILOT_PROFILE_SET" in
    screen) printf '%s\n%s\n' "$baseline" "$responsive" ;;
    responsive) printf '%s\n' "$responsive" ;;
    stable) printf '%s\n' "$stable" ;;
    arima_refine)
      case "$dataset_name" in
        mnist)
          printf '%s\n' \
            'refine_i10|3|0.05|0.05|4|10|0.9|1' \
            'refine_i15|3|0.05|0.05|4|15|0.9|1'
          ;;
        imdb)
          printf '%s\n' \
            'refine_i10|3|0.10|0.015|4|10|0.9|1' \
            'refine_i15|3|0.10|0.015|4|15|0.9|1'
          ;;
        *) die 'arima_refine profile records support only mnist and imdb' ;;
      esac
      ;;
    confirm7)
      case "$dataset_name" in
        mnist)
          printf '%s\n' \
            'confirm_i10|3|0.05|0.05|4|10|0.9|1' \
            'confirm_i15|3|0.05|0.05|4|15|0.9|1'
          ;;
        imdb)
          printf '%s\n' \
            'confirm_i10|3|0.10|0.015|4|10|0.9|1' \
            'confirm_i15|3|0.10|0.015|4|15|0.9|1'
          ;;
        *) die 'confirm7 ARIMA records support only mnist and imdb' ;;
      esac
      ;;
  esac
}

# Moirai record fields:
# name|tune_lr|tune_steps|tune_start_round|adapt_interval|residual_decay|residual_clip|ema_beta|err_threshold|min_after_start|trim_history
moirai_profile_records() {
  dataset_name=$1
  case "$dataset_name" in
    tictactoe)
      baseline='baseline|0.001|1|10|10|0.9|5.0|0.9|0.005|2|1'
      responsive='responsive|0.05|1|10|4|0.95|5.0|0.9|0.025|4|1'
      stable='stable|0.10|1|10|0|1.0|5.0|0.9|0.04|5|1'
      ;;
    adult)
      baseline='baseline|0.001|1|10|10|0.9|5.0|0.9|0.005|2|1'
      responsive='responsive|0.003|1|10|4|0.95|5.0|0.9|0.12|4|1'
      stable='stable|0.01|1|10|0|1.0|5.0|0.9|0.18|5|1'
      ;;
    dota2)
      baseline='baseline|0.001|1|10|10|0.9|5.0|0.9|0.005|2|1'
      responsive='responsive|0.001|1|10|4|0.95|5.0|0.9|0.25|4|1'
      stable='stable|0.003|1|10|0|1.0|5.0|0.9|0.35|5|1'
      ;;
    mnist)
      baseline='baseline|0.001|1|10|10|0.9|5.0|0.9|0.045|4|1'
      responsive='responsive|0.003|1|10|4|0.95|5.0|0.9|0.06|4|1'
      stable='stable|0.01|1|10|0|1.0|5.0|0.9|0.10|5|1'
      ;;
    imdb)
      baseline='baseline|0.001|1|10|10|0.9|5.0|0.9|0.015|4|1'
      responsive='responsive|0.05|1|10|4|0.95|5.0|0.9|0.025|4|1'
      stable='stable|0.10|1|10|0|1.0|5.0|0.9|0.04|5|1'
      ;;
    *) die "unknown dataset: $dataset_name" ;;
  esac
  case "$PILOT_PROFILE_SET" in
    screen) printf '%s\n%s\n%s\n' "$baseline" "$responsive" "$stable" ;;
    responsive) printf '%s\n' "$responsive" ;;
    stable) printf '%s\n' "$stable" ;;
    confirm7)
      case "$dataset_name" in
        mnist) : ;;
        imdb)
          printf '%s\n' \
            'confirm_lr01|0.01|1|10|4|0.95|5.0|0.9|0.025|4|1' \
            'confirm_lr02|0.02|1|10|4|0.95|5.0|0.9|0.025|4|1' \
            'confirm_lr03|0.03|1|10|4|0.95|5.0|0.9|0.025|4|1'
          ;;
        *) die 'confirm7 Moirai records support only imdb' ;;
      esac
      ;;
  esac
}

# Offline ARIMA is de-duplicated when profiles share the same readiness gate:
# reset/retrain settings do not affect the offline model.  IMDB gets a second
# offline control for its early-readiness (0.10) candidates.  A robustness
# profile has exactly four variants per condition.
variant_records() {
  dataset_name=$1
  if contains_word arima $PILOT_FAMILIES; then
    if [ "$PILOT_PROFILE_SET" = arima_refine ] || [ "$PILOT_PROFILE_SET" = confirm7 ]; then
      for arima_record in $(arima_profile_records "$dataset_name"); do
        printf 'arima_adaptive|%s\n' "$arima_record"
      done
    elif [ "$PILOT_PROFILE_SET" = screen ]; then
      first_arima=1
      for arima_record in $(arima_profile_records "$dataset_name"); do
        saved_profile_ifs=$IFS
        IFS='|'
        set -- $arima_record
        IFS=$saved_profile_ifs
        arima_profile_name=$1
        if [ "$first_arima" -eq 1 ]; then
          printf 'arima_offline|%s\n' "$arima_record"
          first_arima=0
        elif [ "$dataset_name" = imdb ] && [ "$arima_profile_name" = responsive ]; then
          printf 'arima_offline|%s\n' "$arima_record"
        fi
        printf 'arima_adaptive|%s\n' "$arima_record"
      done
    else
      for arima_record in $(arima_profile_records "$dataset_name"); do
        printf 'arima_offline|%s\n' "$arima_record"
        printf 'arima_adaptive|%s\n' "$arima_record"
      done
    fi
  fi
  if contains_word moirai $PILOT_FAMILIES; then
    if [ "$PILOT_PROFILE_SET" = confirm7 ]; then
      for moirai_record in $(moirai_profile_records "$dataset_name"); do
        printf 'moirai_online|%s\n' "$moirai_record"
      done
    else
      printf '%s\n' 'moirai_offline|control'
      for moirai_record in $(moirai_profile_records "$dataset_name"); do
        printf 'moirai_online|%s\n' "$moirai_record"
      done
    fi
  fi
}

fixed_honest_fake_counts() {
  case "$1" in
    2) printf '%s\n' '25:25' ;;
    3) printf '%s\n' '17:33' ;;
    6) printf '%s\n' '9:41' ;;
    10) printf '%s\n' '5:45' ;;
    *) die "unknown canonical fixed gap: $1" ;;
  esac
}

print_command() {
  printf 'command:'
  for command_arg do printf ' <%s>' "$command_arg"; done
  printf '\n'
}

# Build one exact argv and either print it or execute it in a task-specific
# run/log/tmp/cache/results tree.  No argv is serialized back through a shell.
handle_task() {
  mode=$1
  task_number=$2
  assigned_gpu_id=$3
  shift 3
  trial_id=$1
  scenario=$2
  gap=$3
  probability=$4
  dataset=$5
  alpha=$6
  client_num=$7
  variant_record=$8

  saved_ifs=$IFS
  IFS='|'
  set -- $variant_record
  IFS=$saved_ifs
  variant_kind=$1
  profile_name=$2
  attack_method=$(method_for_variant "$dataset" "$variant_kind")

  arima_var_window=na
  arima_var_threshold=na
  arima_err_threshold=na
  arima_min_rounds=na
  arima_retrain_interval=na
  arima_ema_beta=na
  arima_trim_history=na
  moirai_tune_lr=na
  moirai_tune_steps=na
  moirai_tune_start_round=na
  moirai_adapt_interval=na
  moirai_residual_decay=na
  moirai_residual_clip=na
  moirai_ema_beta=na
  moirai_err_threshold=na
  moirai_min_rounds=na
  moirai_trim_history=na

  case "$variant_kind" in
    arima_offline|arima_adaptive)
      arima_var_window=$3
      arima_var_threshold=$4
      arima_err_threshold=$5
      arima_min_rounds=$6
      arima_retrain_interval=$7
      arima_ema_beta=$8
      arima_trim_history=$9
      ;;
    moirai_offline)
      [ "$profile_name" = control ] || die 'invalid Moirai offline record'
      ;;
    moirai_online)
      moirai_tune_lr=$3
      moirai_tune_steps=$4
      moirai_tune_start_round=$5
      moirai_adapt_interval=$6
      moirai_residual_decay=$7
      moirai_residual_clip=$8
      moirai_ema_beta=$9
      shift 9
      moirai_err_threshold=$1
      moirai_min_rounds=$2
      moirai_trim_history=$3
      ;;
    *) die "unknown variant kind: $variant_kind" ;;
  esac

  task_label=$(printf 'task_%06d_%s_%s_%s_a%s_n%s_%s_g%s_t%s' \
    "$task_number" "$dataset" "$variant_kind" "$profile_name" "$alpha" \
    "$client_num" "$scenario" "$gap" "$trial_id")
  task_root=$TASKS_ROOT/$task_label
  task_results=$RESULTS_CAMPAIGN_ROOT/$task_label

  set -- "$PYTHON_BIN" "$ENTRY" \
    --dataset "$dataset" \
    --alpha "$alpha" \
    --client_num "$client_num" \
    --attacker_id "$ATTACKER_ID" \
    --attack_method "$attack_method" \
    --contribution_method "$CONTRIBUTION_METHOD" \
    --use_last_round_sv \
    --trial_id "$trial_id" \
    --num_epoch "$NUM_EPOCH" \
    --num_local_epochs "$NUM_LOCAL_EPOCHS" \
    --random_round "$RANDOM_ROUND" \
    --results_root "$task_results" \
    --use_attack \
    --use_lazy \
    --lazy_schedule "$scenario" \
    --lazy_honest_gap "$gap" \
    --lazy_honest_phase "$LAZY_HONEST_PHASE" \
    --lazy_fake_policy "$LAZY_FAKE_POLICY" \
    --use_hessian \
    --lambda_h "$LAMBDA_H"

  if [ "$scenario" = probabilistic ]; then
    set -- "$@" --attack_probability "$probability"
  fi
  if is_layerwise_dataset "$dataset"; then
    set -- "$@" \
      --layerwise_selection_metric "$LAYERWISE_SELECTION_METRIC" \
      --layerwise_selection_fraction "$LAYERWISE_SELECTION_FRACTION" \
      --layerwise_pca_threshold "$LAYERWISE_PCA_THRESHOLD" \
      --layerwise_pca_components "$LAYERWISE_PCA_COMPONENTS"
    if [ -n "$LAYERWISE_SELECTION_TOP_N" ]; then
      set -- "$@" --layerwise_selection_top_n "$LAYERWISE_SELECTION_TOP_N"
    fi
  fi
  case "$variant_kind" in
    arima_offline|arima_adaptive)
      set -- "$@" \
        --arima_var_window "$arima_var_window" \
        --arima_var_threshold "$arima_var_threshold"
      ;;
  esac
  if [ "$variant_kind" = arima_adaptive ]; then
    set -- "$@" \
      --arima_err_threshold "$arima_err_threshold" \
      --arima_min_rounds_after_start "$arima_min_rounds" \
      --arima_retrain_interval "$arima_retrain_interval" \
      --arima_err_ema_beta "$arima_ema_beta"
    if [ "$arima_trim_history" -eq 1 ]; then
      set -- "$@" --arima_trim_history
    else
      set -- "$@" --no-arima_trim_history
    fi
  fi
  if [ "$variant_kind" = moirai_online ]; then
    set -- "$@" \
      --moirai_tune_lr "$moirai_tune_lr" \
      --moirai_tune_steps "$moirai_tune_steps" \
      --moirai_tune_start_round "$moirai_tune_start_round" \
      --moirai_adapt_interval "$moirai_adapt_interval" \
      --moirai_residual_decay "$moirai_residual_decay" \
      --moirai_residual_clip "$moirai_residual_clip" \
      --moirai_err_ema_beta "$moirai_ema_beta" \
      --moirai_err_threshold "$moirai_err_threshold" \
      --moirai_min_rounds_after_start "$moirai_min_rounds"
    if [ "$moirai_trim_history" -eq 1 ]; then
      set -- "$@" --moirai_trim_history
    else
      set -- "$@" --no-moirai_trim_history
    fi
  fi

  if [ "$mode" = print ]; then
    printf 'task=%s dataset=%s alpha=%s client=%s scenario=%s gap=%s paired_probability=%s variant=%s profile=%s\n' \
      "$task_number" "$dataset" "$alpha" "$client_num" "$scenario" "$gap" \
      "$probability" "$variant_kind" "$profile_name"
    print_command "$@"
    return 0
  fi

  [ ! -e "$task_root" ] || die "run-root collision: $task_root"
  [ ! -e "$task_results" ] || die "Results collision: $task_results"
  mkdir "$task_root" "$task_results"
  mkdir "$task_root/logs" "$task_root/tmp" "$task_root/cache"

  case "$assigned_gpu_id" in
    ''|*[!0-9]*) die "task $task_number has invalid assigned GPU id: $assigned_gpu_id" ;;
  esac
  expected_fixed=$(fixed_honest_fake_counts "$gap")
  manifest=$task_root/manifest.txt
  {
    printf 'task_number=%s\ntask_label=%s\n' "$task_number" "$task_label"
    printf 'pilot_stage=%s\npilot_profile_set=%s\npilot_not_full_matrix=1\n' \
      "$PILOT_STAGE" "$PILOT_PROFILE_SET"
    printf 'dataset=%s\nalpha=%s\nclient_num=%s\n' "$dataset" "$alpha" "$client_num"
    printf 'variant_kind=%s\nprofile_name=%s\nattack_method=%s\n' \
      "$variant_kind" "$profile_name" "$attack_method"
    printf 'trial_id=%s\nscenario=%s\ngap=%s\npaired_probability=%s\n' \
      "$trial_id" "$scenario" "$gap" "$probability"
    printf 'fixed_expected_honest_fake=%s\n' "$expected_fixed"
    printf 'num_epoch=%s\nnum_local_epochs=%s\nrandom_round=%s\n' \
      "$NUM_EPOCH" "$NUM_LOCAL_EPOCHS" "$RANDOM_ROUND"
    printf 'use_hessian=%s\nlambda_h=%s\n' "$USE_HESSIAN" "$LAMBDA_H"
    printf 'lazy_honest_phase=%s\nlazy_fake_policy=%s\n' \
      "$LAZY_HONEST_PHASE" "$LAZY_FAKE_POLICY"
    printf 'arima_var_window=%s\narima_var_threshold=%s\n' \
      "$arima_var_window" "$arima_var_threshold"
    printf 'arima_err_threshold=%s\narima_min_rounds_after_start=%s\n' \
      "$arima_err_threshold" "$arima_min_rounds"
    printf 'arima_retrain_interval=%s\narima_err_ema_beta=%s\narima_trim_history=%s\n' \
      "$arima_retrain_interval" "$arima_ema_beta" "$arima_trim_history"
    printf 'moirai_tune_lr=%s\nmoirai_tune_steps=%s\nmoirai_tune_start_round=%s\n' \
      "$moirai_tune_lr" "$moirai_tune_steps" "$moirai_tune_start_round"
    printf 'moirai_adapt_interval=%s\nmoirai_residual_decay=%s\nmoirai_residual_clip=%s\n' \
      "$moirai_adapt_interval" "$moirai_residual_decay" "$moirai_residual_clip"
    printf 'moirai_err_ema_beta=%s\nmoirai_err_threshold=%s\n' \
      "$moirai_ema_beta" "$moirai_err_threshold"
    printf 'moirai_min_rounds_after_start=%s\nmoirai_trim_history=%s\n' \
      "$moirai_min_rounds" "$moirai_trim_history"
    printf 'gpu_id=%s\nrun_root=%s\nresults_root=%s\n' "$assigned_gpu_id" "$task_root" "$task_results"
    printf '%s\n' 'command_argv:'
    printf '  %s\n' "$@"
  } > "$manifest"

  log_file=$task_root/logs/run.log
  started=$(date +%s)
  if (
    cd "$CANDIDATE_ROOT"
    CUDA_VISIBLE_DEVICES=$assigned_gpu_id \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TMPDIR=$task_root/tmp \
    HF_HOME=$task_root/cache/huggingface \
    TORCH_HOME=$task_root/cache/torch \
    XDG_CACHE_HOME=$task_root/cache/xdg \
    "$@"
  ) > "$log_file" 2>&1; then
    ended=$(date +%s)
    rounds=$(grep -c '^\[ROUND ' "$log_file" || true)
    honest=$(grep -c '^\[ROUND .* honest=True ' "$log_file" || true)
    fake=$(grep -c '^\[ROUND .* honest=False ' "$log_file" || true)
    hessian=$(grep -c '^\[ROUND .* hessian=on ' "$log_file" || true)
    protocol=pass
    [ "$rounds" -eq 50 ] || protocol=fail
    [ "$hessian" -eq 50 ] || protocol=fail
    [ $((honest + fake)) -eq 50 ] || protocol=fail
    if [ "$scenario" = fixed ]; then
      expected_honest=${expected_fixed%%:*}
      expected_fake=${expected_fixed#*:}
      [ "$honest" -eq "$expected_honest" ] || protocol=fail
      [ "$fake" -eq "$expected_fake" ] || protocol=fail
    fi
    printf 'success protocol_%s runtime_seconds=%s rounds=%s honest=%s fake=%s hessian=%s\n' \
      "$protocol" "$((ended - started))" "$rounds" "$honest" "$fake" "$hessian" \
      > "$task_root/status.txt"
    [ "$protocol" = pass ] && return 0
    return 3
  else
    exit_code=$?
    ended=$(date +%s)
    printf 'failed exit_code=%s runtime_seconds=%s\n' \
      "$exit_code" "$((ended - started))" > "$task_root/status.txt"
    return "$exit_code"
  fi
}

# Real runs use a shared filesystem queue and one long-lived serial worker per
# active GPU.  A worker claims its next spec only after its previous task has
# finished, so a slow task never creates a global batch barrier.  Claims use an
# atomic same-filesystem rename; argv is rebuilt from validated fields and is
# never evaluated as shell source.
enqueue_task() {
  queued_task_number=$1
  shift
  queued_spec=$(printf '%s/task_%06d.spec' "$SCHEDULER_PENDING" "$queued_task_number")
  [ ! -e "$queued_spec" ] || die "scheduler spec collision: $queued_spec"
  {
    printf '%s\n' "$queued_task_number"
    printf '%s\n' "$@"
  } > "$queued_spec"
}

claim_next_task() {
  claim_gpu_id=$1
  for pending_spec in "$SCHEDULER_PENDING"/*.spec; do
    [ -f "$pending_spec" ] || return 1
    pending_name=${pending_spec##*/}
    claimed_spec=$SCHEDULER_RUNNING/${pending_name%.spec}.gpu_${claim_gpu_id}.spec
    if mv "$pending_spec" "$claimed_spec" 2>/dev/null; then
      printf '%s\n' "$claimed_spec"
      return 0
    fi
  done
  return 1
}

run_gpu_worker() {
  worker_gpu_id=$1
  worker_log=$SCHEDULER_WORKERS/gpu_${worker_gpu_id}.log
  worker_status=$SCHEDULER_WORKERS/gpu_${worker_gpu_id}.status
  worker_attempted=0
  worker_failed=0
  printf 'gpu=%s state=started\n' "$worker_gpu_id" > "$worker_log"

  while claimed_spec=$(claim_next_task "$worker_gpu_id"); do
    if ! {
      IFS= read -r claimed_task_number
      IFS= read -r claimed_trial_id
      IFS= read -r claimed_scenario
      IFS= read -r claimed_gap
      IFS= read -r claimed_probability
      IFS= read -r claimed_dataset
      IFS= read -r claimed_alpha
      IFS= read -r claimed_client_num
      IFS= read -r claimed_variant_record
    } < "$claimed_spec"; then
      printf 'gpu=%s state=invalid_spec path=%s\n' "$worker_gpu_id" "$claimed_spec" >> "$worker_log"
      return 1
    fi

    worker_attempted=$((worker_attempted + 1))
    printf 'gpu=%s task=%s state=started spec=%s\n' \
      "$worker_gpu_id" "$claimed_task_number" "$claimed_spec" >> "$worker_log"
    task_exit_code=0
    if handle_task run "$claimed_task_number" "$worker_gpu_id" \
      "$claimed_trial_id" "$claimed_scenario" "$claimed_gap" \
      "$claimed_probability" "$claimed_dataset" "$claimed_alpha" \
      "$claimed_client_num" "$claimed_variant_record"; then
      task_state=success
    else
      task_exit_code=$?
      task_state=failed
      worker_failed=$((worker_failed + 1))
      failure_marker=$(printf '%s/task_%06d.failed' "$SCHEDULER_FAILURES" "$claimed_task_number")
      printf 'task=%s gpu=%s exit_code=%s spec=%s\n' \
        "$claimed_task_number" "$worker_gpu_id" "$task_exit_code" "$claimed_spec" \
        > "$failure_marker"
    fi
    completed_name=${claimed_spec##*/}
    mv "$claimed_spec" "$SCHEDULER_COMPLETED/$completed_name"
    printf 'gpu=%s task=%s state=%s exit_code=%s\n' \
      "$worker_gpu_id" "$claimed_task_number" "$task_state" "$task_exit_code" >> "$worker_log"
  done

  printf 'gpu=%s attempted=%s failed=%s state=finished\n' \
    "$worker_gpu_id" "$worker_attempted" "$worker_failed" > "$worker_status"
  printf 'gpu=%s state=finished attempted=%s failed=%s\n' \
    "$worker_gpu_id" "$worker_attempted" "$worker_failed" >> "$worker_log"
}

count_scheduler_files() {
  count_directory=$1
  count_suffix=$2
  counted_files=0
  for counted_file in "$count_directory"/*"$count_suffix"; do
    [ -f "$counted_file" ] || continue
    counted_files=$((counted_files + 1))
  done
  printf '%s\n' "$counted_files"
}

# Independently count every selected task before any write or process launch.
total_tasks=0
for trial_id in $TRIAL_IDS; do
  for dataset in $PILOT_DATASETS; do
    dataset_variants=$(variant_records "$dataset")
    for alpha in $(selected_alpha_values "$dataset"); do
      for client_num in $PILOT_CLIENT_NUMS; do
        for pair in $PILOT_GAP_PROB_PAIRS; do
          for scenario in $PILOT_SCENARIOS; do
            for ignored_variant in $dataset_variants; do
              total_tasks=$((total_tasks + 1))
            done
          done
        done
      done
    done
  done
done
[ "$total_tasks" -gt 0 ] || die 'pilot task selection is empty'

printf '%s\n' '=== MOIRAI/ARIMA STAGED TUNING PILOT ==='
printf '%s\n' 'WARNING: this is an explicit screening subset, not the full experiment matrix.'
printf 'candidate_root       : %s\n' "$CANDIDATE_ROOT"
printf 'pilot_stage/profile  : %s / %s\n' "$PILOT_STAGE" "$PILOT_PROFILE_SET"
printf 'selected_families    : %s\n' "$PILOT_FAMILIES"
printf 'selected_datasets    : %s\n' "$PILOT_DATASETS"
for dataset in $PILOT_DATASETS; do
  printf 'selected_alpha[%s]  : %s (canonical: %s)\n' \
    "$dataset" "$(selected_alpha_values "$dataset")" "$(canonical_alpha_values "$dataset")"
done
printf 'selected_clients     : %s (canonical: %s)\n' "$PILOT_CLIENT_NUMS" "$CANONICAL_CLIENT_NUMS"
printf 'selected_pairs       : %s (canonical: %s)\n' \
  "$PILOT_GAP_PROB_PAIRS" "$CANONICAL_GAP_PROB_PAIRS"
printf 'selected_scenarios   : %s\n' "$PILOT_SCENARIOS"
printf 'trial_ids            : %s\n' "$TRIAL_IDS"
printf 'rounds/Hessian/lambda: %s / %s / %s\n' "$NUM_EPOCH" "$USE_HESSIAN" "$LAMBDA_H"
printf 'total_tasks          : %s\n' "$total_tasks"
printf 'active_gpu_slots     : %s (one serial worker per GPU)\n' "$active_gpu_ids"
printf '%s\n' 'fixed counts          : gap2=25/25 gap3=17/33 gap6=9/41 gap10=5/45'

if [ "$DRY_RUN" = 1 ]; then
  printed=0
  enumerated=0
  for trial_id in $TRIAL_IDS; do
    for dataset in $PILOT_DATASETS; do
      dataset_variants=$(variant_records "$dataset")
      for alpha in $(selected_alpha_values "$dataset"); do
        for client_num in $PILOT_CLIENT_NUMS; do
          for pair in $PILOT_GAP_PROB_PAIRS; do
            gap=${pair%%:*}
            probability=${pair#*:}
            for scenario in $PILOT_SCENARIOS; do
              for variant_record in $dataset_variants; do
                enumerated=$((enumerated + 1))
                if [ "$printed" -lt "$DRY_RUN_PRINT_LIMIT" ]; then
                  handle_task print "$enumerated" 0 "$trial_id" "$scenario" "$gap" \
                    "$probability" "$dataset" "$alpha" "$client_num" "$variant_record"
                  printed=$((printed + 1))
                fi
              done
            done
          done
        done
      done
    done
  done
  [ "$enumerated" -eq "$total_tasks" ] || die 'DRY_RUN task enumeration drifted'
  printf 'DRY_RUN rendered %s commands and enumerated %s tasks.\n' "$printed" "$enumerated"
  printf '%s\n' 'No directory was created and no experiment was started.'
  exit 0
fi

[ "$CONFIRM_TUNING" -eq 1 ] || {
  printf '%s\n' 'Tuning is locked. Re-run with DRY_RUN=0 CONFIRM_TUNING=1 after reviewing the plan.' >&2
  exit 2
}
[ -f "$CANDIDATE_ROOT/.candidate_deployment_manifest.txt" ] ||
  die 'real tuning requires a candidate created by deploy_and_smoke_moirai_arima.sh'
[ -f "$CANDIDATE_ROOT/.smoke_passed_manifest.txt" ] ||
  die 'real tuning requires a successful end-to-end smoke manifest'
[ -f "$CANDIDATE_ROOT/.smoke_passed.sha256" ] ||
  die 'real tuning requires smoke-passed candidate checksums'
(
  cd "$CANDIDATE_ROOT"
  sha256sum -c .smoke_passed.sha256
) || die 'candidate production files changed after the successful smoke'
[ "$CANDIDATE_ROOT" != "$PROJECT_ROOT" ] || die 'refusing to run from the immutable base project root'
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python executable not found: $PYTHON_BIN"
[ ! -e "$CAMPAIGN_ROOT" ] || die "campaign path already exists: $CAMPAIGN_ROOT"
[ ! -e "$RESULTS_CAMPAIGN_ROOT" ] || die "Results path already exists: $RESULTS_CAMPAIGN_ROOT"

mkdir -p "$CAMPAIGN_PARENT" "$PROJECT_ROOT/Results"
mkdir "$CAMPAIGN_ROOT" "$TASKS_ROOT" "$RESULTS_CAMPAIGN_ROOT"
SCHEDULER_ROOT=$CAMPAIGN_ROOT/scheduler
SCHEDULER_PENDING=$SCHEDULER_ROOT/pending
SCHEDULER_RUNNING=$SCHEDULER_ROOT/running
SCHEDULER_COMPLETED=$SCHEDULER_ROOT/completed
SCHEDULER_FAILURES=$SCHEDULER_ROOT/failures
SCHEDULER_WORKERS=$SCHEDULER_ROOT/workers
mkdir "$SCHEDULER_ROOT" "$SCHEDULER_PENDING" "$SCHEDULER_RUNNING" \
  "$SCHEDULER_COMPLETED" "$SCHEDULER_FAILURES" "$SCHEDULER_WORKERS"

campaign_manifest=$CAMPAIGN_ROOT/campaign_manifest.txt
{
  printf 'campaign_id=%s\ncreated_utc=%s\n' "$CAMPAIGN_ID" "$timestamp"
  printf 'pilot_stage=%s\npilot_profile_set=%s\npilot_not_full_matrix=1\nnot_for_final_reporting=1\n' \
    "$PILOT_STAGE" "$PILOT_PROFILE_SET"
  printf 'selected_families=%s\n' "$PILOT_FAMILIES"
  printf 'candidate_root=%s\nresults_root=%s\n' "$CANDIDATE_ROOT" "$RESULTS_CAMPAIGN_ROOT"
  printf 'selected_datasets=%s\nselected_client_nums=%s\n' "$PILOT_DATASETS" "$PILOT_CLIENT_NUMS"
  printf 'selected_gap_probability_pairs=%s\nselected_scenarios=%s\n' \
    "$PILOT_GAP_PROB_PAIRS" "$PILOT_SCENARIOS"
  printf 'trial_ids=%s\ntotal_tasks=%s\n' "$TRIAL_IDS" "$total_tasks"
  printf 'scheduler=dynamic_shared_queue\nactive_gpu_ids=%s\nmax_parallel=%s\n' \
    "$active_gpu_ids" "$MAX_PARALLEL"
  printf 'canonical_client_nums=%s\ncanonical_gap_probability_pairs=%s\n' \
    "$CANONICAL_CLIENT_NUMS" "$CANONICAL_GAP_PROB_PAIRS"
  printf 'canonical_fixed_counts=gap2:25:25 gap3:17:33 gap6:9:41 gap10:5:45\n'
  for dataset in $CANONICAL_DATASETS; do
    printf 'canonical_alpha_%s=%s\n' "$dataset" "$(canonical_alpha_values "$dataset")"
  done
  for dataset in $PILOT_DATASETS; do
    printf 'selected_alpha_%s=%s\n' "$dataset" "$(selected_alpha_values "$dataset")"
    if contains_word arima $PILOT_FAMILIES; then
      for profile_record in $(arima_profile_records "$dataset"); do
        printf 'arima_profile_%s=%s\n' "$dataset" "$profile_record"
      done
    fi
    if contains_word moirai $PILOT_FAMILIES; then
      for profile_record in $(moirai_profile_records "$dataset"); do
        printf 'moirai_profile_%s=%s\n' "$dataset" "$profile_record"
      done
    fi
  done
  printf 'num_epoch=%s\nnum_local_epochs=%s\nrandom_round=%s\n' \
    "$NUM_EPOCH" "$NUM_LOCAL_EPOCHS" "$RANDOM_ROUND"
  printf 'use_hessian=%s\nlambda_h=%s\n' "$USE_HESSIAN" "$LAMBDA_H"
  printf 'lazy_honest_phase=%s\nlazy_fake_policy=%s\n' \
    "$LAZY_HONEST_PHASE" "$LAZY_FAKE_POLICY"
  sha256sum "$ENTRY" "$NONLAYER_MODULE" "$LAYER_MODULE" "$SCRIPT_PATH"
} > "$campaign_manifest"

task_number=0
for trial_id in $TRIAL_IDS; do
  for dataset in $PILOT_DATASETS; do
    dataset_variants=$(variant_records "$dataset")
    for alpha in $(selected_alpha_values "$dataset"); do
      for client_num in $PILOT_CLIENT_NUMS; do
        for pair in $PILOT_GAP_PROB_PAIRS; do
          gap=${pair%%:*}
          probability=${pair#*:}
          for scenario in $PILOT_SCENARIOS; do
            for variant_record in $dataset_variants; do
              task_number=$((task_number + 1))
              enqueue_task "$task_number" "$trial_id" "$scenario" "$gap" \
                "$probability" "$dataset" "$alpha" "$client_num" "$variant_record"
            done
          done
        done
      done
    done
  done
done
[ "$task_number" -eq "$total_tasks" ] || die 'executed task enumeration drifted'

queued_tasks=$(count_scheduler_files "$SCHEDULER_PENDING" .spec)
[ "$queued_tasks" -eq "$total_tasks" ] || die 'scheduler queue enumeration drifted'

worker_pids=''
for active_gpu_id in $active_gpu_ids; do
  run_gpu_worker "$active_gpu_id" &
  worker_pids="$worker_pids $!"
done
scheduler_worker_failures=0
for worker_pid in $worker_pids; do
  if ! wait "$worker_pid"; then
    scheduler_worker_failures=$((scheduler_worker_failures + 1))
  fi
done

attempted_tasks=$(count_scheduler_files "$SCHEDULER_COMPLETED" .spec)
task_failures=$(count_scheduler_files "$SCHEDULER_FAILURES" .failed)
unfinished_tasks=$((total_tasks - attempted_tasks))
[ "$unfinished_tasks" -ge 0 ] || die 'scheduler completed more tasks than were enumerated'
successful_tasks=$((attempted_tasks - task_failures))
failures=$((task_failures + unfinished_tasks))

printf 'completed_tasks=%s\nsuccessful_tasks=%s\nfailed_or_protocol_invalid_tasks=%s\n' \
  "$attempted_tasks" "$successful_tasks" "$task_failures" >> "$campaign_manifest"
printf 'unfinished_tasks=%s\nscheduler_worker_failures=%s\n' \
  "$unfinished_tasks" "$scheduler_worker_failures" >> "$campaign_manifest"
printf 'campaign_root=%s\nresults_root=%s\ntasks=%s successful=%s failures=%s unfinished=%s worker_failures=%s\n' \
  "$CAMPAIGN_ROOT" "$RESULTS_CAMPAIGN_ROOT" "$total_tasks" "$successful_tasks" \
  "$failures" "$unfinished_tasks" "$scheduler_worker_failures"
[ "$failures" -eq 0 ] && [ "$scheduler_worker_failures" -eq 0 ]

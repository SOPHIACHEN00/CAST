#!/usr/bin/env bash
# Offline-only supplement: 108 tasks, 36 per dataset; preview unless DRY_RUN=0.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PARTITION_SETTING=uniform DETECTORS=shapley ATTACKS="arima moirai"
export SEED=42 TRIAL_IDS="0 1 2" ATTACKER_SELECTION_SEED=2026
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../Results_multi_freerider_uniform_offline}"
exec bash "${SCRIPT_DIR}/run_multi_freerider.sh" "$@"

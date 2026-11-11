#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
test_root=$(mktemp -d "${TMPDIR:-/tmp}/passive-qualitative-test.XXXXXX")
launcher_pid=''
watchdog_pid=''
sentinel_pid=''

cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  [[ -z "$watchdog_pid" ]] || kill -TERM "$watchdog_pid" 2>/dev/null || true
  [[ -z "$launcher_pid" ]] || kill -TERM "$launcher_pid" 2>/dev/null || true
  if [[ -n "$sentinel_pid" ]]; then
    kill -TERM "$sentinel_pid" 2>/dev/null || true
    wait "$sentinel_pid" 2>/dev/null || true
  fi
  case "$test_root" in
    "${TMPDIR:-/tmp}"/passive-qualitative-test.*) rm -rf -- "$test_root" ;;
    *) printf 'unsafe cleanup path: %s\n' "$test_root" >&2 ;;
  esac
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

project_root="$test_root/project"
mock_bin="$test_root/bin"
mkdir -p "$project_root/script" "$mock_bin"
cp "$repo_root/script/run_tictactoe_passive_qualitative.sh" "$project_root/script/"
printf '# mock entry\n' > "$project_root/FLCE_updated.py"

real_python=$(command -v python || command -v python3)

cat > "$mock_bin/mock-flce" <<'MOCK'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == -c ]]; then exit 0; fi
[[ "${PYTHONUNBUFFERED:-}" == 1 && "${PYTHONDONTWRITEBYTECODE:-}" == 1 ]] || exit 29
dataset='' alpha='' client='' attacker='' attack='' detector='' trial='' results=''
use_hessian=0 use_lazy=0 lazy_schedule=fixed lazy_gap=3 lazy_phase=0 lazy_policy=attack
while (($#)); do
  case "$1" in
    --dataset) dataset=$2; shift 2 ;; --alpha) alpha=$2; shift 2 ;;
    --client_num) client=$2; shift 2 ;; --attacker_id) attacker=$2; shift 2 ;;
    --attack_method) attack=$2; shift 2 ;; --detecting_method) detector=$2; shift 2 ;;
    --trial_id) trial=$2; shift 2 ;; --results_root) results=$2; shift 2 ;;
    --use_hessian) use_hessian=1; shift ;; --use_lazy) use_lazy=1; shift ;;
    --lazy_schedule) lazy_schedule=$2; shift 2 ;; --lazy_honest_gap) lazy_gap=$2; shift 2 ;;
    --lazy_honest_phase) lazy_phase=$2; shift 2 ;; --lazy_fake_policy) lazy_policy=$2; shift 2 ;;
    *) shift ;;
  esac
done
task_id=${results%/results}; task_id=${task_id##*/}
if [[ "${MOCK_HANG:-0}" == 1 ]]; then
  mkdir -p "${MOCK_PID_ROOT:?}"
  printf '%s\n' "$$" > "$MOCK_PID_ROOT/$task_id.pid"
  trap '' HUP INT TERM
  while :; do sleep 1; done
fi
[[ "$task_id" != "${MOCK_FAIL_TASK:-}" ]] || exit 23

regime=pure
if ((use_lazy)); then regime=lazy; fi
printf -v raw_alpha '%.1f' "$alpha"
raw_task_id="${dataset}_${attack}_${detector}_alpha${raw_alpha}_clients${client}_attacker${attacker}_trial${trial}"
lambda=N/A
if ((use_hessian)); then lambda=0.9; fi
detect_dir="$results/Detecting_qualitative_$detector"
time_dir="$results/Time_qualitative_$detector"
accuracy_dir="$results/Accuracy_qualitative_$detector"
mkdir -p "$detect_dir" "$time_dir" "$accuracy_dir" "$results/manifests"

accuracy="$accuracy_dir/global_accuracy_log_trial$trial.csv"
printf '%s\n' 'Dataset,AttackMethod,Alpha,ClientNum,AttackerID,TrialID,UseAttack,DetectingType,DetectingMethod,UseHessian,UseLazy,LazySchedule,AttackProbability,LazyHonestGap,LazyHonestPhase,LazyFakePolicy,Round,WithAttackerAccuracy,WithoutAttackerAccuracy' > "$accuracy"
for ((round=0; round<50; round++)); do
  printf '%s,%s,%s,%s,%s,%s,True,qualitative,%s,%s,%s,%s,1.0,%s,%s,%s,%s,0.5,0.6\n' \
    "$dataset" "$attack" "$alpha" "$client" "$attacker" "$trial" "$detector" "$use_hessian" "$use_lazy" "$lazy_schedule" "$lazy_gap" "$lazy_phase" "$lazy_policy" "$round" >> "$accuracy"
done

detection="$detect_dir/round_client_detection_trial$trial.csv"
printf '%s\n' 'TaskID,CampaignComplete,Dataset,AttackMethod,AttackRegime,Alpha,ClientNum,AttackerID,TrialID,UseHessian,LambdaH,UseLazy,LazySchedule,LazyHonestGap,LazyHonestPhase,LazyFakePolicy,DetectingMethod,QualitativeAggregationMode,Round,ClientID,IsDesignatedAttacker,GroundTruthActiveAttack,DesignatedAttackerRoundState,RoundHasActiveAttack,RoundIsHonestForDesignatedAttacker,WEFPayloadSource,DetectionSkipped,DetectorScore,DetectorThreshold,DeltaThresholdPathology,RoundPredictedSuspicious,FinalPredictedSuspicious,DecisionMode,DetectorDetailsJSON' > "$detection"
for ((round=0; round<50; round++)); do
  for ((cid=0; cid<client; cid++)); do
    designated=0; [[ "$cid" -ne "$attacker" ]] || designated=1
    state=fake; active_round=1; honest_round=0
    if ((use_lazy && round % 6 == 0)); then state=honest; active_round=0; honest_round=1; fi
    ground=0; ((designated && active_round)) && ground=1
    payload=N/A
    if [[ "$detector" == wef || "$detector" == s2_wef ]]; then
      payload=observed_local_training
      if ((ground)) && [[ "$attack" != synthetic_replacement ]]; then payload=counterfeit_from_submitted_update; fi
    fi
    skipped=0; score=0.1; threshold=0.2; round_prediction=0; final_prediction=0
    if [[ "$detector" == s2_wef && "$round" -eq 0 ]]; then
      skipped=1; score=''; threshold=''; round_prediction=''; final_prediction=''
    fi
    printf '%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,passive,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s,%s,score_only,{}\n' \
      "$raw_task_id" "$dataset" "$attack" "$regime" "$alpha" "$client" "$attacker" "$trial" "$use_hessian" "$lambda" "$use_lazy" "$lazy_schedule" "$lazy_gap" "$lazy_phase" "$lazy_policy" "$detector" "$round" "$cid" \
      "$designated" "$ground" "$state" "$active_round" "$honest_round" "$payload" "$skipped" "$score" "$threshold" "$round_prediction" "$final_prediction" >> "$detection"
  done
done

time_file="$time_dir/attacking_time_log_trial$trial.csv"
printf '%s\n' 'TaskID,CampaignComplete,Dataset,AttackMethod,AttackRegime,Alpha,ClientNum,AttackerID,TrialID,UseHessian,LambdaH,UseLazy,LazySchedule,LazyHonestGap,LazyHonestPhase,LazyFakePolicy,DetectingMethod,QualitativeAggregationMode,AttackGenerationWithAttackerTotalSec,FakeRoundsTimed,DetectorDecisionWithAttackerTotalSec,DetectorDecisionWithoutAttackerTotalSec,DetectorDecisionWithoutAttackerMeasured,DetectorDecisionWithoutAttackerNotMeasuredReason,FLWithAttackerWallSec,FLWithoutAttackerWallSec,TaskWallSec' > "$time_file"
printf '%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,passive,1,41,2,,0,posthoc_detector_not_run_for_internal_n_minus_1_control,3,4,5\n' \
  "$raw_task_id" "$dataset" "$attack" "$regime" "$alpha" "$client" "$attacker" "$trial" "$use_hessian" "$lambda" "$use_lazy" "$lazy_schedule" "$lazy_gap" "$lazy_phase" "$lazy_policy" "$detector" >> "$time_file"
printf '{}\n' > "$results/manifests/run_manifest_${dataset}_${attack}_trial${trial}_mock.json"
MOCK
chmod +x "$mock_bin/mock-flce"
cat > "$mock_bin/mock-flock" <<'MOCK_FLOCK'
#!/usr/bin/env bash
exit 0
MOCK_FLOCK
chmod +x "$mock_bin/mock-flock"
cat > "$mock_bin/mock-nvidia-smi" <<'MOCK_NVIDIA'
#!/usr/bin/env bash
while (($#)); do
  if [[ "$1" == -i ]]; then printf '%s\n' "$2"; exit 0; fi
  shift
done
exit 1
MOCK_NVIDIA
chmod +x "$mock_bin/mock-nvidia-smi"

run_launcher() {
  local campaign=$1 fail_task=${2:-}
  set +e
  PROJECT_ROOT="$project_root" PYTHON_BIN="$mock_bin/mock-flce" MERGER_PYTHON_BIN="$real_python" \
    FLOCK_BIN="$mock_bin/mock-flock" \
    NVIDIA_SMI_BIN="$mock_bin/mock-nvidia-smi" \
    GPU_IDS=1 WORKERS_PER_GPU=32 CPU_THREADS_PER_WORKER=2 DRY_RUN=0 \
    CONFIRM_QUALITATIVE_RUN=1 CAMPAIGN_ID="$campaign" MOCK_FAIL_TASK="$fail_task" \
    bash "$project_root/script/run_tictactoe_passive_qualitative.sh" \
    > "$test_root/$campaign.out" 2> "$test_root/$campaign.err"
  local status=$?
  set -e
  if [[ "$status" -ne 0 && -z "$fail_task" ]]; then
    printf '%s\n' "unexpected launcher failure for $campaign" >&2
    sed -n '1,200p' "$test_root/$campaign.err" >&2
    sed -n '1,80p' "$test_root/$campaign.out" >&2
    [[ ! -f "$project_root/Results/$campaign/audit/tasks.csv" ]] || sed -n '1,5p' "$project_root/Results/$campaign/audit/tasks.csv" >&2
  fi
  return "$status"
}

clone_campaign() {
  local source=$1 target=$2
  cp -R "$project_root/.qualitative_runs/$source" "$project_root/.qualitative_runs/$target"
  "$real_python" - "$project_root/.qualitative_runs/$target" "$source" "$target" <<'REWRITE_CLONED_PROVENANCE'
import csv, pathlib, sys
root, source, target = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
old, new = f"/.qualitative_runs/{source}/", f"/.qualitative_runs/{target}/"
for manifest_path in root.glob("tasks/*/manifest.tsv"):
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t")); header = list(rows[0])
    for name in ("ResultsRoot", "CommandFile", "CommandArgv"):
        rows[0][name] = rows[0][name].replace(old, new)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    command_file = manifest_path.parent / "command.argv"
    command_file.write_text(command_file.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
REWRITE_CLONED_PROVENANCE
}

run_launcher complete
public="$project_root/Results/complete"
[[ $(($(wc -l < "$public/accuracy/accuracy.csv") - 1)) -eq 11200 ]]
[[ $(($(wc -l < "$public/time/time.csv") - 1)) -eq 224 ]]
[[ $(($(wc -l < "$public/detection/detection.csv") - 1)) -eq 60800 ]]
[[ $(($(wc -l < "$public/audit/tasks.csv") - 1)) -eq 256 ]]
[[ $(head -n 1 "$public/accuracy/accuracy.csv" | grep -o 'TaskID' | wc -l) -eq 1 ]]
[[ $(tail -n +2 "$public/detection/detection.csv" | cut -d, -f1,19,20 | sort -u | wc -l) -eq 60800 ]]
"$real_python" - "$public" "$project_root/.qualitative_runs/complete" <<'VERIFY_COMPLETE'
import collections, csv, pathlib, sys
public, private = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
accuracy = rows(public / "accuracy" / "accuracy.csv")
audit = rows(public / "audit" / "tasks.csv")
required_accuracy = {"TaskID", "CampaignComplete", "AttackRegime", "LambdaH", "QualitativeAggregationMode", "AccuracyControlDefinition"}
assert required_accuracy <= set(accuracy[0]), set(accuracy[0])
assert len({(row["TaskID"], row["Round"]) for row in accuracy}) == 11200
assert {row["CampaignComplete"] for row in accuracy} == {"1"}
required_audit = {
    "GPU", "CPUThreadsPerWorker", "UseHessian", "LambdaH",
    "QualitativeAggregationMode", "AccuracyControlDefinition", "BaselineAdaptationDisclosure", "CommandArgv",
}
assert required_audit <= set(audit[0]), set(audit[0])
assert {row["CampaignComplete"] for row in audit} == {"1"}
hessian = {"arima_adaptive", "arima_adaptive_mnist_imdb_layerwise", "moirai_online", "moirai_online_mnist_imdb_layerwise"}
for row in audit:
    expected_na = row["ClientNum"] == "2" and row["DetectingMethod"] in {"wef", "s2_wef"}
    assert (row["Status"] == "N/A") == expected_na
    expected_hessian = row["AttackMethod"] in hessian
    assert row["UseHessian"] == ("1" if expected_hessian else "0")
    assert row["LambdaH"] == ("0.9" if expected_hessian else "N/A")
    assert row["QualitativeAggregationMode"] == "passive"
    assert row["AccuracyControlDefinition"] == "internal_n_minus_1_remove_designated_attacker_and_shard"
    if row["Status"] == "success":
        assert row["GPU"] == "1" and row["CPUThreadsPerWorker"] == "2"
        assert row["CommandArgv"] and row["BaselineAdaptationDisclosure"]
manifest = next(private.glob("tasks/*/manifest.tsv"))
with manifest.open(encoding="utf-8", newline="") as handle:
    header = next(csv.reader(handle, delimiter="\t"))
assert required_audit <= set(header), header
assert {row["AccuracyControlDefinition"] for row in accuracy} == {"internal_n_minus_1_remove_designated_attacker_and_shard"}
VERIFY_COMPLETE

first_task=$(sed -n '2s/\t.*//p' "$project_root/.qualitative_runs/complete/planned.tsv")
if run_launcher incomplete "$first_task"; then
  printf 'injected failure did not propagate\n' >&2
  exit 1
fi
partial="$project_root/Results/incomplete"
[[ $(($(wc -l < "$partial/accuracy/accuracy.csv") - 1)) -eq 11150 ]]
[[ $(($(wc -l < "$partial/time/time.csv") - 1)) -eq 223 ]]
[[ $(($(wc -l < "$partial/detection/detection.csv") - 1)) -lt 60800 ]]
[[ $(($(wc -l < "$partial/audit/tasks.csv") - 1)) -eq 256 ]]
tail -n +2 "$partial/accuracy/accuracy.csv" | cut -d, -f2 | grep -qx '0'
grep -q ',failed,' "$partial/audit/tasks.csv"

rm -rf -- "$partial"
set +e
PROJECT_ROOT="$project_root" MERGER_PYTHON_BIN="$real_python" DRY_RUN=0 MERGE_ONLY=1 \
  CAMPAIGN_ID=incomplete bash "$project_root/script/run_tictactoe_passive_qualitative.sh" \
  > "$test_root/merge-only.out" 2> "$test_root/merge-only.err"
merge_status=$?
set -e
[[ "$merge_status" -ne 0 ]]
[[ -f "$partial/audit/tasks.csv" ]]

cp -R "$project_root/.qualitative_runs/complete" "$project_root/.qualitative_runs/duplicate"
sed -n '2p' "$project_root/.qualitative_runs/duplicate/planned.tsv" >> "$project_root/.qualitative_runs/duplicate/planned.tsv"
set +e
PROJECT_ROOT="$project_root" MERGER_PYTHON_BIN="$real_python" DRY_RUN=0 MERGE_ONLY=1 \
  CAMPAIGN_ID=duplicate bash "$project_root/script/run_tictactoe_passive_qualitative.sh" >/dev/null 2>&1
duplicate_status=$?
set -e
[[ "$duplicate_status" -ne 0 ]]
[[ ! -e "$project_root/Results/duplicate" ]]
[[ ! -e "$project_root/Results/.duplicate.staging" ]]

clone_campaign complete malformed
bad_accuracy=$(find "$project_root/.qualitative_runs/malformed/tasks" -name 'global_accuracy_log_trial0.csv' | head -n 1)
sed -i '1s/^Dataset/Broken/' "$bad_accuracy"
set +e
PROJECT_ROOT="$project_root" MERGER_PYTHON_BIN="$real_python" DRY_RUN=0 MERGE_ONLY=1 \
  CAMPAIGN_ID=malformed bash "$project_root/script/run_tictactoe_passive_qualitative.sh" >/dev/null 2>&1
malformed_status=$?
set -e
[[ "$malformed_status" -ne 0 ]]
[[ -f "$project_root/Results/malformed/audit/tasks.csv" ]]
grep -q ',failed,protocol_invalid:' "$project_root/Results/malformed/audit/tasks.csv"
tail -n +2 "$project_root/Results/malformed/accuracy/accuracy.csv" | cut -d, -f2 | grep -qx '0'
[[ ! -e "$project_root/Results/.malformed.staging" ]]

clone_campaign complete protocol
bad_accuracy=$(find "$project_root/.qualitative_runs/protocol/tasks" -name 'global_accuracy_log_trial0.csv' | head -n 1)
"$real_python" - "$bad_accuracy" <<'BREAK_PROTOCOL'
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle)); header = list(rows[0])
rows[0]["UseAttack"] = "False"
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header); writer.writeheader(); writer.writerows(rows)
BREAK_PROTOCOL
set +e
PROJECT_ROOT="$project_root" MERGER_PYTHON_BIN="$real_python" DRY_RUN=0 MERGE_ONLY=1 \
  CAMPAIGN_ID=protocol bash "$project_root/script/run_tictactoe_passive_qualitative.sh" >/dev/null 2>&1
protocol_status=$?
set -e
[[ "$protocol_status" -ne 0 ]]
grep -q ',failed,protocol_invalid:' "$project_root/Results/protocol/audit/tasks.csv"

clone_campaign complete truth
bad_detection=$(find "$project_root/.qualitative_runs/truth/tasks" -name 'round_client_detection_trial0.csv' | head -n 1)
"$real_python" - "$bad_detection" <<'BREAK_TRUTH'
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle)); header = list(rows[0])
rows[0]["GroundTruthActiveAttack"] = "1" if rows[0]["GroundTruthActiveAttack"] == "0" else "0"
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header); writer.writeheader(); writer.writerows(rows)
BREAK_TRUTH
set +e
PROJECT_ROOT="$project_root" MERGER_PYTHON_BIN="$real_python" DRY_RUN=0 MERGE_ONLY=1 \
  CAMPAIGN_ID=truth bash "$project_root/script/run_tictactoe_passive_qualitative.sh" >/dev/null 2>&1
truth_status=$?
set -e
[[ "$truth_status" -ne 0 ]]
"$real_python" - "$project_root/Results/truth/audit/tasks.csv" "$project_root/Results/truth/accuracy/accuracy.csv" <<'CHECK_TRUTH_REJECTION'
import csv, sys
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    audit = list(csv.DictReader(handle))
assert sum(row["Status"] == "failed" and row["Reason"].startswith("protocol_invalid:") for row in audit) == 1
assert any(row["Status"] == "success" for row in audit)
with open(sys.argv[2], encoding="utf-8", newline="") as handle:
    accuracy = list(csv.DictReader(handle))
assert accuracy and {row["CampaignComplete"] for row in accuracy} == {"0"}
CHECK_TRUTH_REJECTION

pid_root="$test_root/hanging-pids"
sleep 300 &
sentinel_pid=$!
PROJECT_ROOT="$project_root" PYTHON_BIN="$mock_bin/mock-flce" MERGER_PYTHON_BIN="$real_python" \
  FLOCK_BIN="$mock_bin/mock-flock" \
  NVIDIA_SMI_BIN="$mock_bin/mock-nvidia-smi" \
  GPU_IDS=1 WORKERS_PER_GPU=15 CPU_THREADS_PER_WORKER=2 DRY_RUN=0 \
  CONFIRM_QUALITATIVE_RUN=1 CAMPAIGN_ID=interrupt MOCK_HANG=1 MOCK_PID_ROOT="$pid_root" \
  bash "$project_root/script/run_tictactoe_passive_qualitative.sh" \
  > "$test_root/interrupt.out" 2> "$test_root/interrupt.err" &
launcher_pid=$!
for _ in {1..20}; do
  count=0
  [[ ! -d "$pid_root" ]] || count=$(find "$pid_root" -name '*.pid' | wc -l)
  ((count >= 15)) && break
  sleep 1
done
((count >= 15))
kill -TERM "$launcher_pid"
(
  sleep 20
  kill -0 "$launcher_pid" 2>/dev/null && kill -KILL "$launcher_pid" 2>/dev/null || true
) &
watchdog_pid=$!
set +e
wait "$launcher_pid"
interrupt_status=$?
set -e
launcher_pid=''
kill -TERM "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
watchdog_pid=''
[[ "$interrupt_status" -eq 130 ]]
if ! kill -0 "$sentinel_pid" 2>/dev/null; then
  printf 'unrelated sentinel was terminated by launcher cleanup\n' >&2
  exit 1
fi
for file in "$pid_root"/*.pid; do
  pid=$(<"$file")
  if kill -0 "$pid" 2>/dev/null; then
    printf 'interrupted child is still alive: %s\n' "$pid" >&2
    exit 1
  fi
done
kill -TERM "$sentinel_pid" 2>/dev/null || true
wait "$sentinel_pid" 2>/dev/null || true
sentinel_pid=''

printf '%s\n' 'passive_qualitative_scheduler_mock=PASS complete=11200/224/60800/256 partial=PASS merge_only=PASS duplicate=PASS malformed=PASS protocol=PASS truth=PASS interrupt_cleanup=PASS'

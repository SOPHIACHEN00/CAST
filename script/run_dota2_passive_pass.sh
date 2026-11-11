#!/usr/bin/env bash
set -Eeuo pipefail
# Standalone PASS-only launcher. Default: DRY_RUN=1 (print commands only).
# Full run: DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 bash this_script.sh
# All alphas, two pure attacks once, two lazy attacks at gaps 3/6/10.
# One CSV per dataset after all tasks succeed. See README_passive_pass.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
exec "${MERGER_PYTHON_BIN:-${PYTHON_BIN:-python}}" - --project-root "${PROJECT_ROOT}" --dataset dota2 "$@" <<'PY_PASS_CAMPAIGN'
"""Standalone PASS campaign payload; embedded into the five Bash launchers.

Only the Python standard library is used for scheduling and consolidation.
Experiment parameters are copied from run_<dataset>_passive_qualitative.sh.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading

PROFILES = {
    'tictactoe': dict(alphas='10 100 1000 20000', pure='moirai_online advanced_scaled_delta', lazy='moirai_online synthetic_replacement', gpu='1', arima=('0.005','0.016'), moirai=('0.05','0.025'), dkfr=''),
    'adult': dict(alphas='1000 100000 1000000 20000000', pure='moirai_online adam_local_deviation', lazy='arima_adaptive synthetic_replacement', gpu='2', arima=('0.005','0.08'), moirai=('0.003','0.12'), dkfr=''),
    'dota2': dict(alphas='10000 1000000 50000000 200000000', pure='arima_adaptive dkfr', lazy='synthetic_replacement moirai_online', gpu='3', arima=('0.005','0.18'), moirai=('0.001','0.25'), dkfr='32 2 2 5e-4 5e-4 2 0.1 1 1 128 1 0 0 0 0 5 0 3'),
    'mnist': dict(alphas='10 1000 100000 10000000', pure='moirai_online_mnist_imdb_layerwise dkfr', lazy='synthetic_replacement moirai_online_mnist_imdb_layerwise', gpu='0', arima=('0.05','0.05'), moirai=('0.003','0.06'), dkfr='32 2 2 1e-5 5e-6 12 0 0.05 1 128 0 1 5 8 0.15 0.2 0.1 2'),
    'imdb': dict(alphas='10 100 1000 200000', pure='dkfr moirai_online_mnist_imdb_layerwise', lazy='synthetic_replacement arima_adaptive_mnist_imdb_layerwise', gpu='1', arima=('0.10','0.015'), moirai=('0.01','0.025'), dkfr='32 2 2 5e-5 5e-5 4 0 0.1 1 128 0 1 2 4 0 0.5 0.1 2'),
}
for _profile in PROFILES.values():
    _profile['uniform'] = _profile['alphas'].split()[-1]
HESSIAN = {'arima_adaptive','arima_adaptive_mnist_imdb_layerwise','moirai_online','moirai_online_mnist_imdb_layerwise'}
DKFR_FLAGS = ['--dkfr_' + s for s in 'synth_batch_size generator_steps kd_steps lr_g lr_student temperature alpha_feat alpha_kl alpha_cls hidden_dim lambda_kl_g kl_cap student_kl_cap student_loss_cap teacher_conf_min grad_clip max_update_norm guard_reset_patience'.split()]
PASS_SETTINGS = dict(alpha=0.95, beta=1.75, c0=None, report_low=-1.0, report_high=1.0, seed=42)

@dataclass(frozen=True)
class Task:
    dataset: str
    trial: int
    alpha: str
    clients: int
    regime: str
    attack: str
    gap: int | None

    @property
    def task_id(self):
        regime = 'pure' if self.gap is None else f'lazy_gap{self.gap}'
        return f'pass_{self.dataset}_t{self.trial}_a{self.alpha}_n{self.clients}_{regime}_{self.attack}'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def plan_tasks(dataset, trials, gaps):
    check(dataset in PROFILES, 'Unknown dataset')
    check(bool(trials) and len(set(trials)) == len(trials) and all(isinstance(t,int) and t >= 0 for t in trials), 'Invalid/duplicate TRIAL_IDS')
    check(bool(gaps) and len(set(gaps)) == len(gaps) and set(gaps) <= {3,6,10}, 'LAZY_GAPS must be a unique subset of 3 6 10')
    p = PROFILES[dataset]
    return [Task(dataset,t,a,n,regime,attack,gap)
            for t in trials for a in p['alphas'].split() for n in (2,4,6,8)
            for regime,gap in [('pure',None)] + [('lazy',g) for g in gaps]
            for attack in p[regime].split()]


def build_command(task, runner, results, python_bin, audit_batch):
    t = task
    p = PROFILES[t.dataset]
    args = [str(python_bin),str(runner),'--dataset',t.dataset,'--alpha',t.alpha,'--client_num',str(t.clients),
        '--attacker_id','1','--attack_method',t.attack,'--detecting_method','pass','--detecting_type','qualitative',
        '--qualitative_aggregation_mode','passive','--trial_id',str(t.trial),'--num_epoch','50','--num_local_epochs','1',
        '--random_round','2' if t.attack == 'advanced_scaled_delta' else '10','--results_root',str(results),'--use_attack',
        '--pass_alpha','0.95','--pass_beta','1.75','--pass_report_low','-1.0','--pass_report_high','1.0',
        '--pass_seed','42','--pass_audit_batch_size',str(audit_batch),'--pass_max_audit_samples','0']
    if 'arima' in t.attack:
        args += ['--arima_var_window','3','--arima_var_threshold',p['arima'][0],'--arima_err_threshold',p['arima'][1],
            '--arima_min_rounds_after_start','4','--arima_retrain_interval','15','--arima_err_ema_beta','0.9','--arima_trim_history']
    elif 'moirai' in t.attack:
        args += ['--moirai_online_tuning','--moirai_tune_lr',p['moirai'][0],'--moirai_tune_steps','1','--moirai_tune_start_round','10',
            '--moirai_adapt_interval','4','--moirai_residual_decay','0.95','--moirai_residual_clip','5','--moirai_err_ema_beta','0.9',
            '--moirai_err_threshold',p['moirai'][1],'--moirai_min_rounds_after_start','4','--moirai_trim_history']
    elif t.attack == 'advanced_scaled_delta':
        args += '--scaled_delta_c_const 1.0 --scaled_delta_noise_fraction 1.0 --scaled_delta_max_noise_ratio 5.0 --scaled_delta_max_scale 10.0'.split()
    elif t.attack == 'adam_local_deviation':
        args += '--wang_adam_lr 8e-3 --wang_adam_beta1 0.9 --wang_adam_beta2 0.999 --wang_adam_eps 1e-8 --adam_public_fraction 0.05 --adam_public_seed 20260726'.split()
    elif t.attack == 'dkfr':
        values = p['dkfr'].split()
        check(len(values) == len(DKFR_FLAGS), 'Missing DKFR profile')
        args += [item for pair in zip(DKFR_FLAGS,values) for item in pair]
        if t.dataset == 'mnist':
            args += '--dkfr_mnist_tv 0.1 --dkfr_mnist_pixel_reg 0.2 --dkfr_mnist_logit_reg 1e-4'.split()
    elif t.attack == 'synthetic_replacement':
        args += '--synthetic_mixup_alpha 0.4 --synthetic_noise_std 0.02 --synthetic_tabular_noise_std 0.05 --synthetic_tabular_flip_prob 0.02 --synthetic_seed 0'.split()
    if t.attack.endswith('_mnist_imdb_layerwise'):
        args += '--layerwise_selection_metric normalized_rms --layerwise_selection_fraction 0.5 --layerwise_pca_threshold 512 --layerwise_pca_components 8'.split()
    if t.attack in HESSIAN:
        args += ['--use_hessian','--lambda_h','0.9']
    if t.regime == 'lazy':
        args += ['--use_lazy','--lazy_schedule','fixed','--lazy_honest_gap',str(t.gap),'--lazy_honest_phase','0','--lazy_fake_policy','attack']
    return args


def new_campaign_config(dataset,trials,gaps,project_root,runner,python_bin,audit_batch,campaign_id):
    return dict(schema=1,dataset=dataset,trials=trials,gaps=gaps,epochs=50,campaign_id=campaign_id,
        project_root=str(project_root),runner=str(runner),python_bin=str(python_bin),audit_batch=audit_batch,
        pass_settings=PASS_SETTINGS,tasks=[asdict(t) for t in plan_tasks(dataset,trials,gaps)])


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def compact(data):
    return json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)


TIME_COLUMNS = dict(AttackGenerationWithAttackerTotalSec='TaskAttackGenerationSec', FakeRoundsTimed='TaskFakeRoundsTimed',
    DetectorDecisionWithAttackerTotalSec='TaskDetectorWithAttackerSec', DetectorDecisionWithoutAttackerTotalSec='TaskDetectorWithoutAttackerSec',
    FLWithAttackerWallSec='TaskFLWithAttackerSec', FLWithoutAttackerWallSec='TaskFLWithoutAttackerSec', TaskWallSec='TaskWallSec')
DET_FIELDS = 'TaskID CampaignComplete Dataset AttackMethod AttackRegime Alpha ClientNum AttackerID TrialID UseHessian LambdaH UseLazy LazySchedule LazyHonestGap LazyHonestPhase LazyFakePolicy DetectingMethod QualitativeAggregationMode Round ClientID IsDesignatedAttacker GroundTruthActiveAttack DesignatedAttackerRoundState RoundHasActiveAttack RoundIsHonestForDesignatedAttacker WEFPayloadSource DetectionSkipped DetectorScore DetectorThreshold DeltaThresholdPathology RoundPredictedSuspicious FinalPredictedSuspicious DecisionMode DetectorDetailsJSON'.split()
EXTRA_FIELDS = 'CampaignID RunnerTaskID Distribution SubmittedRound AuditRound DecisionRound IncomingReportsJSON WithAttackerAccuracy WithoutAttackerAccuracy AccuracyControlDefinition ControlClientID ControlClientScore ControlDecisionValid ControlInvalidReason ControlCoverage TaskValidDecisions TaskInvalidAttemptedDecisions TaskUnauditedDecisions TaskCoverage TaskEvaluationJSON PassAlpha PassBeta PassC0 PassReportLow PassReportHigh PassSeed PassAuditBatchSize PassMaxAuditSamples'.split() + list(TIME_COLUMNS.values())


def validate_metadata(row,task):
    expected = dict(Dataset=task.dataset,AttackMethod=task.attack,DetectingMethod='pass',
        Alpha=task.alpha,ClientNum=task.clients,AttackerID=1,TrialID=task.trial)
    for name,value in expected.items():
        check(name in row, f'Missing metadata: {name}')
        equal = float(row[name]) == float(value) if name in ('Alpha','ClientNum','AttackerID','TrialID') else row[name] == value
        check(equal, f'{task.task_id}: incorrect {name}: {row[name]}')
    if 'UseLazy' in row:
        check(str(row['UseLazy']).lower() in (('true','1') if task.gap else ('false','0')), 'UseLazy mismatch')
    if task.gap:
        check(int(row['LazyHonestGap']) == task.gap and int(row['LazyHonestPhase']) == 0, 'Lazy cadence mismatch')
        check(row['LazySchedule'] == 'fixed' and row['LazyFakePolicy'] == 'attack', 'Lazy policy mismatch')
    if 'QualitativeAggregationMode' in row:
        check(row['QualitativeAggregationMode'] == 'passive', 'Aggregation mode mismatch')


def task_rows(directory,config,task):
    root = directory/'tasks'/task.task_id/'results'
    trial = task.trial
    detection = csv_rows(root/f'Detecting_qualitative_pass/round_client_detection_trial{trial}.csv')
    accuracy = csv_rows(root/f'Accuracy_qualitative_pass/global_accuracy_log_trial{trial}.csv')
    timing = csv_rows(root/f'Time_qualitative_pass/attacking_time_log_trial{trial}.csv')
    audit = read_json(root/f'Detecting_qualitative_pass/pass_audit_trial{trial}.json')
    control = read_json(root/f'Detecting_qualitative_pass/pass_audit_trial{trial}_without_attacker.json')
    check(len(timing) == 1 and len(accuracy) == 50 and len(detection) == 50*task.clients, 'Incomplete or duplicated task outputs')
    for row in detection+accuracy+timing:
        validate_metadata(row,task)
    acc = {int(row['Round']):row for row in accuracy}
    records = {int(row['submitted_round']):row for row in audit['round_records']}
    controls = {int(row['submitted_round']):row for row in control['round_records']}
    check(set(acc) == set(records) == set(controls) == set(range(50)), 'Missing submitted rounds')
    check(len(audit['round_records']) == len(control['round_records']) == 50, 'Duplicate audit rounds')
    check({(int(row['Round']),int(row['ClientID'])) for row in detection} == {(r,i) for r in range(50) for i in range(task.clients)}, 'Missing or duplicate detection rows')
    original_ids = [i for i in range(task.clients) if i != 1]
    check(control['experiment']['original_client_ids'] == original_ids, 'Wrong N-1 control mapping')
    for name,value in PASS_SETTINGS.items():
        expected = 42+trial*100003 if name == 'seed' else value
        check(audit['config'][name] == expected, f'Wrong PASS setting: {name}')
    check(audit['config']['audit_batch_size'] == config['audit_batch'], 'Audit batch mismatch')
    check(audit['config']['max_audit_samples'] is None, 'Expected full local audit partition')
    check(math.isclose(audit['config']['effective_c0'],1/task.clients), 'Wrong initial contribution')
    valid_count = 0
    for row in sorted(detection,key=lambda d:(int(d['Round']),int(d['ClientID']))):
        r,i = int(row['Round']),int(row['ClientID'])
        rec,ctl = records[r],controls[r]
        expected_decision = r+1 if r < 49 else None
        for record in (rec,ctl):
            check(record['audit_round'] == record['decision_round'] == expected_decision, 'PASS decision/submission misalignment')
        details = json.loads(row['DetectorDetailsJSON'])
        check(details['submitted_round'] == r and details['decision_round'] == expected_decision, 'Detection CSV is misaligned')
        valid = rec['valid_by_client'][str(i)]
        valid_count += int(valid)
        score = rec['client_scores'][str(i)]
        if valid:
            check(score is not None and math.isfinite(score) and math.isclose(float(row['DetectorScore']),score), 'Invalid score')
            check(int(row['RoundPredictedSuspicious']) == int(i in rec['suspicious_clients']), 'Prediction mismatch')
        else:
            check(score is None and row['DetectorScore'] == row['RoundPredictedSuspicious'] == '', 'Invalid decisions must remain blank')
        check(int(row['DetectionSkipped']) == int(not valid), 'Validity mismatch')
        check(r != 49 or not valid, 'Last submission must remain unaudited')
        honest = task.gap is not None and r % task.gap == 0
        warm = task.attack == 'adam_local_deviation' and r == 0
        active = not honest and not warm
        check(int(row['GroundTruthActiveAttack']) == int(i == 1 and active), 'Submitted-round attack label mismatch')
        matrix = rec['report_matrix']
        check(len(matrix) == task.clients and all(len(v) == task.clients for v in matrix), 'Wrong report matrix dimensions')
        incoming = [dict(auditor=j,report=matrix[j][i],source=rec['report_sources'][str(j)],samples=rec['audit_sample_counts'][str(j)]) for j in range(task.clients) if j != i]
        ci = original_ids.index(i) if i in original_ids else None
        cv = ctl['valid_by_client'][str(ci)] if ci is not None else None
        cr = ctl['invalid_reasons'][str(ci)] if ci is not None else 'attacker_removed'
        result = {key:row[key] for key in DET_FIELDS}
        result.update(TaskID=task.task_id,CampaignComplete=1,RunnerTaskID=row['TaskID'],CampaignID=config['campaign_id'],
            LazyHonestGap=task.gap if task.gap is not None else 'N/A',
            Distribution='uniform' if task.alpha == PROFILES[task.dataset]['uniform'] else 'quantity skew',
            SubmittedRound=r,AuditRound=expected_decision,DecisionRound=expected_decision,IncomingReportsJSON=compact(incoming),
            WithAttackerAccuracy=acc[r]['WithAttackerAccuracy'],WithoutAttackerAccuracy=acc[r]['WithoutAttackerAccuracy'],
            AccuracyControlDefinition='existing_remove_attacker_N_minus_1',ControlClientID=ci,
            ControlClientScore=ctl['client_scores'][str(ci)] if ci is not None else None,
            ControlDecisionValid=int(cv) if cv is not None else None,ControlInvalidReason=cr,ControlCoverage=control['coverage']['coverage'],
            TaskValidDecisions=audit['coverage']['valid_decisions'],TaskInvalidAttemptedDecisions=audit['coverage']['invalid_attempted_decisions'],
            TaskUnauditedDecisions=audit['coverage']['unaudited_decisions'],TaskCoverage=audit['coverage']['coverage'],TaskEvaluationJSON=compact(audit.get('evaluation',{})),
            PassAlpha=0.95,PassBeta=1.75,PassC0=1/task.clients,PassReportLow=-1.0,PassReportHigh=1.0,PassSeed=42+trial*100003,
            PassAuditBatchSize=config['audit_batch'],PassMaxAuditSamples='all')
        result.update({target:timing[0][source] for source,target in TIME_COLUMNS.items()})
        yield result
    check(valid_count == audit['coverage']['valid_decisions'], 'Coverage count mismatch')
    check(math.isclose(audit['coverage']['coverage'],valid_count/(50*task.clients)), 'Coverage fraction mismatch')


def merge_campaign(directory,output):
    directory,output = Path(directory),Path(output)
    if output.exists():
        raise FileExistsError(f'Already published: {output}')
    config = read_json(directory/'campaign.json')
    tasks = plan_tasks(config['dataset'],config['trials'],config['gaps'])
    check(config['tasks'] == [asdict(t) for t in tasks], 'Campaign plan changed or is incomplete')
    check(config['epochs'] == 50 and config['pass_settings'] == PASS_SETTINGS, 'Campaign settings changed')
    for task in tasks:
        status_path = directory/'tasks'/task.task_id/'status.json'
        check(status_path.is_file(), f'Missing task: {task.task_id}')
        status = read_json(status_path)
        check(status.get('state') == 'success' and status.get('exit_code') == 0, f'Failed task: {task.task_id}; see its run.log')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8-sig',newline='',dir=output.parent,prefix='.'+output.stem,suffix='.tmp',delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream,fieldnames=DET_FIELDS+EXTRA_FIELDS)
            writer.writeheader()
            for task in tasks:
                for row in task_rows(directory,config,task):
                    writer.writerow(row)
                    count += 1
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes atomically and refuses to overwrite.
        os.link(temporary,output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'MERGED|dataset={config["dataset"]}|tasks={len(tasks)}|rows={count}|file={output}',flush=True)
    return count


class CampaignRunner:
    def __init__(self,directory,config,gpus,workers,threads):
        self.directory,self.config = directory,config
        self.gpus,self.workers,self.threads = gpus,workers,threads
        self.pending = queue.Queue()
        for spec in config['tasks']:
            self.pending.put(Task(**spec))
        self.stop = threading.Event()

    def environment(self,gpu,folder):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[key] = str(self.threads)
        for key,name in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch')]:
            path = folder/name
            path.mkdir(parents=True,exist_ok=True)
            env[key] = str(path)
        env['HF_HOME'] = os.environ.get('SHARED_HF_HOME',os.environ.get('HF_HOME','/output/huggingface'))
        env['HF_HUB_OFFLINE'] = os.environ.get('HF_HUB_OFFLINE','1')
        env['TRANSFORMERS_OFFLINE'] = os.environ.get('TRANSFORMERS_OFFLINE','1')
        return env

    def run_one(self,task,gpu):
        folder = self.directory/'tasks'/task.task_id
        folder.mkdir(parents=True)
        command = build_command(task,Path(self.config['runner']),folder/'results',self.config['python_bin'],self.config['audit_batch'])
        write_json(folder/'command.json',dict(task=asdict(task),argv=command,gpu=gpu,cpu_threads=self.threads))
        write_json(folder/'status.json',dict(state='running'))
        print(f'START|gpu={gpu}|{task.task_id}',flush=True)
        try:
            env = self.environment(gpu,folder)
            with (folder/'run.log').open('w',encoding='utf-8') as log:
                kwargs = dict(start_new_session=True) if os.name == 'posix' else dict(creationflags=subprocess.CREATE_NO_WINDOW)
                with subprocess.Popen(command,cwd=self.config['project_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**kwargs) as process:
                    while True:
                        try:
                            code = process.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            if self.stop.is_set():
                                if os.name == 'posix':
                                    os.killpg(process.pid,signal.SIGTERM)
                                else:
                                    process.terminate()
                                try:
                                    code = process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    if os.name == 'posix':
                                        os.killpg(process.pid,signal.SIGKILL)
                                    else:
                                        process.kill()
                                    code = process.wait()
                                break
            status = dict(state='success' if code == 0 else 'failed',exit_code=code)
        except Exception as error:
            status = dict(state='failed',exit_code=-1,error=str(error))
        write_json(folder/'status.json',status)
        print(f'END|{task.task_id}|{status["state"]}|exit={status["exit_code"]}',flush=True)

    def worker(self,gpu):
        while not self.stop.is_set():
            try:
                task = self.pending.get_nowait()
            except queue.Empty:
                return
            self.run_one(task,gpu)

    def run(self):
        previous = {}
        def stop(signum,frame):
            self.stop.set()
        try:
            for sig in (signal.SIGINT,signal.SIGTERM):
                previous[sig] = signal.signal(sig,stop)
            with ThreadPoolExecutor(max_workers=len(self.gpus)*self.workers) as pool:
                futures = [pool.submit(self.worker,gpu) for gpu in self.gpus for _ in range(self.workers)]
                for future in futures:
                    future.result()
        finally:
            for sig,handler in previous.items():
                signal.signal(sig,handler)
        check(not self.stop.is_set(), 'Campaign interrupted; private logs retained, no final CSV published')


def env_flag(name,default):
    value = os.environ.get(name,default)
    check(value in ('0','1'), f'{name} must be 0 or 1')
    return value == '1'


def env_ints(name,default):
    values = os.environ.get(name,default).split()
    check(bool(values) and all(re.fullmatch(r'0|[1-9][0-9]*',v) for v in values), f'Invalid {name}')
    result = [int(v) for v in values]
    check(len(set(result)) == len(result), f'Duplicate {name}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=PROFILES)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    dry = env_flag('DRY_RUN','1')
    merge_only = env_flag('MERGE_ONLY','0')
    confirm = env_flag('CONFIRM_QUALITATIVE_RUN','0')
    check(os.environ.get('ATTACKER_ID','1') == '1' and os.environ.get('NUM_EPOCH','50') == '50', 'Keep ATTACKER_ID=1 and NUM_EPOCH=50')
    cid = os.environ.get('CAMPAIGN_ID',f'passive_pass_{datetime.now():%Y%m%d-%H%M%S}_pid{os.getpid()}')
    check(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',cid) is not None, 'Invalid CAMPAIGN_ID')
    directory = root/'.qualitative_runs'/cid/args.dataset
    output = root/'Results'/cid/f'{args.dataset}_pass.csv'
    if merge_only:
        check(not dry and 'CAMPAIGN_ID' in os.environ, 'MERGE_ONLY=1 requires DRY_RUN=0 and explicit CAMPAIGN_ID')
        saved = read_json(directory/'campaign.json')
        check(saved['dataset'] == args.dataset and saved['campaign_id'] == cid, 'Wrong saved campaign')
        merge_campaign(directory,output)
        return
    trials,gaps = env_ints('TRIAL_IDS','0'),env_ints('LAZY_GAPS','3 6 10')
    gpus = env_ints('GPU_IDS',PROFILES[args.dataset]['gpu'])
    workers = int(os.environ.get('WORKERS_PER_GPU','1'))
    threads = int(os.environ.get('CPU_THREADS_PER_WORKER','8'))
    audit_batch = int(os.environ.get('PASS_AUDIT_BATCH_SIZE','256'))
    check(min(workers,threads,audit_batch) > 0, 'Worker/thread/audit batch sizes must be positive')
    runner = Path(os.environ.get('SCRIPT','FLCE_updated.py'))
    if not runner.is_absolute():
        runner = root/runner
    config = new_campaign_config(args.dataset,trials,gaps,root,runner,os.environ.get('PYTHON_BIN','python'),audit_batch,cid)
    config.update(gpus=gpus,workers_per_gpu=workers,cpu_threads_per_worker=threads)
    print(f'CONFIG|dataset={args.dataset}|alphas={PROFILES[args.dataset]["alphas"]}|gaps={gaps}|gpu={gpus}|workers={workers}|threads={threads}',flush=True)
    print(f'OUTPUT|{output}',flush=True)
    if dry:
        for spec in config['tasks']:
            task = Task(**spec)
            print('TASK|'+compact(dict(task_id=task.task_id,**spec)))
            print('command: '+shlex.join(build_command(task,runner,directory/'tasks'/task.task_id/'results',config['python_bin'],audit_batch)))
        print(f'DRY_RUN|tasks={len(config["tasks"])}|no experiments launched')
        return
    check(confirm, 'Use DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 to run, as in the existing passive scripts')
    check(os.name == 'posix', 'Training launcher requires a Linux server; dry run and merge are portable')
    check(not directory.exists() and not output.exists(), 'Campaign already exists; use a new CAMPAIGN_ID or MERGE_ONLY=1')
    check(runner.is_file(), f'Missing runner: {runner}')
    import fcntl
    lock_root = Path(os.environ.get('QUALITATIVE_GPU_LOCK_ROOT',str(Path(os.environ.get('TMPDIR','/tmp'))/'passive_qualitative_gpu_locks')))
    lock_root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            handle = stack.enter_context((lock_root/f'gpu_{gpu}.lock').open('a'))
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f'GPU {gpu} is locked by another passive campaign') from error
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([config['python_bin'],'-c','import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'],env=env,check=True)
        subprocess.run([config['python_bin'],str(runner),'--detecting_method','pass','--detecting_type','qualitative','--check_detecting_setup'],cwd=root,check=True)
        directory.mkdir(parents=True)
        write_json(directory/'campaign.json',config)
        CampaignRunner(directory,config,gpus,workers,threads).run()
        merge_campaign(directory,output)


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
PY_PASS_CAMPAIGN

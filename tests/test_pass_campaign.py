"""PASS campaign matrix, exact inherited flags, atomic merge and isolation tests."""
from __future__ import annotations
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'script/passive_pass_campaign.py'
EXPECTED={
 'tictactoe':(['10','100','1000','20000'],['moirai_online','advanced_scaled_delta'],['moirai_online','synthetic_replacement']),
 'adult':(['1000','100000','1000000','20000000'],['moirai_online','adam_local_deviation'],['arima_adaptive','synthetic_replacement']),
 'dota2':(['10000','1000000','50000000','200000000'],['arima_adaptive','dkfr'],['synthetic_replacement','moirai_online']),
 'mnist':(['10','1000','100000','10000000'],['moirai_online_mnist_imdb_layerwise','dkfr'],['synthetic_replacement','moirai_online_mnist_imdb_layerwise']),
 'imdb':(['10','100','1000','200000'],['dkfr','moirai_online_mnist_imdb_layerwise'],['synthetic_replacement','arima_adaptive_mnist_imdb_layerwise']),
}


def option(argv,key):
    assert argv.count(key)==1,(key,argv)
    return argv[argv.index(key)+1]


class PASSCampaignTests(unittest.TestCase):
    def api(self):
        self.assertTrue(SOURCE.exists(),'PASS campaign launcher has not been implemented')
        if '_pass_campaign_test' not in sys.modules:
            spec=importlib.util.spec_from_file_location('_pass_campaign_test',SOURCE)
            module=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=module
            spec.loader.exec_module(module)
        return sys.modules['_pass_campaign_test']

    def test_matrix_has_exact_dataset_alphas_attacks_and_no_repeated_pure_runs(self):
        api=self.api()
        for dataset,(alphas,pure,lazy) in EXPECTED.items():
            tasks=api.plan_tasks(dataset,[0],[3,6,10])
            self.assertEqual(len(tasks),128)
            self.assertEqual(len({t.task_id for t in tasks}),128)
            self.assertEqual(sum(t.regime=='pure' for t in tasks),32)
            self.assertEqual({t.alpha for t in tasks},set(alphas))
            self.assertEqual({t.attack for t in tasks if t.regime=='pure'},set(pure))
            self.assertEqual({t.attack for t in tasks if t.regime=='lazy'},set(lazy))
            for gap,fake in [(3,33),(6,41),(10,45)]:
                group=[t for t in tasks if t.gap==gap]
                self.assertEqual(len(group),32)
                self.assertEqual(sum(r%gap!=0 for r in range(50)),fake)
            self.assertEqual({t.clients for t in tasks},{2,4,6,8})
            self.assertEqual(api.PROFILES[dataset]['uniform'],alphas[-1])
        with self.assertRaises(ValueError): api.plan_tasks('tictactoe',[0],[2])
        with self.assertRaises(ValueError): api.plan_tasks('tictactoe',[0,0],[3,6,10])

    def test_attack_arguments_match_the_old_fixed_gap_scripts(self):
        api=self.api()
        for dataset in EXPECTED:
            for task in api.plan_tasks(dataset,[0],[3,6,10]):
                command=api.build_command(task,Path('/project/FLCE_updated.py'),Path('/private/result'),'python',256)
                self.assertEqual(option(command,'--dataset'),dataset)
                self.assertEqual(option(command,'--alpha'),task.alpha)
                self.assertEqual(option(command,'--detecting_method'),'pass')
                self.assertEqual(option(command,'--qualitative_aggregation_mode'),'passive')
                self.assertEqual(option(command,'--attacker_id'),'1')
                self.assertEqual(option(command,'--num_epoch'),'50')
                self.assertEqual(option(command,'--num_local_epochs'),'1')
                self.assertEqual(option(command,'--pass_alpha'),'0.95')
                self.assertEqual(option(command,'--pass_max_audit_samples'),'0')
                self.assertEqual('--use_hessian' in command,task.attack in api.HESSIAN)
                self.assertEqual('--use_lazy' in command,task.regime=='lazy')
                if task.regime=='lazy': self.assertEqual(option(command,'--lazy_honest_gap'),str(task.gap))
                if 'moirai' in task.attack:
                    source=(ROOT/f'script/run_{dataset}_passive_qualitative.sh').read_text(encoding='utf-8')
                    for source_name,flag in [('MOIRAI_LR','--moirai_tune_lr'),('MOIRAI_ERR_THRESHOLD','--moirai_err_threshold')]:
                        value=re.search(r'^'+source_name+r'=(.*)$',source,re.M).group(1).strip()
                        self.assertEqual(option(command,flag),value)
                if 'arima' in task.attack:
                    source=(ROOT/f'script/run_{dataset}_passive_qualitative.sh').read_text(encoding='utf-8')
                    for source_name,flag in [('ARIMA_VAR_THRESHOLD','--arima_var_threshold'),('ARIMA_ERR_THRESHOLD','--arima_err_threshold')]:
                        value=re.search(r'^'+source_name+r'=(.*)$',source,re.M).group(1).strip()
                        self.assertEqual(option(command,flag),value)
                if task.attack=='advanced_scaled_delta': self.assertEqual(option(command,'--random_round'),'2')
                if task.attack=='dkfr':
                    source=(ROOT/f'script/run_{dataset}_passive_qualitative.sh').read_text(encoding='utf-8')
                    values=re.search(r"^DKFR_VALUES='(.*)'$",source,re.M).group(1).split()
                    self.assertEqual([option(command,k) for k in api.DKFR_FLAGS],values)

    def test_five_bash_files_embed_the_complete_same_python_payload(self):
        self.api()
        for dataset in EXPECTED:
            file=ROOT/f'script/run_{dataset}_passive_pass.sh'
            self.assertTrue(file.is_file(),str(file))
            text=file.read_text(encoding='utf-8')
            self.assertNotIn('\r',text)
            self.assertIn('--dataset '+dataset,text)
            payload=text.split("<<'PY_PASS_CAMPAIGN'\n",1)[1].rsplit('\nPY_PASS_CAMPAIGN',1)[0]
            self.assertEqual(payload,SOURCE.read_text(encoding='utf-8').rstrip())
            compile(payload,str(file),'exec')

    def make_campaign(self,root):
        api=self.api()
        config=api.new_campaign_config('tictactoe',[0],[6],Path(root),Path(root)/'FLCE_updated.py','python',256,'test_campaign')
        directory=Path(root)/'.qualitative_runs/test_campaign/tictactoe'
        directory.mkdir(parents=True)
        (directory/'campaign.json').write_text(json.dumps(config),encoding='utf-8')
        for task in api.plan_tasks('tictactoe',[0],[6]):
            folder=directory/'tasks'/task.task_id
            folder.mkdir(parents=True)
            (folder/'status.json').write_text(json.dumps({'state':'success','exit_code':0}),encoding='utf-8')
            self.write_fixture(api,folder,task)
        return directory,Path(root)/'Results/test_campaign/tictactoe_pass.csv'

    def write_fixture(self,api,folder,task):
        n=task.clients; T=50; root=folder/'results'; trial=task.trial
        raw_id=f'{task.dataset}_{task.attack}_pass_alpha{float(task.alpha)}_clients{n}_attacker1_trial{trial}'
        common=dict(TaskID=raw_id,CampaignComplete=1,Dataset=task.dataset,AttackMethod=task.attack,
          AttackRegime=task.regime,Alpha=task.alpha,ClientNum=n,AttackerID=1,TrialID=trial,
          UseHessian=int(task.attack in api.HESSIAN),LambdaH=.9 if task.attack in api.HESSIAN else 'N/A',
          UseLazy=int(task.regime=='lazy'),LazySchedule='fixed',LazyHonestGap=task.gap or 3,
          LazyHonestPhase=0,LazyFakePolicy='attack',DetectingMethod='pass',QualitativeAggregationMode='passive')
        det=[]; acc=[]; reports=[]
        for r in range(T):
            honest=bool(task.gap and r%task.gap==0)
            active=not honest
            row=dict(round_idx=r,submitted_round=r,audit_round=r+1 if r<T-1 else None,
                decision_round=r+1 if r<T-1 else None,
                client_scores={str(i):1.0 if r<T-1 else None for i in range(n)},
                valid_by_client={str(i):r<T-1 for i in range(n)},
                invalid_reasons={str(i):None if r<T-1 else 'not_audited_before_training_end' for i in range(n)},
                suspicious_clients=[],report_matrix=[[None if i==j or r==T-1 else 0.0 for j in range(n)] for i in range(n)],
                report_sources={str(i):('not_audited' if r==T-1 else 'random_uniform' if i==1 and task.regime=='pure' else 'real_local_data') for i in range(n)},
                audit_sample_counts={str(i):2 for i in range(n)},
                ground_truth_labels={str(i):int(i==1 and active) for i in range(n)})
            reports.append(row)
            for i in range(n):
                det.append(dict(common,Round=r,ClientID=i,IsDesignatedAttacker=int(i==1),GroundTruthActiveAttack=int(i==1 and active),
                  DesignatedAttackerRoundState='honest' if honest else 'fake',RoundHasActiveAttack=int(active),
                  RoundIsHonestForDesignatedAttacker=int(honest),WEFPayloadSource='N/A',DetectionSkipped=int(r==T-1),
                  DetectorScore='' if r==T-1 else 1.,DetectorThreshold='' if r==T-1 else 1/(1.75*n),
                  DeltaThresholdPathology=0,RoundPredictedSuspicious='' if r==T-1 else 0,
                  FinalPredictedSuspicious='' if r==T-1 else 0,DecisionMode='passive_previous_round_contribution_below_threshold',
                  DetectorDetailsJSON=json.dumps({k:row[k] for k in ['submitted_round','audit_round','decision_round']})))
            acc.append(dict(common,Round=r,WithAttackerAccuracy=.7,WithoutAttackerAccuracy=.8,UseAttack=True,DetectingType='qualitative',AttackProbability=1.0))
        timing=dict(common,AttackGenerationWithAttackerTotalSec=1.,FakeRoundsTimed=41 if task.gap else 50,
            DetectorDecisionWithAttackerTotalSec=2.,DetectorDecisionWithoutAttackerTotalSec=1.,
            DetectorDecisionWithoutAttackerMeasured=1,DetectorDecisionWithoutAttackerNotMeasuredReason='',
            FLWithAttackerWallSec=4.,FLWithoutAttackerWallSec=3.,TaskWallSec=8.)
        def write_csv(path,rows):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write_csv(root/f'Detecting_qualitative_pass/round_client_detection_trial{trial}.csv',det)
        write_csv(root/f'Accuracy_qualitative_pass/global_accuracy_log_trial{trial}.csv',acc)
        write_csv(root/f'Time_qualitative_pass/attacking_time_log_trial{trial}.csv',[timing])
        config=dict(api.PASS_SETTINGS,seed=42+trial*100003,effective_c0=1/n,max_audit_samples=None,audit_batch_size=256)
        payload=dict(method='pass',round_records=reports,config=config,qualitative_aggregation_mode='passive',
            coverage=dict(total_submission_decisions=n*T,attempted_decisions=n*(T-1),valid_decisions=n*(T-1),
              invalid_attempted_decisions=0,unaudited_decisions=n,coverage=(T-1)/T,valid_fraction_of_attempted=1),
            final_valid_by_client={str(i):True for i in range(n)},final_audited_submission_round=48,
            experiment=dict(original_client_ids=list(range(n))),evaluation={})
        file=root/f'Detecting_qualitative_pass/pass_audit_trial{trial}.json'
        file.write_text(json.dumps(payload),encoding='utf-8')
        control=copy.deepcopy(payload);cn=n-1
        control['experiment']['original_client_ids']=[i for i in range(n) if i!=1]
        control['coverage']['coverage']=0 if cn==1 else .98
        for row in control['round_records']:
            row['client_scores']={str(i):None if cn==1 or row['submitted_round']==49 else 1. for i in range(cn)}
            row['valid_by_client']={str(i):cn>1 and row['submitted_round']<49 for i in range(cn)}
            row['invalid_reasons']={str(i):'not_audited_before_training_end' if row['submitted_round']==49 else 'no_other_auditor' if cn==1 else None for i in range(cn)}
        file.with_name(f'pass_audit_trial{trial}_without_attacker.json').write_text(json.dumps(control),encoding='utf-8')

    def test_single_dataset_file_joins_rounds_reports_time_and_invalid_control(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            directory,out=self.make_campaign(tmp)
            self.assertFalse(out.exists())
            api.merge_campaign(directory,out)
            self.assertEqual(list(out.parent.iterdir()),[out])
            with out.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),16000)
            self.assertEqual(len({r['TaskID'] for r in rows}),64)
            tail=[r for r in rows if r['SubmittedRound']=='49']
            self.assertTrue(all(r['DecisionRound']=='' and r['RoundPredictedSuspicious']=='' for r in tail))
            control=[r for r in rows if r['ClientNum']=='2' and r['ClientID']=='0' and r['SubmittedRound']=='0']
            self.assertTrue(all(r['ControlInvalidReason']=='no_other_auditor' and r['ControlClientScore']=='' for r in control))
            self.assertEqual(rows[0]['WithAttackerAccuracy'],'0.7')
            self.assertEqual(rows[0]['TaskAttackGenerationSec'],'1.0')
            self.assertEqual(rows[0]['PassAlpha'],'0.95')
            with self.assertRaises(FileExistsError): api.merge_campaign(directory,out)

    def test_no_public_file_when_any_task_is_missing_or_artifacts_are_misaligned(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            directory,out=self.make_campaign(tmp)
            first=api.plan_tasks('tictactoe',[0],[6])[0]
            status=directory/'tasks'/first.task_id/'status.json'
            status.write_text(json.dumps({'state':'failed','exit_code':9}),encoding='utf-8')
            with self.assertRaises(ValueError): api.merge_campaign(directory,out)
            self.assertFalse(out.exists())
            status.write_text(json.dumps({'state':'success','exit_code':0}),encoding='utf-8')
            file=status.parent/'results/Detecting_qualitative_pass/pass_audit_trial0.json'
            data=json.loads(file.read_text(encoding='utf-8'));data['round_records'][0]['decision_round']=0
            file.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError): api.merge_campaign(directory,out)
            self.assertFalse(out.exists())
            self.assertFalse(list(out.parent.glob('*.tmp')) if out.parent.exists() else [])

    def test_merger_rejects_wrong_dataset_alpha_or_duplicate_plan(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            directory,out=self.make_campaign(tmp)
            path=directory/'campaign.json';original=json.loads(path.read_text(encoding='utf-8'))
            for mutation in ('alpha','duplicate','empty'):
                data=copy.deepcopy(original)
                if mutation=='alpha': data['tasks'][0]['alpha']='200000'
                elif mutation=='duplicate': data['tasks'].append(data['tasks'][0])
                else: data['tasks']=[]
                path.write_text(json.dumps(data),encoding='utf-8')
                with self.assertRaises(ValueError): api.merge_campaign(directory,out)
                self.assertFalse(out.exists())

if __name__=='__main__': unittest.main()

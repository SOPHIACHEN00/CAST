import importlib.util
import json
from pathlib import Path
import pytest
import torch
from test_rq1_single_attacker import arguments, api, actual_parser, loader, ScoreModel
from test_multi_freerider import training_namespace, ToyAttack, ToyModel
from test_rq1_fl_aggregation_campaign import fixture_result

ROOT=Path(__file__).resolve().parents[1]

def campaign():
    spec=importlib.util.spec_from_file_location('supplement',ROOT/'script/rq1_adam_dkfr_campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class PublicAttack(ToyAttack):
    def initialize_public_warm_start(self, before, after):
        return [b.detach()-a.detach() for a,b in zip(before.parameters(),after.parameters())]

@pytest.mark.parametrize('attack',['adam_local_deviation','dkfr'])
@pytest.mark.parametrize('method',['fedavg','fedadam','weighted_median'])
def test_real_single_loop_and_result_consumer(monkeypatch,tmp_path,attack,method):
    previous=torch.get_num_threads()
    try:
        args=arguments(monkeypatch,tmp_path,method,0);args.attack_method=attack
        helper=api(monkeypatch);data=loader();model=ScoreModel();attacker=PublicAttack()
        context=helper.prepare_run(args=args,loader=data,model=model)
        ns=training_namespace();ns['args']=args;agg={};ToyModel.trained_clients=[]
        result=ns['run_federated_learning'](data,model,0,attacker,True,50,1,torch.device('cpu'),
            public_data=(torch.full((3,),99.),torch.zeros(3)) if attack=='adam_local_deviation' else None,
            last_round_aggregation_out=agg,single_attacker_audit_out=context['audit'])
        assert ToyModel.trained_clients.count(0)==0
        assert ToyModel.trained_clients.count(99)==(1 if attack=='adam_local_deviation' else 0)
        artifact=helper.finish_run(args=args,loader=data,model=model,attacker=attacker,run_context=context,
            last_round_state=result[3],last_round_client_grads=result[4],last_round_weights=result[5],
            aggregation_context=agg,accuracy=result[2],attack_times=result[1],training_seconds=.1,provenance={})
        warm=[0] if attack=='adam_local_deviation' else []
        assert artifact['audit']['public_warm_start_rounds']==warm
        assert artifact['audit']['fake_call_rounds']==list(range(len(warm),50))
        c=campaign();task=next(t for t in c.build_tasks(['adult']) if t['attacker_id']==0 and t['attack_method']==attack and t['fl_aggregate']==method)
        c.validate_result(task,artifact)
        artifact['audit']['public_warm_start_rounds']=[9]
        with pytest.raises(ValueError):c.validate_result(task,artifact)
    finally:torch.set_num_threads(previous)

def test_all_commands_parse_and_validate(monkeypatch,tmp_path):
    c=campaign();tasks=c.build_tasks();assert len(tasks)==108
    parser=actual_parser(monkeypatch);helper=api(monkeypatch)
    for task in tasks:
        argv=c.command_for(task,'python',ROOT,tmp_path/task['task_id'])
        args=parser.parse_args(argv[2:]);helper.validate_config(args)
        assert args.attack_method in ['adam_local_deviation','dkfr'] and args.attacker_ids is None

def test_dataset_merge_allows_different_public_partition_between_attacks(tmp_path):
    c=campaign();tasks=c.build_tasks(['adult'])
    for task in tasks:
        data=fixture_result(task);warm=task['attack_method']=='adam_local_deviation'
        audit=data['audit'];audit['public_warm_start_rounds']=[0] if warm else []
        audit['fake_call_rounds']=audit['fake_rounds']=list(range(int(warm),50));audit['fake_sources']=['fake']*(50-int(warm))
        if warm:
            audit['round_activity'][0]['active_attackers']=[]
            data['fingerprints']['shards'][0]['x_sha256']='public-pool-held-out'
        directory=tmp_path/'tasks'/task['task_id'];path=c.result_path(task,directory);path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data));(directory/'status.json').write_text('{"returncode":0}')
    assert c.merge_results(tasks,tmp_path)==dict(complete=36,failed=0)
    import csv
    with (tmp_path/'merged/adult/rq1_detail.csv').open() as f:rows=list(csv.DictReader(f))
    assert all(int(r['fake_count'])+int(r['public_warm_start_count'])==50 and r['honest_count']=='0' for r in rows)
    path=c.result_path(tasks[0],tmp_path/'tasks'/tasks[0]['task_id']);data=json.loads(path.read_text());data['fingerprints']['test_x_sha256']='mismatch';path.write_text(json.dumps(data))
    assert c.merge_results(tasks,tmp_path)==dict(complete=18,failed=18)

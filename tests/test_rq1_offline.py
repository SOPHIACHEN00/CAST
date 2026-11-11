import ast
import importlib.util
from pathlib import Path
import json,csv
import pytest
import torch
from test_rq1_single_attacker import arguments,actual_parser,api,loader,ScoreModel
from test_multi_freerider import training_namespace,ToyAttack,ToyModel
from test_rq1_fl_aggregation_campaign import fixture_result

ROOT=Path(__file__).resolve().parents[1]

def campaign():
    spec=importlib.util.spec_from_file_location('offline_campaign',ROOT/'script/rq1_offline_campaign.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c

@pytest.mark.parametrize('dataset',['adult','mnist','imdb'])
@pytest.mark.parametrize('family',['arima','moirai'])
@pytest.mark.parametrize('method',['fedavg','fedadam','weighted_median'])
def test_offline_single_training_and_evaluation(monkeypatch,tmp_path,dataset,family,method):
    previous=torch.get_num_threads()
    try:
        c=campaign();task=next(t for t in c.build_tasks([dataset]) if t['attacker_id']==0 and t['attack_family']==family and t['fl_aggregate']==method)
        args=actual_parser(monkeypatch).parse_args(c.command_for(task,'python',ROOT,tmp_path)[2:])
        helper=api(monkeypatch);data=loader();model=ScoreModel();attacker=ToyAttack()
        context=helper.prepare_run(args=args,loader=data,model=model)
        ns=training_namespace();ns['args']=args
        tree=ast.parse((ROOT/'FLCE_updated.py').read_text(encoding='utf-8'))
        ns['FORECAST_FREE_RIDER_METHODS']=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORECAST_FREE_RIDER_METHODS' for t in n.targets)))
        agg={};ToyModel.trained_clients=[]
        result=ns['run_federated_learning'](data,model,0,attacker,True,50,1,torch.device('cpu'),last_round_aggregation_out=agg,single_attacker_audit_out=context['audit'])
        assert ToyModel.trained_clients.count(0)==0
        artifact=helper.finish_run(args=args,loader=data,model=model,attacker=attacker,run_context=context,last_round_state=result[3],last_round_client_grads=result[4],last_round_weights=result[5],aggregation_context=agg,accuracy=result[2],attack_times=result[1],training_seconds=.1,provenance={})
        assert artifact['audit']['fake_call_rounds']==list(range(50)) and artifact['audit']['public_warm_start_rounds']==[]
        c.validate_result(task,artifact)
    finally:torch.set_num_threads(previous)

def test_full_matrix_routes_offline_constructors(monkeypatch,tmp_path):
    c=campaign();tasks=c.build_tasks();assert len(tasks)==108
    parser=actual_parser(monkeypatch);helper=api(monkeypatch)
    tree=ast.parse((ROOT/'FLCE_updated.py').read_text(encoding='utf-8'))
    registry=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ATTACK_IMPLEMENTATIONS' for t in n.targets))
    mapping={ast.literal_eval(k):v.id for k,v in zip(registry.keys,registry.values)}
    for task in tasks:
        argv=c.command_for(task,'python',ROOT,tmp_path/task['task_id']);args=parser.parse_args(argv[2:]);helper.validate_config(args)
        assert 'Offline' in mapping[args.attack_method]
        assert '--moirai_online_tuning' not in argv and '--arima_retrain_interval' not in argv
        assert not args.rq1_strict_data_free and args.attacker_ids is None and not args.use_lazy

def test_offline_dataset_csv(tmp_path):
    c=campaign();tasks=c.build_tasks(['adult'])
    for task in tasks:
        data=fixture_result(task);data['audit']['public_warm_start_rounds']=[]
        directory=tmp_path/'tasks'/task['task_id'];path=c.result_path(task,directory)
        path.parent.mkdir(parents=True);path.write_text(json.dumps(data));(directory/'status.json').write_text('{"returncode":0}')
    assert c.merge_results(tasks,tmp_path)==dict(complete=36,failed=0)
    with (tmp_path/'merged/adult/rq1_detail.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==36 and all(r['fake_count']=='50' for r in rows)

import ast
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from test_rq1_single_attacker import arguments,actual_parser,api,loader,ScoreModel
from test_multi_freerider import training_namespace,ToyAttack,ToyModel

ROOT=Path(__file__).resolve().parents[1]

def campaign():
    spec=importlib.util.spec_from_file_location('strict_campaign',ROOT/'script/rq1_strict_pure_campaign.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c

@pytest.mark.parametrize('attack',['advanced_scaled_delta','dkfr'])
@pytest.mark.parametrize('method',['fedavg','fedadam','weighted_median'])
def test_strict_actual_loop(monkeypatch,tmp_path,attack,method):
    previous=torch.get_num_threads()
    try:
        args=arguments(monkeypatch,tmp_path,method,0);args.attack_method=attack;args.rq1_strict_data_free=True
        helper=api(monkeypatch);data=loader();model=ScoreModel();attacker=ToyAttack()
        context=helper.prepare_run(args=args,loader=data,model=model)
        ns=training_namespace();ns['args']=args;agg={};ToyModel.trained_clients=[]
        result=ns['run_federated_learning'](data,model,0,attacker,True,50,1,torch.device('cpu'),last_round_aggregation_out=agg,single_attacker_audit_out=context['audit'])
        assert ToyModel.trained_clients.count(0)==0
        artifact=helper.finish_run(args=args,loader=data,model=model,attacker=attacker,run_context=context,last_round_state=result[3],last_round_client_grads=result[4],last_round_weights=result[5],aggregation_context=agg,accuracy=result[2],attack_times=result[1],training_seconds=.1,provenance={})
        assert artifact['audit']['fake_call_rounds']==list(range(50))
        assert artifact['audit']['public_warm_start_rounds']==[]
        assert artifact['metadata']['rq1_strict_data_free'] is True
        c=campaign();task=next(t for t in c.build_tasks(['adult']) if t['attacker_id']==0 and t['attack_method']==attack and t['fl_aggregate']==method)
        c.validate_result(task,artifact)
        artifact['metadata']['rq1_strict_data_free']=False
        with pytest.raises(ValueError):c.validate_result(task,artifact)
    finally:torch.set_num_threads(previous)

@pytest.mark.parametrize('dataset,dim,classes',[('adult',105,2),('mnist',784,10),('imdb',768,2)])
def test_dkfr_initialization_never_reads_training_data(dataset,dim,classes):
    tree=ast.parse((ROOT/'FLCE_updated.py').read_text(encoding='utf-8'))
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="attack_method == 'dkfr'" and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='attacker' for t in x.targets) for x in n.body))
    # Run actual parameter-preparation statements, stopping before the constructor.
    nodes=branch.body[:next(i for i,n in enumerate(branch.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='attacker' for t in n.targets))]
    class ForbiddenLoader:
        def __getattribute__(self,name):raise AssertionError('Private data access: '+name)
    from types import SimpleNamespace
    ns=dict(torch=torch,args=SimpleNamespace(rq1_strict_data_free=True),dataset=dataset,model=torch.nn.Sequential(torch.nn.Linear(dim,8),torch.nn.Linear(8,1 if classes==2 else classes)),loader=ForbiddenLoader())
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual DKFR initializer>','exec'),ns)
    assert ns['input_dim']==dim and ns['num_classes']==classes
    assert all(ns[k] is None for k in ['feature_min','feature_max','feature_mean','feature_std'])

def test_matrix_and_cli(monkeypatch,tmp_path):
    c=campaign();tasks=c.build_tasks();assert len(tasks)==108
    parser=actual_parser(monkeypatch);helper=api(monkeypatch)
    for task in tasks:
        argv=c.command_for(task,'python',ROOT,tmp_path/task['task_id']);args=parser.parse_args(argv[2:]);helper.validate_config(args)
        assert args.rq1_strict_data_free and args.attacker_ids is None and not args.use_lazy
    args.attack_method='adam_local_deviation'
    with pytest.raises(ValueError):helper.validate_config(args)

def test_strict_dataset_csv_and_pairing(tmp_path):
    import csv
    from test_rq1_fl_aggregation_campaign import fixture_result
    c=campaign();tasks=c.build_tasks(['adult'])
    for task in tasks:
        data=fixture_result(task);data['metadata']['rq1_strict_data_free']=True
        data['audit']['public_warm_start_rounds']=[]
        directory=tmp_path/'tasks'/task['task_id'];path=c.result_path(task,directory)
        path.parent.mkdir(parents=True);path.write_text(json.dumps(data))
        (directory/'status.json').write_text('{"returncode":0}')
    assert c.merge_results(tasks,tmp_path)==dict(complete=36,failed=0)
    with (tmp_path/'merged/adult/rq1_detail.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==36
    assert all(r['rq1_strict_data_free']=='True' and r['fake_count']=='50' and r['public_warm_start_count']=='0' for r in rows)
    path=c.result_path(tasks[0],tmp_path/'tasks'/tasks[0]['task_id'])
    data=json.loads(path.read_text());data['fingerprints']['test_x_sha256']='unpaired';path.write_text(json.dumps(data))
    assert c.merge_results(tasks,tmp_path)==dict(complete=0,failed=36)

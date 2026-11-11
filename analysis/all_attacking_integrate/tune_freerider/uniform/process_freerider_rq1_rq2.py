"""Build four Uniform tables with explicit per-client plotting metrics.

RQ1: mean attacker SV / total SV within each run, without multiplying by 100.
RQ2: mean absolute matched SV change across the 8-k honest clients in each run.
Summaries average the three assignment groups and use sample std (ddof=1).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARTITION_ALPHAS = {
    'uniform': {'adult': 20000000, 'mnist': 10000000, 'imdb': 200000},
}
KEYS = ['Dataset', 'AttackMethod', 'Alpha', 'ClientNum', 'AttackerCount',
        'ContributionMethod', 'AttackMode', 'Seed', 'AttackerSelectionSeed']
RQ1_PRIMARY = {'PrimaryMetric': 'Q1_AttackerMeanRatio',
               'PrimaryMetricUnit': 'contribution_share'}
RQ2_PRIMARY = {'PrimaryMetric': 'Q2_AvgAbsImpact',
               'PrimaryMetricUnit': 'raw_sv'}
Q1_METRICS = ['Q1_AttackerMeanRatio', 'Q1_AttackerRatio', 'AttackerMeanContribution',
              'AttackerContribution', 'TotalContribution']
Q2_METRICS = ['Q2_AvgAbsImpact', 'Q2_TotalAbsImpact', 'Q2_AvgSignedDelta',
              'Q2_AvgSignedHonestLoss', 'Q2_TotalSignedHonestLoss',
              'Q2_TotalPositiveHonestLoss', 'Q2_AvgPositiveHonestLoss',
              'Q2_HonestLossShare', 'Q2_HarmedClientFraction', 'Q2_HelpedClientFraction']


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', choices=tuple(PARTITION_ALPHAS), default='uniform',
                        help='This processor reads/writes its own Uniform directory')
    args = parser.parse_args()
    data_root = HERE
    for relative, expected in json.loads((data_root / 'sha256.json').read_text()).items():
        require(hashlib.sha256((data_root / relative).read_bytes()).hexdigest() == expected,
                f'Raw file differs from verified download: {relative}')
    path = ROOT / 'Analysis Code/Q1/compute_q1_summary.py'
    spec = importlib.util.spec_from_file_location('established_q1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q2_path = ROOT / 'Analysis Code/Q2/compute_q2_honest_impact_summary.py'
    q2_spec = importlib.util.spec_from_file_location('established_q2', q2_path)
    q2_module = importlib.util.module_from_spec(q2_spec)
    q2_spec.loader.exec_module(q2_module)
    q1, q2 = [], []
    control_usage = {}
    for dataset, alpha in PARTITION_ALPHAS[args.partition].items():
        source = pd.read_csv(data_root / dataset / 'contributions.csv')
        controls = pd.read_csv(data_root / 'controls' / dataset / 'control_contributions.csv')
        audit = pd.read_csv(data_root / 'controls' / dataset / 'control_audit.csv')
        require(len(controls) == 54 and len(audit) == 9 and audit['task_id'].is_unique
                and audit['status'].eq('complete').all(), f'{dataset}: controls incomplete')
        control_map = {}
        for control_id, cb in controls.groupby('task_id'):
            for col in ['dataset','alpha','trial_id','attacker_count','excluded_ids','seed',
                        'original_client_num','control_client_num','attacker_selection_seed']:
                require(cb[col].nunique(dropna=False) == 1, f'{control_id}: inconsistent {col}')
            c = cb.iloc[0]
            excluded = tuple(sorted(json.loads(c['excluded_ids'])))
            require(c['dataset'] == dataset and c['alpha'] == alpha and c['seed'] == 42
                    and c['original_client_num'] == 8 and c['attacker_selection_seed'] == 2026,
                    f'{control_id}: baseline configuration mismatch')
            require(len(excluded) == len(set(excluded)) == int(c['attacker_count'])
                    and len(cb) == int(c['control_client_num']) == 8-len(excluded)
                    and set(cb['client_id']) == set(range(8))-set(excluded), f'{control_id}: IDs mismatch')
            require(all(math.isfinite(v) for v in cb['shapley_value']), f'{control_id}: nonfinite SV')
            key = (int(c['trial_id']), len(excluded), excluded)
            require(key not in control_map, f'{dataset}: duplicate baseline key')
            control_map[key] = cb
            control_usage[control_id] = 0
        require(set(controls['task_id']) == set(audit['task_id']), f'{dataset}: audit IDs mismatch')
        require(len(source) == 288, f'{dataset}: expected 36 tasks x 8 clients')
        require(source['dataset'].eq(dataset).all() and source['alpha'].eq(alpha).all(), 'Dataset/alpha mismatch')
        require(source['detecting_method'].eq('shapley').all(), 'Only Shapley scores are applicable')
        for task, block in source.groupby('task_id', sort=False):
            first = block.iloc[0]
            ids = json.loads(first['attacker_ids'])
            k = int(first['attacker_count'])
            require(len(block) == 8 and set(block['client_id']) == set(range(8)), f'{task}: client coverage')
            require(len(set(ids)) == k and set(block.loc[block['is_attacker'].eq(1), 'client_id']) == set(ids), f'{task}: attacker labels')
            require(block['seed'].eq(42).all(), f'{task}: training seed mismatch')
            require(block['scope'].eq('actual_last_round_submissions').all(), f'{task}: SV scope mismatch')
            values = block['shapley_value'].astype(float)
            require(all(math.isfinite(v) for v in values), f'{task}: nonfinite SV')
            total = math.fsum(values)
            attacker_sum = math.fsum(block.loc[block['is_attacker'].eq(1), 'shapley_value'])
            ratios = []
            for attacker in ids:
                # Apply the single-attacker definition using this run's total SV.
                # Their mean is the primary metric; their sum remains auxiliary.
                frame = pd.DataFrame({'ClientID': block['client_id'],
                                      'AttackerID': attacker, 'WithAttacker': values})
                ratio = module.compute_attacker_ratio(frame)
                if ratio is not None:
                    ratios.append(float(ratio))
            valid = len(ratios) == k and all(math.isfinite(v) for v in ratios)
            if valid:
                require(math.isclose(math.fsum(ratios)/k, attacker_sum/(k*total),
                                     rel_tol=1e-10, abs_tol=1e-12),
                        f'{task}: per-attacker RQ1 formula mismatch')
            meta = dict(TaskID=task, Dataset=dataset, AttackMethod=first['attack_method'],
                        Alpha=alpha, ClientNum=8, AttackerCount=k,
                        AttackerIDs=json.dumps(ids, separators=(',', ':')),
                        TrialID=int(first['trial_id']), Seed=42,
                        AttackerSelectionSeed=int(first['attacker_selection_seed']),
                        ContributionMethod='shapley', UseHessian=True, LambdaH=0.9,
                        UseLazy=first['regime'] == 'lazy',
                        AttackMode='1real_5fake' if first['regime'] == 'lazy' else 'pure_fake',
                        ShapleyRound=49, SourceFile=f'{dataset}/contributions.csv')
            q1.append(dict(meta, **RQ1_PRIMARY,
                           Q1_AttackerMeanRatio=math.fsum(ratios)/k if valid else float('nan'),
                           Q1_AttackerRatio=math.fsum(ratios) if valid else float('nan'),
                           AttackerMeanContribution=attacker_sum/k,
                           AttackerContribution=attacker_sum, TotalContribution=total,
                           Status='complete' if valid else 'undefined_total_sv'))
            key = (int(first['trial_id']), k, tuple(sorted(ids)))
            require(key in control_map, f'{task}: no exact matched baseline')
            cb = control_map[key]
            require(int(first['attacker_selection_seed']) == 2026, f'{task}: selection seed mismatch')
            honest = block.loc[block['is_attacker'].eq(0), ['client_id','shapley_value','shard_size']]
            paired = honest.merge(cb[['client_id','shapley_value','shard_size']], on='client_id',
                                  how='inner', validate='one_to_one', suffixes=('_with','_without'))
            require(len(paired) == 8-k and paired['shard_size_with'].eq(paired['shard_size_without']).all(),
                    f'{task}: honest IDs or shard sizes differ')
            # Average across honest clients (8-k), without dividing by k or total SV.
            metrics = q2_module.compute_q2_metrics(paired.rename(columns={
                'shapley_value_with':'WithAttacker','shapley_value_without':'WithoutAttacker'}))
            require(metrics is not None and metrics['Q2_HonestClientCount'] == 8-k, f'{task}: incomplete RQ2')
            direct = math.fsum(abs(a-b) for a,b in zip(paired['shapley_value_with'],paired['shapley_value_without']))/(8-k)
            require(math.isclose(metrics['Q2_AvgAbsImpact'], direct, rel_tol=1e-10, abs_tol=1e-12),
                    f'{task}: RQ2 formula mismatch')
            control_id = cb.iloc[0]['task_id']
            control_usage[control_id] += 1
            q2.append(dict(meta, **RQ2_PRIMARY, ExpectedHonestClientCount=8-k, ControlTaskID=control_id,
                           ControlSourceFile=f'controls/{dataset}/control_contributions.csv',
                           Status='complete', **metrics))
    detail1, detail2 = pd.DataFrame(q1), pd.DataFrame(q2)
    require(len(detail1) == 108 and detail1['TaskID'].is_unique, 'Expected 108 distinct Shapley tasks')
    require(len(control_usage) == 27 and set(control_usage.values()) == {4}, 'Each control must match four attack tasks')
    groups1, groups2 = [], []
    for key, block in detail1.groupby(KEYS, dropna=False, sort=True):
        require(len(block) == 3 and set(block['TrialID']) == {0, 1, 2}, 'Incomplete assignment groups')
        meta = dict(zip(KEYS, key))
        row = dict(meta, **RQ1_PRIMARY, TaskCount=3, ExpectedTaskCount=3,
                   ValidTaskCount=int(block['Status'].eq('complete').sum()),
                   Status='complete' if block['Status'].eq('complete').all() else 'partial',
                   StdDDoF=1)
        for metric in Q1_METRICS:
            row[metric] = block[metric].mean()
            row[metric+'_Std'] = block[metric].std(ddof=1)
            row[metric+'_N'] = block[metric].count()
        groups1.append(row)
    for key, block in detail2.groupby(KEYS, dropna=False, sort=True):
        require(len(block) == 3 and set(block['TrialID']) == {0,1,2}, 'Incomplete RQ2 groups')
        meta = dict(zip(KEYS, key))
        summary = dict(meta, **RQ2_PRIMARY, TaskCount=3, ExpectedTaskCount=3, ValidTaskCount=3,
                       ExpectedHonestClientCount=8-int(meta['AttackerCount']),
                       Status='complete', StdDDoF=1)
        for metric in ['Q2_HonestClientCount'] + Q2_METRICS:
            summary[metric] = block[metric].mean()
            summary[metric+'_Std'] = block[metric].std(ddof=1)
            summary[metric+'_N'] = block[metric].count()
        groups2.append(summary)
    tables = {'freerider_RQ1_attacker_ratio_detail.csv': detail1,
              'freerider_RQ1_attacker_ratio_group_mean.csv': pd.DataFrame(groups1),
              'freerider_RQ2_honest_impact_detail.csv': detail2,
              'freerider_RQ2_honest_impact_group_mean.csv': pd.DataFrame(groups2)}
    existing = [data_root / name for name in tables if (data_root / name).exists()]
    if existing:
        backup = data_root / 'backups' / ('rq12_tables_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        backup.mkdir(parents=True)
        for target in existing:
            shutil.copy2(target, backup / target.name)
    for name, table in tables.items():
        target = data_root / name
        temporary = target.with_suffix('.csv.tmp')
        table.to_csv(temporary, index=False, float_format='%.15g')
        temporary.replace(target)
        print(f'{name}: {len(table)} rows')
    print('RQ1/RQ2 ready: 27 controls matched to 108 attacked tasks; 36 configurations, 3 groups each.')
    print('Plot RQ1_AttackerMeanRatio (contribution share, no percent conversion) and Q2_AvgAbsImpact (raw SV).')


if __name__ == '__main__':
    main()

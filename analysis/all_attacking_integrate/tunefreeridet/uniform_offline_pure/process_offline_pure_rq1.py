"""Derive RQ1 from the 54 completed pure runs; never modify raw dataset CSVs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import shutil
import pandas as pd

HERE = Path(__file__).resolve().parent
FORMULA = HERE.parents[2] / 'Analysis Code/Q1/compute_q1_summary.py'
KEYS = ['Dataset', 'AttackMethod', 'Alpha', 'ClientNum', 'AttackerCount',
        'ContributionMethod', 'AttackMode', 'Seed', 'AttackerSelectionSeed']
METRICS = ['Q1_AttackerMeanRatio', 'Q1_AttackerRatio', 'AttackerMeanContribution',
           'AttackerContribution', 'TotalContribution']
PRIMARY = dict(PrimaryMetric='Q1_AttackerMeanRatio', PrimaryMetricUnit='contribution_share')


def main():
    spec = importlib.util.spec_from_file_location('existing_q1_formula', FORMULA)
    established = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(established)
    raw_files = [p for d in ('adult', 'mnist', 'imdb') for p in (HERE / d).glob('*.csv')]
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_files}
    detail = []
    for dataset, alpha in [('adult', 20000000), ('mnist', 10000000), ('imdb', 200000)]:
        source = pd.read_csv(HERE / dataset / 'contributions.csv')
        audit = pd.read_csv(HERE / dataset / 'audit.csv')
        assert len(source) == 144 and len(audit) == 18
        assert audit['status'].eq('complete').all()
        assert set(source['task_id']) == set(audit['task_id'])
        suffix = '' if dataset == 'adult' else '_mnist_imdb_layerwise'
        assert set(source['attack_method']) == {'arima' + suffix, 'moirai' + suffix}
        for task_id, block in source.groupby('task_id', sort=True):
            first = block.iloc[0]
            for col, expected in dict(dataset=dataset, alpha=alpha, client_num=8, regime='pure',
                                      detecting_method='shapley', seed=42, attacker_selection_seed=2026,
                                      fake_rounds_per_attacker=50, real_rounds_per_attacker=0,
                                      scope='actual_last_round_submissions').items():
                assert block[col].eq(expected).all(), (task_id, col)
            ids = json.loads(first['attacker_ids'])
            k = int(first['attacker_count'])
            assert len(block) == 8 and set(block['client_id']) == set(range(8))
            assert len(set(ids)) == len(ids) == k
            assert set(block.loc[block['is_attacker'].eq(1), 'client_id']) == set(ids)
            values = block['shapley_value'].astype(float)
            assert all(math.isfinite(v) for v in values)
            total = math.fsum(values)
            attacker_sum = math.fsum(block.loc[block['is_attacker'].eq(1), 'shapley_value'])
            ratios = [established.compute_attacker_ratio(pd.DataFrame({
                'ClientID': block['client_id'], 'AttackerID': attacker, 'WithAttacker': values
            })) for attacker in ids]
            valid = all(v is not None and math.isfinite(v) for v in ratios)
            mean_ratio = math.fsum(ratios) / k if valid else float('nan')
            if valid:
                assert math.isclose(mean_ratio, attacker_sum / (k * total), rel_tol=1e-10, abs_tol=1e-12)
            detail.append(dict(TaskID=task_id, Dataset=dataset, AttackMethod=first['attack_method'],
                Alpha=alpha, ClientNum=8, AttackerCount=k, AttackerIDs=json.dumps(ids, separators=(',', ':')),
                TrialID=int(first['trial_id']), Seed=42, AttackerSelectionSeed=2026,
                ContributionMethod='shapley', UseHessian=True, LambdaH=0.9, UseLazy=False,
                AttackMode='pure_fake', ShapleyRound=49, SourceFile=f'{dataset}/contributions.csv',
                **PRIMARY, Q1_AttackerMeanRatio=mean_ratio,
                Q1_AttackerRatio=math.fsum(ratios) if valid else float('nan'),
                AttackerMeanContribution=attacker_sum / k, AttackerContribution=attacker_sum,
                TotalContribution=total, Status='complete' if valid else 'undefined_total_sv'))
    detail = pd.DataFrame(detail)
    assert len(detail) == 54 and detail['TaskID'].is_unique
    grouped = []
    for key, block in detail.groupby(KEYS, sort=True, dropna=False):
        assert len(block) == 3 and set(block['TrialID']) == {0, 1, 2}
        row = dict(zip(KEYS, key))
        row.update(PRIMARY, TaskCount=3, ExpectedTaskCount=3,
                   ValidTaskCount=int(block['Status'].eq('complete').sum()),
                   Status='complete' if block['Status'].eq('complete').all() else 'partial', StdDDoF=1)
        for metric in METRICS:
            row[metric] = block[metric].mean()
            row[metric + '_Std'] = block[metric].std(ddof=1)
            row[metric + '_N'] = block[metric].count()
        grouped.append(row)
    grouped = pd.DataFrame(grouped)
    assert len(grouped) == 18
    outputs = {'freerider_RQ1_attacker_ratio_detail.csv': detail,
               'freerider_RQ1_attacker_ratio_group_mean.csv': grouped}
    existing = [HERE / n for n in (*outputs, 'rq1_processing_manifest.json') if (HERE / n).exists()]
    if existing:
        backup = HERE / 'backups' / ('rq1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        backup.mkdir(parents=True)
        for p in existing:
            shutil.copy2(p, backup / p.name)
    for name, frame in outputs.items():
        frame.to_csv(HERE / name, index=False, encoding='utf-8-sig')
    for relative, expected in hashes.items():
        assert hashlib.sha256((HERE / relative).read_bytes()).hexdigest() == expected
    manifest = dict(formula_source=str(FORMULA), formula_sha256=hashlib.sha256(FORMULA.read_bytes()).hexdigest(),
        input_sha256=hashes, detail_rows=54, group_rows=18,
        valid_rq1=int(detail['Status'].eq('complete').sum()),
        primary_formula='mean(attacker raw SV) / sum(all 8 raw SV)',
        aggregation='compute each trial ratio first; mean and sample std ddof=1 across trial_ids 0,1,2',
        percent_conversion=False, clipping=False, zero_denominator='missing')
    (HERE / 'rq1_processing_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('detail_rows', 'group_rows', 'valid_rq1')}))
    print('All original CSV hashes unchanged.')


if __name__ == '__main__':
    main()

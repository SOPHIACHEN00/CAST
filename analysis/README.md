# RQ1 / RQ2 postprocessing

This directory contains shared metric definitions and experiment adapters from the current analysis project. The relative directory structure is preserved because adapters load metric functions from `Analysis Code/` by file path.

## Combined contribution CSVs

Inputs are existing `combined_contributions_trial*.csv` files with `ClientID`, `AttackerID`, `WithAttacker`, `WithoutAttacker`, and the experiment configuration columns required by each script.

Run from the repository root:

```bash
python "analysis/Analysis Code/Q1/compute_q1_summary.py" combined_contributions_trial0.csv -o q1_attacker_ratio_summary.csv
python "analysis/Analysis Code/Q2/compute_q2_honest_impact_summary.py" combined_contributions_trial0.csv -o q2_honest_impact_summary.csv
```

- `Q1/compute_q1_summary.py`: the attacker's signed Shapley value divided by the sum across all clients.
- `Q1/compute_q1_positive_credit_summary.py`: also reports the positive-credit share after clipping negative contributions to zero.
- `Q2/compute_q2_honest_impact_summary.py`: absolute changes, signed losses, and positive losses for matched honest clients.
- `Q2/compute_q2_summary.py`: the original mean absolute-change summary.
- `q1_*group_mean.py`, `q2_*group_mean.py`, `q1_avg_steal_summary.py`, and `q2_avg_impact_summary.py`: grouped summaries. Some older entry points read default CSV filenames from the current working directory.
- `Q2/compute_q2_honest_client_boxplot.py`: produces metrics at the honest-client level.

Metric definitions and column names are preserved. Signed contribution ratios, positive-credit shares, absolute impact, and directional losses represent different quantities and should be interpreted separately.

## Multiple attackers with Uniform partitions

Entry point: `all_attacking_integrate/tune_freerider/uniform/process_freerider_rq1_rq2.py`.

Place the corresponding experiment exports beside the entry point using this layout:

```text
uniform/
  process_freerider_rq1_rq2.py
  sha256.json
  adult/contributions.csv
  mnist/contributions.csv
  imdb/contributions.csv
  controls/
    adult/{control_contributions.csv,control_audit.csv}
    mnist/{control_contributions.csv,control_audit.csv}
    imdb/{control_contributions.csv,control_audit.csv}
```

`sha256.json` is the verified raw-download manifest: keys are paths relative to this directory and values are SHA-256 hashes. Supply the manifest that matches the actual inputs. The source repository does not bundle experiment data.

```bash
python analysis/all_attacking_integrate/tune_freerider/uniform/process_freerider_rq1_rq2.py --partition uniform
```

This adapter validates the established complete experiment design: 36 attack tasks and 9 honest controls per dataset, N=8, 1/2/3 attackers, and three attacker-assignment groups. The primary RQ1 metric is **mean contribution per attacker divided by total contribution**; total attacker share is retained separately. RQ2 measures the mean absolute Shapley change across matched honest clients. Group-level sample standard deviations use `ddof=1`. The adapter expects this specific task matrix.

## Other experiment adapters

| File | Purpose |
| --- | --- |
| `all_attacking_integrate/tunefreeridet/uniform_offline_pure/process_offline_pure_rq1.py` | Offline pure multi-attacker RQ1; preserves the existing directory name and completeness checks for 54 tasks |
| `all_attacking_integrate/tune_lambda/process_lambda_rq1_rq2.py` | LambdaH sensitivity; retains LambdaH in grouping keys |
| `all_attacking_integrate/finetuneinterval/process_interval_rq1_rq2.py` | Adaptation-interval sensitivity; retains the tuning parameter and interval |

Each adapter expects the original experiment directories and filenames beside its entry point. Supply the corresponding raw CSVs first. For the newer single-attacker experiments comparing three aggregation methods, `script/rq1_*_campaign.py` already converts result JSON into `rq1_detail`, `rq1_mean_std`, and related CSVs through `--merge-only CAMPAIGN_DIRECTORY`.

For analysis alone, install `requirements-analysis.txt`. Generated plots, raw results, and server-connection scripts are not bundled here.

# Recomputing metrics from archived experiment exports

This directory contains shared metric definitions and experiment adapters from the current analysis project. The relative directory structure is preserved because adapters load metric functions from `Analysis Code/` by file path.

The historical `Q2` directory measures changes in honest-client credit, which appear in Figure 2 and the paper's RQ5. The paper's RQ2 concerns detector Recall and FPR.

## Combined contribution CSVs

Inputs are existing `combined_contributions_trial*.csv` files with `ClientID`, `AttackerID`, `WithAttacker`, `WithoutAttacker`, and the experiment configuration columns required by each script.

Run from the repository root:

```bash
python "analysis/Analysis Code/Q1/compute_q1_summary.py" combined_contributions_trial0.csv -o q1_attacker_ratio_summary.csv
python "analysis/Analysis Code/Q2/compute_q2_honest_impact_summary.py" combined_contributions_trial0.csv -o q2_honest_impact_summary.csv
python "analysis/Analysis Code/Q2/compute_q2_honest_client_boxplot.py" combined_contributions_trial0.csv -o q2_honest_clients.csv
```

- `Q1/compute_q1_summary.py`: the attacker's signed Shapley value divided by the sum across all clients.
- `Q1/compute_q1_positive_credit_summary.py`: also reports the positive-credit share after clipping negative contributions to zero.
- `Q2/compute_q2_honest_impact_summary.py`: absolute changes, signed losses, and positive losses for matched honest clients.
- `Q2/compute_q2_summary.py`: the original mean absolute-change summary.
- `q1_*group_mean.py`, `q2_*group_mean.py`, `q1_avg_steal_summary.py`, and `q2_avg_impact_summary.py`: grouped summaries. Some older entry points read default CSV filenames from the current working directory.
- `Q2/compute_q2_honest_client_boxplot.py`: produces metrics at the honest-client level.

Metric definitions and column names are preserved. Signed contribution ratios, positive-credit shares, absolute impact, and directional losses represent different quantities and should be interpreted separately.

The three commands above accept either the historical `ContributionMethod` column or the current `DetectingMethod`. Keep the partition seed, server aggregation and other run settings with each export. Some older CSV schemas omit those fields.

The default exact Shapley utility replays the final round's submitted updates, fixes the preceding server state and assigns utility zero to the empty coalition. Raw Shapley values sum to the grand-coalition accuracy for this game. Dividing by their nonzero signed sum gives shares that sum to one and can include negatives. Estimating the excess credit of an intermittently participating client additionally requires its genuine-participation counterfactual on a common scale. The historical share export contains the awarded credit only. `WithoutAttacker` contains scores for the surviving clients.

## Matched excess contribution

To measure excess contribution in a separate experiment, match the awarded run to two final-accuracy measurements. In one control the designated client trains only on its recorded honest rounds. In the other the same client is excluded throughout. Use the same original partition, initial state, participation schedule, training protocol and evaluation protocol. The raw gap equals the awarded Shapley value minus the difference between the two control accuracies.

```bash
python analysis/counterfactual_gap.py runs.csv --controls controls.csv --signed-share -o gaps.csv
```

The [input schema](counterfactual_gap.py) lists the CSV columns, SHA-256 fields and required evidence of private-data access. The optional signed share divides both terms by the awarded run's total signed credit. With a zero total it leaves the share empty. Historical Figure 1 inputs provide the awarded share alone.

## Figures 1–4

From an archived export of the five datasets, rebuild the two combined inputs for Figures 1 and 2 with the command below. `--archive-root` contains the dataset summaries and the separately measured pure-attack baseline exports. `--selected-dir` contains the five dataset subdirectories and `forecast_source_selection_manifest.csv`. The command reads their relative layout and writes the two combined CSVs. The synthetic reference for Figure 1 is a separate input.

```bash
python analysis/prepare_credit_exports.py --archive-root EXPORT_ROOT --selected-dir EXPORT_ROOT/rq1_rq2_modified_forecast_20260818/derived_with_dota2_gap6 --output-dir rebuilt
```

The input filenames below are the archived **analysis exports**. Figure 1 also uses a separately measured synthetic reference. Supply the files from the relevant export directory or pass their paths. Install the plotting dependencies with `python -m pip install -r requirements-figures.txt`.

```bash
python analysis/plot_credit_figures.py 1 combined_RQ1_plot_data.csv --synthetic-reference synthetic_baseline_RQ1.csv --paired-csv figure1_pairs.csv -o figure1.pdf
python analysis/plot_credit_figures.py 2 combined_RQ2_box_plot_data.csv -o figure2.pdf
python analysis/plot_credit_figures.py 3 RQ1_four_methods_group_mean.csv -o figure3.pdf
python analysis/plot_credit_figures.py 4 lambda_RQ1_attacker_ratio_group_mean.csv -o figure4.pdf
```

| Figure | Observation and scope |
| --- | --- |
| 1 | Signed credit share. The two displayed sweeps intersect at six clients and the first partition. The optional paired CSV counts that setting once per dataset and mode. The archived Dota2 third concentration is 50,000,000. The Syn reference was measured separately under the non-lazy schedule. |
| 2 | One observation per matched honest client in trial 0 with designated client 1. Boxes summarize `abs(WithAttacker − WithoutAttacker)` in raw Shapley units. |
| 3 | Mean signed credit per free-rider divided by total credit. The archived sweep used uniform partitions, eight clients and three assignment groups. |
| 4 | Signed credit share by the configured mixture weight for the historical correction proxy. The displayed panels use six clients. |

## Passive detection

Pass the root of the archived `Organized_Tidy_Passive_Detection_Results` export or the matching complete client-round CSVs. Each table cell averages setting-level rates equally over four partitions and three client counts (4, 6, 8). The archived panel has trial 0 and designated client 1. The command checks round and client coverage. Recall uses evaluated submissions on active free-riding rounds. FPR uses evaluated submissions of never-designated clients. A designated client's honest rounds and decisions marked unevaluated are outside those denominators.

```bash
python analysis/paper_detection.py EXPORT_ROOT --panel dagmm -o table_ii.csv
python analysis/paper_detection.py EXPORT_ROOT --panel patches -o table_iv.csv
python analysis/paper_detection.py EXPORT_ROOT --panel pass -o pass_rates.csv
```

Outputs include the number of evaluated observations, the decision rule and setting coverage. The patches panel also includes the recorded source of the auxiliary reports. STD-DAGMM flags energy strictly above the round median. Delta-DAGMM uses its per-round energy rule for this table and supports a separate final client decision. PASS uses the contribution audit on local validation performance. WEF, S²-WEF and the peer log detector in these exports observed a passive training trajectory. [STD-DAGMM](https://arxiv.org/abs/1911.12560) · [Delta-DAGMM](https://doi.org/10.1155/2022/8928790) · [PASS](https://doi.org/10.1109/JIOT.2023.3288936)

## Timing

With the corresponding data prepared, this example records five new honest rounds. Choose an unused output filename for each measurement.

```bash
python FLCE_updated.py --dataset tictactoe --client_num 4 --num_epoch 5 --num_local_epochs 1 --detecting_method loo --use_last_round_sv --honest_round_timing_csv honest_tictactoe.csv
python analysis/timing_summary.py honest honest_tictactoe.csv -o honest_tictactoe_summary.csv
```

To summarize existing timing exports, run the applicable command below.

```bash
python analysis/timing_summary.py honest global_training_time_trial0_no_attack.csv -o honest_rounds.csv
python analysis/timing_summary.py generation attacking_time_log_trial0.csv -o generation_totals.csv
python analysis/timing_summary.py grid full_grid_selected_time_tasks.csv --fl-rounds 50 -o timing_grid.csv
```

The honest probe measures one simulated server round through all clients' local training, aggregation and broadcast. It ends before accuracy scoring. The generation CSV contains a run total. A per-FL-round mean requires the actual FL round count, and a per-timed-fake-round mean requires its separate count. For the archived 50-round timing grid the script takes the mean within each setting and then the median across settings. Unrecorded denominators remain empty.

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

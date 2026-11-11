# PASS qualitative experiments: server Bash launchers

These experiments evaluate PASS. Each of the five Bash launchers embeds the complete scheduling and result-consolidation code and can run from the server project's `script/` directory. They require `FLCE_updated.py`, the PASS module, datasets, and the attack dependencies; separate Python scheduling files are not required for these standalone launchers.

## Datasets, partition settings, and attack profiles

| Dataset | Alpha values; the final value selects Uniform partitioning in the code | Pure 1 | Pure 2 | Lazy 1 | Lazy 2 | Default GPU |
| --- | --- | --- | --- | --- | --- | --- |
| tictactoe | 10 / 100 / 1000 / **20000** | moirai_online | advanced_scaled_delta | moirai_online | synthetic_replacement | 1 |
| adult | 1000 / 100000 / 1000000 / **20000000** | moirai_online | adam_local_deviation | arima_adaptive | synthetic_replacement | 2 |
| dota2 | 10000 / 1000000 / 50000000 / **200000000** | arima_adaptive | dkfr | synthetic_replacement | moirai_online | 3 |
| mnist | 10 / 1000 / 100000 / **10000000** | moirai_online_mnist_imdb_layerwise | dkfr | synthetic_replacement | moirai_online_mnist_imdb_layerwise | 0 |
| imdb | 10 / 100 / 1000 / **200000** | dkfr | moirai_online_mnist_imdb_layerwise | synthetic_replacement | arima_adaptive_mnist_imdb_layerwise | 1 |

An attack can appear under both pure and lazy schedules, so these are four configurations rather than necessarily four distinct algorithms. The profiles preserve the existing passive-experiment classifications and data-access behavior, including the Adam attack's public-data warm start at round 0.

## Fixed lazy intervals

Each lazy attack runs with gaps 3, 6, and 10, with `lazy_honest_phase=0` and `lazy_fake_policy=attack`.

| Gap | Repeating schedule | Honest rounds, indexed from zero | Honest / fake rounds out of 50 |
| --- | --- | --- | --- |
| 3 | 1 honest + 2 fake | 0, 3, 6, ... 48 | 17 / 33 |
| 6 | 1 honest + 5 fake | 0, 6, 12, ... 48 | 9 / 41 |
| 10 | 1 honest + 9 fake | 0, 10, 20, 30, 40 | 5 / 45 |

The base passive launcher uses gap 6; `fixed_gap3` and `fixed_gap10` use gaps 3 and 10. The `moirai_rerun` launcher reruns a Moirai subset at gap 6. Gap 2, used in other experiments, is not part of this PASS profile. Each pure configuration runs once.

## Fixed experiment parameters

- Detector: `pass`, with `qualitative_aggregation_mode=passive` and the existing aggregation behavior.
- Client counts: 2, 4, 6, and 8; one attacker with `attacker_id=1`; default `trial_id=0`.
- Fifty global rounds and one local epoch per round. Existing defaults remain: MLP hidden size 16, training batch size 16, learning rate 0.008, seed 42, and a 90/10 split.
- `random_round=10`, except `advanced_scaled_delta`, which uses 2.
- ARIMA/Moirai use `use_hessian` and `lambda_h=0.9`, preserving the current curvature approximation.
- Per dataset/trial: 4 alpha values x 4 client counts x (2 pure + 2 lazy x 3 gaps) = **128 tasks**, comprising 32 pure and 96 lazy tasks. All five datasets yield 640 tasks. Each task also retains the existing attacker-removal control run.
- GPU assignments follow the existing launchers. Defaults are one worker per GPU and eight CPU threads per worker, inherited from the Moirai rerun configuration. Environment variables can override resource settings.

PASS uses `pass_alpha=0.95` and `pass_beta=1.75`. Omitting `pass_c0` allows each run to use 1/N. Forged pure-attack audit reports use independent Uniform(-1,1) values with base seed 42 and an effective seed offset of `trial_id * 100003`; lazy attackers report genuine audits. The complete local training partition is audited with batch size 256 and `pass_max_audit_samples=0`. PASS alpha and data-partition alpha are distinct parameters.

## Inherited attack parameters

| Dataset | ARIMA variance / error threshold | Moirai tuning rate / error threshold |
| --- | --- | --- |
| tictactoe | 0.005 / 0.016 | 0.05 / 0.025 |
| adult | 0.005 / 0.08 | 0.003 / 0.12 |
| dota2 | 0.005 / 0.18 | 0.001 / 0.25 |
| mnist | 0.05 / 0.05 | 0.003 / 0.06 |
| imdb | 0.10 / 0.015 | 0.01 / 0.025 |

Attack-specific arguments are passed only to tasks using the corresponding attack:

- ARIMA: `var_window=3`, `min_rounds_after_start=4`, `retrain_interval=15`, `err_ema_beta=0.9`, and history trimming.
- Moirai: online tuning, `tune_steps=1`, `tune_start_round=10`, `adapt_interval=4`, `residual_decay=0.95`, `residual_clip=5`, `err_ema_beta=0.9`, `min_rounds_after_start=4`, and history trimming.
- Layerwise: `selection_metric=normalized_rms`, `selection_fraction=0.5`, `pca_threshold=512`, `pca_components=8`.
- Advanced scaled delta: `c_const=1.0`, `noise_fraction=1.0`, `max_noise_ratio=5.0`, `max_scale=10.0`.
- Adam: `lr=8e-3`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `public_fraction=0.05`, `public_seed=20260726`.
- Synthetic replacement: `mixup_alpha=0.4`, `noise_std=0.02`, `tabular_noise_std=0.05`, `tabular_flip_prob=0.02`, `seed=0`.
- DKFR: the three relevant datasets retain all 18 original `DKFR_VALUES`. MNIST additionally uses `tv=0.1`, `pixel_reg=0.2`, and `logit_reg=1e-4`. Use the dry-run output to inspect the complete commands embedded in each Bash file.

## Server execution

Activate the experiment's Python environment and place the five LF-terminated Bash files in the project's `script/` directory.

Preview commands without training:

```bash
bash script/run_tictactoe_passive_pass.sh
```

Run the required dataset commands. The following sequence avoids simultaneous use of the shared default GPU 1 by TicTacToe and IMDB:

```bash
DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 CAMPAIGN_ID=pass_all bash script/run_tictactoe_passive_pass.sh
DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 CAMPAIGN_ID=pass_all bash script/run_adult_passive_pass.sh
DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 CAMPAIGN_ID=pass_all bash script/run_dota2_passive_pass.sh
DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 CAMPAIGN_ID=pass_all bash script/run_mnist_passive_pass.sh
DRY_RUN=0 CONFIRM_QUALITATIVE_RUN=1 CAMPAIGN_ID=pass_all bash script/run_imdb_passive_pass.sh
```

Common overrides include `GPU_IDS="0 1"`, `WORKERS_PER_GPU=1`, `CPU_THREADS_PER_WORKER=8`, `TRIAL_IDS="0 1 2"`, `PYTHON_BIN=/path/to/python`, and `PASS_AUDIT_BATCH_SIZE=128`. Set `LAZY_GAPS="6"` to select one gap; the default is `3 6 10`. `PROJECT_ROOT` selects the project directory.

Moirai retains the rerun launcher's shared-cache settings: `SHARED_HF_HOME` defaults to `HF_HOME`, otherwise `/output/huggingface`; `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The server must already have the local weights and dependencies. GPU locks are shared with the passive launchers; select different GPUs for concurrent campaigns.

## One final CSV per dataset

Using the same campaign ID produces:

```text
Results/pass_all/tictactoe_pass.csv
Results/pass_all/adult_pass.csv
Results/pass_all/dota2_pass.csv
Results/pass_all/mnist_pass.csv
Results/pass_all/imdb_pass.csv
```

Each CSV combines all alpha values, client counts, pure/lazy schedules, gaps, and trials for that dataset. The default is 32,000 rows per file, with one row per task, submitted round, and client. A dataset's CSV is published only after all its tasks succeed. Existing final files are not overwritten; use a new campaign ID for a new experiment.

Rows contain scores and decisions, submitted and decision rounds, received audit reports as JSON, accuracy, the existing N-1 control, PASS settings, coverage, and task timing. `TaskID` includes the gap; `RunnerTaskID` retains the entry point's original task ID.

- `DecisionRound=SubmittedRound+1`. Round 49 has no subsequent audit in a 50-round experiment; its decision stays missing and is not counted as normal. With no other invalid cases, coverage is 49/50.
- PASS runs at N=2. Its attacker-removal control has N=1 and cannot audit another client; the invalid reason remains `no_other_auditor`.
- Accuracy is matched by submitted round. `WithoutAttackerAccuracy` comes from the existing N-1 control rather than an N-client honest run.
- Timing, coverage, and `TaskEvaluationJSON` repeat across a task's rows. Deduplicate by `TaskID` before adding task times. Detection time is already included in training wall time.
- Per-task raw outputs and logs remain under `.qualitative_runs/<CAMPAIGN_ID>/<dataset>/tasks/` for diagnosis.

Rebuild a final CSV without training:

```bash
DRY_RUN=0 MERGE_ONLY=1 CAMPAIGN_ID=pass_all bash script/run_tictactoe_passive_pass.sh
```

Merge-only mode requires complete successful outputs for every task in the original campaign; it does not retry failed tasks.

For maintenance, edit `script/passive_pass_campaign.py`, then run `python script/generate_passive_pass_launchers.py` to synchronize the payload embedded in all five Bash launchers.

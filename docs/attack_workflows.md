# Current pure and lazy attack workflows

The shared `FLCE_updated.py` runner contains 16 attack CLI choices. The repository includes all 71 current `script/run_*.sh` launchers and their support code. Historical v11-v20 runners and their root-level launchers are outside this release.

## Attack implementations

Use the following values with `--attack_method`. Standard implementations live in `module/method/attack_updated_nonlayerwise.py`; layerwise implementations live in `module/method/attack_updated_layerwise.py`.

| CLI value | Implementation |
| --- | --- |
| `random` | `RandomAttack` |
| `replay` | `ReplayAttack` |
| `fraboni_disguised` | `FraboniDisguisedAttack` |
| `scaled_delta` | `ScaledDeltaAttack` |
| `advanced_scaled_delta` | `AdvancedScaledDeltaAttack` |
| `adam_local_deviation` | `WangLocalDeviationAttack` |
| `dkfr` | `DKFRLiteAttack` |
| `synthetic_replacement` | `SyntheticReplacementAttack` |
| `arima` | `OfflineARIMAAttack` |
| `arima_adaptive` | `AdaptiveARIMAAttack` |
| `moirai` | `OfflineMoiraiAttack` |
| `moirai_online` | `OnlineMoiraiAttack` |
| `arima_mnist_imdb_layerwise` | `LayerwiseOfflineARIMAAttackMNISTIMDB` |
| `arima_adaptive_mnist_imdb_layerwise` | `LayerwiseAdaptiveARIMAAttackMNISTIMDB` |
| `moirai_mnist_imdb_layerwise` | `LayerwiseOfflineMoiraiAttackMNISTIMDB` |
| `moirai_online_mnist_imdb_layerwise` | `LayerwiseOnlineMoiraiAttackMNISTIMDB` |

Pure and lazy are scheduling modes of the selected attack. Method-specific public warm starts, synthetic-data training, and forecast-readiness rules remain part of the implementation. A pure schedule alone does not imply that every attack has the same data-access protocol. The dedicated strict-pure RQ1 launcher has its own narrower supported-method checks.

## Scheduling modes

| Mode | Arguments |
| --- | --- |
| Pure | `--use_attack` with `--use_lazy` omitted |
| Fixed lazy | `--use_attack --use_lazy --lazy_schedule fixed --lazy_honest_gap 6 --lazy_honest_phase 0 --lazy_fake_policy attack` |
| Probabilistic lazy | `--use_attack --use_lazy --lazy_schedule probabilistic --attack_probability 0.82 --lazy_fake_policy attack` |

The probabilistic scheduler samples the positions of a fixed fake-round budget. Forecast launchers pair gaps 2/3/6/10 with fake ratios 0.50/0.66/0.82/0.90, respectively. Existing attack-specific warm-start handling is preserved.

Example after preparing TicTacToe data:

```bash
python FLCE_updated.py --dataset tictactoe --client_num 4 --num_epoch 50 --use_attack --attack_method random --detecting_method shapley --fl_aggregate fedavg
```

Add the fixed-lazy or probabilistic-lazy arguments from the table to change the schedule. Select `fedavg`, `fedadam`, or `weighted_median` through `--fl_aggregate` in the shared entry point. Individual campaign launchers retain their existing fixed protocols and may support a narrower set of choices.

## Launcher families

Dataset-specific names below use `adult`, `dota2`, `imdb`, `mnist`, or `tictactoe`.

| Family | Entry points and coverage |
| --- | --- |
| Forecast pure | `run_<dataset>_forecast_pure_shapley.sh`: offline/adaptive ARIMA and offline/online Moirai; layerwise variants for MNIST/IMDB |
| Forecast lazy | `run_<dataset>_forecast_lazy_shapley.sh`: the same forecast methods with fixed and probabilistic lazy schedules |
| Offline lazy | `run_offline_lazy_shapley.sh`: offline ARIMA/Moirai with attacker-position and lazy-schedule sweeps |
| Passive qualitative detection | `run_<dataset>_passive_qualitative.sh`, plus `fixed_gap3`, `fixed_gap10`, and Moirai rerun variants; STD-DAGMM, Delta-DAGMM, WEF, S2-WEF |
| PASS | `run_<dataset>_passive_pass.sh`; see [the PASS guide](../script/README_passive_pass.md) |
| Log-based detection | `run_<dataset>_passive_log_based.sh`, backed by `log_based_campaign.py` and `passive_pass_campaign.py` |
| Sampling contribution methods | `run_<dataset>_sampling_shapley.sh` and `run_<dataset>_sampling_least_core.sh` |
| Multiple attackers | `run_multi_freerider.sh`, per-dataset wrappers, Uniform Shapley/STD-DAGMM entries, offline Uniform entry, and matched honest controls |
| Single-attacker RQ1 | `run_rq1_fl_aggregation.sh`, `run_rq1_strict_pure.sh`, `run_rq1_offline.sh`, `run_rq1_adam_dkfr.sh` |
| Pure sensitivity studies | `run_lambda_pure_shapley.sh` and `run_adaptation_interval_pure_shapley.sh` |
| Forecast diagnostics and matrices | Corrected single-run, full matrix, tuning pilot, missing-task rerun, and end-to-end smoke entry points |

The passive/PASS/log-based/sampling preset families include non-forecast baselines as well as forecasting attacks. The established per-dataset pure/lazy profiles are:

| Dataset | Pure profiles | Lazy profiles |
| --- | --- | --- |
| TicTacToe | Online Moirai; advanced scaled delta | Online Moirai; synthetic replacement |
| Adult | Online Moirai; Adam local deviation | Adaptive ARIMA; synthetic replacement |
| Dota2 | Adaptive ARIMA; DKFR | Synthetic replacement; online Moirai |
| MNIST | Layerwise online Moirai; DKFR | Synthetic replacement; layerwise online Moirai |
| IMDB | DKFR; layerwise online Moirai | Synthetic replacement; layerwise adaptive ARIMA |

The full CLI attack registry above remains available for additional runs. Preset launchers preserve the original experimental selections rather than sweeping every attack under every detector.

## Execution and support files

Run launchers from the repository root in Bash on the intended Linux experiment host. Most campaign entry points have a dry-run mode; inspect each header because deployment and diagnostic scripts can have different defaults. The three examples below explicitly request previews:

```bash
DRY_RUN=1 bash script/run_adult_forecast_pure_shapley.sh
DRY_RUN=1 bash script/run_adult_forecast_lazy_shapley.sh
DRY_RUN=1 bash script/run_adult_sampling_shapley.sh
```

GPU assignments, worker counts, CPU threads, cache paths, model availability, and dataset locations retain their existing environment-variable controls. Some server-oriented workflows retain defaults such as `/output/gradpred_exp/attack` or a server Python path; set `PROJECT_ROOT`, `CANDIDATE_ROOT`, and `PYTHON_BIN` for the target installation as required by that script.

`deploy_and_smoke_moirai_arima.sh` is included because the full matrix and tuning pilot use its candidate/smoke provenance. It stages an already available source tree and uses local experiment assets; it is not an SSH uploader. `preview_run_logfr_nohup.sh` previews log-based launch commands. PASS payload generation, PASS CSV summarization, forecast tuning analysis, result mergers, and the referenced smoke/scheduler tests are also included.

## Complete current launcher inventory

- [`run_adaptation_interval_pure_shapley.sh`](../script/run_adaptation_interval_pure_shapley.sh)
- [`run_adult_forecast_lazy_shapley.sh`](../script/run_adult_forecast_lazy_shapley.sh)
- [`run_adult_forecast_pure_shapley.sh`](../script/run_adult_forecast_pure_shapley.sh)
- [`run_adult_multi_freerider.sh`](../script/run_adult_multi_freerider.sh)
- [`run_adult_passive_log_based.sh`](../script/run_adult_passive_log_based.sh)
- [`run_adult_passive_pass.sh`](../script/run_adult_passive_pass.sh)
- [`run_adult_passive_qualitative.sh`](../script/run_adult_passive_qualitative.sh)
- [`run_adult_passive_qualitative_fixed_gap10.sh`](../script/run_adult_passive_qualitative_fixed_gap10.sh)
- [`run_adult_passive_qualitative_fixed_gap3.sh`](../script/run_adult_passive_qualitative_fixed_gap3.sh)
- [`run_adult_passive_qualitative_moirai_rerun.sh`](../script/run_adult_passive_qualitative_moirai_rerun.sh)
- [`run_adult_sampling_least_core.sh`](../script/run_adult_sampling_least_core.sh)
- [`run_adult_sampling_shapley.sh`](../script/run_adult_sampling_shapley.sh)
- [`run_corrected_forecast_attack.sh`](../script/run_corrected_forecast_attack.sh)
- [`run_dota2_forecast_lazy_shapley.sh`](../script/run_dota2_forecast_lazy_shapley.sh)
- [`run_dota2_forecast_pure_shapley.sh`](../script/run_dota2_forecast_pure_shapley.sh)
- [`run_dota2_missing_moirai_online_shapley.sh`](../script/run_dota2_missing_moirai_online_shapley.sh)
- [`run_dota2_passive_log_based.sh`](../script/run_dota2_passive_log_based.sh)
- [`run_dota2_passive_pass.sh`](../script/run_dota2_passive_pass.sh)
- [`run_dota2_passive_qualitative.sh`](../script/run_dota2_passive_qualitative.sh)
- [`run_dota2_passive_qualitative_fixed_gap10.sh`](../script/run_dota2_passive_qualitative_fixed_gap10.sh)
- [`run_dota2_passive_qualitative_fixed_gap3.sh`](../script/run_dota2_passive_qualitative_fixed_gap3.sh)
- [`run_dota2_passive_qualitative_moirai_rerun.sh`](../script/run_dota2_passive_qualitative_moirai_rerun.sh)
- [`run_dota2_passive_qualitative_moirai_rerun2_failed_only.sh`](../script/run_dota2_passive_qualitative_moirai_rerun2_failed_only.sh)
- [`run_dota2_sampling_least_core.sh`](../script/run_dota2_sampling_least_core.sh)
- [`run_dota2_sampling_shapley.sh`](../script/run_dota2_sampling_shapley.sh)
- [`run_freerider_controls.sh`](../script/run_freerider_controls.sh)
- [`run_full_moirai_arima_matrix.sh`](../script/run_full_moirai_arima_matrix.sh)
- [`run_imdb_forecast_lazy_shapley.sh`](../script/run_imdb_forecast_lazy_shapley.sh)
- [`run_imdb_forecast_pure_shapley.sh`](../script/run_imdb_forecast_pure_shapley.sh)
- [`run_imdb_multi_freerider.sh`](../script/run_imdb_multi_freerider.sh)
- [`run_imdb_passive_log_based.sh`](../script/run_imdb_passive_log_based.sh)
- [`run_imdb_passive_pass.sh`](../script/run_imdb_passive_pass.sh)
- [`run_imdb_passive_qualitative.sh`](../script/run_imdb_passive_qualitative.sh)
- [`run_imdb_passive_qualitative_fixed_gap10.sh`](../script/run_imdb_passive_qualitative_fixed_gap10.sh)
- [`run_imdb_passive_qualitative_fixed_gap3.sh`](../script/run_imdb_passive_qualitative_fixed_gap3.sh)
- [`run_imdb_passive_qualitative_moirai_rerun.sh`](../script/run_imdb_passive_qualitative_moirai_rerun.sh)
- [`run_imdb_sampling_least_core.sh`](../script/run_imdb_sampling_least_core.sh)
- [`run_imdb_sampling_shapley.sh`](../script/run_imdb_sampling_shapley.sh)
- [`run_lambda_pure_shapley.sh`](../script/run_lambda_pure_shapley.sh)
- [`run_mnist_forecast_lazy_shapley.sh`](../script/run_mnist_forecast_lazy_shapley.sh)
- [`run_mnist_forecast_pure_shapley.sh`](../script/run_mnist_forecast_pure_shapley.sh)
- [`run_mnist_multi_freerider.sh`](../script/run_mnist_multi_freerider.sh)
- [`run_mnist_passive_log_based.sh`](../script/run_mnist_passive_log_based.sh)
- [`run_mnist_passive_pass.sh`](../script/run_mnist_passive_pass.sh)
- [`run_mnist_passive_qualitative.sh`](../script/run_mnist_passive_qualitative.sh)
- [`run_mnist_passive_qualitative_fixed_gap10.sh`](../script/run_mnist_passive_qualitative_fixed_gap10.sh)
- [`run_mnist_passive_qualitative_fixed_gap3.sh`](../script/run_mnist_passive_qualitative_fixed_gap3.sh)
- [`run_mnist_passive_qualitative_moirai_rerun.sh`](../script/run_mnist_passive_qualitative_moirai_rerun.sh)
- [`run_mnist_sampling_least_core.sh`](../script/run_mnist_sampling_least_core.sh)
- [`run_mnist_sampling_shapley.sh`](../script/run_mnist_sampling_shapley.sh)
- [`run_moirai_arima_tuning_pilot.sh`](../script/run_moirai_arima_tuning_pilot.sh)
- [`run_multi_freerider.sh`](../script/run_multi_freerider.sh)
- [`run_offline_lazy_shapley.sh`](../script/run_offline_lazy_shapley.sh)
- [`run_offline_multi_freerider_uniform.sh`](../script/run_offline_multi_freerider_uniform.sh)
- [`run_rq1_adam_dkfr.sh`](../script/run_rq1_adam_dkfr.sh)
- [`run_rq1_fl_aggregation.sh`](../script/run_rq1_fl_aggregation.sh)
- [`run_rq1_offline.sh`](../script/run_rq1_offline.sh)
- [`run_rq1_strict_pure.sh`](../script/run_rq1_strict_pure.sh)
- [`run_server_e2e_smoke_matrix.sh`](../script/run_server_e2e_smoke_matrix.sh)
- [`run_tictactoe_forecast_lazy_shapley.sh`](../script/run_tictactoe_forecast_lazy_shapley.sh)
- [`run_tictactoe_forecast_pure_shapley.sh`](../script/run_tictactoe_forecast_pure_shapley.sh)
- [`run_tictactoe_passive_log_based.sh`](../script/run_tictactoe_passive_log_based.sh)
- [`run_tictactoe_passive_pass.sh`](../script/run_tictactoe_passive_pass.sh)
- [`run_tictactoe_passive_qualitative.sh`](../script/run_tictactoe_passive_qualitative.sh)
- [`run_tictactoe_passive_qualitative_fixed_gap10.sh`](../script/run_tictactoe_passive_qualitative_fixed_gap10.sh)
- [`run_tictactoe_passive_qualitative_fixed_gap3.sh`](../script/run_tictactoe_passive_qualitative_fixed_gap3.sh)
- [`run_tictactoe_passive_qualitative_moirai_rerun.sh`](../script/run_tictactoe_passive_qualitative_moirai_rerun.sh)
- [`run_tictactoe_sampling_least_core.sh`](../script/run_tictactoe_sampling_least_core.sh)
- [`run_tictactoe_sampling_shapley.sh`](../script/run_tictactoe_sampling_shapley.sh)
- [`run_uniform_freerider_rq1_rq2.sh`](../script/run_uniform_freerider_rq1_rq2.sh)
- [`run_uniform_freerider_std_dagmm.sh`](../script/run_uniform_freerider_std_dagmm.sh)

# CAST

Research code for free-rider attacks, contribution evaluation, and detection in federated learning. Initial source release: 2026-11-10.

## Code layout

| Component | Location |
| --- | --- |
| Current FLCE entry point | [`FLCE_updated.py`](FLCE_updated.py) |
| Three server aggregation methods | [`module/method/fl_aggregation.py`](module/method/fl_aggregation.py) |
| Standard and layerwise attacks | [`attack_updated_nonlayerwise.py`](module/method/attack_updated_nonlayerwise.py), [`attack_updated_layerwise.py`](module/method/attack_updated_layerwise.py) |
| Qualitative detection | [`module/method/qualitative_detecting/`](module/method/qualitative_detecting/) |
| Quantitative contribution evaluation | [`module/method/quantitative_detecting/`](module/method/quantitative_detecting/) |
| Data loading, preprocessing, and partitioning | [`module/data_manager/`](module/data_manager/), [`script/datapre.py`](script/datapre.py) |
| Local models and training | [`module/model/`](module/model/) |
| Pure and lazy attack workflows | [`docs/attack_workflows.md`](docs/attack_workflows.md) |
| RQ1/RQ2 calculations and summaries | [`analysis/`](analysis/README.md) |
| Experiment launchers and result mergers | [`script/`](script/) |
| Existing regression tests | [`tests/`](tests/) |

The shared entry point selects `fedavg`, `fedadam`, or `weighted_median` through `--fl_aggregate`; `median` is an alias for `weighted_median`. These are server aggregation rules. Hierarchical clustering in the log-based detector is configured separately with `--logfr_linkage single|average|complete`.

Attack choices include `random`, `replay`, `fraboni_disguised`, `scaled_delta`, `advanced_scaled_delta`, `adam_local_deviation`, `dkfr`, `synthetic_replacement`, offline/adaptive ARIMA, offline/online Moirai, and the corresponding layerwise forecast variants for MNIST/IMDB. Pure and lazy schedules are configured by the same runner. See the [complete attack and launcher inventory](docs/attack_workflows.md) for the available combinations.

Qualitative methods include EDS, STD-DAGMM, Delta-DAGMM, WEF, S2-WEF, PASS-CE, and log-based free-rider detection. EDS implements the ensemble of similarity-distance metrics. RFFL has a separate reputation and reward protocol with its own [author implementation](https://github.com/XinyiYS/Robust-and-Fair-Federated-Learning). Quantitative methods include exact Shapley, sampling Shapley, sampling least core, and leave-one-out (`loo`). Leave-one-out uses the same coalition utility as exact Shapley and returns the signed difference between the grand coalition and the coalition excluding the client. The current STD-DAGMM implementation imports its base network and configuration from `std_dagmm_old_v1.py`.

## Environment

Use a dedicated Python 3.10 environment and run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python FLCE_updated.py --help
```

Install forecast attack or IMDB preprocessing dependencies as needed:

```bash
python -m pip install -r requirements-forecast.txt
python -m pip install -r requirements-imdb.txt
```

These dependency lists were assembled from the source environment and direct imports; they are not a complete lockfile validated in a fresh environment. Select a PyTorch/CUDA installation suitable for the target hardware. `requirements-forecast.txt` installs Uni2TS; the existing model loader uses local weights or Hugging Face.

The source snapshot pins PyTorch 2.3.1 and torchvision 0.18.1. To set up the reported PyTorch 2.4.1 / CUDA 12.4 core stack in a separate Python 3.10.17 environment, use the commands below. The optional requirements files above include the snapshot pins. New run manifests record the observed Python version, PyTorch version and PyTorch CUDA build. The hardware and original package lock for each archived run need separate provenance. [PyTorch release matrix](https://pytorch.org/get-started/previous-versions/#v241)

```bash
python -m pip install -r requirements-analysis.txt scipy==1.11.4 scikit-learn==1.3.0
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

## Data and preprocessing

Prepare raw datasets, preprocessed tensors, model weights, and experiment results separately. Main dataset paths:

| Dataset | Path used by the code |
| --- | --- |
| Adult | `data/raw/adult/adult.data` |
| TicTacToe | `data/raw/tic+tac+toe+endgame/tic-tac-toe.data` |
| Dota2 | `data/raw/dota2+games+results/dota2.csv` |
| MNIST | `data/raw/`, downloaded through torchvision |
| IMDB | `data/raw/imdb_preprocessed/imdb_distilbert-base-uncased_maxlen256.pt` |

For MNIST, run `python MNIST_download.py` first. The IMDB download script saves to `imdb_dataset/` at the repository root, so pass that directory explicitly to preprocessing:

```bash
python download_IMDB.py
python script/download_distilbert_model.py
python script/preprocess_imdb_embeddings.py --dataset_dir imdb_dataset
```

IMDB preprocessing pools the two labelled splits and encodes each review as the first-token hidden state of frozen DistilBERT (768 features for the supplied checkpoint, truncation at 256 tokens). MNIST pools 70,000 images and flattens them to 784 pixels. Both loaders re-split the pool 90/10. The Adult loader drops incomplete rows before splitting (30,162 complete rows in the standard 32,561-row file). The runner uses quantity-skew Dirichlet proportions to allocate the shuffled pool. The uniform setting divides it into equal-sized shards up to rounding. A trial ID and the seed controlling the partition are distinct fields.

For Dota2, place the two headerless UCI files in the Dota2 directory and concatenate train followed by test. The historical combined input has 102,944 rows and SHA-256 `f2bf7b2544cb93f8748f51f041fc098d723c9fcb27276149e7bae95c0938b6c0`. Run this command in the directory containing the two downloaded CSVs. [UCI Dota2 files](https://archive.ics.uci.edu/dataset/367/dota2+games+results)

```bash
python -c "from pathlib import Path; Path('dota2.csv').write_bytes(Path('dota2Train.csv').read_bytes() + Path('dota2Test.csv').read_bytes())"
```

## Running experiments

After preparing TicTacToe data, run a random attack with Shapley evaluation. Change the aggregation argument to use either of the other server rules:

```bash
python FLCE_updated.py --dataset tictactoe --client_num 4 --num_epoch 50 --use_attack --attack_method random --detecting_method shapley --fl_aggregate fedavg
```

The following Python entry points manage online, strict-pure, offline, and Adam/DKFR single-attacker RQ1 experiments. They preview tasks by default. Use `--run` for execution; their GPU task queues require Linux.

```bash
python script/rq1_fl_aggregation_campaign.py --dry-run
python script/rq1_strict_pure_campaign.py --dry-run
python script/rq1_offline_campaign.py --dry-run
python script/rq1_adam_dkfr_campaign.py --dry-run
```

Multiple attackers and matched honest controls:

```bash
python script/multi_freerider_campaign.py --help
python script/freerider_control_campaign.py --help
bash script/run_uniform_freerider_rq1_rq2.sh
```

Additional per-dataset pure/lazy, passive-detection, PASS, log-based, and sampling launchers are documented in [attack workflows](docs/attack_workflows.md). Each launcher retains its existing experiment matrix and execution controls. Inspect its header before changing GPU or concurrency settings: some dedicated entry points fix these values. Use `--help` for the complete FLCE argument reference.

See the [analysis guide](analysis/README.md) for figure inputs, paired credit comparisons, client-round Recall/FPR, timing denominators and postprocessing commands. Historical `use_hessian` labels denote a constant-tensor correction proxy. The online Moirai variant updates a forecast residual with a frozen backbone. Layerwise selection counts parameterized modules. These distinctions affect the interpretation of ablation results.

## Validation and provenance

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

The [original source manifest](docs/source_manifest.json) records the copied files and hashes in the [archived release](https://github.com/SOPHIACHEN00/CAST/tree/8d4d1bfd75df41d8630637aa5a0cf32ba07cac59). Git tracks subsequent changes, including the contribution, detection and timing analysis commands. The [validation record](docs/validation.md) identifies the checks performed for each release.

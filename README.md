# CAST

联邦学习 free-rider 攻击、贡献评估与检测的研究代码。首批整理日期：2026-11-10。

## 代码入口

| 内容 | 位置 |
| --- | --- |
| 当前 FLCE 主函数 | [`FLCE_updated.py`](FLCE_updated.py) |
| 三种服务器聚合方法 | [`module/method/fl_aggregation.py`](module/method/fl_aggregation.py) |
| 攻击：普通 / 分层实现 | [`attack_updated_nonlayerwise.py`](module/method/attack_updated_nonlayerwise.py)、[`attack_updated_layerwise.py`](module/method/attack_updated_layerwise.py) |
| 质化检测 | [`module/method/qualitative_detecting/`](module/method/qualitative_detecting/) |
| 量化检测与贡献评估 | [`module/method/quantitative_detecting/`](module/method/quantitative_detecting/) |
| 数据读取、预处理与划分 | [`module/data_manager/`](module/data_manager/)、[`script/datapre.py`](script/datapre.py) |
| 本地模型与训练 | [`module/model/`](module/model/) |
| RQ1/RQ2 计算与汇总 | [`analysis/`](analysis/README.md) |
| 实验启动与结果合并 | [`script/`](script/) |
| 现有聚合、攻击与 RQ1 回归测试 | [`tests/`](tests/) |

主函数保持原有模块路径，统一通过 `--fl_aggregate` 选择 `fedavg`、`fedadam` 或 `weighted_median`（`median` 是后者的别名）。这三项是服务器聚合规则。Log-based 检测中的层次聚类另有 `--logfr_linkage single|average|complete`。

支持的攻击参数包括 `random`、`replay`、`fraboni_disguised`、`scaled_delta`、`advanced_scaled_delta`、`adam_local_deviation`、`dkfr`、`synthetic_replacement`、离线/自适应 ARIMA、离线/在线 Moirai，以及 MNIST/IMDB 对应的分层版本。

质化检测包含 EDS、STD-DAGMM、Delta-DAGMM、WEF、S2-WEF、PASS 和 log-based free-rider detection。量化方法包含 exact Shapley、sampling Shapley 和 sampling least core。`std_dagmm_old_v1.py` 为当前 STD-DAGMM 提供基础网络和配置，是必要依赖。

## 环境

使用独立的 Python 3.10 环境，从仓库根目录运行命令：

```bash
python -m pip install -r requirements.txt
python FLCE_updated.py --help
```

按需安装预测攻击或 IMDB 预处理依赖：

```bash
python -m pip install -r requirements-forecast.txt
python -m pip install -r requirements-imdb.txt
```

依赖文件根据工作区环境记录和直接导入整理；它们不是经过全新环境安装验证的完整锁文件。PyTorch/CUDA 的安装方式需匹配运行设备。Uni2TS 通过 `requirements-forecast.txt` 安装，模型权重按原有代码从本地模型目录或 Hugging Face 加载。

## 数据与预处理

原始数据、预处理张量、模型权重及实验结果需单独准备。当前主要数据路径：

| 数据集 | 代码使用的路径 |
| --- | --- |
| Adult | `data/raw/adult/adult.data` |
| TicTacToe | `data/raw/tic+tac+toe+endgame/tic-tac-toe.data` |
| Dota2 | `data/raw/dota2+games+results/dota2.csv` |
| MNIST | `data/raw/`，由 torchvision 下载 |
| IMDB | `data/raw/imdb_preprocessed/imdb_distilbert-base-uncased_maxlen256.pt` |

MNIST 可先运行 `python MNIST_download.py`。IMDB 的下载脚本默认保存到根目录的 `imdb_dataset/`，因此预处理时显式传入该路径：

```bash
python download_IMDB.py
python script/download_distilbert_model.py
python script/preprocess_imdb_embeddings.py --dataset_dir imdb_dataset
```

## 运行示例

准备 TicTacToe 数据后，使用 random 攻击与 Shapley 评估；将聚合参数替换为另外两种即可使用同一主函数：

```bash
python FLCE_updated.py --dataset tictactoe --client_num 4 --num_epoch 50 --use_attack --attack_method random --detecting_method shapley --fl_aggregate fedavg
```

单攻击者 RQ1 的在线、严格 pure、离线、Adam/DKFR 补充实验分别由以下 Python 入口管理。默认只预览任务；执行时使用各入口的 `--run`，实际 GPU 队列执行要求 Linux。

```bash
python script/rq1_fl_aggregation_campaign.py --dry-run
python script/rq1_strict_pure_campaign.py --dry-run
python script/rq1_offline_campaign.py --dry-run
python script/rq1_adam_dkfr_campaign.py --dry-run
```

多攻击者及配对诚实对照：

```bash
python script/multi_freerider_campaign.py --help
python script/freerider_control_campaign.py --help
bash script/run_uniform_freerider_rq1_rq2.sh
```

Bash 启动器保留现有实验矩阵和资源设置，默认预览。修改 GPU/并发设置时请检查对应启动器；部分专用入口会固定资源参数。通用主函数参数以 `--help` 为准。

RQ1/RQ2 的输入格式、不同指标口径和后处理命令见 [`analysis/README.md`](analysis/README.md)。

## 验证与来源

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

本次整理保留源程序逻辑，只统一文本编码/换行并补充说明与依赖清单。逐文件来源与 SHA-256 见 [`docs/source_manifest.json`](docs/source_manifest.json)，验证范围见 [`docs/validation.md`](docs/validation.md)。

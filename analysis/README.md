# RQ1 / RQ2 后处理

此目录收录当前数据分析项目中的共享公式与实验适配器。保留原有相对目录结构，是因为适配器通过文件路径加载 `Analysis Code/` 下的公式。

## 通用贡献 CSV

输入为已有的 `combined_contributions_trial*.csv`，包含 `ClientID`、`AttackerID`、`WithAttacker`、`WithoutAttacker` 及各脚本要求的实验配置列。

从仓库根目录执行：

```bash
python "analysis/Analysis Code/Q1/compute_q1_summary.py" combined_contributions_trial0.csv -o q1_attacker_ratio_summary.csv
python "analysis/Analysis Code/Q2/compute_q2_honest_impact_summary.py" combined_contributions_trial0.csv -o q2_honest_impact_summary.csv
```

- `Q1/compute_q1_summary.py`：攻击者有符号 Shapley 值 / 所有客户端 Shapley 值之和。
- `Q1/compute_q1_positive_credit_summary.py`：另外保留正贡献份额口径，将负值截为零后计算份额。
- `Q2/compute_q2_honest_impact_summary.py`：配对诚实客户端的绝对变化、方向性损失和正损失等指标。
- `Q2/compute_q2_summary.py`：原始平均绝对变化汇总入口。
- `q1_*group_mean.py`、`q2_*group_mean.py`、`q1_avg_steal_summary.py`、`q2_avg_impact_summary.py`：相应表格的分组汇总。部分旧入口按当前工作目录读取默认 CSV 名称。
- `Q2/compute_q2_honest_client_boxplot.py`：输出诚实客户端层面的指标表。

不同脚本的指标定义和列名保持原样；有符号贡献比例、正贡献份额、绝对影响及方向性损失应分别解释。

## 多攻击者 Uniform 实验

入口：`all_attacking_integrate/tune_freerider/uniform/process_freerider_rq1_rq2.py`。

将实际实验导出的输入放在该入口旁，保持以下结构：

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

`sha256.json` 是经过核对的原始下载文件清单，键为相对此目录的路径、值为 SHA-256。需要使用与实际输入配套的清单。源代码仓库不附带这些实验输入。

```bash
python analysis/all_attacking_integrate/tune_freerider/uniform/process_freerider_rq1_rq2.py --partition uniform
```

该适配器检查完整的既定实验设计：每数据集 36 个攻击任务、9 个诚实对照，N=8，攻击者数 1/2/3，三个攻击者分配组。主 RQ1 指标是**每名攻击者的平均贡献 / 总贡献**，攻击者总份额另外保留；RQ2 为配对诚实客户端的平均绝对 Shapley 变化。组间标准差使用 `ddof=1`。这不是任意任务矩阵的通用读取器。

## 其他实验适配器

| 文件 | 作用 |
| --- | --- |
| `all_attacking_integrate/tunefreeridet/uniform_offline_pure/process_offline_pure_rq1.py` | 离线 pure 多攻击者 RQ1，保留既有目录名和 54 个任务的完整性要求 |
| `all_attacking_integrate/tune_lambda/process_lambda_rq1_rq2.py` | LambdaH 调参，分组时保留 LambdaH |
| `all_attacking_integrate/finetuneinterval/process_interval_rq1_rq2.py` | 调参间隔实验，保留调参参数与间隔 |

各适配器使用入口旁的既有实验目录/文件名；需先放入对应原始 CSV。若分析单攻击者三种聚合方法的新实验，`script/rq1_*_campaign.py` 已包含 JSON 到 `rq1_detail` / `rq1_mean_std` 等 CSV 的计算逻辑，可使用其 `--merge-only CAMPAIGN_DIRECTORY`。

只运行分析时可安装 `requirements-analysis.txt`。本仓库不含绘图产物、原始结果或服务器连接脚本。

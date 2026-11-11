import pandas as pd


df = pd.read_csv("q2_avg_impact_summary.csv")

group_cols = [
    "Dataset",
    "AttackMethod",
    "Alpha",
    "ClientNum",
    "AttackMode",
]

q2_group_mean = (
    df.groupby(group_cols)["Q2_AvgImpact"]
    .mean()
    .reset_index()
)

q2_group_mean.to_csv("q2_avg_impact_group_mean.csv", index=False)

print(q2_group_mean)

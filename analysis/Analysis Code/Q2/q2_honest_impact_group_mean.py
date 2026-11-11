import argparse

import pandas as pd


DEFAULT_METRICS = [
    "Q2_HonestClientCount",
    "Q2_AvgAbsImpact",
    "Q2_TotalAbsImpact",
    "Q2_AvgSignedDelta",
    "Q2_AvgSignedHonestLoss",
    "Q2_TotalSignedHonestLoss",
    "Q2_TotalPositiveHonestLoss",
    "Q2_AvgPositiveHonestLoss",
    "Q2_HonestLossShare",
    "Q2_HarmedClientFraction",
    "Q2_HelpedClientFraction",
    "Q2_MinMaxAvgAbsImpact",
    "Q2_MinMaxAvgSignedHonestLoss",
    "Q2_MinMaxHarmedClientFraction",
]


def summarize_q2(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    group_cols = [
        "Dataset",
        "AttackMethod",
        "Alpha",
        "ClientNum",
        "AttackMode",
    ]

    metric_cols = [col for col in DEFAULT_METRICS if col in df.columns]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    summary.to_csv(output_csv_path, index=False)
    print(f"Q2 honest-impact group mean saved to: {output_csv_path}")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Average RQ2 honest-impact metrics by experiment setting.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="q2_honest_impact_summary.csv",
        help="Path to q2_honest_impact_summary.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="q2_honest_impact_group_mean.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    summarize_q2(args.input_csv, args.output)


if __name__ == "__main__":
    main()

import argparse

import pandas as pd


DEFAULT_METRICS = [
    "Q1_AttackerSignedScore",
    "Q1_SignedShare",
    "Q1_AttackerPositiveCredit",
    "Q1_PositiveCreditShare",
    "Q1_AttackerMinMaxScore",
    "Q1_AttackerWasNegative",
    "Q1_NegativeClientCount",
]


def summarize_q1(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    group_cols = [
        "Dataset",
        "AttackMethod",
        "Alpha",
        "ClientNum",
        "AttackMode",
    ]
    if "UseHessian" in df.columns:
        group_cols.append("UseHessian")
    if "ContributionMethod" in df.columns:
        group_cols.append("ContributionMethod")

    metric_cols = [col for col in DEFAULT_METRICS if col in df.columns]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    summary.to_csv(output_csv_path, index=False)
    print(f"Q1 positive-credit group mean saved to: {output_csv_path}")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Average RQ1 positive-credit metrics by experiment setting.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="q1_positive_credit_summary.csv",
        help="Path to q1_positive_credit_summary.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="q1_positive_credit_group_mean.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    summarize_q1(args.input_csv, args.output)


if __name__ == "__main__":
    main()

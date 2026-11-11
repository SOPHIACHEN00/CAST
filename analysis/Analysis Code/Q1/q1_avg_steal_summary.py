import argparse
import pandas as pd


def summarize_q1(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    group_cols = [
        "Dataset",
        "AttackMethod",
        "Alpha",
        "ClientNum",
        "AttackMode",
    ]

    q1_group_mean = (
        df.groupby(group_cols, dropna=False)["Q1_AttackerRatio"]
        .mean()
        .reset_index()
    )

    q1_group_mean.to_csv(output_csv_path, index=False)
    print(f"Q1 ratio group mean saved to: {output_csv_path}")
    print(q1_group_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average RQ1 attacker contribution ratio by experiment setting.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="q1_attacker_ratio_summary.csv",
        help="Path to q1_attacker_ratio_summary.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="q1_attacker_ratio_group_mean.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()
    summarize_q1(args.input_csv, args.output)

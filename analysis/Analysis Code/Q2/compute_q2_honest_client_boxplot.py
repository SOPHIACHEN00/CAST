import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contribution_csv import read_contribution_runs


def process_combined_csv_for_q2_boxplot(input_csv_path, output_csv_path):
    df, run_keys = read_contribution_runs(input_csv_path, require_controls=True)

    # The attacker row has no no-attacker baseline. Keep only honest clients.
    df = df[df["ClientID"] != df["AttackerID"]].copy()
    df["WithoutAttacker"] = pd.to_numeric(df["WithoutAttacker"], errors="raise")

    df["Q2_Delta"] = df["WithAttacker"] - df["WithoutAttacker"]
    df["Q2_AbsImpact"] = df["Q2_Delta"].abs()
    df["Q2_HonestLoss"] = df["WithoutAttacker"] - df["WithAttacker"]
    df["Q2_PositiveHonestLoss"] = df["Q2_HonestLoss"].clip(lower=0)

    output_cols = run_keys + [
        "ClientID",
        "WithAttacker",
        "WithoutAttacker",
        "Q2_Delta",
        "Q2_AbsImpact",
        "Q2_HonestLoss",
        "Q2_PositiveHonestLoss",
    ]

    df[output_cols].to_csv(output_csv_path, index=False)
    print(f"Q2 honest-client boxplot data saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export honest-client-level RQ2 impact data for boxplots."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="combined_contributions_trial0.csv",
        help="Path to a combined shapley/contribution CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="q2_honest_client_boxplot.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    process_combined_csv_for_q2_boxplot(args.input_csv, args.output)


if __name__ == "__main__":
    main()

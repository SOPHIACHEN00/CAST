import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contribution_csv import read_contribution_runs


def compute_attacker_ratio(group_df):
    attacker_ids = group_df["AttackerID"].dropna().unique()
    if len(attacker_ids) != 1:
        return None

    attacker_id = attacker_ids[0]
    total_contribution = pd.to_numeric(group_df["WithAttacker"], errors="coerce").sum()
    if pd.isna(total_contribution) or total_contribution == 0:
        return None

    attacker_rows = group_df[group_df["ClientID"] == attacker_id].copy()
    if len(attacker_rows) != 1:
        return None

    attacker_value = pd.to_numeric(attacker_rows["WithAttacker"], errors="coerce").iloc[0]
    if pd.isna(attacker_value):
        return None

    return attacker_value / total_contribution


def process_combined_csv_for_q1(input_csv_path, output_csv_path):
    df, group_cols = read_contribution_runs(input_csv_path)

    results = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        ratio = compute_attacker_ratio(group_df)
        if ratio is None:
            continue

        record = dict(zip(group_cols, group_key))
        record["Q1_AttackerRatio"] = ratio
        results.append(record)

    df_result = pd.DataFrame(results, columns=group_cols + ["Q1_AttackerRatio"])
    df_result.to_csv(output_csv_path, index=False)
    print(f"\nQ1 ratio summary saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute RQ1 attacker contribution ratio from combined Shapley contributions."
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
        default="q1_attacker_ratio_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()
    process_combined_csv_for_q1(args.input_csv, args.output)

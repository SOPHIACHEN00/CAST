"""
RQ2 processing for honest-client impact and contribution stealing.

RQ2 asks how adding a free-rider changes the contribution scores assigned to
honest clients. There are two related but different effects:

1. Impact / disruption:
   The honest clients' scores may change in either direction. We measure this
   with the absolute difference:

       abs(C_with_attacker - C_without_attacker)

2. Stealing / directional loss:
   If the attacker steals contribution from honest clients, honest clients'
   scores should decrease relative to the no-attacker baseline. We measure this
   with:

       honest_loss = C_without_attacker - C_with_attacker

   Positive honest_loss means the honest client lost contribution. Negative
   honest_loss means the honest client gained contribution after the attacker
   was added, so it should not be counted as stolen credit.

The main RQ2 metrics should usually be the raw-score metrics:

    Q2_AvgAbsImpact
    Q2_AvgSignedHonestLoss
    Q2_TotalPositiveHonestLoss
    Q2_HonestLossShare

Min-max metrics are reported only as supplementary evidence of relative ranking
or position distortion. They should not be interpreted as the amount of credit
stolen, because min-max normalization changes the meaning of the original
Shapley scores.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contribution_csv import read_contribution_runs


def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or math.isclose(float(denominator), 0.0):
        return float("nan")
    return numerator / denominator


def compute_q2_metrics(group_df):
    df_clean = group_df.dropna(subset=["WithAttacker", "WithoutAttacker"]).copy()
    if df_clean.empty:
        return None

    with_scores = pd.to_numeric(df_clean["WithAttacker"], errors="coerce")
    without_scores = pd.to_numeric(df_clean["WithoutAttacker"], errors="coerce")
    valid_mask = with_scores.notna() & without_scores.notna()
    if not valid_mask.any():
        return None

    with_scores = with_scores[valid_mask]
    without_scores = without_scores[valid_mask]

    delta = with_scores - without_scores
    honest_loss = without_scores - with_scores
    positive_honest_loss = honest_loss.clip(lower=0)
    positive_without_pool = without_scores.clip(lower=0).sum()

    metrics = {
        "Q2_HonestClientCount": int(valid_mask.sum()),
        "Q2_AvgAbsImpact": float(delta.abs().mean()),
        "Q2_TotalAbsImpact": float(delta.abs().sum()),
        "Q2_AvgSignedDelta": float(delta.mean()),
        "Q2_AvgSignedHonestLoss": float(honest_loss.mean()),
        "Q2_TotalSignedHonestLoss": float(honest_loss.sum()),
        "Q2_TotalPositiveHonestLoss": float(positive_honest_loss.sum()),
        "Q2_AvgPositiveHonestLoss": float(positive_honest_loss.mean()),
        "Q2_HonestLossShare": safe_divide(
            float(positive_honest_loss.sum()),
            float(positive_without_pool),
        ),
        "Q2_HarmedClientFraction": float((honest_loss > 0).mean()),
        "Q2_HelpedClientFraction": float((honest_loss < 0).mean()),
    }

    if {"WithAttackerMinMax", "WithoutAttackerMinMax"}.issubset(df_clean.columns):
        with_minmax = pd.to_numeric(df_clean["WithAttackerMinMax"], errors="coerce")[valid_mask]
        without_minmax = pd.to_numeric(df_clean["WithoutAttackerMinMax"], errors="coerce")[valid_mask]
        minmax_valid = with_minmax.notna() & without_minmax.notna()
        if minmax_valid.any():
            minmax_delta = with_minmax[minmax_valid] - without_minmax[minmax_valid]
            minmax_loss = without_minmax[minmax_valid] - with_minmax[minmax_valid]
            metrics.update(
                {
                    "Q2_MinMaxAvgAbsImpact": float(minmax_delta.abs().mean()),
                    "Q2_MinMaxAvgSignedHonestLoss": float(minmax_loss.mean()),
                    "Q2_MinMaxHarmedClientFraction": float((minmax_loss > 0).mean()),
                }
            )

    return metrics


def process_combined_csv_for_q2(input_csv_path, output_csv_path):
    df, group_cols = read_contribution_runs(input_csv_path, require_controls=True)
    honest_clients = df.loc[df["ClientID"].ne(df["AttackerID"])]

    results = []
    for group_key, group_df in honest_clients.groupby(group_cols, dropna=False):
        metrics = compute_q2_metrics(group_df)
        if metrics is None:
            continue

        record = dict(zip(group_cols, group_key))
        record.update(metrics)
        results.append(record)

    df_result = pd.DataFrame(results) if results else pd.DataFrame(columns=group_cols + ["Q2_AvgAbsImpact"])
    df_result.to_csv(output_csv_path, index=False)
    print(f"Q2 honest-impact summary saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RQ2 honest-client impact and contribution-stealing metrics."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="combined_contributions_trial0.csv",
        help="Path to combined_contributions_trial*.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="q2_honest_impact_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    process_combined_csv_for_q2(args.input_csv, args.output)


if __name__ == "__main__":
    main()

"""
RQ1 processing with signed and positive-credit Shapley scores.

Shapley-based FLCE can assign negative scores when a free-rider is easy to
detect, such as RandGrad. A plain ratio,

    attacker_score / sum(all_scores)

can become hard to interpret when some scores are negative, because the
denominator is a signed payoff total rather than an amount of earned credit.
For this reason, this script reports both:

1. signed quantities, which preserve the FLCE penalty signal; and
2. positive-credit quantities, where negative scores are clipped to zero before
   computing the attacker's share of the credit pool.

The main RQ1 metric should usually be Q1_PositiveCreditShare:

    max(attacker_score, 0) / sum_i max(client_score_i, 0)

Signed scores remain useful for showing that detectable attacks receive
negative contribution, while positive-credit share answers how much credit a
free-rider can earn when FLCE still assigns it positive value.
"""

import argparse
import math

import pandas as pd


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_attack_mode(use_lazy, lazy_honest_gap):
    if parse_bool(use_lazy):
        gap = int(float(lazy_honest_gap))
        if gap == 2:
            return "1real_1fake"
        if gap == 3:
            return "1real_2fake"
        if gap == 10:
            return "1real_9fake"
        return f"1real_{gap - 1}fake"
    return "pure_fake"


def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or math.isclose(float(denominator), 0.0):
        return float("nan")
    return numerator / denominator


def compute_q1_metrics(group_df):
    attacker_ids = group_df["AttackerID"].dropna().unique()
    if len(attacker_ids) != 1:
        return None

    attacker_id = attacker_ids[0]
    scores = pd.to_numeric(group_df["WithAttacker"], errors="coerce")
    client_ids = pd.to_numeric(group_df["ClientID"], errors="coerce")

    attacker_scores = scores[client_ids == attacker_id].dropna()
    if len(attacker_scores) != 1:
        return None

    attacker_score = float(attacker_scores.iloc[0])
    positive_scores = scores.clip(lower=0)

    total_signed_score = float(scores.sum(skipna=True))
    total_positive_credit = float(positive_scores.sum(skipna=True))
    attacker_positive_credit = max(attacker_score, 0.0)

    metrics = {
        "Q1_AttackerSignedScore": attacker_score,
        "Q1_TotalSignedScore": total_signed_score,
        "Q1_SignedShare": safe_divide(attacker_score, total_signed_score),
        "Q1_AttackerPositiveCredit": attacker_positive_credit,
        "Q1_TotalPositiveCredit": total_positive_credit,
        "Q1_PositiveCreditShare": safe_divide(attacker_positive_credit, total_positive_credit),
        "Q1_AttackerWasNegative": attacker_score < 0,
        "Q1_NegativeClientCount": int((scores < 0).sum()),
    }

    if "WithAttackerMinMax" in group_df.columns:
        minmax_scores = pd.to_numeric(group_df["WithAttackerMinMax"], errors="coerce")
        attacker_minmax = minmax_scores[client_ids == attacker_id].dropna()
        metrics["Q1_AttackerMinMaxScore"] = (
            float(attacker_minmax.iloc[0]) if len(attacker_minmax) == 1 else float("nan")
        )

    return metrics


def process_combined_csv_for_q1(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    df["WithAttacker"] = pd.to_numeric(df["WithAttacker"], errors="coerce")
    df["ClientID"] = pd.to_numeric(df["ClientID"], errors="coerce")
    df["AttackerID"] = pd.to_numeric(df["AttackerID"], errors="coerce")
    df["UseAttackBool"] = df["UseAttack"].apply(parse_bool)
    df = df[df["UseAttackBool"]].copy()

    df["AttackMode"] = df.apply(
        lambda row: infer_attack_mode(row["UseLazy"], row["LazyHonestGap"]),
        axis=1,
    )

    group_cols = [
        "Dataset",
        "AttackMethod",
        "Alpha",
        "ClientNum",
        "AttackerID",
        "TrialID",
        "ContributionMethod",
        "UseHessian",
        "UseLazy",
        "LazyHonestGap",
        "LazyHonestPhase",
        "LazyFakePolicy",
        "AttackMode",
    ]

    results = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        metrics = compute_q1_metrics(group_df)
        if metrics is None:
            continue

        record = dict(zip(group_cols, group_key))
        record.update(metrics)
        results.append(record)

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_csv_path, index=False)
    print(f"Q1 positive-credit summary saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RQ1 signed and positive-credit attacker metrics from combined contributions."
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
        default="q1_positive_credit_summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    process_combined_csv_for_q1(args.input_csv, args.output)


if __name__ == "__main__":
    main()

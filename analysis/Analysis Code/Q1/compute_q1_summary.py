import argparse
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
        ratio = compute_attacker_ratio(group_df)
        if ratio is None:
            continue

        record = dict(zip(group_cols, group_key))
        record["Q1_AttackerRatio"] = ratio
        results.append(record)

    df_result = pd.DataFrame(results)
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

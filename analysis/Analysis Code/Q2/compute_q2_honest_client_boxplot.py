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


def process_combined_csv_for_q2_boxplot(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    df["WithAttacker"] = pd.to_numeric(df["WithAttacker"], errors="coerce")
    df["WithoutAttacker"] = pd.to_numeric(df["WithoutAttacker"], errors="coerce")
    df["ClientID"] = pd.to_numeric(df["ClientID"], errors="coerce")
    df["AttackerID"] = pd.to_numeric(df["AttackerID"], errors="coerce")
    df["UseAttackBool"] = df["UseAttack"].apply(parse_bool)
    df = df[df["UseAttackBool"]].copy()

    df["AttackMode"] = df.apply(
        lambda row: infer_attack_mode(row["UseLazy"], row["LazyHonestGap"]),
        axis=1,
    )

    # The attacker row has no no-attacker baseline. Keep only honest clients.
    df = df[df["ClientID"] != df["AttackerID"]].copy()
    df = df.dropna(subset=["WithAttacker", "WithoutAttacker"])

    df["Q2_Delta"] = df["WithAttacker"] - df["WithoutAttacker"]
    df["Q2_AbsImpact"] = df["Q2_Delta"].abs()
    df["Q2_HonestLoss"] = df["WithoutAttacker"] - df["WithAttacker"]
    df["Q2_PositiveHonestLoss"] = df["Q2_HonestLoss"].clip(lower=0)

    output_cols = [
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

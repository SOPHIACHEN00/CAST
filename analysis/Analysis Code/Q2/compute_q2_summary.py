import pandas as pd


def infer_attack_mode(use_lazy, lazy_honest_gap):
    if bool(use_lazy):
        gap = int(lazy_honest_gap)
        if gap == 2:
            return "1real_1fake"
        if gap == 3:
            return "1real_2fake"
        if gap == 10:
            return "1real_9fake"
        return f"1real_{gap - 1}fake"
    return "pure_fake"


def compute_avg_impact(group_df):
    df_clean = group_df.dropna(subset=["WithAttacker", "WithoutAttacker"]).copy()
    if df_clean.empty:
        return None

    df_clean["AbsDelta"] = (
        pd.to_numeric(df_clean["WithAttacker"], errors="coerce")
        - pd.to_numeric(df_clean["WithoutAttacker"], errors="coerce")
    ).abs()

    abs_delta_mean = df_clean["AbsDelta"].mean()
    if pd.isna(abs_delta_mean):
        return None
    return abs_delta_mean


def process_combined_csv_for_q2(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    df["WithAttacker"] = pd.to_numeric(df["WithAttacker"], errors="coerce")
    df["WithoutAttacker"] = pd.to_numeric(df["WithoutAttacker"], errors="coerce")
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
        avg_impact = compute_avg_impact(group_df)
        if avg_impact is None:
            continue

        record = dict(zip(group_cols, group_key))
        record["Q2_AvgImpact"] = avg_impact
        results.append(record)

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_csv_path, index=False)
    print(f"\nQ2 impact summary saved to: {output_csv_path}")


if __name__ == "__main__":
    input_file = "combined_contributions_trial0.csv"
    output_file = "q2_avg_impact_summary.csv"
    process_combined_csv_for_q2(input_file, output_file)

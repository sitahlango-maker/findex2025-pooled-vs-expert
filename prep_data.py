"""
Script 1: prepare the Findex 2025 Sub-Saharan Africa sample and check it.

Run in Cloud Shell (the data never leaves your environment):
    python3 prep_data.py --data findex_microdata_2025_labelled_update112425.csv

Writes:
    data/ssa_2025_clean.csv        individual rows   -> PRIVATE, never share or commit
    results/country_summary.csv    country averages  -> aggregated, OK to share
    results/data_checks.txt        checks and counts -> aggregated, OK to share

Codes follow the Global Findex 2025 codebook (WLD_2024_FINDEX_v02_M):
    female: 1 Female, 2 Male          urbanicity: 1 Rural, 2 Urban
    educ: 1 Primary or less, 2 Secondary, 3 Tertiary or more
    inc_q: 1 poorest 20% ... 5 richest 20% (within economy)
    emp_in: 1 In the workforce, 2 Out of the workforce
    internet_use: 0 No/DK/Ref, 1 Yes
    con1 (has a mobile phone): 1 Yes, 2 No, 3 DK, 4 Refused
    con9 (main phone type): 1 Smartphone, 2 Basic, 8 DK, 9 Refused
    dig_account (target): 0 No/DK/Ref, 1 Yes
"""

import argparse
import os

import pandas as pd

TARGET = "dig_account"
FEATURES = ["female", "age", "educ", "inc_q", "urbanicity", "emp_in",
            "internet_use", "has_phone", "has_smartphone"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log = []

    df = pd.read_csv(args.data, low_memory=False)
    log.append(f"All economies: {len(df):,} rows, {df['economy'].nunique()} economies")

    ssa = df[df["regionwb"].astype(str).str.startswith("Sub-Saharan")].copy()
    log.append(f"Sub-Saharan Africa: {len(ssa):,} rows, {ssa['economy'].nunique()} economies")

    # Connectivity features (new in Findex 2025)
    ssa["has_phone"] = ssa["con1"].map({1: 1, 2: 0})                 # DK/Refused -> missing
    phone_type_known = ssa["con9"].isin([1, 2])
    ssa["has_smartphone"] = 0
    ssa.loc[(ssa["has_phone"] == 1) & (ssa["con9"] == 1), "has_smartphone"] = 1
    ssa.loc[(ssa["has_phone"] == 1) & ~phone_type_known, "has_smartphone"] = pd.NA
    ssa.loc[ssa["has_phone"].isna(), "has_smartphone"] = pd.NA

    # Missing-value report per variable (aggregated)
    log.append("\nMissing share by variable (SSA):")
    for col in FEATURES + [TARGET]:
        log.append(f"  {col:15s} {ssa[col].isna().mean():6.1%}")

    keep = ["economy", "economycode", "wgt"] + FEATURES + [TARGET]
    clean = ssa[keep].dropna().copy()
    for col in FEATURES + [TARGET]:
        clean[col] = clean[col].astype(int)
    log.append(f"\nAfter dropping rows with missing values: {len(clean):,} rows "
               f"({len(clean) / len(ssa):.1%} kept)")

    # Code checks: stop if anything unexpected appears
    allowed = {"female": {1, 2}, "educ": {1, 2, 3}, "inc_q": {1, 2, 3, 4, 5},
               "urbanicity": {1, 2}, "emp_in": {1, 2}, "internet_use": {0, 1},
               "has_phone": {0, 1}, "has_smartphone": {0, 1}, TARGET: {0, 1}}
    for col, ok in allowed.items():
        bad = set(clean[col].unique()) - ok
        if bad:
            raise ValueError(f"{col}: unexpected codes {sorted(bad)}")
    log.append("Code checks: all variables within expected codes")

    # Sanity check: urban respondents should have higher account ownership
    by_res = clean.groupby("urbanicity")[TARGET].mean()
    log.append(f"Account rate rural (1): {by_res[1]:.1%}, urban (2): {by_res[2]:.1%}")

    # Country-level summary (weighted with survey weights = national estimates)
    def wmean(g, col):
        return (g[col] * g["wgt"]).sum() / g["wgt"].sum()

    summary = clean.groupby(["economy", "economycode"]).apply(lambda g: pd.Series({
        "n": len(g),
        "dig_account_rate": wmean(g, TARGET),
        "phone_rate": wmean(g, "has_phone"),
        "smartphone_rate": wmean(g, "has_smartphone"),
        "internet_rate": wmean(g, "internet_use"),
        "rural_share": ((g["urbanicity"] == 1) * g["wgt"]).sum() / g["wgt"].sum(),
    })).reset_index().sort_values("dig_account_rate")
    summary.round(3).to_csv("results/country_summary.csv", index=False)

    clean.to_csv("data/ssa_2025_clean.csv", index=False)
    with open("results/data_checks.txt", "w") as f:
        f.write("\n".join(log) + "\n")

    print("\n".join(log))
    print("\nCountry summary (weighted):")
    print(summary.round(2).to_string(index=False))
    print("\nSaved: results/country_summary.csv, results/data_checks.txt")
    print("PRIVATE (do not share): data/ssa_2025_clean.csv")


if __name__ == "__main__":
    main()

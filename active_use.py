"""
Script 5: robustness check - does the pooling result hold for ACTIVE USE?

Active use = anydigpayment = 1 (made or received a digital payment in the past
12 months), the World Bank's headline usage indicator.

Two versions of the question:
  A_all_adults      all adults: who actively pays digitally?
  B_account_owners  digitally enabled account owners only: active vs dormant

Same five strategies, same 5-fold cross-validation and the same paired tests as
Script 2 (compare_models.py), so the results are directly comparable.

Run in ~/paper (needs compare_models.py in the same folder):
    python3 active_use.py --raw findex_microdata_2025_labelled_update112425.csv

Writes (aggregated only, OK to share):
    results/active_use_rates.csv       weighted active-use and dormancy rates by country
    results/active_use_by_country.csv  within-country AUC per strategy and version
    results/active_use_summary.csv     averages per strategy and version
    results/active_use_tests.csv       paired tests vs XGB_pooled per version
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from compare_models import BASELINE, FEATURES, SEED, STRATEGIES, fit_predict

warnings.filterwarnings("ignore")
ACTIVE = "anydigpayment"
MIN_CLASS = 20          # need at least 20 people in each class to score a country


def load(raw_path):
    """Same cleaning as Script 1 (prep_data.py), plus the active-use variable."""
    df = pd.read_csv(raw_path, low_memory=False)
    ssa = df[df["regionwb"].astype(str).str.startswith("Sub-Saharan")].copy()
    ssa["has_phone"] = ssa["con1"].map({1: 1, 2: 0})
    known = ssa["con9"].isin([1, 2])
    ssa["has_smartphone"] = 0
    ssa.loc[(ssa["has_phone"] == 1) & (ssa["con9"] == 1), "has_smartphone"] = 1
    ssa.loc[(ssa["has_phone"] == 1) & ~known, "has_smartphone"] = np.nan
    ssa.loc[ssa["has_phone"].isna(), "has_smartphone"] = np.nan
    keep = ["economy", "economycode", "wgt"] + FEATURES + ["dig_account", ACTIVE]
    clean = ssa[keep].dropna().copy()
    for c in FEATURES + ["dig_account", ACTIVE]:
        clean[c] = clean[c].astype(int)
    return clean.reset_index(drop=True)


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def run_version(df, name):
    X, y, country = df[FEATURES], df[ACTIVE], df["economycode"]
    countries = sorted(country.unique())
    oof = pd.DataFrame(index=df.index, columns=STRATEGIES, dtype=float)
    strat = country + "_" + y.astype(str)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat)
    for k, (tr, te) in enumerate(folds, 1):
        for s in STRATEGIES:
            oof.iloc[te, oof.columns.get_loc(s)] = fit_predict(
                s, X.iloc[tr], y.iloc[tr], country.iloc[tr],
                X.iloc[te], country.iloc[te], countries)
        print(f"  [{name}] fold {k}/5 done")

    rows = []
    for c in countries:
        m = (country == c).values
        n1, n0 = int(y[m].sum()), int((1 - y[m]).sum())
        if min(n1, n0) < MIN_CLASS:
            continue
        for s in STRATEGIES:
            rows.append({"version": name, "country": c, "strategy": s, "n": int(m.sum()),
                         "active_share": round(n1 / m.sum(), 3),
                         "auc": roc_auc_score(y[m], oof.loc[m, s])})
    bc = pd.DataFrame(rows)
    wide = bc.pivot(index="country", columns="strategy", values="auc")

    summ = bc.groupby("strategy").agg(mean_auc=("auc", "mean")).reindex(STRATEGIES)
    summ["countries_best"] = wide.idxmax(axis=1).value_counts().reindex(STRATEGIES).fillna(0).astype(int)
    summ["pooled_overall_auc"] = [roc_auc_score(y, oof[s]) for s in STRATEGIES]
    summ.insert(0, "countries_scored", len(wide))
    summ.insert(0, "people", len(df))
    summ.insert(0, "version", name)

    rng = np.random.default_rng(SEED)
    tests = []
    for s in STRATEGIES:
        if s == BASELINE:
            continue
        d = (wide[s] - wide[BASELINE]).values
        boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        tests.append({"version": name, "strategy": s, "vs": BASELINE,
                      "mean_auc_difference": d.mean(),
                      "ci95_low": np.percentile(boot, 2.5), "ci95_high": np.percentile(boot, 97.5),
                      "countries_better": int((d > 0).sum()), "countries_worse": int((d < 0).sum()),
                      "wilcoxon_p": wilcoxon(d).pvalue})
    return bc, summ.reset_index(), pd.DataFrame(tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="original Findex 2025 CSV")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    df = load(args.raw)
    print(f"Rows: {len(df):,} | countries: {df.economycode.nunique()} "
          f"(Script 1 had 33,826 rows; any difference = rows missing {ACTIVE})")

    # Descriptive rates (survey-weighted), aggregated by country
    rates = df.groupby(["economy", "economycode"]).apply(lambda g: pd.Series({
        "active_use_rate_adults": wmean(g[ACTIVE], g["wgt"]),
        "dig_account_rate": wmean(g["dig_account"], g["wgt"]),
        "active_among_owners": wmean(g.loc[g.dig_account == 1, ACTIVE], g.loc[g.dig_account == 1, "wgt"])
        if (g.dig_account == 1).any() else np.nan,
    })).reset_index()
    rates["dormant_among_owners"] = 1 - rates["active_among_owners"]
    rates.sort_values("active_use_rate_adults").round(3).to_csv("results/active_use_rates.csv", index=False)

    versions = {"A_all_adults": df,
                "B_account_owners": df[df.dig_account == 1].reset_index(drop=True)}
    out_bc, out_sum, out_tests = [], [], []
    for name, d in versions.items():
        bc, summ, tests = run_version(d, name)
        out_bc.append(bc); out_sum.append(summ); out_tests.append(tests)

    pd.concat(out_bc).round(4).to_csv("results/active_use_by_country.csv", index=False)
    summary = pd.concat(out_sum)
    summary.round(4).to_csv("results/active_use_summary.csv", index=False)
    tests = pd.concat(out_tests)
    tests.round(4).to_csv("results/active_use_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nWeighted rates (median across countries): active use among adults "
          f"{rates.active_use_rate_adults.median():.1%}, dormant among account owners "
          f"{rates.dormant_among_owners.median():.1%}")
    print("\nSummary:\n", summary.round(3).to_string(index=False))
    print("\nPaired tests vs", BASELINE, ":\n", tests.round(4).to_string(index=False))
    print("\nSaved results/active_use_*.csv (aggregated only)")


if __name__ == "__main__":
    main()

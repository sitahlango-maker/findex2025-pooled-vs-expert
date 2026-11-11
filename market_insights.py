"""
Script 3: market and policy insights for 34 Sub-Saharan African markets.

Run (after Scripts 1 and 2) in Cloud Shell or Colab:
    python3 market_insights.py --raw findex_microdata_2025_labelled_update112425.csv

All outputs are country-level aggregates (OK to share):
    results/market_insights.csv     one row per market
    results/market_types.csv        country typology (cluster averages)
    results/reachable_unbanked.png  chart
    results/top_barriers.png        chart

Important: these are associations in survey data, not proven causes.
"What-if" figures show what the model expects, not guaranteed effects.

Definitions
  Unbanked            no digitally enabled account (dig_account = 0)
  Reachable unbanked  unbanked, owns a mobile phone, and has a cross-validated
                      predicted probability >= 0.5 (looks like account holders)
  Phone what-if       expected change in account ownership if every adult without
                      a phone had a (basic) phone, all else unchanged
  Gender gap          men's minus women's account rate; "unexplained" is the part
                      that remains when women keep their own characteristics but
                      are scored as men
  Top barrier         among the unbanked, the feature that pushes predictions
                      down the most on average (TreeSHAP contributions)
"""

import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "dig_account"
FEATURES = ["female", "age", "educ", "inc_q", "urbanicity", "emp_in",
            "internet_use", "has_phone", "has_smartphone"]
LABELS = {"female": "Being a woman", "age": "Age", "educ": "Low education",
          "inc_q": "Low income", "urbanicity": "Rural residence",
          "emp_in": "Not in workforce", "internet_use": "No internet use",
          "has_phone": "No mobile phone", "has_smartphone": "No smartphone"}


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="original Findex 2025 CSV")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv("data/ssa_2025_clean.csv")
    oof = pd.read_csv("data/oof_predictions.csv")
    assert (oof["economycode"].values == df["economycode"].values).all(), "row order mismatch"
    df["p_oof"] = oof["XGB_pooled"].values             # cross-validated predictions

    # Country-level extras from the raw file (aggregated only)
    raw = pd.read_csv(args.raw, low_memory=False,
                      usecols=["economy", "economycode", "regionwb", "pop_adult", "wgt",
                               "account", "account_fin", "account_mob"])
    raw = raw[raw["regionwb"].astype(str).str.startswith("Sub-Saharan")]
    extras = raw.groupby("economycode").apply(lambda g: pd.Series({
        "economy": g["economy"].iloc[0],
        "adults_m": g["pop_adult"].iloc[0] / 1e6,
        # among account holders: share with mobile money vs financial institution
        "mobile_money_share": wmean(g.loc[g.account == 1, "account_mob"].fillna(0),
                                    g.loc[g.account == 1, "wgt"]),
        "bank_fi_share": wmean(g.loc[g.account == 1, "account_fin"],
                               g.loc[g.account == 1, "wgt"]),
    }))

    # One pooled model on all data, used for what-if and explanations
    countries = sorted(df["economycode"].unique())
    def design(d):
        X = d[FEATURES].copy()
        for c in countries:
            X[f"c_{c}"] = (d["economycode"] == c).astype(int)
        return X
    model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
                              eval_metric="logloss", random_state=SEED, n_jobs=2)
    X = design(df)
    model.fit(X, df[TARGET])
    p_now = model.predict_proba(X)[:, 1]

    # What-if: everyone without a phone gets a basic phone
    X_phone = X.copy()
    X_phone.loc[df["has_phone"] == 0, "has_phone"] = 1
    p_phone = model.predict_proba(X_phone)[:, 1]

    # What-if for the gender gap: women scored as men
    X_male = X.copy()
    X_male["female"] = 2
    p_male = model.predict_proba(X_male)[:, 1]

    # TreeSHAP contributions (log-odds), individual features only
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :len(FEATURES)]
    contrib = pd.DataFrame(contrib, columns=FEATURES, index=df.index)

    rows = []
    for c in countries:
        m = (df["economycode"] == c).values
        d, w = df[m], df.loc[m, "wgt"]
        unb = d[TARGET] == 0
        women, men = d["female"] == 1, d["female"] == 2
        reachable = unb & (d["has_phone"] == 1) & (d["p_oof"] >= 0.5)
        barrier = contrib[m][unb.values].mul(w[unb], axis=0).sum() / w[unb].sum()
        top = barrier.sort_values()
        rows.append({
            "economycode": c,
            "account_rate": wmean(d[TARGET], w),
            "unbanked_share": wmean(unb, w),
            "reachable_share_of_adults": wmean(reachable, w),
            "reachable_share_of_unbanked": wmean(reachable[unb], w[unb]) if unb.any() else np.nan,
            "phone_whatif_pp": wmean(p_phone[m] - p_now[m], w) * 100,
            "gender_gap_pp": (wmean(d.loc[men, TARGET], w[men]) - wmean(d.loc[women, TARGET], w[women])) * 100,
            "gender_gap_unexplained_pp": wmean((p_male[m] - p_now[m])[women.values], w[women]) * 100,
            "top_barrier": LABELS[top.index[0]],
            "second_barrier": LABELS[top.index[1]],
            **{f"barrier_{f}": barrier[f] for f in FEATURES},
        })
    ins = pd.DataFrame(rows).set_index("economycode").join(extras)
    for col, share in [("unbanked_m", "unbanked_share"),
                       ("reachable_unbanked_m", "reachable_share_of_adults")]:
        ins[col] = ins[share] * ins["adults_m"]
    ins["phone_whatif_m"] = ins["phone_whatif_pp"] / 100 * ins["adults_m"]

    # Country typology (k-means on standardised country indicators)
    summary = pd.read_csv("results/country_summary.csv").set_index("economycode")
    ins = ins.join(summary[["phone_rate", "smartphone_rate", "internet_rate", "rural_share"]])
    typ_cols = ["account_rate", "mobile_money_share", "smartphone_rate", "phone_rate", "internet_rate"]
    Z = StandardScaler().fit_transform(ins[typ_cols])
    ins["market_type"] = KMeans(4, n_init=50, random_state=SEED).fit_predict(Z)
    types = ins.groupby("market_type")[typ_cols].mean().round(2)
    types["countries"] = ins.groupby("market_type")["economy"].apply(lambda s: ", ".join(sorted(s)))
    types.to_csv("results/market_types.csv")

    cols = ["economy", "adults_m", "account_rate", "unbanked_m", "reachable_unbanked_m",
            "reachable_share_of_unbanked", "phone_whatif_pp", "phone_whatif_m",
            "gender_gap_pp", "gender_gap_unexplained_pp", "mobile_money_share",
            "bank_fi_share", "top_barrier", "second_barrier", "market_type"]
    out = ins[cols + [f"barrier_{f}" for f in FEATURES]].sort_values("reachable_unbanked_m", ascending=False)
    out.round(3).to_csv("results/market_insights.csv")

    # Charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = out.sort_values("unbanked_m")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(s["economy"], s["unbanked_m"], color="#CFD8DC", label="Unbanked adults")
    ax.barh(s["economy"], s["reachable_unbanked_m"], color="#1E88E5",
            label="Reachable unbanked (has phone, profile like account holders)")
    ax.set_xlabel("Millions of adults (15+)")
    ax.set_title("Unbanked and reachable unbanked adults, Findex 2025")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig("results/reachable_unbanked.png", dpi=200); plt.close(fig)

    heat = out.set_index("economy")[[f"barrier_{f}" for f in FEATURES]]
    heat.columns = [LABELS[f] for f in FEATURES]
    heat = heat.loc[out.sort_values("account_rate")["economy"]]
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(heat.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(heat.shape[1])); ax.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(heat.shape[0])); ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title("What holds the unbanked back? (red = pushes account ownership down)")
    fig.colorbar(im, ax=ax, label="Average contribution among the unbanked (log-odds)")
    fig.tight_layout(); fig.savefig("results/top_barriers.png", dpi=200); plt.close(fig)

    pd.set_option("display.width", 200)
    show = ["economy", "adults_m", "account_rate", "unbanked_m", "reachable_unbanked_m",
            "phone_whatif_pp", "gender_gap_pp", "gender_gap_unexplained_pp",
            "mobile_money_share", "top_barrier", "market_type"]
    print(out[show].round(2).to_string())
    print("\nMarket types (cluster averages):\n", types.to_string())
    print(f"\nTotals across {len(out)} markets: unbanked {out.unbanked_m.sum():.0f}m, "
          f"reachable unbanked {out.reachable_unbanked_m.sum():.0f}m, "
          f"phone what-if {out.phone_whatif_m.sum():.0f}m adults")
    print("\nSaved results/market_insights.csv, market_types.csv, reachable_unbanked.png, top_barriers.png")


if __name__ == "__main__":
    main()

"""
Script 4: learning curves (when would country experts win?) and fairness checks.

Run (after Scripts 1 and 2) in Cloud Shell or Colab:
    python3 curves_fairness.py

Reads:   data/ssa_2025_clean.csv, data/oof_predictions.csv   (private)
Writes (aggregated, OK to share):
    results/learning_curves.csv        mean AUC by strategy and country sample size
    results/learning_curves.png
    results/fairness_by_group.csv      within-country AUC and calibration by gender and residence
    results/calibration.png

Learning curves: for each country, 20% of its people are held out for testing
(5 rotations). The expert sees n of the remaining people of that country; the
shared (pooled) model sees the same n plus every other country's data.
Logistic regression is used for speed; in Script 2 it matched XGBoost.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "dig_account"
FEATURES = ["female", "age", "educ", "inc_q", "urbanicity", "emp_in",
            "internet_use", "has_phone", "has_smartphone"]
SIZES = [50, 100, 200, 400, None]          # None = all available training rows


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def with_country(X, country, countries):
    D = X.copy()
    for c in countries:
        D[f"c_{c}"] = (country == c).astype(int).values
    return D


def learning_curves(df, countries):
    rng = np.random.default_rng(SEED)
    X, y, cty = df[FEATURES], df[TARGET], df["economycode"]
    rows = []
    for c in countries:
        m = (cty == c).values
        Xc, yc = X[m], y[m]
        others = ~m
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xc, yc)):
            Xtr_all, ytr_all = Xc.iloc[tr], yc.iloc[tr]
            Xte, yte = Xc.iloc[te], yc.iloc[te]
            if yte.nunique() < 2:
                continue
            for n in SIZES:
                n_use = len(tr) if n is None else min(n, len(tr))
                idx = rng.choice(len(tr), n_use, replace=False)
                Xs, ys = Xtr_all.iloc[idx], ytr_all.iloc[idx]
                if ys.nunique() < 2:
                    continue
                # Expert: this country's n people only
                p_exp = lr().fit(Xs, ys).predict_proba(Xte)[:, 1]
                # Shared: every other country + this country's n people
                Xp = pd.concat([X[others], Xs])
                yp = pd.concat([y[others], ys])
                cp = pd.concat([cty[others], pd.Series(c, index=Xs.index)])
                p_pool = lr().fit(with_country(Xp, cp, countries), yp).predict_proba(
                    with_country(Xte, pd.Series(c, index=Xte.index), countries))[:, 1]
                for s, p in [("Expert (own country only)", p_exp), ("Shared model", p_pool)]:
                    rows.append({"country": c, "fold": k, "n_country": n_use, "size_label": "all" if n is None else n,
                                 "strategy": s, "auc": roc_auc_score(yte, p)})
        print(f"  learning curve done: {c}")
    return pd.DataFrame(rows)


def fairness(df, oof, countries):
    rows = []
    for name, col, groups in [("Gender", "female", {1: "Women", 2: "Men"}),
                              ("Residence", "urbanicity", {1: "Rural", 2: "Urban"})]:
        for code, label in groups.items():
            aucs, gaps, ns = [], [], []
            for c in countries:
                m = ((df["economycode"] == c) & (df[col] == code)).values
                y, p = df.loc[m, TARGET], oof.loc[m, "XGB_pooled"]
                if m.sum() < 30 or y.nunique() < 2:
                    continue
                aucs.append(roc_auc_score(y, p))
                gaps.append(p.mean() - y.mean())        # + = model too optimistic
                ns.append(int(m.sum()))
            rows.append({"attribute": name, "group": label, "countries": len(aucs),
                         "people": int(np.sum(ns)), "mean_within_country_auc": np.mean(aucs),
                         "mean_calibration_gap_pp": np.mean(gaps) * 100,
                         "max_abs_calibration_gap_pp": np.max(np.abs(gaps)) * 100})
    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv("data/ssa_2025_clean.csv")
    oof = pd.read_csv("data/oof_predictions.csv")
    assert (oof["economycode"].values == df["economycode"].values).all(), "row order mismatch"
    countries = sorted(df["economycode"].unique())

    lc = learning_curves(df, countries)
    lc.round(4).to_csv("results/learning_curves_raw.csv", index=False)
    summ = (lc.groupby(["strategy", "size_label"], sort=False)
              .agg(mean_auc=("auc", "mean"), mean_n=("n_country", "mean")).reset_index())
    wide = lc.pivot_table(index=["country", "fold", "size_label"], columns="strategy", values="auc").reset_index()
    wide["gap"] = wide["Expert (own country only)"] - wide["Shared model"]
    gap = wide.groupby("size_label", sort=False).agg(
        mean_gap=("gap", "mean"), share_expert_wins=("gap", lambda g: (g > 0).mean())).reset_index()
    summ = summ.merge(gap, on="size_label")
    summ.round(4).to_csv("results/learning_curves.csv", index=False)

    fair = fairness(df, oof, countries)
    fair.round(3).to_csv("results/fairness_by_group.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, colr in [("Shared model", "#1E88E5"), ("Expert (own country only)", "#E53935")]:
        d = summ[summ["strategy"] == s]
        ax.plot(d["mean_n"], d["mean_auc"], marker="o", color=colr, lw=2, label=s)
    ax.set_xscale("log")
    ax.set_xlabel("People from the country used for training (log scale)")
    ax.set_ylabel("Mean ROC-AUC on held-out people")
    ax.set_title("How much country data does an expert need?")
    ax.legend()
    fig.tight_layout(); fig.savefig("results/learning_curves.png", dpi=200); plt.close(fig)

    # Calibration: predicted vs observed, in 10 bins, by gender and residence
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (col, groups) in zip(axes, [("female", {1: "Women", 2: "Men"}), ("urbanicity", {1: "Rural", 2: "Urban"})]):
        for code, label in groups.items():
            m = (df[col] == code).values
            b = pd.cut(oof.loc[m, "XGB_pooled"], np.linspace(0, 1, 11), include_lowest=True)
            g = pd.DataFrame({"p": oof.loc[m, "XGB_pooled"], "y": df.loc[m, TARGET], "b": b}).groupby("b", observed=True).mean()
            ax.plot(g["p"], g["y"], marker="o", lw=1.5, label=label)
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax.set_xlabel("Predicted probability"); ax.legend()
    axes[0].set_ylabel("Observed share with an account")
    fig.suptitle("Calibration of the shared model by group")
    fig.tight_layout(); fig.savefig("results/calibration.png", dpi=200); plt.close(fig)

    pd.set_option("display.width", 160)
    print("\nLearning curves (mean across countries and folds):\n", summ.round(3).to_string(index=False))
    print("\nFairness (within-country, shared XGBoost model):\n", fair.round(3).to_string(index=False))
    print("\nSaved results/learning_curves.csv/.png, fairness_by_group.csv, calibration.png")


if __name__ == "__main__":
    main()

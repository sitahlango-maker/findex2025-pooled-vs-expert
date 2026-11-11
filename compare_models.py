"""
Script 2: the main experiment. Pooled vs country-specific vs partial pooling.

Question: with about 1,000 survey respondents per country, is it better to train
one model for all 34 countries, one model per country, or something in between?

    python3 compare_models.py

Reads:   data/ssa_2025_clean.csv            (private, from Script 1)
Writes:  results/auc_by_country.csv          per-country scores    (aggregated, OK to share)
         results/strategy_summary.csv        averages across countries (OK to share)
         results/significance_tests.csv      paired tests across countries (OK to share)
         results/auc_by_country.png          chart (OK to share)
         data/oof_predictions.csv            individual predictions (PRIVATE, for Script 3)

Strategies compared (all evaluated with the same 5-fold cross-validation, so every
person is scored by a model that never saw them):
    LR_pooled      one logistic regression, all countries, country indicators
    XGB_pooled     one XGBoost model, all countries, country indicators
    LR_partial     logistic regression with country-specific adjustments for every
                   feature, shrunk towards the shared model (partial pooling)
    LR_expert      one logistic regression per country
    XGB_expert     one XGBoost model per country (the "mixture of experts" idea)
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:  # only used to test the script without XGBoost
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False

SEED = 42
TARGET = "dig_account"
FEATURES = ["female", "age", "educ", "inc_q", "urbanicity", "emp_in",
            "internet_use", "has_phone", "has_smartphone"]
STRATEGIES = ["LR_pooled", "XGB_pooled", "LR_partial", "LR_expert", "XGB_expert"]
BASELINE = "XGB_pooled"


# ------------------------------------------------------------ models
def make_xgb(n_rows):
    """Smaller trees for small (single-country) samples to limit overfitting."""
    small = n_rows < 5000
    if HAVE_XGB:
        return xgb.XGBClassifier(
            n_estimators=200 if small else 400, max_depth=3 if small else 5,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            min_child_weight=5 if small else 2, eval_metric="logloss",
            random_state=SEED, n_jobs=2)
    return HistGradientBoostingClassifier(max_depth=3 if small else 5, learning_rate=0.05,
                                          max_iter=200 if small else 400, random_state=SEED)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def country_dummies(country, countries):
    return pd.DataFrame({f"c_{c}": (country == c).astype(int) for c in countries},
                        index=country.index)


def partial_pooling_matrix(X, country, countries):
    """Shared features + country indicators + country x feature interactions.
    The L2 penalty shrinks the interactions towards zero, i.e. towards the
    shared model, unless a country's data clearly supports a difference."""
    parts = [X, country_dummies(country, countries)]
    for c in countries:
        m = (country == c).astype(int).values[:, None]
        parts.append(pd.DataFrame(X.values * m, index=X.index,
                                  columns=[f"{f}_x_{c}" for f in X.columns]))
    return pd.concat(parts, axis=1)


def fit_predict(strategy, X_tr, y_tr, c_tr, X_te, c_te, countries):
    if strategy in ("LR_pooled", "XGB_pooled"):
        A_tr = pd.concat([X_tr, country_dummies(c_tr, countries)], axis=1)
        A_te = pd.concat([X_te, country_dummies(c_te, countries)], axis=1)
        model = make_lr() if strategy == "LR_pooled" else make_xgb(len(A_tr))
        return model.fit(A_tr, y_tr).predict_proba(A_te)[:, 1]

    if strategy == "LR_partial":
        A_tr = partial_pooling_matrix(X_tr, c_tr, countries)
        A_te = partial_pooling_matrix(X_te, c_te, countries)
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=[0.003, 0.01, 0.03, 0.1, 0.3], cv=3, scoring="roc_auc", max_iter=3000))
        return model.fit(A_tr, y_tr).predict_proba(A_te)[:, 1]

    # Experts: one model per country
    pred = pd.Series(np.nan, index=X_te.index)
    for c in countries:
        tr, te = (c_tr == c).values, (c_te == c).values
        if te.sum() == 0:
            continue
        if y_tr[tr].nunique() < 2:            # a country with only one class
            pred[te] = y_tr[tr].mean()
            continue
        model = make_lr() if strategy == "LR_expert" else make_xgb(tr.sum())
        pred[te] = model.fit(X_tr[tr], y_tr[tr]).predict_proba(X_te[te])[:, 1]
    return pred.values


# ------------------------------------------------------------- main
def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv("data/ssa_2025_clean.csv")
    X, y, country = df[FEATURES], df[TARGET], df["economycode"]
    countries = sorted(country.unique())
    print(f"Rows: {len(df):,} | countries: {len(countries)} | engine: "
          f"{'XGBoost ' + xgb.__version__ if HAVE_XGB else 'sklearn (TEST ONLY)'}")

    # Out-of-fold predictions: each person is scored by models that never saw them
    oof = pd.DataFrame(index=df.index, columns=STRATEGIES, dtype=float)
    strat = country + "_" + y.astype(str)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat)
    for k, (tr, te) in enumerate(folds, 1):
        for s in STRATEGIES:
            oof.iloc[te, oof.columns.get_loc(s)] = fit_predict(
                s, X.iloc[tr], y.iloc[tr], country.iloc[tr],
                X.iloc[te], country.iloc[te], countries)
        print(f"  fold {k}/5 done")

    # Per-country scores
    rows = []
    for c in countries:
        m = (country == c).values
        if y[m].nunique() < 2:
            continue
        for s in STRATEGIES:
            rows.append({"country": c, "strategy": s, "n": int(m.sum()),
                         "account_rate": round(y[m].mean(), 3),
                         "auc": roc_auc_score(y[m], oof.loc[m, s]),
                         "brier": brier_score_loss(y[m], oof.loc[m, s])})
    by_country = pd.DataFrame(rows)
    by_country.round(4).to_csv("results/auc_by_country.csv", index=False)

    wide = by_country.pivot(index="country", columns="strategy", values="auc")

    # Summary across countries (each country counts once)
    summary = by_country.groupby("strategy").agg(
        mean_auc=("auc", "mean"), median_auc=("auc", "median"),
        mean_brier=("brier", "mean")).reindex(STRATEGIES)
    summary["countries_best"] = wide.idxmax(axis=1).value_counts().reindex(STRATEGIES).fillna(0).astype(int)
    summary["pooled_overall_auc"] = [roc_auc_score(y, oof[s]) for s in STRATEGIES]
    summary.round(4).to_csv("results/strategy_summary.csv")

    # Paired tests: is each strategy different from the baseline across countries?
    rng = np.random.default_rng(SEED)
    tests = []
    for s in STRATEGIES:
        if s == BASELINE:
            continue
        d = (wide[s] - wide[BASELINE]).values
        boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        tests.append({"strategy": s, "vs": BASELINE,
                      "mean_auc_difference": d.mean(),
                      "ci95_low": np.percentile(boot, 2.5),
                      "ci95_high": np.percentile(boot, 97.5),
                      "countries_better": int((d > 0).sum()),
                      "countries_worse": int((d < 0).sum()),
                      "wilcoxon_p": wilcoxon(d).pvalue})
    tests = pd.DataFrame(tests)
    tests.round(4).to_csv("results/significance_tests.csv", index=False)

    # Private: individual-level predictions for Script 3
    out = df[["economycode"]].join(oof)
    out.to_csv("data/oof_predictions.csv", index=False)

    # Chart: per-country AUC, countries sorted by account rate
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = by_country.drop_duplicates("country").sort_values("account_rate")["country"]
    fig, ax = plt.subplots(figsize=(11, 5))
    styles = {"XGB_pooled": ("#1E88E5", "o"), "LR_partial": ("#43A047", "s"),
              "XGB_expert": ("#E53935", "^"), "LR_expert": ("#FB8C00", "v"),
              "LR_pooled": ("#8E24AA", "D")}
    for s, (col, mk) in styles.items():
        ax.scatter(range(len(order)), wide.loc[order, s], color=col, marker=mk,
                   s=28, label=s, alpha=0.85)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_ylabel("ROC-AUC (5-fold cross-validated)")
    ax.set_xlabel("Countries, sorted from lowest to highest account ownership")
    ax.set_title("Model strategy performance by country (Findex 2025, Sub-Saharan Africa)")
    ax.legend(fontsize=8, ncol=5, loc="lower right")
    fig.tight_layout()
    fig.savefig("results/auc_by_country.png", dpi=200)

    pd.set_option("display.width", 160)
    print("\nAverage across countries:\n", summary.round(3).to_string())
    print("\nCompared with", BASELINE, ":\n", tests.round(4).to_string(index=False))
    print("\nSaved results/*.csv and results/auc_by_country.png")
    print("PRIVATE (do not share): data/oof_predictions.csv")


if __name__ == "__main__":
    main()

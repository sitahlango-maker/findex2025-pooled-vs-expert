"""
Robustness check: did partial pooling stop at the edge of its penalty range?

The main run let cross-validation choose C from [0.003 ... 0.3] and it picked 0.003,
the strongest penalty on offer. This script widens the range to stronger penalties
(0.001, 0.0003, 0.0001) and reruns partial pooling on the SAME 5 outer folds.

Run in ~/paper (after Script 2):
    python3 penalty_check.py

Reads data/ssa_2025_clean.csv (private) and results/auc_by_country.csv.
Writes results/penalty_check.csv (aggregated only). Small C = high tax.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_models import FEATURES, SEED, TARGET, partial_pooling_matrix

warnings.filterwarnings("ignore")
WIDE_CS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

df = pd.read_csv("data/ssa_2025_clean.csv")
X, y, country = df[FEATURES], df[TARGET], df["economycode"]
countries = sorted(country.unique())

# Same outer folds as the main experiment (same seed, same stratification)
oof = pd.Series(np.nan, index=df.index)
chosen = []
strat = country + "_" + y.astype(str)
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat), 1):
    A_tr = partial_pooling_matrix(X.iloc[tr], country.iloc[tr], countries)
    A_te = partial_pooling_matrix(X.iloc[te], country.iloc[te], countries)
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=WIDE_CS, cv=3, scoring="roc_auc", max_iter=3000))
    model.fit(A_tr, y.iloc[tr])
    chosen.append(model[-1].C_[0])
    oof.iloc[te] = model.predict_proba(A_te)[:, 1]
    print(f"  fold {k}/5 done, chosen C = {model[-1].C_[0]}")

# Within-country AUC, same rule as the main experiment
aucs = {}
for c in countries:
    m = (country == c).values
    if y[m].nunique() == 2:
        aucs[c] = roc_auc_score(y[m], oof[m])
wide_partial = pd.Series(aucs)

main = pd.read_csv("results/auc_by_country.csv").pivot(index="country", columns="strategy", values="auc")
main = main.loc[wide_partial.index]
d = (wide_partial - main["XGB_pooled"]).values

out = pd.DataFrame([{
    "Cs_offered": " ".join(map(str, WIDE_CS)),
    "C_chosen_by_fold": " ".join(map(str, chosen)),
    "mean_auc_partial_original": round(main["LR_partial"].mean(), 4),
    "mean_auc_partial_wider_range": round(wide_partial.mean(), 4),
    "mean_auc_LR_pooled": round(main["LR_pooled"].mean(), 4),
    "mean_auc_XGB_pooled": round(main["XGB_pooled"].mean(), 4),
    "diff_vs_XGB_pooled": round(d.mean(), 4),
    "countries_better_than_XGB_pooled": int((d > 0).sum()),
    "pooled_overall_auc_wider_range": round(roc_auc_score(y, oof), 4),
}])
out.to_csv("results/penalty_check.csv", index=False)
print("\n" + out.T.to_string(header=False))

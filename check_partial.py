"""
Check the partial-pooling model: which "tax" (C) did cross-validation pick,
and how big are the country adjustments compared with the shared rules?

Run in ~/paper (after Script 1):
    python3 check_partial.py

Reads data/ssa_2025_clean.csv (private). Prints aggregated numbers only.
Small C = high tax (adjustments pushed towards zero); large C = low tax.
"""
import numpy as np
import pandas as pd

from compare_models import FEATURES, TARGET, partial_pooling_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/ssa_2025_clean.csv")
X, y, country = df[FEATURES], df[TARGET], df["economycode"]
countries = sorted(country.unique())

A = partial_pooling_matrix(X, country, countries)
model = make_pipeline(StandardScaler(), LogisticRegressionCV(
    Cs=[0.003, 0.01, 0.03, 0.1, 0.3], cv=3, scoring="roc_auc", max_iter=3000))
model.fit(A, y)
lr = model[-1]
coef = pd.Series(lr.coef_[0], index=A.columns)

print(f"Chosen C (price): {lr.C_[0]}   options were 0.003 (highest tax) ... 0.3 (lowest tax)")
print("Mean cross-validated AUC for each C:",
      dict(zip(lr.Cs_, np.round(lr.scores_[1].mean(axis=0), 4))))

shared = coef[FEATURES].abs()
adjust = coef[[c for c in A.columns if "_x_" in c]].abs()
print("\nShared rules, average size:          ", round(shared.mean(), 4))
print("Country adjustments, average size:   ", round(adjust.mean(), 4))
print("Adjustments as % of shared rules:    ", round(100 * adjust.mean() / shared.mean(), 1), "%")
print("Share of adjustments that are tiny (< 0.01):", round(100 * (adjust < 0.01).mean(), 1), "%")

by_feature = {f: round(coef[[f"{f}_x_{c}" for c in countries]].abs().mean(), 4) for f in FEATURES}
print("\nAverage country adjustment by feature (vs shared rule):")
for f in FEATURES:
    print(f"  {f:15s} shared {coef[f]:+.3f}   avg adjustment {by_feature[f]:.3f}")

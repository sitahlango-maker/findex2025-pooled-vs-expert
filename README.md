# When Do Country-Specific Models Help?

**Pooled versus expert models for predicting digital financial inclusion in Sub-Saharan Africa**

Winnie MejaSitah Lang'o · Independent researcher, Nairobi, Kenya

This repository contains the code and aggregated results for the paper. Using Global Findex 2025 microdata for 33,826 adults in 34 Sub-Saharan African economies, it compares five strategies for predicting ownership of a digitally enabled account: pooled logistic regression and XGBoost with country indicators, penalised partial pooling, and country-specific (expert) models. A shared model performs as well as any alternative, and country-specific XGBoost is worse in 33 of 34 economies.

## Data

The analysis uses the **Global Findex 2025 microdata** (World Bank, `WLD_2024_FINDEX_v02_M`), available from the [World Bank Microdata Library](https://microdata.worldbank.org/catalog/7860) after accepting its terms of use.

**The survey data are not included in this repository.** The World Bank's terms do not permit redistribution. Download the file yourself, then run the scripts below. Only aggregated, country-level results are published here.

## Reproduce the results

Requires Python 3.11.

```bash
pip install -r requirements.txt

# 1. Prepare the Sub-Saharan Africa sample (writes data/, which stays private)
python3 prep_data.py --data findex_microdata_2025_labelled_update112425.csv

# 2. Main comparison of the five strategies
python3 compare_models.py

# 3. Market estimates: reachable unbanked, barriers, gender gaps, market types
python3 market_insights.py --raw findex_microdata_2025_labelled_update112425.csv

# 4. Learning curves and fairness checks
python3 curves_fairness.py

# 5. Robustness check: active use instead of ownership
python3 active_use.py --raw findex_microdata_2025_labelled_update112425.csv

# Optional: inspect the partial-pooling penalty and draw the Africa maps
python3 check_partial.py
python3 africa_maps.py
```

## What is in `results/`

All files are aggregated (country level or above).

| File | Contents |
|---|---|
| `strategy_summary.csv` | Mean and pooled AUC per strategy |
| `auc_by_country.csv` | Within-country AUC and Brier score, every strategy and country |
| `significance_tests.csv` | Paired tests against the pooled XGBoost baseline |
| `learning_curves.csv` | Expert versus shared model by country sample size |
| `fairness_by_group.csv` | AUC and calibration by gender and residence |
| `market_insights.csv`, `market_types.csv` | Market estimates and the four market types |
| `active_use_*.csv` | Robustness check on active digital use |
| `*.png` | Figures used in the paper |

## Citation

Lang'o, W. M. (2026). *When Do Country-Specific Models Help? Pooled Versus Expert Models for Predicting Digital Financial Inclusion in Sub-Saharan Africa*. Working paper.

Data: World Bank (2025). *Global Financial Inclusion (Global Findex) Database 2025*. https://microdata.worldbank.org/catalog/7860

## Licence

Code: MIT licence. Aggregated results may be reused with citation of the paper and the Global Findex 2025 database.

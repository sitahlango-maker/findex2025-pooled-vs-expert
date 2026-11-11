"""
Two separate Africa maps from aggregated results:
    results/map_model_comparison.png  country-specific XGBoost minus shared model (within-country AUC)
    results/map_market_types.png      the four market types

Run in ~/paper after Scripts 2 and 3 (geopandas must be installed):
    python3 africa_maps.py

Reads only aggregated files: results/auc_by_country.csv, results/market_insights.csv
Downloads Natural Earth country borders (public domain) once.
"""
import os

import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BORDERS = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
GREY, EDGE = "#E4E7EB", "white"
SMALL = ["COM", "MUS"]      # island states too small to see at this scale
PALETTE = {"Mobile money mature": "#2F6FB0", "Mobile money growing, basic phones": "#5FB0A0",
           "Bank-led, connected": "#B7791F", "Low-connectivity frontier": "#B5475B"}
os.makedirs("results", exist_ok=True)

# ---- aggregated results ----
auc = pd.read_csv("results/auc_by_country.csv")
wide = auc.pivot(index="country", columns="strategy", values="auc")
diff = (wide["XGB_expert"] - wide["XGB_pooled"]).rename("diff").reset_index().rename(columns={"country": "iso3"})

ins = pd.read_csv("results/market_insights.csv")
ins = ins.rename(columns={ins.columns[0]: "iso3"})
prof = ins.groupby("market_type")[["account_rate", "mobile_money_share"]].mean()
names = {}
k = prof["account_rate"].idxmin(); names[k] = "Low-connectivity frontier"; rest = prof.drop(index=k)
k = rest["mobile_money_share"].idxmin(); names[k] = "Bank-led, connected"; rest = rest.drop(index=k)
k = rest["account_rate"].idxmax(); names[k] = "Mobile money mature"; rest = rest.drop(index=k)
for k in rest.index:
    names[k] = "Mobile money growing, basic phones"
ins["type_name"] = ins["market_type"].map(names)
data = diff.merge(ins[["iso3", "type_name"]], on="iso3", how="outer")

# ---- borders ----
world = gpd.read_file(BORDERS)
code = "ADM0_A3" if "ADM0_A3" in world.columns else "ISO_A3"
africa = world[(world["CONTINENT"] == "Africa") | world[code].isin(data["iso3"])].merge(data, left_on=code, right_on="iso3", how="left")
missing = sorted(set(data["iso3"]) - set(africa[code]))
if missing:
    print("Not found on the map (check codes):", missing)
islands = africa[africa[code].isin(SMALL)].copy()
islands["pt"] = islands.geometry.representative_point()
xmin, ymin, xmax, ymax = africa.total_bounds


def base_map(figsize=(7.5, 8.6)):
    fig, ax = plt.subplots(figsize=figsize)
    africa.plot(ax=ax, color=GREY, edgecolor=EDGE, linewidth=0.5)
    ax.set_xlim(xmin - 1, xmax + 1); ax.set_ylim(ymin - 1, ymax + 1)
    ax.set_aspect("equal"); ax.set_axis_off()
    return fig, ax


island_handle = Line2D([], [], marker="o", color="w", markerfacecolor="#999", markeredgecolor="#333",
                       markersize=7, label="Small island state (Comoros, Mauritius)")

# ---- Map 1: model comparison ----
fig, ax = base_map()
lim = max(0.03, float(data["diff"].abs().max()))
norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
africa[africa["diff"].notna()].plot(ax=ax, column="diff", cmap="RdBu", norm=norm, edgecolor=EDGE, linewidth=0.5)
for _, r in islands[islands["diff"].notna()].iterrows():
    ax.scatter(r.pt.x, r.pt.y, s=60, color=cm.RdBu(norm(r["diff"])), edgecolor="#333", linewidth=0.6, zorder=5)
worse, n = int((data["diff"] < 0).sum()), int(data["diff"].notna().sum())
ax.set_title(f"Country-specific XGBoost is worse than the shared model\nin {worse} of {n} markets",
             fontsize=12, loc="left")
cb = fig.colorbar(cm.ScalarMappable(cmap="RdBu", norm=norm), ax=ax, orientation="horizontal",
                  fraction=0.04, pad=0.02, shrink=0.8)
cb.set_label("AUC difference: country model minus shared model (red = country model worse)", fontsize=8.5)
ax.legend(handles=[Patch(color=GREY, label="Not in study"), island_handle],
          loc="lower left", fontsize=8.5, frameon=False)
fig.savefig("results/map_model_comparison.png", dpi=220, facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---- Map 2: market types ----
fig, ax = base_map()
for t, c in PALETTE.items():
    sub = africa[africa["type_name"] == t]
    if len(sub):
        sub.plot(ax=ax, color=c, edgecolor=EDGE, linewidth=0.5)
for _, r in islands[islands["type_name"].notna()].iterrows():
    ax.scatter(r.pt.x, r.pt.y, s=60, color=PALETTE[r["type_name"]], edgecolor="#333", linewidth=0.6, zorder=5)
ax.set_title("Four market types in Sub-Saharan Africa", fontsize=12, loc="left")
handles = [Patch(color=c, label=f"{t} ({int((data['type_name'] == t).sum())})") for t, c in PALETTE.items()]
handles += [Patch(color=GREY, label="Not in study"), island_handle]
ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False)
fig.savefig("results/map_market_types.png", dpi=220, facecolor="white", bbox_inches="tight")
plt.close(fig)

print("Saved results/map_model_comparison.png and results/map_market_types.png")

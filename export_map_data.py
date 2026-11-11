"""
Export simplified Africa country borders + aggregated results as one GeoJSON file
for an interactive web map.

Run in ~/paper (geopandas installed):
    python3 export_map_data.py

Reads only aggregated files: results/auc_by_country.csv, results/market_insights.csv
Writes: results/africa_interactive.geojson   (country borders + country-level figures only)
"""
import geopandas as gpd
import pandas as pd

BORDERS = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"

auc = pd.read_csv("results/auc_by_country.csv")
wide = auc.pivot(index="country", columns="strategy", values="auc")
models = pd.DataFrame({
    "iso3": wide.index,
    "auc_shared": wide["XGB_pooled"].round(3).values,
    "auc_country_xgb": wide["XGB_expert"].round(3).values,
    "auc_diff": (wide["XGB_expert"] - wide["XGB_pooled"]).round(4).values,
})

ins = pd.read_csv("results/market_insights.csv")
ins = ins.rename(columns={ins.columns[0]: "iso3"})
prof = ins.groupby("market_type")[["account_rate", "mobile_money_share"]].mean()
names = {}
k = prof["account_rate"].idxmin(); names[k] = "Low-connectivity frontier"; rest = prof.drop(index=k)
k = rest["mobile_money_share"].idxmin(); names[k] = "Bank-led, connected"; rest = rest.drop(index=k)
k = rest["account_rate"].idxmax(); names[k] = "Mobile money mature"; rest = rest.drop(index=k)
for k in rest.index:
    names[k] = "Mobile money growing, basic phones"
ins["market_type_name"] = ins["market_type"].map(names)
keep = ["iso3", "economy", "adults_m", "account_rate", "unbanked_m", "reachable_unbanked_m",
        "reachable_share_of_unbanked", "phone_whatif_pp", "gender_gap_pp",
        "gender_gap_unexplained_pp", "mobile_money_share", "top_barrier", "market_type_name"]
data = ins[keep].merge(models, on="iso3", how="left")
for c in data.columns:
    if data[c].dtype.kind == "f":
        data[c] = data[c].round(3)

world = gpd.read_file(BORDERS)
code = "ADM0_A3" if "ADM0_A3" in world.columns else "ISO_A3"
geo = world[(world["CONTINENT"] == "Africa") | world[code].isin(data["iso3"])][[code, "NAME", "geometry"]]
geo = geo.rename(columns={code: "iso3", "NAME": "name"})
geo["geometry"] = geo.geometry.simplify(0.05, preserve_topology=True)   # smaller file
geo = geo.merge(data, on="iso3", how="left")
geo["in_study"] = geo["economy"].notna()
geo.to_file("results/africa_interactive.geojson", driver="GeoJSON")
print(f"Saved results/africa_interactive.geojson ({len(geo)} countries, {int(geo['in_study'].sum())} in study)")

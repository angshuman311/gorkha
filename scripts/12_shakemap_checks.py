"""Sensitivity checks of the ShakeMap features.

Check A: the mainshock ShakeMap version of 2015-07-02 in place of the version of 2020.
Check B: the ShakeMap of the Mw 7.3 aftershock of 2015-05-12 as more features.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from gorkha import baselines, checks, download, features, graph, metrics, survey_sim
from gorkha.design import TARGET, TRANSFORMS, design_matrix
from gorkha.paths import INTERIM, PROCESSED, RAW, RESULTS

SHAKE = ["mmi", "pga", "pgv", "vs30"]
BANDS = {"mmi": "MMI", "pga": "PGA", "pgv": "PGV", "vs30": "SVEL"}
REALIZATIONS = 200
# The 16 features of the first feature table (2026-10-06).
OLD_FEATURES = ["mmi", "pga", "pgv", "vs30", "elev_mean", "slope_mean", "dpm_cover",
                "dpm_valid_frac", "dpm_mean", "dpm_p90", "ls_mapped", "ls_frac", "dist_epi_km",
                "dist_road_km", "dist_hq_km", "area_km2"]
MODELS = ["ridge", "forest", "ok", "rk", "rk_gdif"]


def ward_means(grid, name: str, wards) -> pd.DataFrame:
    raster = features.shakemap_raster(grid, INTERIM / f"shakemap_{name}.tif")
    out = {}
    for col, band in BANDS.items():
        i = features.SHAKEMAP_BANDS.index(band) + 1
        out[col] = features.zonal(wards.geometry, raster, ["mean"], band=i, all_touched=True)["mean"]
    return pd.DataFrame(out)


def design(df: pd.DataFrame, aftershock: bool) -> pd.DataFrame:
    x = design_matrix(df, OLD_FEATURES)
    if aftershock:
        x["mmi_after"] = df["mmi_after"].to_numpy()
        x["pga_after"] = np.log(df["pga_after"].to_numpy())
        x["pgv_after"] = np.log(df["pgv_after"].to_numpy())
        # Largest intensity of the two events.
        x["mmi_max"] = np.maximum(df["mmi"], df["mmi_after"]).to_numpy()
    return x


def one(scheme, budget, r, x, xy, y, n_b, district, pairs, road, gdif):
    s = survey_sim.survey_mask(scheme, budget, r, pairs, road)
    preds = baselines.predict_all(x, xy, y, s, seed=r, gdif_columns=gdif)
    return [{"scheme": scheme, "budget": budget, "model": m,
             "mae": metrics.evaluate(y, p, s, n_b, district, sd)["mae"]}
            for m, (p, sd) in preds.items() if m in MODELS]


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    grids = download.shakemap_variants()
    wards = features.nodes()
    base = pd.read_parquet(PROCESSED / "wards.parquet")
    pairs = graph.load()["adj_pairs"]
    w = checks.weights(pairs, len(base))
    means = {k: ward_means(g, k, wards) for k, g in grids.items()}
    means["main_2020"] = ward_means(RAW / "shakemap" / "grid.xml", "main_2020", wards)
    base = base.copy()
    base[SHAKE] = means["main_2020"][SHAKE].to_numpy()

    print("correlation of the 2015 version with the 2020 version (mainshock, ward means):")
    print({c: round(float(np.corrcoef(base[c], means["main_2015"][c])[0, 1]), 3) for c in SHAKE})
    print("mean of each version:", {c: (round(float(base[c].mean()), 2),
                                        round(float(means["main_2015"][c].mean()), 2)) for c in SHAKE})

    variants = {}
    variants["2020 mainshock (first feature table)"] = (base, False)
    d = base.copy()
    d[SHAKE] = means["main_2015"][SHAKE].to_numpy()
    variants["2015 mainshock"] = (d, False)
    d = base.copy()
    for c in ("mmi", "pga", "pgv"):
        d[f"{c}_after"] = means["after_2020"][c].to_numpy()
    variants["2020 mainshock + 2020 aftershock"] = (d, True)
    d = variants["2015 mainshock"][0].copy()
    for c in ("mmi", "pga", "pgv"):
        d[f"{c}_after"] = means["after_2015"][c].to_numpy()
    variants["2015 mainshock + 2015 aftershock"] = (d, True)

    y = base[TARGET].to_numpy(dtype=float)
    xy = base[["x_km", "y_km"]].to_numpy()
    rows, tables, by_district = [], [], {}
    for name, (df, aftershock) in variants.items():
        xd = design(df, aftershock)
        x = StandardScaler().fit_transform(xd)
        fit = LinearRegression().fit(x, y)
        res = y - fit.predict(x)
        m = checks.moran(res, w)
        semi = checks.semivariance_by_distance(xy, {"r": res})["r"]
        rows.append({"variant": name, "features": x.shape[1], "R2": fit.score(x, y),
                     "residual Moran I": m["I"], "semivariance 0-5 km": semi.iloc[0],
                     "semivariance 60-80 km": semi.iloc[-2]})
        by_district[name] = pd.Series(res).groupby(base["district_name"].values).mean()

        gdif_names = ["mmi", "elev_mean", "dpm_cover", "dpm_p90"] + (["mmi_after"] if aftershock else [])
        gdif = [xd.columns.get_loc(c) for c in gdif_names]
        shared = (x, xy, y, base["n_buildings"].to_numpy(dtype=float),
                  base["district_name"].to_numpy(), pairs, base["dist_road_km"].to_numpy(), gdif)
        tasks = [(s, b, r) for s in survey_sim.SCHEMES for b in survey_sim.BUDGETS
                 for r in range(REALIZATIONS)]
        out = Parallel(n_jobs=-1, batch_size=8)(delayed(one)(*t, *shared) for t in tasks)
        t = pd.DataFrame([q for part in out for q in part])
        t["variant"] = name
        tables.append(t)

    print("\nlinear regression on all wards:")
    print(pd.DataFrame(rows).set_index("variant").round(4).to_string())
    print("\nmean residual of the linear regression by district:")
    print(pd.DataFrame(by_district).round(3).to_string())

    allt = pd.concat(tables)
    allt.to_parquet(RESULTS / "shakemap_checks.parquet", index=False)
    mean = allt.groupby(["scheme", "variant", "model", "budget"])["mae"].mean().unstack("budget")
    for scheme in ("random", "clustered"):
        print(f"\nmean absolute error, {scheme} survey, {REALIZATIONS} realizations:")
        print(mean.loc[scheme].round(4).to_string())

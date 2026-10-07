"""Sentinel-2 land context values for each ward, and a test of their value.

Source: Sentinel-2 L2A from the Microsoft Planetary Computer (no account necessary).
The images are from October 2015 to March 2016. They are a proxy for an image from before
the earthquakes. The landslide areas of the USGS inventory are not used.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from joblib import Parallel, delayed
from rasterio import features as rio_features
from rasterio.enums import Resampling
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from gorkha import baselines, checks, graph, metrics, survey_sim
from gorkha.design import TARGET, design_matrix
from gorkha.features import LANDSLIDES
from gorkha.paths import CRS_METRIC, PROCESSED, RESULTS

SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
_token = {}
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
SIZE = 1830  # 60 m pixels for a tile of 109.8 km
# Scene classes that show the land surface: vegetation, bare, water, unclassified.
VALID_SCL = [4, 5, 6, 7]
VALUES = ["ndvi", "ndbi", "bright"]
S2_FEATURES = [f"s2_{v}_{s}" for v in VALUES for s in ("mean", "sd")]
REALIZATIONS = 200
MODELS = ["ridge", "mlp", "forest", "ok", "rk"]


def best_items(bbox) -> dict:
    """Item with the smallest cloud cover for each tile."""
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 500,
            "datetime": "2015-10-01/2016-03-31", "query": {"eo:cloud_cover": {"lt": 60}}}
    items = requests.post(SEARCH, json=body, timeout=120).json()["features"]
    best = {}
    for it in items:
        tile, cloud = it["properties"]["s2:mgrs_tile"], it["properties"]["eo:cloud_cover"]
        if tile not in best or cloud < best[tile]["properties"]["eo:cloud_cover"]:
            best[tile] = it
    return best


def read_band(item: dict, band: str) -> tuple:
    if "value" not in _token:  # one access token for all files of the collection
        _token["value"] = requests.get(TOKEN, timeout=60).json()["token"]
    href = item["assets"][band]["href"] + "?" + _token["value"]
    with rasterio.open(href) as src:
        how = Resampling.nearest if band == "SCL" else Resampling.average
        data = src.read(1, out_shape=(SIZE, SIZE), resampling=how).astype("float32")
        transform = src.transform * src.transform.scale(src.width / SIZE, src.height / SIZE)
        return data, transform, src.crs


def tile_sums(item: dict, wards: gpd.GeoDataFrame, slides) -> pd.DataFrame:
    """Sum, sum of squares, and pixel count of each value for each ward in one tile."""
    data = {}
    for band in BANDS:
        data[band], transform, crs = read_band(item, band)
    w = wards.to_crs(crs)
    shapes = [(g, i + 1) for i, g in zip(w.index, w.geometry)]
    label = rio_features.rasterize(shapes, out_shape=(SIZE, SIZE), transform=transform, fill=0,
                                   dtype="int32")
    slide = rio_features.rasterize([(g, 1) for g in slides.to_crs(crs).geometry],
                                   out_shape=(SIZE, SIZE), transform=transform, fill=0,
                                   all_touched=True, dtype="uint8")
    ok = (label > 0) & np.isin(data["SCL"], VALID_SCL) & (slide == 0) & (data["B04"] > 0)
    eps = 1e-6
    value = {
        "ndvi": (data["B08"] - data["B04"]) / (data["B08"] + data["B04"] + eps),
        "ndbi": (data["B11"] - data["B08"]) / (data["B11"] + data["B08"] + eps),
        "bright": (data["B02"] + data["B03"] + data["B04"]) / 30000.0,
    }
    n = len(wards) + 1
    out = {"count": np.bincount(label[ok], minlength=n)}
    for k, v in value.items():
        out[f"{k}_sum"] = np.bincount(label[ok], weights=v[ok], minlength=n)
        out[f"{k}_sq"] = np.bincount(label[ok], weights=v[ok] ** 2, minlength=n)
    return pd.DataFrame(out).iloc[1:].reset_index(drop=True)


def build() -> pd.DataFrame:
    wards = gpd.read_file(PROCESSED / "wards.gpkg")
    slides = gpd.read_file(LANDSLIDES / "Full20170209.shp")
    slides["geometry"] = slides.geometry.buffer(0)
    w, s, e, n = wards.total_bounds
    items = best_items([w, s, e, n])
    total = None
    for tile, item in sorted(items.items()):
        p = item["properties"]
        print(f"tile {tile}: {p['datetime'][:10]}, cloud cover {p['eo:cloud_cover']:.1f}%")
        t = tile_sums(item, wards, slides)
        total = t if total is None else total + t
    c = total["count"].clip(lower=1)
    out = pd.DataFrame({"ward_id": wards["ward_id"], "s2_pixels": total["count"]})
    for k in VALUES:
        mean = total[f"{k}_sum"] / c
        out[f"s2_{k}_mean"] = mean
        out[f"s2_{k}_sd"] = np.sqrt((total[f"{k}_sq"] / c - mean ** 2).clip(lower=0))
    out.to_parquet(PROCESSED / "s2_context.parquet", index=False)
    return out


def one(scheme, budget, r, x, xy, y, n_b, district, pairs, road):
    s = survey_sim.survey_mask(scheme, budget, r, pairs, road)
    preds = baselines.predict_all(x, xy, y, s, seed=r)
    return [{"scheme": scheme, "budget": budget, "model": m,
             "mae": metrics.evaluate(y, p, s, n_b, district, sd)["mae"]}
            for m, (p, sd) in preds.items() if m in MODELS]


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    path = PROCESSED / "s2_context.parquet"
    s2 = pd.read_parquet(path) if path.exists() else build()
    df = pd.read_parquet(PROCESSED / "wards.parquet").merge(s2, on="ward_id")
    area = df["area_km2"] * 1e6 / 3600.0
    print(f"\nwards without Sentinel-2 pixels: {int((df['s2_pixels'] == 0).sum())}, "
          f"median share of valid pixels: {(df['s2_pixels'] / area).clip(upper=1).median():.2f}")
    y = df[TARGET].to_numpy(dtype=float)
    xy = df[["x_km", "y_km"]].to_numpy()
    pairs = graph.load()["adj_pairs"]
    w = checks.weights(pairs, len(df))
    print("correlation with the damage fraction:")
    print(df[S2_FEATURES].corrwith(df[TARGET]).round(3).to_string())

    base = design_matrix(df)
    variants = {"present features": base,
                "present features + Sentinel-2 context": pd.concat([base, df[S2_FEATURES]], axis=1),
                "Sentinel-2 context only": df[S2_FEATURES]}
    rows, tables = [], []
    for name, xd in variants.items():
        x = StandardScaler().fit_transform(xd)
        fit = LinearRegression().fit(x, y)
        res = y - fit.predict(x)
        semi = checks.semivariance_by_distance(xy, {"r": res})["r"]
        rows.append({"variant": name, "features": x.shape[1], "R2": fit.score(x, y),
                     "residual Moran I": checks.moran(res, w)["I"],
                     "semivariance 0-5 km": semi.iloc[0]})
        shared = (x, xy, y, df["n_buildings"].to_numpy(dtype=float),
                  df["district_name"].to_numpy(), pairs, df["dist_road_km"].to_numpy())
        tasks = [(s, b, r) for s in ("random", "clustered") for b in survey_sim.BUDGETS
                 for r in range(REALIZATIONS)]
        out = Parallel(n_jobs=-1, batch_size=8)(delayed(one)(*t, *shared) for t in tasks)
        t = pd.DataFrame([q for part in out for q in part])
        t["variant"] = name
        tables.append(t)
    print("\nlinear regression on all wards (semivariance of the target at 0 to 5 km: 0.0212):")
    print(pd.DataFrame(rows).set_index("variant").round(4).to_string())
    allt = pd.concat(tables)
    allt.to_parquet(RESULTS / "s2_context_checks.parquet", index=False)
    mean = allt.groupby(["scheme", "variant", "model", "budget"])["mae"].mean().unstack("budget")
    for scheme in ("random", "clustered"):
        print(f"\nmean absolute error, {scheme} survey, {REALIZATIONS} realizations:")
        print(mean.loc[scheme].round(4).to_string())

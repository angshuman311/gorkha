"""Sentinel-2 patch data set for the CNN (mode 2: land context).

Each patch is 64 x 64 pixels of 10 m (640 m) with 6 bands. The script reads each map tile
from the Microsoft Planetary Computer and keeps only the patches in the study wards.
Output: data/patches/s2_<tile>.npy (uint16, N x 6 x 64 x 64) and data/patches/s2_index.parquet.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio import features as rio_features
from rasterio.enums import Resampling

from gorkha.paths import PROCESSED, ROOT

SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
PATCH = 64
FULL = 10980
VALID_SCL = [4, 5, 6, 7]
MIN_WARD_SHARE = 0.5   # share of the patch pixels in study wards
MIN_VALID_SHARE = 0.6  # share of the patch pixels without cloud, shadow, or snow
OUT = ROOT / "data" / "patches"


def best_items(bbox) -> dict:
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 1000,
            "datetime": "2015-10-01/2016-03-31", "query": {"eo:cloud_cover": {"lt": 90}}}
    best = {}
    for it in requests.post(SEARCH, json=body, timeout=120).json()["features"]:
        tile, cloud = it["properties"]["s2:mgrs_tile"], it["properties"]["eo:cloud_cover"]
        if tile not in best or cloud < best[tile]["properties"]["eo:cloud_cover"]:
            best[tile] = it
    return best


def read(item, band, token, how):
    with rasterio.open(item["assets"][band]["href"] + "?" + token) as src:
        data = src.read(1, out_shape=(FULL, FULL), resampling=how)
        transform = src.transform * src.transform.scale(src.width / FULL, src.height / FULL)
        return data, transform, src.crs


def blocks(a: np.ndarray) -> np.ndarray:
    """Cut a (FULL, FULL) array into (rows, cols, PATCH, PATCH) blocks."""
    k = FULL // PATCH
    return a[: k * PATCH, : k * PATCH].reshape(k, PATCH, k, PATCH).swapaxes(1, 2)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    wards = gpd.read_file(PROCESSED / "wards.gpkg")
    items = best_items(list(wards.total_bounds))
    token = requests.get(TOKEN, timeout=60).json()["token"]
    index = []
    for tile, item in sorted(items.items()):
        path = OUT / f"s2_{tile}.npy"
        if path.exists():
            print(f"skip   {tile} (on disk)", flush=True)
            continue
        scl, transform, crs = read(item, "SCL", token, Resampling.nearest)
        w = wards.to_crs(crs)
        label = rio_features.rasterize([(g, i + 1) for i, g in zip(w.index, w.geometry)],
                                       out_shape=(FULL, FULL), transform=transform, fill=0,
                                       dtype="int32")
        lab = blocks(label)
        in_ward = (lab > 0).mean(axis=(2, 3))
        valid = blocks(np.isin(scl, VALID_SCL)).mean(axis=(2, 3))
        keep = (in_ward >= MIN_WARD_SHARE) & (valid >= MIN_VALID_SHARE)
        rows, cols = np.nonzero(keep)
        if len(rows) == 0:
            print(f"none   {tile}: no patch in the study wards", flush=True)
            np.save(path, np.zeros((0, len(BANDS), PATCH, PATCH), dtype="uint16"))
            continue
        out = np.zeros((len(rows), len(BANDS), PATCH, PATCH), dtype="uint16")
        for b, band in enumerate(BANDS):
            data, _, _ = read(item, band, token, Resampling.bilinear)
            out[:, b] = blocks(data)[rows, cols]
        np.save(path, out)
        # The ward of a patch is the ward of its center pixel, or the most frequent ward.
        centre = lab[rows, cols, PATCH // 2, PATCH // 2]
        for i, (r, c) in enumerate(zip(rows, cols)):
            node = centre[i]
            if node == 0:
                ids = lab[r, c][lab[r, c] > 0]
                node = np.bincount(ids).argmax()
            x, y = transform * ((c + 0.5) * PATCH, (r + 0.5) * PATCH)
            index.append({"tile": tile, "i": i, "node": int(node) - 1, "x": x, "y": y,
                          "ward_share": float(in_ward[r, c]), "valid_share": float(valid[r, c]),
                          "date": item["properties"]["datetime"][:10]})
        pd.DataFrame(index).to_parquet(OUT / f"s2_index_part_{tile}.parquet", index=False)
        print(f"done   {tile} ({item['properties']['datetime'][:10]}): {len(rows)} patches, "
              f"{out.nbytes / 1e9:.2f} GB", flush=True)
        index = []
    parts = [pd.read_parquet(p) for p in sorted(OUT.glob("s2_index_part_*.parquet"))]
    full = pd.concat(parts, ignore_index=True)
    full["ward_id"] = wards["ward_id"].to_numpy()[full["node"].to_numpy()]
    full.to_parquet(OUT / "s2_index.parquet", index=False)
    per_ward = full.groupby("node").size().reindex(range(len(wards)), fill_value=0)
    print(f"patches: {len(full)}, wards without a patch: {int((per_ward == 0).sum())}, "
          f"median patches for each ward: {per_ward.median():.0f}")

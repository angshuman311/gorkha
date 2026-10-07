"""Second pass for the Sentinel-2 patches: wards without a patch get images of other dates.

The first pass (16_s2_patches.py) uses one image for each map tile. At the edge of a
satellite pass, that image has no data for a part of the tile.
Output: data/patches/s2_fill_<k>.npy and new rows in data/patches/s2_index.parquet.
"""

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio import features as rio_features
from rasterio.enums import Resampling

from gorkha.paths import PROCESSED

spec = importlib.util.spec_from_file_location("p16", Path(__file__).with_name("16_s2_patches.py"))
p16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p16)

if __name__ == "__main__":
    wards = gpd.read_file(PROCESSED / "wards.gpkg")
    index = pd.read_parquet(p16.OUT / "s2_index.parquet")
    index = index[~index["tile"].str.startswith("fill")]
    missing = sorted(set(range(len(wards))) - set(index["node"]))
    print(f"wards without a patch: {len(missing)}", flush=True)
    box = list(wards.iloc[missing].total_bounds)
    body = {"collections": ["sentinel-2-l2a"], "bbox": box, "limit": 500,
            "datetime": "2015-10-01/2016-04-30", "query": {"eo:cloud_cover": {"lt": 40}}}
    items = requests.post(p16.SEARCH, json=body, timeout=120).json()["features"]
    items.sort(key=lambda it: it["properties"]["eo:cloud_cover"])
    token = requests.get(p16.TOKEN, timeout=60).json()["token"]
    new_rows, k = [], 0
    for item in items:
        if not missing:
            break
        scl, transform, crs = p16.read(item, "SCL", token, Resampling.nearest)
        w = wards.iloc[missing].to_crs(crs)
        label = rio_features.rasterize([(g, i + 1) for i, g in zip(w.index, w.geometry)],
                                       out_shape=(p16.FULL, p16.FULL), transform=transform,
                                       fill=0, dtype="int32")
        lab = p16.blocks(label)
        in_ward = (lab > 0).mean(axis=(2, 3))
        valid = p16.blocks(np.isin(scl, p16.VALID_SCL)).mean(axis=(2, 3))
        rows, cols = np.nonzero((in_ward >= p16.MIN_WARD_SHARE) & (valid >= p16.MIN_VALID_SHARE))
        tile, date = item["properties"]["s2:mgrs_tile"], item["properties"]["datetime"][:10]
        if len(rows) == 0:
            print(f"none   {tile} {date}", flush=True)
            continue
        out = np.zeros((len(rows), len(p16.BANDS), p16.PATCH, p16.PATCH), dtype="uint16")
        for b, band in enumerate(p16.BANDS):
            data, _, _ = p16.read(item, band, token, Resampling.bilinear)
            out[:, b] = p16.blocks(data)[rows, cols]
        name = f"fill{k}"
        np.save(p16.OUT / f"s2_{name}.npy", out)
        for i, (r, c) in enumerate(zip(rows, cols)):
            ids = lab[r, c][lab[r, c] > 0]
            node = int(np.bincount(ids).argmax()) - 1
            x, y = transform * ((c + 0.5) * p16.PATCH, (r + 0.5) * p16.PATCH)
            new_rows.append({"tile": name, "i": i, "node": node, "x": x, "y": y,
                             "ward_share": float(in_ward[r, c]), "valid_share": float(valid[r, c]),
                             "date": date, "ward_id": wards["ward_id"].iat[node]})
        got = {r["node"] for r in new_rows}
        missing = [m for m in missing if m not in got]
        print(f"done   {tile} {date}: {len(rows)} patches, wards still without a patch: {len(missing)}",
              flush=True)
        k += 1
    full = pd.concat([index, pd.DataFrame(new_rows)], ignore_index=True)
    full.to_parquet(p16.OUT / "s2_index.parquet", index=False)
    print(f"patches: {len(full)}, wards with a patch: {full['node'].nunique()} of {len(wards)}")

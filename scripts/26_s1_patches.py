"""Sentinel-1 patch data set for the CNN (mode 1: damage evidence).

Input: the ASF HyP3 coherence products in data/raw/s1_coherence (one zip for each scene
pair). Each product has a coherence image (*_corr.tif) and an amplitude image (*_amp.tif)
at 40 m.
The script makes three mosaics of the coherence (before the first earthquake, across the
first earthquake, across the second earthquake) and two of the amplitude (the reference
and the secondary scene of the pairs across the first earthquake). Where products overlap,
the mosaic keeps the pixel with the larger coherence of the "before" pair, because that is
the pair with the smaller noise.
Then it cuts patches of 32 x 32 pixels (1.28 km) with 5 channels for each ward.
Output: data/patches/s1_patches.npy (float16, N x 5 x 32 x 32), s1_index.parquet.
"""

import re
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

from gorkha.paths import CRS_METRIC, PROCESSED, RAW, ROOT

SRC = RAW / "s1_coherence"
OUT = ROOT / "data" / "patches"
PATCH = 32
RES = 40.0
MIN_WARD_SHARE = 0.5
MIN_VALID_SHARE = 0.6
CHANNELS = ["coh_pre", "coh_main", "coh_after", "amp_before", "amp_after"]


def pair_kind(name: str) -> str:
    d1, d2 = re.findall(r"(\d{8})T\d{6}", name)[:2]
    if d2 <= "20150424":
        return "pre"
    return "main" if d1 <= "20150424" else "after"


def unzip_all() -> list:
    products = []
    for z in sorted(SRC.glob("*.zip")):
        folder = SRC / z.stem
        if not folder.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(SRC)
        corr = next(folder.glob("*_corr.tif"), None)
        amp = next(folder.glob("*_amp.tif"), None)
        if corr is not None:
            products.append({"kind": pair_kind(z.stem), "corr": corr, "amp": amp, "name": z.stem})
    return products


def grid(wards) -> tuple:
    x0, y0, x1, y1 = wards.total_bounds
    width = int(np.ceil((x1 - x0) / RES / PATCH)) * PATCH
    height = int(np.ceil((y1 - y0) / RES / PATCH)) * PATCH
    transform = rasterio.transform.from_origin(x0, y1, RES, RES)
    return transform, width, height


def mosaic(paths: list, transform, width, height, how="max") -> np.ndarray:
    """Reproject each product to the common grid and combine them (maximum or first valid)."""
    out = np.full((height, width), np.nan, dtype="float32")
    for p in paths:
        with rasterio.open(p) as src:
            data = src.read(1).astype("float32")
            nodata = src.nodata if src.nodata is not None else 0.0
            data[data == nodata] = np.nan
            dst = np.full((height, width), np.nan, dtype="float32")
            reproject(data, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=CRS_METRIC, src_nodata=np.nan,
                      dst_nodata=np.nan, resampling=Resampling.average)
        if how == "max":
            out = np.fmax(out, dst)
        else:
            out = np.where(np.isnan(out), dst, out)
    return out


def blocks(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[0] // PATCH, a.shape[1] // PATCH
    return a[: h * PATCH, : w * PATCH].reshape(h, PATCH, w, PATCH).swapaxes(1, 2)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    wards = gpd.read_file(PROCESSED / "wards.gpkg").to_crs(CRS_METRIC)
    products = unzip_all()
    print(f"products: {len(products)}", {k: sum(p['kind'] == k for p in products) for k in ('pre', 'main', 'after')})
    transform, width, height = grid(wards)
    layers = {}
    for kind in ("pre", "main", "after"):
        layers[f"coh_{kind}"] = mosaic([p["corr"] for p in products if p["kind"] == kind],
                                       transform, width, height)
    # The amplitude product of a pair has the two scenes. HyP3 gives one amplitude image for
    # the pair (the reference scene). The "before" amplitude is the reference of the main
    # pairs, and the "after" amplitude is the reference of the pairs across the second
    # earthquake (scenes between the two earthquakes).
    amp_main = [p["amp"] for p in products if p["kind"] == "main" and p["amp"] is not None]
    amp_after = [p["amp"] for p in products if p["kind"] == "after" and p["amp"] is not None]
    layers["amp_before"] = np.log1p(mosaic(amp_main, transform, width, height, how="first"))
    layers["amp_after"] = np.log1p(mosaic(amp_after, transform, width, height, how="first"))
    for k, v in layers.items():
        print(f"{k}: valid share {np.isfinite(v).mean():.3f}, mean {np.nanmean(v):.3f}")
    np.savez_compressed(OUT / "s1_layers.npz", transform=np.array(transform)[:6], **layers)

    label = rio_features.rasterize([(g, i + 1) for i, g in zip(wards.index, wards.geometry)],
                                   out_shape=(height, width), transform=transform, fill=0, dtype="int32")
    lab = blocks(label)
    in_ward = (lab > 0).mean(axis=(2, 3))
    valid = blocks(np.isfinite(layers["coh_main"])).mean(axis=(2, 3))
    rows, cols = np.nonzero((in_ward >= MIN_WARD_SHARE) & (valid >= MIN_VALID_SHARE))
    stack = np.stack([blocks(layers[c])[rows, cols] for c in CHANNELS], axis=1)
    stack = np.nan_to_num(stack, nan=0.0).astype("float16")
    np.save(OUT / "s1_patches.npy", stack)
    index = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        ids = lab[r, c][lab[r, c] > 0]
        node = int(np.bincount(ids).argmax()) - 1
        x, y = transform * ((c + 0.5) * PATCH, (r + 0.5) * PATCH)
        index.append({"i": i, "node": node, "x": x, "y": y, "ward_share": float(in_ward[r, c]),
                      "valid_share": float(valid[r, c]), "ward_id": wards["ward_id"].iat[node]})
    index = pd.DataFrame(index)
    index.to_parquet(OUT / "s1_index.parquet", index=False)
    per_ward = index.groupby("node").size().reindex(range(len(wards)), fill_value=0)
    print(f"patches: {len(index)} ({stack.nbytes / 1e9:.2f} GB), wards without a patch: "
          f"{int((per_ward == 0).sum())}, median patches for each ward: {per_ward.median():.0f}")

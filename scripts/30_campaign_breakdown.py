"""Where is the error? Campaigns with the estimate of each ward kept, then three breakdowns:
error by damage class, building-weighted error, and the error of the grade 4 or 5 building
count in each district.

Usage: python scripts/30_campaign_breakdown.py [--levels 0,100] [--image FILE]
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gorkha import campaign, graph
from gorkha.design import TARGET, design_matrix
from gorkha.paths import PROCESSED, RESULTS

CLASSES = {"high (above 0.8)": (0.8, 1.01), "middle (0.5 to 0.8)": (0.5, 0.8), "low (below 0.5)": (-0.01, 0.5)}
ROUNDS = [1, 3, 7, 16]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="0,100")
    parser.add_argument("--image", default="s2_embedding_ssl4eo_ft.parquet")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--name", default="mode2")
    args = parser.parse_args()
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    g = graph.load()
    pairs = g["adj_pairs"]
    edges = np.unique(np.vstack([pairs, g["knn_pairs"]]), axis=0)
    x = StandardScaler().fit_transform(design_matrix(df)).astype("float32")
    emb = pd.read_parquet(PROCESSED / args.image)
    x_sherpa = np.hstack([x, emb.drop(columns="ward_id").to_numpy(dtype="float32")])
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    n_b = df["n_buildings"].to_numpy(dtype=float)
    district = df["district_name"].to_numpy()
    start = [int(i) for i in df.groupby("district_name")["dist_hq_km"].idxmin()]
    with np.load(PROCESSED / "roads_levels.npz") as f:
        matrices = {k: f[k].astype(float) for k in f.files}

    rows, t0 = [], time.time()
    for level in [int(v) for v in args.levels.split(",")]:
        travel = matrices[f"t_{level}_0"]
        for method in ("random", "kriging", "sherpa"):
            rec = campaign.run(method, x, xy, y, pairs, edges, travel, start, args.rounds, seed=0,
                               x_sherpa=x_sherpa, keep_estimates=True)
            print(f"level {level} {method}: {time.time() - t0:.0f} s", flush=True)
            for r in rec:
                est, s = np.array(r["estimate"]), np.array(r["surveyed"])
                u = ~s
                err = np.abs(est - y)
                row = {"level": level, "method": method, "round": r["round"], "cost_hours": r["cost_hours"],
                       "mae": float(err[u].mean()),
                       "mae_weighted": float((err[u] * n_b[u]).sum() / n_b[u].sum())}
                for name, (lo, hi) in CLASSES.items():
                    m = u & (y > lo) & (y <= hi)
                    row[f"mae {name}"] = float(err[m].mean()) if m.any() else np.nan
                count_est = np.where(s, y, est) * n_b
                by = pd.DataFrame({"d": district, "e": count_est, "t": y * n_b}).groupby("d").sum()
                row["district count error"] = float((np.abs(by["e"] - by["t"]) / by["t"]).mean())
                rows.append(row)
    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS / f"campaign_breakdown_{args.name}.parquet", index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in res.columns if c.startswith("mae") or c == "district count error"]
    for level in res["level"].unique():
        print(f"\nblockage {level}%, rounds {ROUNDS}:")
        t = res[(res["level"] == level) & res["round"].isin(ROUNDS)].set_index(["method", "round"])[cols]
        print(t.round(3).to_string())

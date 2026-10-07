"""Estimator against selection: a 2 by 2 comparison with the same labels.

Rows: the estimator (regression kriging, GORKHA). Columns: the surveyed wards (the random
campaign, the GORKHA campaign). The surveyed sets come from the campaign records that kept
them (scripts/24_campaign.py --keep).
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gorkha import campaign, graph
from gorkha.design import TARGET, design_matrix
from gorkha.paths import PROCESSED, RESULTS

ROUNDS = [1, 3, 7, 16]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="campaign2_mode2.parquet")
    parser.add_argument("--image", default="s2_embedding_ssl4eo_ft.parquet")
    parser.add_argument("--levels", default="0,obs")
    args = parser.parse_args()
    res = pd.read_parquet(RESULTS / args.file)
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    g = graph.load()
    pairs = g["adj_pairs"]
    edges = np.unique(np.vstack([pairs, g["knn_pairs"]]), axis=0)
    x = StandardScaler().fit_transform(design_matrix(df)).astype("float32")
    emb = pd.read_parquet(PROCESSED / args.image)
    x_g = np.hstack([x, emb.drop(columns="ward_id").to_numpy(dtype="float32")])
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    rows = []
    res["level"] = res["level"].astype(str)
    for level in args.levels.split(","):
        for sel in ("random", "gorkha"):
            d = res[(res["level"] == level) & (res["method"] == sel) & (res["draw"] == 0) & res["round"].isin(ROUNDS)]
            seed = d["seed"].min()
            d = d[d["seed"] == seed]
            for r in d.itertuples():
                s = np.array(r.surveyed)
                mk, _, _ = campaign.regression_kriging(x, xy, y, s)
                mg, _, _ = campaign.gorkha_estimate(x_g, xy, y, s, edges, seed=int(r.round))
                for est, m in (("regression kriging", mk), ("GORKHA", mg)):
                    rows.append({"level": level, "selection": sel, "estimator": est, "round": r.round,
                                 "wards": int(s.sum()), "mae": float(np.abs(m - y)[~s].mean())})
                print(f"level {level} selection {sel} round {r.round} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "ablation_2x2.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["level", "round"], columns=["selection", "estimator"], values="mae").round(4).to_string())

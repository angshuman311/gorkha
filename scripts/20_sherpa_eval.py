"""Test of the SHERPA model with one protocol and three survey schemes.

In each run, each model learns only from the surveyed wards of the run.
Models: mean, ordinary kriging, regression kriging, GNN on the feature table, and the GNN
with the image vector of each ward (CNN on Sentinel-2 patches, pretrained without labels).
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from gorkha import baselines, gnn, graph, survey_sim
from gorkha.design import TARGET, design_matrix
from gorkha.paths import PROCESSED, RESULTS


def regression_kriging(x, xy, y, s):
    alpha = RidgeCV(alphas=baselines.RIDGE_ALPHAS).fit(x[s], y[s]).alpha_
    trend = Ridge(alpha=alpha).fit(x[s], y[s]).predict(x)
    r, _ = baselines.krige(xy[s], y[s] - trend[s], xy)
    return np.clip(trend + r, 0, 1)


def gnn_estimate(x, xy, y, s, edges, seed):
    n = len(y)
    everything = np.arange(n)
    rng = np.random.default_rng([7, seed])
    tasks = []
    for _ in range(12):
        hide = s & (rng.random(n) < 0.4)
        if hide.sum() and (s & ~hide).sum() >= 3:
            tasks.append(gnn.make_task(everything, s & ~hide, hide))
    if not tasks:
        return np.full(n, y[s].mean())
    model = gnn.train(tasks, x, y, edges, xy, epochs=200, seed=seed)
    return gnn.predict(model, gnn.make_task(everything, s, ~s), x, y, edges, xy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--embedding", default="s2_embedding.parquet")
    parser.add_argument("--out", default="sherpa_eval_s2.parquet")
    args = parser.parse_args()
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    emb = pd.read_parquet(PROCESSED / args.embedding)
    assert (emb["ward_id"].to_numpy() == df["ward_id"].to_numpy()).all()
    g = graph.load()
    pairs = g["adj_pairs"]
    edges = np.unique(np.vstack([pairs, g["knn_pairs"]]), axis=0)
    x_tab = StandardScaler().fit_transform(design_matrix(df)).astype("float32")
    x_img = emb.drop(columns="ward_id").to_numpy(dtype="float32")
    x_all = np.hstack([x_tab, x_img])
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    road = df["dist_road_km"].to_numpy()

    rows, start = [], time.time()
    for scheme in survey_sim.SCHEMES:
        for budget in survey_sim.BUDGETS:
            for r in range(args.runs):
                s = survey_sim.survey_mask(scheme, budget, r, pairs, road)
                ok, _ = baselines.krige(xy[s], y[s], xy)
                preds = {
                    "mean": np.full(len(y), y[s].mean()),
                    "ok": np.clip(ok, 0, 1),
                    "rk": regression_kriging(x_tab, xy, y, s),
                    "rk_img": regression_kriging(x_all, xy, y, s),
                    "gnn": gnn_estimate(x_tab, xy, y, s, edges, r),
                    "gnn_img": gnn_estimate(x_all, xy, y, s, edges, r),
                    "gnn_img_only": gnn_estimate(x_img, xy, y, s, edges, r),
                }
                for model, p in preds.items():
                    rows.append({"scheme": scheme, "budget": budget, "realization": r,
                                 "model": model, "mae": float(np.abs(p[~s] - y[~s]).mean())})
        print(f"{scheme} complete, {time.time() - start:.0f} s", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS / args.out, index=False)
    pd.set_option("display.width", 200)
    order = ["mean", "ok", "rk", "rk_img", "gnn", "gnn_img", "gnn_img_only"]
    table = res.pivot_table(index=["scheme", "model"], columns="budget", values="mae")
    print(f"\nmean absolute error, {args.runs} runs for each setting:")
    print(table.reindex(order, level=1).round(4).to_string())
    paired = res.pivot_table(index=["scheme", "budget", "realization"], columns="model", values="mae")
    for a, b in (("gnn_img", "gnn"), ("gnn_img", "ok"), ("gnn", "ok")):
        diff = (paired[a] - paired[b]).groupby(level=[0, 1])
        print(f"\n{a} minus {b}: mean difference, and (standard error)")
        print((diff.mean().round(4).astype(str) + " (" + diff.sem().round(4).astype(str) + ")")
              .unstack().to_string())

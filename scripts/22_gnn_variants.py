"""Compare GNN variants with ordinary kriging on the feature table (no images).

Usage: python scripts/22_gnn_variants.py [--runs N]
Variants: the GNN without and with the distance layer, and a group of 5 models.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gorkha import baselines, gnn, graph, survey_sim
from gorkha.design import TARGET, design_matrix
from gorkha.paths import PROCESSED, RESULTS


def tasks_for(s, n, seed, count=12):
    rng = np.random.default_rng([7, seed])
    everything, tasks = np.arange(n), []
    for _ in range(count):
        hide = s & (rng.random(n) < 0.4)
        if hide.sum() and (s & ~hide).sum() >= 3:
            tasks.append(gnn.make_task(everything, s & ~hide, hide))
    return tasks


def estimate(x, xy, y, s, edges, seed, kernel, members=1):
    n = len(y)
    tasks = tasks_for(s, n, seed)
    if not tasks:
        return np.full(n, y[s].mean()), np.full(n, y[s].std())
    test = gnn.make_task(np.arange(n), s, ~s)
    out = [gnn.predict(gnn.train(tasks, x, y, edges, xy, epochs=200, seed=seed * 10 + m,
                                 kernel=kernel), test, x, y, edges, xy) for m in range(members)]
    mean = np.mean([o[0] for o in out], axis=0)
    # Variance of a group: mean of the variances plus the variance of the estimates.
    var = np.mean([o[1] ** 2 for o in out], axis=0) + np.var([o[0] for o in out], axis=0)
    return mean, np.sqrt(var)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    g = graph.load()
    pairs = g["adj_pairs"]
    edges = np.unique(np.vstack([pairs, g["knn_pairs"]]), axis=0)
    x = StandardScaler().fit_transform(design_matrix(df)).astype("float32")
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    road = df["dist_road_km"].to_numpy()
    rows, start = [], time.time()
    for scheme in ("random", "clustered"):
        for budget in survey_sim.BUDGETS:
            for r in range(args.runs):
                s = survey_sim.survey_mask(scheme, budget, r, pairs, road)
                ok, ok_var = baselines.krige(xy[s], y[s], xy)
                preds = {
                    "ok": (np.clip(ok, 0, 1), np.sqrt(ok_var)),
                    "gnn_plain": estimate(x, xy, y, s, edges, r, kernel=False),
                    "gnn_distance": estimate(x, xy, y, s, edges, r, kernel=True),
                    "gnn_distance_x5": estimate(x, xy, y, s, edges, r, kernel=True, members=5),
                }
                for model, (p, sd) in preds.items():
                    err = np.abs(p[~s] - y[~s])
                    rows.append({"scheme": scheme, "budget": budget, "realization": r,
                                 "model": model, "mae": float(err.mean()),
                                 "coverage_90": float((err <= 1.6449 * sd[~s]).mean())})
        print(f"{scheme} complete, {time.time() - start:.0f} s", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS / "gnn_variants.parquet", index=False)
    pd.set_option("display.width", 200)
    for metric in ("mae", "coverage_90"):
        print(f"\n{metric}, {args.runs} runs for each setting:")
        print(res.pivot_table(index=["scheme", "model"], columns="budget", values=metric)
              .round(4).to_string())

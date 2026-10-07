"""GNN on the hand-crafted features (rung 3), Protocols A and B, with kriging references.

Protocol A: the model trains only on the labels of the simulated survey.
Protocol B: the model trains on fully labeled districts and is tested on held-out districts
with a sparse survey.
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

SCHEMES = ["random", "clustered"]
# Fixed district folds for Protocol B. Each fold is held out one time.
FOLDS = [["Gorkha", "Dhading", "Makwanpur"], ["Nuwakot", "Rasuwa", "Kavrepalanchok"],
         ["Sindhupalchok", "Dolakha"], ["Ramechhap", "Okhaldhunga", "Sindhuli"]]


def region_mask(scheme, budget, r, nodes, pairs, road, n):
    sub = gnn.induced_pairs(pairs, nodes, n)
    return survey_sim.survey_mask(scheme, budget, r, sub, road[nodes])


def references(task, x, xy, y):
    """Ordinary kriging and regression kriging for the same task."""
    nd, v = task.nodes, task.visible
    alpha = RidgeCV(alphas=baselines.RIDGE_ALPHAS).fit(x[nd][v], y[nd][v]).alpha_
    trend = Ridge(alpha=alpha).fit(x[nd][v], y[nd][v]).predict(x[nd])
    r, _ = baselines.krige(xy[nd][v], y[nd][v] - trend[v], xy[nd])
    return {"mean": np.full(len(nd), y[nd][v].mean()), "ok": task.ok,
            "rk": np.clip(trend + r, 0, 1)}


def score(rows, protocol, scheme, budget, r, task, preds, y):
    t = task.target
    for model, p in preds.items():
        rows.append({"protocol": protocol, "scheme": scheme, "budget": budget, "realization": r,
                     "model": model, "mae": float(np.abs(p[t] - y[task.nodes][t]).mean())})


def protocol_a(x, xy, y, pairs, edges, road, runs, rows):
    n = len(y)
    everything = np.arange(n)
    for scheme in SCHEMES:
        for budget in survey_sim.BUDGETS:
            for r in range(runs):
                s = survey_sim.survey_mask(scheme, budget, r, pairs, road)
                test = gnn.make_task(everything, s, ~s, xy, y)
                rng = np.random.default_rng([7, r])
                tasks = []
                for _ in range(12):
                    hide = s & (rng.random(n) < 0.4)
                    if hide.sum() == 0 or (s & ~hide).sum() < 3:
                        continue
                    tasks.append(gnn.make_task(everything, s & ~hide, hide))
                preds = references(test, x, xy, y)
                if tasks:
                    model = gnn.train(tasks, x, y, edges, xy, epochs=200, seed=r)
                    preds["gnn"] = gnn.predict(model, test, x, y, edges, xy)
                else:
                    preds["gnn"] = np.full(n, y[s].mean())
                score(rows, "A", scheme, budget, r, test, preds, y)


def protocol_b(x, xy, y, pairs, edges, road, district, runs, rows):
    n = len(y)
    for k, held in enumerate(FOLDS):
        test_nodes = np.where(np.isin(district, held))[0]
        train_nodes = np.where(~np.isin(district, held))[0]
        tasks = []
        for i in range(60):
            scheme = SCHEMES[i % 2]
            budget = survey_sim.BUDGETS[i % len(survey_sim.BUDGETS)]
            v = region_mask(scheme, budget, 1000 + i, train_nodes, pairs, road, n)
            tasks.append(gnn.make_task(train_nodes, v, ~v))
        model = gnn.train(tasks, x, y, edges, xy, epochs=1200, layers=4, seed=k)
        for scheme in SCHEMES:
            for budget in survey_sim.BUDGETS:
                for r in range(runs):
                    v = region_mask(scheme, budget, r, test_nodes, pairs, road, n)
                    if v.sum() < 3:
                        continue
                    test = gnn.make_task(test_nodes, v, ~v, xy, y)
                    preds = references(test, x, xy, y)
                    preds["gnn"] = gnn.predict(model, test, x, y, edges, xy)
                    score(rows, "B", scheme, budget, r, test, preds, y)
        print(f"fold {k + 1} of {len(FOLDS)} complete: held out {held}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    g = graph.load()
    pairs = g["adj_pairs"]
    # Message passing uses the adjacency edges and the nearest neighbor edges.
    edges = np.unique(np.vstack([pairs, g["knn_pairs"]]), axis=0)
    x = StandardScaler().fit_transform(design_matrix(df)).astype("float32")
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    road = df["dist_road_km"].to_numpy()
    print("device:", gnn.DEVICE, flush=True)

    rows, start = [], time.time()
    protocol_a(x, xy, y, pairs, edges, road, args.runs, rows)
    print(f"protocol A complete, {time.time() - start:.0f} s", flush=True)
    protocol_b(x, xy, y, pairs, edges, road, df["district_name"].to_numpy(), args.runs, rows)
    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS / "gnn_rung3.parquet", index=False)
    pd.set_option("display.width", 200)
    table = res.pivot_table(index=["protocol", "scheme", "model"], columns="budget", values="mae")
    print(f"\nmean absolute error, {args.runs} runs for each setting, {time.time() - start:.0f} s:")
    print(table.round(4).to_string())
    paired = res.pivot_table(index=["protocol", "scheme", "budget", "realization"],
                             columns="model", values="mae")
    print("\nshare of runs where the GNN has a smaller error than ordinary kriging:")
    print((paired["gnn"] < paired["ok"]).groupby(level=[0, 1, 2]).mean().unstack().round(2).to_string())

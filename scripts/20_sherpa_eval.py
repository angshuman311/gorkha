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
    """Ridge trend and kriging of the leave-one-out residuals. Return estimate and sd."""
    alpha = RidgeCV(alphas=baselines.RIDGE_ALPHAS).fit(x[s], y[s]).alpha_
    trend = Ridge(alpha=alpha).fit(x[s], y[s]).predict(x)
    r, var = baselines.krige(xy[s], baselines.ridge_loo_residuals(x[s], y[s], alpha), xy)
    return np.clip(trend + r, 0, 1), np.sqrt(var)


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
        return np.full(n, y[s].mean()), np.full(n, y[s].std())
    model = gnn.train(tasks, x, y, edges, xy, epochs=200, seed=seed)
    return gnn.predict(model, gnn.make_task(everything, s, ~s), x, y, edges, xy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--embeddings", default="cnn=s2_embedding.parquet",
                        help="name=file pairs with commas between them")
    parser.add_argument("--out", default="sherpa_eval_s2.parquet")
    parser.add_argument("--events", type=int, default=2, choices=[1, 2],
                        help="number of earthquakes before the survey")
    args = parser.parse_args()
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    g = graph.load()
    pairs = g["adj_pairs"]
    edges = np.unique(np.vstack([pairs, g["knn_pairs"]]), axis=0)
    x_tab = StandardScaler().fit_transform(design_matrix(df, n_events=args.events)).astype("float32")
    images = {}
    for pair in args.embeddings.split(","):
        name, file = pair.split("=")
        emb = pd.read_parquet(PROCESSED / file)
        assert (emb["ward_id"].to_numpy() == df["ward_id"].to_numpy()).all()
        images[name] = np.hstack([x_tab, emb.drop(columns="ward_id").to_numpy(dtype="float32")])
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    road = df["dist_road_km"].to_numpy()

    rows, start = [], time.time()
    for scheme in survey_sim.SCHEMES:
        for budget in survey_sim.BUDGETS:
            for r in range(args.runs):
                s = survey_sim.survey_mask(scheme, budget, r, pairs, road)
                ok, ok_var = baselines.krige(xy[s], y[s], xy)
                preds = {
                    "mean": (np.full(len(y), y[s].mean()), np.full(len(y), y[s].std(ddof=1))),
                    "ok": (np.clip(ok, 0, 1), np.sqrt(ok_var)),
                    "rk": regression_kriging(x_tab, xy, y, s),
                    "gnn": gnn_estimate(x_tab, xy, y, s, edges, r),
                }
                for name, x_all in images.items():
                    preds[f"rk_{name}"] = regression_kriging(x_all, xy, y, s)
                    preds[f"gnn_{name}"] = gnn_estimate(x_all, xy, y, s, edges, r)
                for model, (p, sd) in preds.items():
                    err = np.abs(p[~s] - y[~s])
                    rows.append({"scheme": scheme, "budget": budget, "realization": r,
                                 "model": model, "mae": float(err.mean()),
                                 "coverage_90": float((err <= 1.6449 * sd[~s]).mean()),
                                 "mean_sd": float(sd[~s].mean())})
        print(f"{scheme} complete, {time.time() - start:.0f} s", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RESULTS / args.out, index=False)
    pd.set_option("display.width", 200)
    order = (["mean", "ok", "rk"] + [f"rk_{n}" for n in images] + ["gnn"]
             + [f"gnn_{n}" for n in images])
    table = res.pivot_table(index=["scheme", "model"], columns="budget", values="mae")
    print(f"\nmean absolute error, {args.runs} runs for each setting:")
    print(table.reindex(order, level=1).round(4).to_string())
    cover = res.pivot_table(index=["scheme", "model"], columns="budget", values="coverage_90")
    print("\nshare of wards with the true value in the 90% interval (correct: 0.90):")
    print(cover.reindex(order, level=1).round(3).to_string())
    paired = res.pivot_table(index=["scheme", "budget", "realization"], columns="model", values="mae")
    for a, b in [("gnn", "ok")] + [(f"gnn_{n}", c) for n in images for c in ("gnn", "ok")]:
        diff = (paired[a] - paired[b]).groupby(level=[0, 1])
        print(f"\n{a} minus {b}: mean difference, and (standard error)")
        print((diff.mean().round(4).astype(str) + " (" + diff.sem().round(4).astype(str) + ")")
              .unstack().to_string())

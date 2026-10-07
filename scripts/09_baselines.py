"""Monte Carlo evaluation of the baselines.

Usage: python scripts/09_baselines.py [--realizations N] [--target frac_g45|mean_grade]
"""

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from gorkha import baselines, graph, metrics, survey_sim
from gorkha.design import TARGET, design_matrix
from gorkha.paths import PROCESSED, RESULTS

# Permitted range of each target.
LIMITS = {"frac_g45": (0.0, 1.0), "mean_grade": (1.0, 5.0)}


def one(scheme, budget, r, x, xy, y, n_buildings, district, pairs, dist_road, gdif_columns,
        limits, counts):
    surveyed = survey_sim.survey_mask(scheme, budget, r, pairs, dist_road)
    rows = []
    preds = baselines.predict_all(x, xy, y, surveyed, seed=r, gdif_columns=gdif_columns,
                                  limits=limits)
    for model, (pred, sd) in preds.items():
        rows.append({
            "scheme": scheme, "budget": budget, "realization": r, "model": model,
            "n_surveyed": int(surveyed.sum()),
            **metrics.evaluate(y, pred, surveyed, n_buildings, district, sd, counts=counts),
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--target", choices=list(LIMITS), default=TARGET)
    args = parser.parse_args()
    fraction = args.target == TARGET

    df = pd.read_parquet(PROCESSED / "wards.parquet")
    pairs = graph.load()["adj_pairs"]
    # The scaler uses the features of all wards. It uses no label.
    design = design_matrix(df)
    x = StandardScaler().fit_transform(design)
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[args.target].to_numpy(dtype=float)
    gdif_columns = [design.columns.get_loc(c) for c in baselines.GDIF_FEATURES]
    shared = (x, xy, y, df["n_buildings"].to_numpy(dtype=float),
              df["district_name"].to_numpy(), pairs, df["dist_road_km"].to_numpy(),
              gdif_columns, LIMITS[args.target], fraction)

    tasks = [(s, b, r) for s in survey_sim.SCHEMES for b in survey_sim.BUDGETS
             for r in range(args.realizations)]
    start = time.time()
    rows = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(one)(*t, *shared) for t in tasks)
    res = pd.DataFrame([row for part in rows for row in part])
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = "baselines.parquet" if fraction else f"baselines_{args.target}.parquet"
    res.to_parquet(RESULTS / name, index=False)
    print(f"target {args.target}: {len(tasks)} survey realizations, {len(res)} rows, "
          f"{time.time() - start:.0f} s, file {name}")
    print(f"variance of the target over the {len(y)} wards: {np.var(y):.4f}")

    pd.set_option("display.width", 250)
    res["model"] = pd.Categorical(res["model"], baselines.MODELS)
    shown = ("mae", "total_rel_err") if fraction else ("mse", "mae")
    for metric in shown:
        print(f"\nmean {metric} on the wards without a survey:")
        print(res.pivot_table(index=["scheme", "budget"], columns="model", values=metric,
                              aggfunc="mean", observed=True).round(4).to_string())
    print("\nmean coverage of the 90% interval:")
    print(res.pivot_table(index=["scheme", "budget"], columns="model", values="coverage_90",
                          aggfunc="mean", observed=True).round(3).to_string())

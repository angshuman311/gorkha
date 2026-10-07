"""Survey campaigns at four levels of road blockage.

Usage: python scripts/24_campaign.py [--rounds N] [--levels 0,30,60,100] [--methods ...]
One team starts at the headquarters of each district (11 teams). In each round, each team
surveys one ward. A blockage level closes a random share of the road pieces that a
landslide crosses. The result is the error of the damage estimate against the survey cost.

Cases: "random" (kriging, random wards), "kriging" (kriging, largest variance, then the
shortest open route), "sherpa" (GNN with a learned visit score that sees the road state).
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gorkha import campaign, graph, roads
from gorkha.design import TARGET, design_matrix
from gorkha.paths import PROCESSED, RESULTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--levels", default="0,30,60,100")
    parser.add_argument("--methods", default="random,kriging,sherpa")
    parser.add_argument("--events", type=int, default=2, choices=[1, 2])
    parser.add_argument("--out", default="campaign.parquet")
    parser.add_argument("--image", default="s2_embedding_ssl4eo_ft.parquet",
                        help="image vectors of SHERPA (input 4), or 'none'")
    args = parser.parse_args()
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    g = graph.load()
    pairs = g["adj_pairs"]
    edges = np.unique(np.vstack([pairs, g["knn_pairs"]]), axis=0)
    x = StandardScaler().fit_transform(design_matrix(df, n_events=args.events)).astype("float32")
    xy = df[["x_km", "y_km"]].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    x_sherpa = x
    if args.image != "none":
        emb = pd.read_parquet(PROCESSED / args.image)
        assert (emb["ward_id"].to_numpy() == df["ward_id"].to_numpy()).all()
        x_sherpa = np.hstack([x, emb.drop(columns="ward_id").to_numpy(dtype="float32")])
    print(f"SHERPA input: {x_sherpa.shape[1]} values for each ward ({args.image})", flush=True)
    with np.load(PROCESSED / "roads_levels.npz") as f:
        matrices = {k: f[k].astype(float) for k in f.files}
    start = [int(i) for i in df.groupby("district_name")["dist_hq_km"].idxmin()]
    methods = args.methods.split(",")
    print(f"teams: {len(start)}, rounds: {args.rounds}, wards at the end: "
          f"{len(start) * (args.rounds + 1)} of {len(df)}", flush=True)

    out_path = RESULTS / args.out
    rows = pd.read_parquet(out_path).to_dict("records") if out_path.exists() else []
    done = {(r["level"], r["draw"], r["method"], r["seed"]) for r in rows}
    t0 = time.time()
    for level in [int(v) for v in args.levels.split(",")]:
        for draw in range(roads.BLOCK_LEVELS[level]):
            travel = matrices[f"t_{level}_{draw}"]
            # With one draw (levels 0 and 100), the seeds give the repetitions.
            reps = 3 if roads.BLOCK_LEVELS[level] == 1 else 1
            plan = [(m, draw * 10 + s) for m in methods
                    for s in range(reps * 2 if m == "random" else (1 if m == "kriging" else reps))]
            for method, seed in plan:
                if (level, draw, method, seed) in done:
                    continue
                rec = campaign.run(method, x, xy, y, pairs, edges, travel, start, args.rounds, seed=seed,
                                   x_sherpa=x_sherpa)
                for row in rec:
                    row.update(level=level, draw=draw)
                rows += rec
                last = rec[-1]
                print(f"level {level:3d} draw {draw} {method:8s} seed {seed}: final error "
                      f"{last['mae']:.4f}, cost {last['cost_hours']:.0f} h, flights {last['flights']}, "
                      f"{time.time() - t0:.0f} s", flush=True)
                pd.DataFrame(rows).to_parquet(out_path, index=False)

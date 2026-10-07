"""Feature correlations and spatial autocorrelation. Check gate 2."""

import json
import sys

import pandas as pd

from gorkha import checks, graph
from gorkha.design import TARGET
from gorkha.paths import INTERIM, PROCESSED

GATE_2_P = 0.05
# The gate uses these two residual sets (see the plan). The third set is for information.
GATE_SETS = ["residual: linear regression, in sample",
             "residual: gradient boosting, spatial block folds"]

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = pd.read_parquet(PROCESSED / "wards.parquet")
    pairs = graph.load()["adj_pairs"]

    print(f"feature correlations with {TARGET} (transformed features):")
    print(checks.correlations(df, TARGET).round(3).to_string())

    out = {}
    for target in (TARGET, "mean_grade"):
        r = checks.run(df, pairs, target)
        out[target] = r
        print(f"\nMoran's I, target = {target} ({checks.PERMUTATIONS} permutations):")
        print(r["table"].round(4).to_string())

    table = out[TARGET]["table"]
    res = pd.DataFrame(out[TARGET]["residuals"])
    res.insert(0, "ward_id", df["ward_id"].values)
    res.to_parquet(INTERIM / "check_residuals.parquet", index=False)
    linear = [c for c in res.columns if c.startswith("linear")][0]
    semi = checks.semivariance_by_distance(
        df[["x_km", "y_km"]].to_numpy(),
        {TARGET: df[TARGET], "residual of the linear regression": res[linear]})
    semi.to_csv(INTERIM / "check_semivariance.csv")
    print(f"\nsemivariance by distance (variance of {TARGET}: {df[TARGET].var():.4f}):")
    print(semi.round(4).to_string())
    summary = {t: out[t]["table"].reset_index().to_dict("records") for t in out}
    (INTERIM / "check_moran.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    ok = all(table.loc[s, "I"] > 0 and table.loc[s, "p"] < GATE_2_P for s in GATE_SETS)
    if not ok:
        print("\nGATE 2 FAILED: the residuals show no positive spatial autocorrelation")
        sys.exit(2)
    print("\nGATE 2 PASSED: the residuals show positive spatial autocorrelation "
          f"(p < {GATE_2_P}) in the two planned residual sets")

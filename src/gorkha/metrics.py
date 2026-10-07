"""Error metrics for the predictions on the wards without a survey."""

import numpy as np
import pandas as pd

Z_90 = 1.6449  # half width of a 90% normal interval in standard deviations


def evaluate(y: np.ndarray, pred: np.ndarray, surveyed: np.ndarray, n_buildings: np.ndarray,
             district: np.ndarray, sd: np.ndarray | None = None, counts: bool = True) -> dict:
    """Metrics for one model in one survey realization.

    `y` and `pred` are the grade 4 or 5 fractions of all wards. The error metrics use the
    wards without a survey. The count metrics use the true label for a surveyed ward.
    Set `counts` to False for a target that is not a fraction of buildings.
    """
    u = ~surveyed
    err = pred[u] - y[u]
    weight = n_buildings[u]
    out = {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mae_weighted": float(np.sum(weight * np.abs(err)) / np.sum(weight)),
    }

    if counts:
        # Count of grade 4 or 5 buildings: known for a surveyed ward, predicted for the others.
        estimate = np.where(surveyed, y, pred) * n_buildings
        truth = y * n_buildings
        out["total_rel_err"] = float(abs(estimate.sum() - truth.sum()) / truth.sum())
        by = pd.DataFrame({"d": district, "e": estimate, "t": truth}).groupby("d").sum()
        out["district_mape"] = float(np.mean(np.abs(by["e"] - by["t"]) / by["t"]))

    if sd is not None:
        out["coverage_90"] = float(np.mean(np.abs(err) <= Z_90 * sd[u]))
        out["mean_sd"] = float(np.mean(sd[u]))
    return out

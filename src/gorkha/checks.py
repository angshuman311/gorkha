"""Sanity checks: feature correlations and spatial autocorrelation (gate 2)."""

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import W
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .design import design_matrix

SEED = 0
N_BLOCKS = 25
PERMUTATIONS = 999


def weights(pairs: np.ndarray, n: int) -> W:
    """Row-standardized spatial weights from the undirected edge array."""
    neighbors = {i: [] for i in range(n)}
    for i, j in pairs:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))
    w = W(neighbors, silence_warnings=True)
    w.transform = "r"
    return w


def moran(values, w: W) -> dict:
    np.random.seed(SEED)
    m = Moran(np.asarray(values, dtype=float), w, permutations=PERMUTATIONS)
    return {"I": m.I, "expected_I": m.EI, "z": m.z_sim, "p": m.p_sim}


def correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    x = design_matrix(df)
    rows = []
    for name in x.columns:
        rows.append({
            "feature": name,
            "pearson": np.corrcoef(x[name], df[target])[0, 1],
            "spearman": spearmanr(x[name], df[target]).statistic,
        })
    return pd.DataFrame(rows).set_index("feature")


def spatial_blocks(xy: np.ndarray, n_blocks: int = N_BLOCKS) -> np.ndarray:
    return KMeans(n_clusters=n_blocks, n_init=10, random_state=SEED).fit_predict(xy)


def residual_sets(df: pd.DataFrame, target: str) -> dict:
    """Residuals of the target after the features, for three model and fold choices.

    Return {name: (residuals, R2)}.
    """
    x = design_matrix(df).to_numpy()
    y = df[target].to_numpy(dtype=float)
    blocks = spatial_blocks(df[["x_km", "y_km"]].to_numpy())

    ols = make_pipeline(StandardScaler(), LinearRegression()).fit(x, y)
    out = {"linear regression, in sample": ols.predict(x)}

    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                        l2_regularization=1.0, random_state=SEED)
    out["gradient boosting, spatial block folds"] = cross_val_predict(
        gbm, x, y, cv=GroupKFold(n_splits=5), groups=blocks)
    out["gradient boosting, random folds"] = cross_val_predict(
        gbm, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=SEED))
    return {k: (y - p, r2_score(y, p)) for k, p in out.items()}


DISTANCE_EDGES_KM = [0, 5, 10, 15, 20, 30, 40, 60, 80, 120]


def semivariance_by_distance(xy: np.ndarray, fields: dict) -> pd.DataFrame:
    """Mean semivariance of each field for the ward pairs in each distance class."""
    from scipy.spatial.distance import pdist

    d = pdist(xy)
    edges = DISTANCE_EDGES_KM
    rows = {}
    for name, z in fields.items():
        g = 0.5 * pdist(np.asarray(z, dtype=float)[:, None], "sqeuclidean")
        rows[name] = [g[(d > a) & (d <= b)].mean() for a, b in zip(edges[:-1], edges[1:])]
    out = pd.DataFrame(rows, index=[f"{a} to {b} km" for a, b in zip(edges[:-1], edges[1:])])
    out.insert(0, "ward pairs", [int(((d > a) & (d <= b)).sum()) for a, b in zip(edges[:-1], edges[1:])])
    return out.rename_axis("distance")


def run(df: pd.DataFrame, pairs: np.ndarray, target: str) -> dict:
    w = weights(pairs, len(df))
    rows = [{"quantity": f"{target}", "R2": np.nan, **moran(df[target], w)}]
    res = residual_sets(df, target)
    for name, (r, r2) in res.items():
        rows.append({"quantity": f"residual: {name}", "R2": r2, **moran(r, w)})
    table = pd.DataFrame(rows).set_index("quantity")
    return {"table": table, "residuals": {k: v[0] for k, v in res.items()}}

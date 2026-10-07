"""Baseline models. Each model sees only the labels of the surveyed wards (protocol A)."""

import warnings

import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor

MODELS = ["mean", "ok", "ridge", "mlp", "forest", "rk", "rk_loo", "rk_gdif"]
MODEL_LABELS = {
    "mean": "Mean of the surveyed wards",
    "ok": "Ordinary kriging (no features)",
    "ridge": "Ridge regression",
    "mlp": "MLP",
    "forest": "Random forest",
    "rk": "Regression kriging, all features",
    "rk_loo": "Regression kriging, all features, leave-one-out residuals",
    "rk_gdif": "Regression kriging, G-DIF covariates",
}
RIDGE_ALPHAS = np.logspace(-2, 3, 16)

# Trend covariates of the G-DIF code (GDIF_nb.Rmd): shaking intensity, elevation, and the
# Damage Proxy Map. The engineering forecast of G-DIF is not available here.
GDIF_FEATURES = ["mmi", "mmi_after", "elev_mean", "dpm_cover", "dpm_p90"]

# Variogram fit, with the defaults of the R package gstat (the G-DIF code uses gstat):
# cutoff at one third of the diagonal of the data, 15 lags, fit weights N / h^2.
N_LAGS = 15
CUTOFF_SHARE = 1.0 / 3.0
# A small survey has too few ward pairs in the cutoff. Then the fit uses all pairs.
MIN_PAIRS = 40
PAIRS_PER_LAG = 20


def exponential(h, nugget, psill, range_):
    """Exponential variogram in the pykrige convention (practical range)."""
    return nugget + psill * (1.0 - np.exp(-3.0 * h / range_))


def experimental_variogram(xy: np.ndarray, z: np.ndarray) -> tuple:
    """Lag distance, semivariance, and pair count for each lag that has a pair."""
    d = pdist(xy)
    g = 0.5 * pdist(z[:, None], "sqeuclidean")
    cutoff = float(np.hypot(*np.ptp(xy, axis=0)) * CUTOFF_SHARE)
    if (d <= cutoff).sum() < MIN_PAIRS:
        cutoff = float(d.max())
    inside = d <= cutoff
    n_lags = int(np.clip(inside.sum() // PAIRS_PER_LAG, 4, N_LAGS))
    lag = np.minimum((d[inside] / cutoff * n_lags).astype(int), n_lags - 1)
    count = np.bincount(lag, minlength=n_lags)
    has = count > 0
    h = np.bincount(lag, weights=d[inside], minlength=n_lags)[has] / count[has]
    gamma = np.bincount(lag, weights=g[inside], minlength=n_lags)[has] / count[has]
    return h, gamma, count[has], cutoff


def fit_variogram(xy: np.ndarray, z: np.ndarray) -> dict:
    """Weighted least squares fit of an exponential variogram with a nugget."""
    h, gamma, count, cutoff = experimental_variogram(xy, z)
    var = float(np.var(z, ddof=1))
    if len(h) < 3:
        return {"psill": 0.0, "range": cutoff, "nugget": var}
    weight = np.sqrt(count) / h
    top = max(float(gamma.max()), var)
    fit = least_squares(
        lambda p: weight * (exponential(h, *p) - gamma),
        x0=[0.25 * var, 0.75 * var, 0.5 * cutoff],
        bounds=([0.0, 0.0, 0.5 * h.min()], [1.5 * top, 3.0 * top, 3.0 * cutoff]),
    )
    nugget, psill, range_ = fit.x
    return {"psill": float(psill), "range": float(range_), "nugget": float(nugget)}


def krige(xy_s: np.ndarray, z_s: np.ndarray, xy_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging with a variogram fitted on the surveyed wards.

    Return the prediction and the kriging variance at `xy_u`.
    """
    if np.ptp(z_s) < 1e-9:
        return np.full(len(xy_u), z_s.mean()), np.zeros(len(xy_u))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OrdinaryKriging(xy_s[:, 0], xy_s[:, 1], z_s, variogram_model="exponential",
                                variogram_parameters=fit_variogram(xy_s, z_s),
                                enable_plotting=False, verbose=False)
        z, ss = model.execute("points", xy_u[:, 0], xy_u[:, 1], backend="vectorized")
    return np.asarray(z, dtype=float), np.maximum(np.asarray(ss, dtype=float), 0.0)


def ridge_loo_residuals(xs: np.ndarray, ys: np.ndarray, alpha: float) -> np.ndarray:
    """Leave-one-out residuals of a ridge regression with an intercept (closed form)."""
    n = len(ys)
    xc = xs - xs.mean(axis=0)
    u, sv, _ = np.linalg.svd(xc, full_matrices=False)
    shrink = sv ** 2 / (sv ** 2 + alpha)
    leverage = np.einsum("ij,j,ij->i", u, shrink, u) + 1.0 / n
    fitted = u @ (shrink * (u.T @ (ys - ys.mean()))) + ys.mean()
    return (ys - fitted) / np.maximum(1.0 - leverage, 1e-6)


def predict_all(x: np.ndarray, xy: np.ndarray, y: np.ndarray, surveyed: np.ndarray,
                seed: int = 0, gdif_columns: list | None = None,
                limits: tuple = (0.0, 1.0)) -> dict:
    """Fit each baseline on the surveyed wards and predict all wards.

    `x` is the standardized design matrix of all wards. The scaler uses the features of
    all wards, which are known before a survey. It uses no label.
    `gdif_columns` gives the columns of `x` for the G-DIF covariates.
    `limits` gives the permitted range of the target (a fraction, or a damage grade).
    Return {model: (prediction, standard deviation or None)}.
    """
    s, n = surveyed, len(y)
    ys = y[s]
    out = {}

    out["mean"] = (np.full(n, ys.mean()), np.full(n, ys.std(ddof=1)))

    z, var = krige(xy[s], ys, xy)
    out["ok"] = (z, np.sqrt(var))

    alpha = RidgeCV(alphas=RIDGE_ALPHAS).fit(x[s], ys).alpha_
    trend = Ridge(alpha=alpha).fit(x[s], ys).predict(x)
    out["ridge"] = (trend, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp = MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, solver="lbfgs", max_iter=500,
                           random_state=seed).fit(x[s], ys)
    out["mlp"] = (mlp.predict(x), None)

    forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.5,
                                   random_state=seed, n_jobs=1).fit(x[s], ys)
    out["forest"] = (forest.predict(x), None)

    # Regression kriging: the ridge trend plus ordinary kriging of the trend residuals.
    r, var = krige(xy[s], ys - trend[s], xy)
    out["rk"] = (trend + r, np.sqrt(var))

    # Variant: the residual of a surveyed ward comes from a trend fitted without that ward.
    # An in-sample residual is too small when the survey has few wards.
    r, var = krige(xy[s], ridge_loo_residuals(x[s], ys, alpha), xy)
    out["rk_loo"] = (trend + r, np.sqrt(var))

    if gdif_columns is not None:
        xg = x[:, gdif_columns]
        trend_g = RidgeCV(alphas=RIDGE_ALPHAS).fit(xg[s], ys).predict(xg)
        r, var = krige(xy[s], ys - trend_g[s], xy)
        out["rk_gdif"] = (trend_g + r, np.sqrt(var))

    return {k: (np.clip(p, *limits), sd) for k, (p, sd) in out.items()}

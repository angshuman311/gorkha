"""Design matrix for the models that use the secondary data."""

import numpy as np
import pandas as pd

TARGET = "frac_g45"

# Transform for each feature. The transforms reduce the skew of the distributions.
TRANSFORMS = {
    "mmi": None,
    "pga": np.log,
    "pgv": np.log,
    "vs30": np.log,
    "dist_epi_km": None,
    "mmi_after": None,
    "pga_after": np.log,
    "pgv_after": np.log,
    "mmi_max": None,
    "dist_epi2_km": None,
    "elev_mean": lambda v: v / 1000.0,
    "slope_mean": None,
    "dpm_cover": None,
    "dpm_valid_frac": None,
    "dpm_mean": None,
    "dpm_p90": None,
    "ls_mapped": None,
    "ls_frac": np.sqrt,
    "dist_road_km": np.log1p,
    "dist_hq_km": np.log1p,
    "area_km2": np.log,
}

# Features of the damage model. The second group is used only if a second earthquake
# occurred before the survey (`n_events` = 2).
FEATURE_GROUPS = {
    "shaking, first earthquake": ["mmi", "pga", "pgv", "vs30", "dist_epi_km"],
    "shaking, second earthquake": ["mmi_after", "pga_after", "pgv_after", "mmi_max", "dist_epi2_km"],
    "terrain": ["elev_mean", "slope_mean"],
    "dpm": ["dpm_cover", "dpm_valid_frac", "dpm_mean", "dpm_p90"],
    "landslide": ["ls_mapped", "ls_frac"],
}

# These values are not in the damage model (decision of 2026-10-07). Their correlation with
# the damage is 0.08 or less. They are inputs of the survey planner and of the survey simulation.
PLANNER_FEATURES = ["dist_road_km", "dist_hq_km", "area_km2"]


def model_features(n_events: int = 2) -> list:
    """Names of the damage model features for 1 or 2 earthquakes before the survey."""
    if n_events not in (1, 2):
        raise ValueError("n_events must be 1 or 2")
    groups = [g for name, g in FEATURE_GROUPS.items() if n_events == 2 or "second" not in name]
    return [f for g in groups for f in g]


def design_matrix(df: pd.DataFrame, features: list | None = None, n_events: int = 2) -> pd.DataFrame:
    """Transformed features, not standardized. Fit the scaler on the training rows only."""
    features = features or model_features(n_events)
    out = {}
    for name in features:
        fn = TRANSFORMS[name]
        v = df[name].to_numpy(dtype=float)
        out[name] = fn(v) if fn else v
    return pd.DataFrame(out, index=df.index)

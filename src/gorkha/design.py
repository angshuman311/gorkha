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
    "mmi_after": None,
    "pga_after": np.log,
    "pgv_after": np.log,
    "mmi_max": None,
    "elev_mean": lambda v: v / 1000.0,
    "slope_mean": None,
    "dpm_cover": None,
    "dpm_valid_frac": None,
    "dpm_mean": None,
    "dpm_p90": None,
    "ls_mapped": None,
    "ls_frac": np.sqrt,
    "dist_epi_km": None,
    "dist_road_km": np.log1p,
    "dist_hq_km": np.log1p,
    "area_km2": np.log,
}

FEATURE_GROUPS = {
    "shaking": ["mmi", "pga", "pgv", "vs30", "mmi_after", "pga_after", "pgv_after", "mmi_max"],
    "terrain": ["elev_mean", "slope_mean"],
    "dpm": ["dpm_cover", "dpm_valid_frac", "dpm_mean", "dpm_p90"],
    "landslide": ["ls_mapped", "ls_frac"],
    "access": ["dist_epi_km", "dist_road_km", "dist_hq_km", "area_km2"],
}


def design_matrix(df: pd.DataFrame, features: list | None = None) -> pd.DataFrame:
    """Transformed features, not standardized. Fit the scaler on the training rows only."""
    features = features or list(TRANSFORMS)
    out = {}
    for name in features:
        fn = TRANSFORMS[name]
        v = df[name].to_numpy(dtype=float)
        out[name] = fn(v) if fn else v
    return pd.DataFrame(out, index=df.index)

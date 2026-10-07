"""Ward labels from the building damage survey."""

import pandas as pd

from .paths import INTERIM, RAW

SURVEY = RAW / "survey"


def name_table() -> pd.DataFrame:
    """Survey wards with the municipality and district names."""
    t = pd.read_csv(SURVEY / "ward_vdcmun_district_name_mapping.csv")
    t["ward_no"] = t["ward_id"] % 100
    return t


def buildings() -> pd.DataFrame:
    cols = ["building_id", "district_id", "vdcmun_id", "ward_id", "damage_grade",
            "technical_solution_proposed"]
    b = pd.read_csv(SURVEY / "csv_building_structure.csv", usecols=cols)
    b["grade"] = pd.to_numeric(b["damage_grade"].str.extract(r"(\d)")[0], errors="coerce")
    return b


def ward_labels(b: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row for each survey ward: building count, grade 4 or 5 fraction, mean grade.

    Buildings without a damage grade are not in the counts.
    """
    if b is None:
        b = buildings()
    b = b[b["grade"].notna()]
    g = b.groupby("ward_id")["grade"]
    out = pd.DataFrame(
        {
            "n_buildings": g.size(),
            "n_g45": g.apply(lambda s: int((s >= 4).sum())),
            "mean_grade": g.mean(),
        }
    )
    out["frac_g45"] = out["n_g45"] / out["n_buildings"]
    out = out.reset_index()
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INTERIM / "ward_labels.parquet", index=False)
    return out

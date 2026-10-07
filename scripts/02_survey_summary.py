"""Summarize the damage survey and write the ward labels."""

import pandas as pd

from gorkha import labels

if __name__ == "__main__":
    names = labels.name_table()
    b = labels.buildings()
    print(f"buildings: {len(b)}")
    print(f"buildings without a damage grade: {int(b['grade'].isna().sum())}")
    print(f"districts: {names['district_name'].nunique()}, "
          f"municipality ids: {names['vdcmun_id'].nunique()}, "
          f"wards in the name table: {len(names)}, "
          f"wards with buildings: {b['ward_id'].nunique()}")
    print("\ndamage grade distribution (share of buildings):")
    print(b["grade"].value_counts(normalize=True).sort_index().round(3).to_string())
    print("\ndamage grade against the proposed technical solution (row shares):")
    print(pd.crosstab(b["grade"], b["technical_solution_proposed"], normalize="index")
          .round(3).to_string())

    w = labels.ward_labels(b)
    w = w.merge(names[["ward_id", "district_name"]], on="ward_id", how="left")
    print("\nward labels by district:")
    by = w.groupby("district_name").agg(
        wards=("ward_id", "size"),
        buildings=("n_buildings", "sum"),
        g45=("n_g45", "sum"),
        ward_frac_min=("frac_g45", "min"),
        ward_frac_median=("frac_g45", "median"),
        ward_frac_max=("frac_g45", "max"),
        ward_buildings_median=("n_buildings", "median"),
    )
    by["frac_g45"] = by["g45"] / by["buildings"]
    print(by.round(3).to_string())
    print(f"\nall districts: frac_g45 = {w['n_g45'].sum() / w['n_buildings'].sum():.3f}, "
          f"ward frac_g45 mean = {w['frac_g45'].mean():.3f}, sd = {w['frac_g45'].std():.3f}")
    print("buildings per ward: "
          + ", ".join(f"{k}={v:.0f}" for k, v in w["n_buildings"].describe().items()))

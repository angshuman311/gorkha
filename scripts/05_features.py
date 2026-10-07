"""Build the ward feature table."""

import pandas as pd

from gorkha import features

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    w = features.build()
    cols = features.FEATURES + ["psa03", "psa10", "psa30", "dpm_has_data"]
    print(f"wards (nodes): {len(w)}")
    print(w[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3).to_string())
    print(f"\nmissing values: {int(w[cols].isna().sum().sum())}")
    print(w[cols].isna().sum()[lambda s: s > 0].to_string())

    print("\nDamage Proxy Map coverage:")
    print(f"  wards with more than 50% of the area in the radar footprint: {int((w['dpm_cover'] > 0.5).sum())}")
    print(f"  wards with a DPM statistic: {int(w['dpm_has_data'].sum())}")
    print(f"  buildings in wards with a DPM statistic: "
          f"{w.loc[w['dpm_has_data'] == 1, 'n_buildings'].sum() / w['n_buildings'].sum():.3f}")
    print(w.groupby("district_name").agg(wards=("ward_id", "size"),
                                          dpm_wards=("dpm_has_data", "sum"),
                                          dpm_cover=("dpm_cover", "mean"),
                                          dpm_valid_frac=("dpm_valid_frac", "mean"),
                                          ls_mapped=("ls_mapped", "mean"),
                                          ls_wards=("ls_frac", lambda s: int((s > 0).sum())))
          .round(3).to_string())
    print("\nLandslide inventory:")
    print(f"  wards with more than 50% of the area in the examined extent: {int((w['ls_mapped'] > 0.5).sum())}")
    print(f"  wards with a landslide: {int((w['ls_frac'] > 0).sum())}")

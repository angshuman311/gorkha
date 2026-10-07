"""Join the survey wards to the OSM ward polygons and check gate 1."""

import sys

import pandas as pd

from gorkha import join

GATE_1 = 0.90

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    r = join.join()
    mun = r["municipalities"]
    cols = ["district", "vdcmun_name", "mun_name_en", "mun_prefix", "score",
            "survey_wards", "osm_wards"]

    count_differs = mun["accepted"] & (mun["survey_wards"] != mun["osm_wards"])
    print(f"municipality pairs: {int(mun['accepted'].sum())} accepted, "
          f"{int((~mun['accepted']).sum())} not accepted")
    print("\npairs with a name score below 100:")
    print(mun[mun["accepted"] & (mun["score"] < 100)].sort_values("score")[cols].to_string(index=False))
    print("\nnot accepted (manual review):")
    print(mun[~mun["accepted"]][cols].to_string(index=False))
    print("\naccepted pairs where the ward counts differ:")
    print(mun[count_differs][cols].to_string(index=False))

    print("\nmatch by district:")
    print(r["by_district"].round(4).to_string())
    print(f"\nOSM wards with a duplicate (municipality, ward number): {len(r['osm_duplicates'])}")
    print(f"OSM wards without a survey ward: {len(r['osm_unused'])}")
    j = r["ward_join"]
    print("survey wards without a polygon:")
    print(j[~j["matched"]][["ward_id", "district_name", "vdcmun_name", "ward_no", "n_buildings"]]
          .to_string(index=False))

    rate = r["building_match_rate"]
    print(f"\nward match rate (wards with buildings): {r['ward_match_rate']:.4f}")
    print(f"building match rate: {rate:.4f}")
    if rate < GATE_1:
        print(f"GATE 1 FAILED: building match rate is less than {GATE_1}")
        sys.exit(2)
    print(f"GATE 1 PASSED: building match rate is at least {GATE_1}")

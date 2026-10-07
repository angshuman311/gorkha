"""Join the survey wards to the OSM ward polygons.

Key: district name + municipality name + ward number.
"""

import re

import geopandas as gpd
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from . import labels
from .paths import INTERIM, REPORTS

# A municipality pair with a lower name score goes to the manual review file.
MIN_SCORE = 80

# Municipality type at the end of a name. The survey spells one type
# "Sub-Metropolitian City".
_SUFFIX = re.compile(
    r"\s*(sub[\s-]*metropolit\w*\s+city|metropolitan\s+city|rural\s+municipality|"
    r"urban\s+municipality|municipality|upamahanagarpalika|mahanagarpalika|"
    r"gaunpalika|nagarpalika)\s*$"
)

# Manual review of the pairs with a score below MIN_SCORE (2026-10-06).
# Key: survey vdcmun_id. Value: OSM ward name prefix.
# In each case the pair is the only unpaired unit on each side in its district, and
# the two sides have the same ward count.
MANUAL_MUNICIPALITIES = {
    2201: "Baiteshwor",  # survey spelling "Baitedhar"
    3603: "Sulikot",  # survey name "Barpak Sulikot"
    3604: "Bhimsen",  # survey name "Bhimsen Thapa"
}

# Key: survey ward_id. Value: OSM ward number in the same municipality.
# Marin has 7 wards on each side. The survey numbers are 1 to 6 and 8, the OSM numbers
# are 1 to 7. Ward 8 and ward 7 are the only unpaired wards.
MANUAL_WARD_NUMBERS = {200608: 7}

# Key: survey ward_id. Value: survey vdcmun_id of the municipality that contains the ward.
# The survey lists ward 230209 as Balephi ward 9. Balephi has 8 wards and Bahrabise has 9
# (official ward map of Balephi, LLRC 2016, checked by the user on 2026-10-06). The only
# polygon without a survey ward was OSM "Barhabise-09", which is adjacent to Balephi wards
# 3, 7, and 8. The ward number 9 is the same on the two sides.
MANUAL_WARD_MUNICIPALITY = {230209: 2301}

# Spelling variants in the romanization of Nepali place names.
_PHONETIC = (
    ("chh", "ch"), ("sh", "s"), ("ow", "o"), ("aa", "a"), ("ee", "i"), ("oo", "u"),
    ("w", "b"), ("v", "b"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
    ("gh", "g"), ("ph", "p"), ("jh", "j"), ("y", "i"),
)


def norm_name(name) -> str:
    """Lowercase letters only, without the municipality type suffix."""
    if not isinstance(name, str):
        return ""
    s = _SUFFIX.sub("", name.lower().strip())
    return re.sub(r"[^a-z]", "", s)


def phonetic_key(norm: str) -> str:
    for a, b in _PHONETIC:
        norm = norm.replace(a, b)
    return norm


def name_score(a, b) -> float:
    """Similarity of two municipality names, 0 to 100."""
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return 0.0
    return max(fuzz.ratio(na, nb), fuzz.ratio(phonetic_key(na), phonetic_key(nb)))


def match_municipalities(survey: pd.DataFrame, osm_wards: pd.DataFrame) -> pd.DataFrame:
    """One to one match of survey municipalities to OSM municipalities in each district."""
    rows = []
    for district, s in survey.groupby("district_name"):
        s_mun = (
            s.groupby("vdcmun_id")
            .agg(vdcmun_name=("vdcmun_name", "first"), survey_wards=("ward_id", "size"))
            .reset_index()
        )
        o = osm_wards[osm_wards["district"] == district]
        o_mun = (
            o.groupby("mun_osm")
            .agg(
                mun_name_en=("mun_name_en", "first"),
                mun_prefix=("mun_prefix", lambda x: x.mode().iat[0]),
                osm_wards=("osm_id", "size"),
            )
            .reset_index()
        )
        score = np.zeros((len(s_mun), len(o_mun)))
        for i, a in enumerate(s_mun["vdcmun_name"]):
            for j, (b1, b2) in enumerate(zip(o_mun["mun_name_en"], o_mun["mun_prefix"])):
                score[i, j] = max(name_score(a, b1), name_score(a, b2))
        ri, ci = linear_sum_assignment(-score)
        assigned = dict(zip(ri, ci))
        for i, r in enumerate(s_mun.itertuples(index=False)):
            row = {"district": district, "vdcmun_id": r.vdcmun_id,
                   "vdcmun_name": r.vdcmun_name, "survey_wards": r.survey_wards}
            if i in assigned:
                om = o_mun.iloc[assigned[i]]
                row.update(mun_osm=om["mun_osm"], mun_name_en=om["mun_name_en"],
                           mun_prefix=om["mun_prefix"], osm_wards=om["osm_wards"],
                           score=score[i, assigned[i]])
            rows.append(row)
        for j in set(range(len(o_mun))) - set(ci):
            om = o_mun.iloc[j]
            rows.append({"district": district, "mun_osm": om["mun_osm"],
                         "mun_name_en": om["mun_name_en"], "mun_prefix": om["mun_prefix"],
                         "osm_wards": om["osm_wards"]})
    out = pd.DataFrame(rows)
    by_score = out["score"].fillna(0) >= MIN_SCORE
    manual = pd.Series(
        [
            norm_name(MANUAL_MUNICIPALITIES.get(v)) == norm_name(p) != ""
            for v, p in zip(out["vdcmun_id"], out["mun_prefix"])
        ],
        index=out.index,
    )
    out["accepted_by"] = np.where(by_score, "score", np.where(manual, "manual", ""))
    out["accepted"] = by_score | manual
    return out


def join() -> dict:
    survey = labels.name_table()
    lab = pd.read_parquet(INTERIM / "ward_labels.parquet")
    osm = gpd.read_file(INTERIM / "osm_wards.gpkg").drop(columns="geometry")

    mun = match_municipalities(survey, osm)
    pairs = mun[mun["accepted"]][["vdcmun_id", "mun_osm", "score"]]

    dup = osm[osm.duplicated(["mun_osm", "ward_no"], keep=False) & osm["ward_no"].notna()]
    osm_unique = osm[osm["ward_no"].notna()].drop_duplicates(["mun_osm", "ward_no"])

    j = survey.merge(pairs, on="vdcmun_id", how="left")
    # A survey ward with a wrong municipality gets the OSM municipality of the correct one.
    mun_of = pairs.set_index("vdcmun_id")["mun_osm"]
    j["mun_osm"] = j["ward_id"].map(MANUAL_WARD_MUNICIPALITY).map(mun_of).fillna(j["mun_osm"])
    j["osm_ward_no"] = j["ward_id"].map(MANUAL_WARD_NUMBERS).fillna(j["ward_no"]).astype("int64")
    j["join_note"] = ""
    j.loc[j["ward_id"].isin(MANUAL_WARD_NUMBERS), "join_note"] = "ward number from manual review"
    j.loc[j["ward_id"].isin(MANUAL_WARD_MUNICIPALITY), "join_note"] = "municipality from manual review"
    j = j.merge(
        osm_unique[["mun_osm", "ward_no", "osm_id"]]
        .astype({"ward_no": "int64"})
        .rename(columns={"ward_no": "osm_ward_no"}),
        on=["mun_osm", "osm_ward_no"],
        how="left",
    )
    if j["osm_id"].dropna().duplicated().any():
        raise RuntimeError("two survey wards point to the same OSM ward")
    j = j.merge(lab[["ward_id", "n_buildings"]], on="ward_id", how="left")
    j["n_buildings"] = j["n_buildings"].fillna(0).astype(int)
    j["matched"] = j["osm_id"].notna()

    by = j.groupby("district_name").apply(
        lambda d: pd.Series(
            {
                "survey_wards": len(d),
                "wards_with_buildings": int((d["n_buildings"] > 0).sum()),
                "matched_wards": int(d["matched"].sum()),
                "buildings": int(d["n_buildings"].sum()),
                "matched_buildings": int(d.loc[d["matched"], "n_buildings"].sum()),
            }
        ),
        include_groups=False,
    )
    by["building_match_rate"] = by["matched_buildings"] / by["buildings"]

    osm_unused = osm[~osm["osm_id"].isin(j["osm_id"].dropna())]

    INTERIM.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    j.to_parquet(INTERIM / "ward_join.parquet", index=False)
    mun.sort_values(["district", "score"]).to_csv(REPORTS / "join_municipalities.csv", index=False)
    pd.concat(
        [
            j[~j["matched"]].assign(side="survey ward without a polygon"),
            osm_unused.assign(side="OSM ward without a survey ward"),
        ]
    ).to_csv(REPORTS / "join_unmatched.csv", index=False)

    with_b = j[j["n_buildings"] > 0]
    return {
        "municipalities": mun,
        "by_district": by,
        "ward_join": j,
        "osm_duplicates": dup,
        "osm_unused": osm_unused,
        "building_match_rate": j.loc[j["matched"], "n_buildings"].sum() / j["n_buildings"].sum(),
        "ward_match_rate": with_b["matched"].mean(),
    }

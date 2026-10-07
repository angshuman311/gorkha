"""District, municipality, and ward polygons from the OSM boundary relations."""

import json

import geopandas as gpd
import pandas as pd
from rapidfuzz import fuzz, process
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

from .paths import CRS_METRIC, DISTRICTS, INTERIM, RAW

WGS84 = "EPSG:4326"


def _lines(element: dict, roles: tuple) -> list:
    out = []
    for m in element["members"]:
        if m["type"] != "way" or m.get("role") not in roles:
            continue
        pts = m.get("geometry") or []
        if len(pts) >= 2:
            out.append(LineString([(p["lon"], p["lat"]) for p in pts]))
    return out


def relation_polygon(element: dict):
    """Assemble the polygon of a boundary relation from its member ways.

    Return None if the outer ways do not close a ring.
    """
    outer = _lines(element, ("outer", ""))
    if not outer:
        return None
    geom = unary_union(list(polygonize(unary_union(outer))))
    inner = _lines(element, ("inner",))
    if inner:
        geom = geom.difference(unary_union(list(polygonize(unary_union(inner)))))
    return None if geom.is_empty else geom


def load_level(level: int) -> gpd.GeoDataFrame:
    path = RAW / "osm" / f"admin_level_{level}.json"
    elements = json.loads(path.read_text(encoding="utf-8"))["elements"]
    rows = []
    for e in elements:
        tags = e.get("tags", {})
        centre = next(
            (m for m in e["members"] if m.get("role") == "admin_centre" and "lat" in m),
            None,
        )
        rows.append(
            {
                "osm_id": e["id"],
                "name": tags.get("name"),
                "name_en": tags.get("name:en") or tags.get("int_name"),
                "suffix_en": tags.get("name:suffix:en") or tags.get("name:suffix"),
                "ward_tag": tags.get("ward"),
                "subareas": [
                    m["ref"]
                    for m in e["members"]
                    if m["type"] == "relation" and m.get("role") == "subarea"
                ],
                "centre": Point(centre["lon"], centre["lat"]) if centre else None,
                "geometry": relation_polygon(e),
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)
    failed = gdf.geometry.isna().sum()
    if failed:
        print(f"level {level}: {failed} of {len(gdf)} relations have no closed ring")
    return gdf[gdf.geometry.notna()].reset_index(drop=True)


def _parent_by_point(children: gpd.GeoDataFrame, parents: gpd.GeoDataFrame) -> pd.Series:
    """Parent osm_id for each child: the parent polygon that contains a point of the child."""
    pts = gpd.GeoDataFrame(
        {"child": children["osm_id"].values},
        geometry=children.geometry.representative_point().values,
        crs=children.crs,
    )
    hit = gpd.sjoin(pts, parents[["osm_id", "geometry"]], predicate="within", how="left")
    hit = hit.drop_duplicates("child").set_index("child")["osm_id"]
    return children["osm_id"].map(hit)


def _parent_by_subarea(children: gpd.GeoDataFrame, parents: gpd.GeoDataFrame) -> pd.Series:
    link = {ref: pid for pid, refs in zip(parents["osm_id"], parents["subareas"]) for ref in refs}
    return children["osm_id"].map(link)


def study_districts() -> gpd.GeoDataFrame:
    d = load_level(6)
    d["district"] = [
        (process.extractOne(n, DISTRICTS, scorer=fuzz.ratio, score_cutoff=90) or [None])[0]
        if n
        else None
        for n in d["name_en"]
    ]
    d = d[d["district"].notna()].reset_index(drop=True)
    if sorted(d["district"]) != sorted(DISTRICTS):
        raise RuntimeError(f"district match is not one to one: {sorted(d['district'])}")
    return d


def build() -> dict:
    """Build the polygons of the study area and write them to data/interim."""
    districts = study_districts()

    mun = load_level(7)
    mun["district_osm"] = _parent_by_point(mun, districts)
    sub = _parent_by_subarea(mun, districts)
    mun["district_osm"] = mun["district_osm"].fillna(sub)
    mun = mun[mun["district_osm"].notna()].reset_index(drop=True)
    mun["district"] = mun["district_osm"].map(districts.set_index("osm_id")["district"])

    wards = load_level(9)
    by_point = _parent_by_point(wards, mun)
    by_sub = _parent_by_subarea(wards, mun)
    both = by_point.notna() & by_sub.notna()
    agree = (by_point[both] == by_sub[both]).mean() if both.any() else float("nan")
    wards["mun_osm"] = by_point.fillna(by_sub)
    wards = wards[wards["mun_osm"].notna()].reset_index(drop=True)

    m = mun.set_index("osm_id")
    wards["district"] = wards["mun_osm"].map(m["district"])
    wards["mun_name_en"] = wards["mun_osm"].map(m["name_en"])
    wards["mun_suffix_en"] = wards["mun_osm"].map(m["suffix_en"])
    # The ward name has the form "<municipality>-<ward number>". The number in the name
    # has priority over the `ward` tag: relation 16011799 "Khijidemba-02" has ward=1.
    from_name = pd.to_numeric(wards["name"].str.extract(r"(\d+)\s*$")[0], errors="coerce")
    from_tag = pd.to_numeric(wards["ward_tag"], errors="coerce")
    number_differs = int((from_name.notna() & from_tag.notna() & (from_name != from_tag)).sum())
    wards["ward_no"] = from_name.fillna(from_tag).astype("Int64")
    wards["mun_prefix"] = wards["name"].str.replace(r"[-\s]*\d+\s*$", "", regex=True).str.strip()

    wards_m = wards.to_crs(CRS_METRIC)
    wards["area_km2"] = wards_m.area.values / 1e6
    mun["area_km2"] = mun.to_crs(CRS_METRIC).area.values / 1e6
    districts["area_km2"] = districts.to_crs(CRS_METRIC).area.values / 1e6

    INTERIM.mkdir(parents=True, exist_ok=True)
    keep_w = ["osm_id", "name", "ward_no", "mun_prefix", "mun_osm", "mun_name_en",
              "mun_suffix_en", "district", "area_km2", "geometry"]
    wards[keep_w].to_file(INTERIM / "osm_wards.gpkg", driver="GPKG")
    mun[["osm_id", "name", "name_en", "suffix_en", "district", "area_km2", "geometry"]].to_file(
        INTERIM / "osm_municipalities.gpkg", driver="GPKG"
    )
    hq = districts[["osm_id", "district", "area_km2"]].copy()
    hq["hq_lon"] = [c.x if c else None for c in districts["centre"]]
    hq["hq_lat"] = [c.y if c else None for c in districts["centre"]]
    gpd.GeoDataFrame(hq, geometry=districts.geometry.values, crs=WGS84).to_file(
        INTERIM / "osm_districts.gpkg", driver="GPKG"
    )

    ward_area = wards.groupby("district")["area_km2"].sum()
    summary = pd.DataFrame(
        {
            "municipalities": mun.groupby("district").size(),
            "wards": wards.groupby("district").size(),
            "ward_area_km2": ward_area.round(0),
            "district_area_km2": districts.set_index("district")["area_km2"].round(0),
        }
    )
    summary["ward_area_share"] = (summary["ward_area_km2"] / summary["district_area_km2"]).round(3)
    return {
        "summary": summary,
        "n_wards": len(wards),
        "n_municipalities": len(mun),
        "parent_agreement": agree,
        "parent_from_subarea_only": int((by_point.isna() & by_sub.notna()).sum()),
        "wards_without_number": int(wards["ward_no"].isna().sum()),
        "ward_number_differs": number_differs,
        "duplicate_ward_numbers": int(wards.duplicated(["mun_osm", "ward_no"]).sum()),
    }

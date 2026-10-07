"""Map of the Balephi and Bahrabise wards: OSM polygons with the survey values.

The script also compares the survey building counts with the OSM building counts.
"""

import json

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from gorkha import download, plots
from gorkha.paths import CRS_METRIC, FIGURES, INTERIM, RAW

# OSM spelling of the two municipalities, with a light tint of series color 1 and 2.
FILL = {"Balefi": "#b7d3f6", "Barhabise": "#f6c4b0"}
LEGEND = {"Balefi": "Balephi (OSM: Balefi)", "Barhabise": "Bahrabise (OSM: Barhabise)"}
MOVED = "municipality from manual review"


def osm_building_counts(sel: gpd.GeoDataFrame) -> pd.Series:
    """Number of OSM building ways with the center in each ward polygon."""
    w, s, e, n = sel.to_crs("EPSG:4326").total_bounds
    query = (f'[out:json][timeout:300];way["building"]({s - 0.01},{w - 0.01},{n + 0.01},'
             f'{e + 0.01});out center;')
    path = download.overpass(query, RAW / "osm" / "buildings_balephi_barhabise.json")
    elements = json.loads(path.read_text(encoding="utf-8"))["elements"]
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([x["center"]["lon"] for x in elements],
                                    [x["center"]["lat"] for x in elements]),
        crs="EPSG:4326").to_crs(sel.crs)
    hit = gpd.sjoin(points, sel[["osm_id", "geometry"]], predicate="within")
    return hit.groupby("osm_id").size()


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    plots.style()
    osm = gpd.read_file(INTERIM / "osm_wards.gpkg").to_crs(CRS_METRIC)
    join = pd.read_parquet(INTERIM / "ward_join.parquet")
    lab = pd.read_parquet(INTERIM / "ward_labels.parquet")[["ward_id", "frac_g45"]]
    survey = join.merge(lab, on="ward_id", how="left")

    sel = osm[(osm["district"] == "Sindhupalchok") & osm["mun_prefix"].isin(FILL)].copy()
    cols = ["osm_id", "ward_id", "vdcmun_name", "n_buildings", "frac_g45", "join_note"]
    matched = survey[survey["matched"]].rename(columns={"ward_no": "survey_ward_no"})
    sel = sel.merge(matched[cols + ["survey_ward_no"]], on="osm_id", how="left")
    sel["osm_buildings"] = sel["osm_id"].map(osm_building_counts(sel))
    sel["survey_per_osm"] = sel["n_buildings"] / sel["osm_buildings"]

    print("OSM polygons with the matched survey ward:")
    print(sel[["name", "area_km2", "ward_id", "vdcmun_name", "n_buildings", "frac_g45",
               "osm_buildings", "survey_per_osm", "join_note"]]
          .sort_values("name").round(2).to_string(index=False))
    ratio = sel.loc[sel["join_note"] != MOVED, "survey_per_osm"].dropna()
    print(f"\nsurvey buildings / OSM buildings, wards without a manual municipality: "
          f"median {ratio.median():.2f}, minimum {ratio.min():.2f}, maximum {ratio.max():.2f}")
    orphan = survey[survey["vdcmun_id"].isin([2301, 2302]) & ~survey["matched"]]
    print(f"survey wards of the two municipalities without a polygon: {len(orphan)}")

    moved = sel[sel["join_note"] == MOVED]
    free = sel[sel["ward_id"].isna()]

    fig, ax = plt.subplots(figsize=(9.5, 8.2), layout="constrained")
    box = sel.total_bounds
    pad = 1500
    around = osm[~osm["osm_id"].isin(sel["osm_id"])].cx[box[0] - pad:box[2] + pad,
                                                        box[1] - pad:box[3] + pad]
    around.plot(ax=ax, color=plots.SURFACE, edgecolor=plots.GRID, linewidth=0.8)
    for prefix, color in FILL.items():
        part = sel[(sel["mun_prefix"] == prefix) & sel["ward_id"].notna()]
        part.plot(ax=ax, color=color, edgecolor=plots.SURFACE, linewidth=1.6)
    if len(free):
        free.plot(ax=ax, color=plots.NO_DATA, edgecolor=plots.INK, linewidth=1.6, hatch="///")
    sel.dissolve("mun_prefix").boundary.plot(ax=ax, color=plots.INK_2, linewidth=1.0)
    if len(moved):
        moved.boundary.plot(ax=ax, color=plots.INK, linewidth=1.8)

    halo = [pe.withStroke(linewidth=2.2, foreground=plots.SURFACE)]
    for _, row in sel.iterrows():
        p = row.geometry.representative_point()
        if pd.isna(row["ward_id"]):
            text = f"{row['ward_no']}\nno survey ward"
        else:
            text = f"{row['ward_no']}\n{int(row['n_buildings']):,} | {row['frac_g45']:.2f}"
            if row["join_note"] == MOVED:
                text += f"\nsurvey: Balephi {int(row['survey_ward_no'])}"
        ax.annotate(text, (p.x, p.y), ha="center", va="center", fontsize=8, color=plots.INK,
                    linespacing=1.15, path_effects=halo)

    x0, y0 = box[0] - pad + 300, box[1] - pad + 400
    ax.plot([x0, x0 + 5000], [y0, y0], color=plots.INK, linewidth=2)
    ax.annotate("5 km", (x0 + 2500, y0), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=8, color=plots.INK_2)
    ax.set_xlim(box[0] - pad, box[2] + pad)
    ax.set_ylim(box[1] - pad, box[3] + pad)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title("Wards of Balephi and Bahrabise: OpenStreetMap polygons with the survey values\n"
                 "Label: ward number, then buildings in the survey | fraction with grade 4 or 5",
                 fontsize=10)
    handles = [Patch(facecolor=FILL[k], edgecolor="none",
                     label=f"{LEGEND[k]}, {int((sel['mun_prefix'] == k).sum())} polygons")
               for k in FILL]
    if len(moved):
        handles.append(Patch(facecolor=FILL["Barhabise"], edgecolor=plots.INK, linewidth=1.8,
                             label="Manual link: the survey lists this ward in Balephi"))
    if len(free):
        handles.append(Patch(facecolor=plots.NO_DATA, edgecolor=plots.INK, hatch="///",
                             label="Polygon without a survey ward"))
    handles.append(Patch(facecolor=plots.SURFACE, edgecolor=plots.GRID,
                         label="Wards of other municipalities"))
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    fig.savefig(FIGURES / "balephi_barhabise_wards.png")
    print(f"\nwrote {FIGURES / 'balephi_barhabise_wards.png'}")

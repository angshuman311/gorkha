"""Write reports/gate_report.md and its figures from the files in data/."""

import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gorkha import checks, features, graph, labels, plots
from gorkha.design import TARGET
from gorkha.paths import CRS_METRIC, FIGURES, INTERIM, PROCESSED, REPORTS

GATE_1 = 0.90
GATE_2_P = 0.05


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}", index: bool = True) -> str:
    if index:
        df = df.reset_index()
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def figures(w: gpd.GeoDataFrame, res: pd.DataFrame, g: dict) -> None:
    plots.style()
    wm = w.to_crs(CRS_METRIC)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    plots.choropleth(ax, wm, w[TARGET].values, plots.SEQUENTIAL, 0, 1,
                     "fraction of buildings", "Buildings with damage grade 4 or 5, by ward",
                     district_names=True)
    fig.savefig(FIGURES / "target_map.png")
    plt.close(fig)

    names = [c for c in res.columns if c != "ward_id"]
    lim = float(np.round(np.percentile(np.abs(res[names].to_numpy()), 98), 1))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    for ax, name in zip(axes, names):
        plots.choropleth(ax, wm, res[name].values, plots.DIVERGING, -lim, lim,
                         "", name.capitalize(), colorbar=False)
    plots.scale_bar(fig, axes, plots.DIVERGING, -lim, lim, "observed minus predicted", shrink=0.7)
    fig.suptitle("Residual of the grade 4 or 5 fraction after the secondary data",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIGURES / "residual_maps.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), layout="constrained")
    plots.choropleth(axes[0, 0], wm, w["mmi"].values, plots.SEQUENTIAL,
                     float(np.floor(w["mmi"].min())), float(np.ceil(w["mmi"].max())),
                     "MMI", "ShakeMap intensity (ward mean)")
    no_dpm = (w["dpm_has_data"] == 0).values
    plots.choropleth(axes[0, 1], wm, w["dpm_p90"].values, plots.SEQUENTIAL, 0,
                     float(np.percentile(w.loc[~no_dpm, "dpm_p90"], 98)),
                     "90th percentile of the change value",
                     "Damage Proxy Map, ALOS-2 (ward 90th percentile)", mask=no_dpm,
                     mask_label="no DPM data")
    foot = gpd.GeoSeries([features.dpm_footprint()], crs="EPSG:4326").to_crs(CRS_METRIC)
    foot.boundary.plot(ax=axes[0, 1], color=plots.INK, linewidth=0.8)
    no_ls = (w["ls_mapped"] < 0.5).values
    plots.choropleth(axes[1, 0], wm, 100 * w["ls_frac"].values, plots.SEQUENTIAL, 0,
                     float(np.percentile(100 * w.loc[~no_ls, "ls_frac"], 98)),
                     "% of the examined ward area", "Landslide area (Roback et al. inventory)",
                     mask=no_ls, mask_label="less than 50% of the ward examined")
    plots.choropleth(axes[1, 1], wm, w["dist_road_km"].values, plots.SEQUENTIAL, 0,
                     float(np.percentile(w["dist_road_km"], 98)),
                     "km", "Distance from the ward centroid to a major road")
    fig.savefig(FIGURES / "feature_maps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    wm.plot(ax=ax, color="#f0efec", edgecolor=plots.SURFACE, linewidth=0.15)
    xy = w[["x_km", "y_km"]].to_numpy() * 1000
    for i, j in g["adj_pairs"]:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=plots.SERIES[0], linewidth=0.45)
    ax.scatter(xy[:, 0], xy[:, 1], s=2.5, color=plots.INK_2, zorder=3, linewidths=0)
    ax.set_title(f"Ward graph: {len(w)} nodes, {len(g['adj_pairs'])} adjacency edges")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(FIGURES / "graph.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    w = gpd.read_file(PROCESSED / "wards.gpkg")
    g = graph.load()
    res = pd.read_parquet(INTERIM / "check_residuals.parquet")
    moran = json.loads((INTERIM / "check_moran.json").read_text(encoding="utf-8"))
    j = pd.read_parquet(INTERIM / "ward_join.parquet")
    mun = pd.read_csv(REPORTS / "join_municipalities.csv")
    b = labels.buildings()
    figures(w, res, g)

    graded = b[b["grade"].notna()]
    grade_share = graded["grade"].astype(int).value_counts(normalize=True).sort_index()
    solution = pd.crosstab(graded["grade"].astype(int), graded["technical_solution_proposed"],
                           normalize="index")
    solution.index.name = "grade"

    n_buildings = int(j["n_buildings"].sum())
    matched_buildings = int(j.loc[j["matched"], "n_buildings"].sum())
    rate = matched_buildings / n_buildings
    with_b = j[j["n_buildings"] > 0]
    unmatched = with_b[~with_b["matched"]]
    pseudo = j[(j["n_buildings"] == 0)]
    by_district = j.groupby("district_name").agg(
        wards_with_buildings=("n_buildings", lambda s: int((s > 0).sum())),
        matched_wards=("matched", "sum"),
        buildings=("n_buildings", "sum"),
    )
    by_district["matched_buildings"] = j[j["matched"]].groupby("district_name")["n_buildings"].sum()
    by_district["building_match_rate"] = by_district["matched_buildings"] / by_district["buildings"]
    manual = mun[mun["accepted_by"] == "manual"]
    n_score = int((mun["accepted_by"] == "score").sum())
    below_100 = int(((mun["accepted_by"] == "score") & (mun["score"] < 100)).sum())
    osm = gpd.read_file(INTERIM / "osm_wards.gpkg")
    if len(unmatched):
        cols = ["ward_id", "district_name", "vdcmun_name", "ward_no", "n_buildings"]
        open_item = (f"Open item: {len(unmatched)} survey ward(s) with buildings have no polygon.\n\n"
                     + md_table(unmatched[cols], index=False))
    else:
        open_item = "All survey wards with buildings have a polygon."
    res_d = (w[["district_name"]].assign(r=res[res.columns[1]].values)
             .groupby("district_name")["r"].mean())

    degree = np.bincount(g["adj_pairs"].ravel(), minlength=len(w))
    feat = w[features.FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
    feat = feat.rename(columns={"50%": "median"}).rename_axis("feature")
    cover = w.groupby("district_name").agg(
        wards=("ward_id", "size"),
        wards_with_dpm=("dpm_has_data", "sum"),
        dpm_footprint_share=("dpm_cover", "mean"),
        landslide_examined_share=("ls_mapped", "mean"),
        wards_with_landslide=("ls_frac", lambda s: int((s > 0).sum())),
    )
    dpm_n = int(w["dpm_has_data"].sum())
    dpm_b = w.loc[w["dpm_has_data"] == 1, "n_buildings"].sum() / w["n_buildings"].sum()
    in_dpm = w[w["dpm_has_data"] == 1]
    dpm_corr = {c: np.corrcoef(in_dpm[c], in_dpm[TARGET])[0, 1] for c in ("dpm_mean", "dpm_p90")}
    ls_n = int((w["ls_mapped"] > 0.5).sum())

    corr = checks.correlations(pd.DataFrame(w.drop(columns="geometry")), TARGET)
    mt = pd.DataFrame(moran[TARGET]).set_index("quantity")[["R2", "I", "z", "p"]]
    mt2 = pd.DataFrame(moran["mean_grade"]).set_index("quantity")[["R2", "I", "z", "p"]]
    gate_rows = [r for r in mt.index if "in sample" in r or "spatial block" in r]
    gate_2 = all(mt.loc[r, "I"] > 0 and mt.loc[r, "p"] < GATE_2_P for r in gate_rows)
    random_row = [r for r in mt.index if "random folds" in r][0]
    semi = pd.read_csv(INTERIM / "check_semivariance.csv", index_col=0)

    text = f"""# Gate report: ward-level data for the Gorkha damage project

Date: 2026-10-06. The script `scripts/08_gate_report.py` writes this file from the files in `data/`.

## 1. Result

| Gate | Criterion | Value | Result |
|---|---|---|---|
| 1 | At least approximately 90% of the surveyed buildings match a ward polygon | {rate:.2%} ({matched_buildings:,} of {n_buildings:,}) | {"Pass" if rate >= GATE_1 else "Fail"} |
| 2 | The residual Moran's I is positive and the permutation p-value is less than {GATE_2_P} | I = {mt.loc[gate_rows[0], "I"]:.3f} and {mt.loc[gate_rows[1], "I"]:.3f}, p = {mt.loc[gate_rows[0], "p"]:.3f} and {mt.loc[gate_rows[1], "p"]:.3f} | {"Pass" if gate_2 else "Fail"} |

The graph has {len(w)} wards (nodes) and {len(g["adj_pairs"])} adjacency edges.

## 2. Survey

- Buildings: {len(b):,}. Buildings without a damage grade: {int(b["grade"].isna().sum())}.
- Wards in the name table: {len(j)}. Wards with buildings: {len(with_b)}.
- The survey uses the municipalities and wards of the federal structure of 2017.
- Buildings with grade 4 or 5: {(graded["grade"] >= 4).mean():.1%}.

Share of buildings for each damage grade:

{md_table(grade_share.rename("share").rename_axis("grade").to_frame())}

Proposed technical solution for each damage grade (row shares):

{md_table(solution)}

The survey proposes reconstruction for {solution.loc[4, "Reconstruction"]:.1%} of the grade 4 buildings and {solution.loc[5, "Reconstruction"]:.1%} of the grade 5 buildings.
Thus the label "grade 4 or 5" agrees with "reconstruction is necessary" in this data.
The grant eligibility rule in the government policy is not verified here.

![Target map](figures/target_map.png)

## 3. Join of the survey wards to the OSM polygons (gate 1)

Key: district name, municipality name, ward number.

{md_table(by_district, fmt="{:.4f}")}

Decisions in the join:

- The OSM polygons give {len(osm)} wards in {osm["mun_osm"].nunique()} municipalities for the 11 districts.
- The name score accepts {n_score} municipality pairs automatically. {below_100} of them have a score below 100 (spelling variants). The file `reports/join_municipalities.csv` lists all pairs.
- Manual decision for {len(manual)} municipality pairs with a score below 80. The pairs are: {", ".join(f"{a} = {b_}" for a, b_ in zip(manual["vdcmun_name"].str.replace(" Rural Municipality", ""), manual["mun_prefix"]))}. In each case, the pair is the only unpaired unit on each side in its district, and the ward counts are equal.
- Manual decision for one ward number: survey ward 200608 (Marin, ward 8) = OSM "Marin-07". Marin has 7 wards on each side. The survey numbers are 1 to 6 and 8.
- Manual decision for one ward with a wrong municipality in the survey: survey ward 230209 ("Balephi ward 9") = OSM "Barhabise-09". The survey has 9 wards for Balephi and 8 wards for Bahrabise. The official numbers are 8 and 9. The user checked the official ward map of Balephi (LLRC, 2016) on 2026-10-06: the area of this polygon is part of Barhabise. The polygon is adjacent to Balephi wards 3, 7, and 8, and the ward number 9 is the same on the two sides. See `figures/balephi_barhabise_wards.png`.
- The ward number comes from the OSM name. The `ward` tag is the alternative. One relation ("Khijidemba-02") has `ward=1`.
- {len(pseudo)} survey rows have ward number 99 and 0 buildings (Langtang National Park, Parsa Wild Life Reserve, and one row of Panchpokhari Thangpal). They are not wards.

{open_item}

## 4. Graph

- Nodes: {len(w)}. Adjacency edges: {len(g["adj_pairs"])}. Strict queen adjacency and adjacency with a 10 m tolerance give the same edges.
- Components: 1. Wards without a neighbor: 0. Added edges: {len(g["added_pairs"])}.
- Degree: mean {degree.mean():.2f}, minimum {degree.min()}, maximum {degree.max()}.
- Edge length (centroid distance): median {np.median(g["adj_dist_km"]):.1f} km, maximum {g["adj_dist_km"].max():.1f} km.
- Nearest neighbor edges (k = 8): {len(g["knn_pairs"])}, for the later ablation.
- Files: `data/processed/wards.gpkg`, `data/processed/wards.parquet`, `data/processed/graph.npz`. The plan named `graph.pt`. PyTorch is not installed until the GNN phase.

![Graph](figures/graph.png)

## 5. Features

All features come from data that do not use the survey building attributes.

{md_table(feat)}

Units: `pga` in %g, `pgv` in cm/s, `vs30` in m/s, `elev_mean` in m, `slope_mean` in degrees. `dpm_mean` and `dpm_p90` use (pixel value - 128) / 127, thus 0 is "no change".

Coverage by district:

{md_table(cover)}

- **Damage Proxy Map:** {dpm_n} of {len(w)} wards have a DPM statistic ({dpm_b:.1%} of the buildings). The ALOS-2 product covers a strip of approximately 70 by 180 km. Dolakha and Okhaldhunga have no DPM data. In the wards with DPM data, the correlation with the target is {dpm_corr["dpm_mean"]:.2f} for `dpm_mean` and {dpm_corr["dpm_p90"]:.2f} for `dpm_p90`.
- **Landslides:** {ls_n} of {len(w)} wards have more than 50% of the area in the examined extent (mapping extent minus obscured areas). `ls_mapped` gives this share for each ward.
- **ShakeMap:** the grid includes Vs30 (`SVEL`), thus the global Vs30 raster is not necessary.

![Feature maps](figures/feature_maps.png)

Correlation of each transformed feature with the target:

{md_table(corr)}

## 6. Spatial autocorrelation (gate 2)

Weights: row-standardized adjacency. {checks.PERMUTATIONS} permutations. Target: `{TARGET}`.

{md_table(mt, fmt="{:.4f}")}

The same check for the mean damage grade:

{md_table(mt2, fmt="{:.4f}")}

- The gate uses the first two residual sets, as the plan specifies. The two sets pass.
- The third residual set is for information. With random folds, gradient boosting trains on 80% of the wards, and these wards are between the test wards. The residual Moran's I decreases to {mt.loc[random_row, "I"]:.3f} (p = {mt.loc[random_row, "p"]:.3f}).
- Interpretation (not a measured result): with many labels, a flexible model can use the smooth features as a location signal. Thus the graph can add the most at low survey budgets. The baselines must include a flexible model without a graph.

![Residual maps](figures/residual_maps.png)

Semivariance by distance between ward centroids (variance of `{TARGET}`: {w[TARGET].var():.4f}):

{md_table(semi, fmt="{:.4f}")}

- For ward pairs at 0 to 5 km, the semivariance of the residual ({semi.iloc[0, 2]:.4f}) is not smaller than the semivariance of the target ({semi.iloc[0, 1]:.4f}). Thus the linear regression on the features does not explain the differences between adjacent wards.
- The features explain the differences at long distances: at 80 to 120 km, the semivariance decreases from {semi.iloc[-1, 1]:.4f} to {semi.iloc[-1, 2]:.4f}.

## 7. Limits of the data

1. **ShakeMap version.** The file is the USGS "atlas" product, updated 2020-07-07. It is not the map of the first week. The USGS product list also has a version of 2015-07-02.
2. **Landslide inventory.** The inventory files have the date 2017-02-09. Thus the full inventory was not available in the first week. The README lists landslide density as a permitted feature.
3. **Roads.** The major roads are the OSM roads of 2026-10-06, not the roads of April 2015.
4. **Damage Proxy Map.** Coverage is partial (Section 5). The value 128 is the fill value outside the radar footprint and also the value of masked pixels in the footprint.
5. **Building counts.** The building count of each ward comes from the survey. The metrics use it as a weight. A real survey plan must use a census count.
6. **Join.** The join has 5 manual decisions (Section 3). An error in one of them puts a label on the wrong polygon.
7. **Aftershock.** The door-to-door survey came after the Mw 7.3 aftershock of 2015-05-12. The G-DIF paper (Loos et al., 2020) states that the survey was completed by July 2016. Thus the damage grades include the aftershock damage. The ShakeMap features describe the mainshock of 2015-04-25 only. The residuals do not isolate an aftershock effect. The mean linear regression residual is {res_d["Dolakha"]:+.2f} in Dolakha, which is near the aftershock. It is {res_d["Dhading"]:+.2f} in Dhading, {res_d["Nuwakot"]:+.2f} in Nuwakot, and {res_d["Sindhupalchok"]:+.2f} in Sindhupalchok.
"""
    (REPORTS / "gate_report.md").write_text(text, encoding="utf-8")
    print(f"wrote {REPORTS / 'gate_report.md'} ({len(text.splitlines())} lines)")
    print(f"gate 1: {rate:.4f}, gate 2: {gate_2}")

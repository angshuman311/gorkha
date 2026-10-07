"""Ward features from information that is available before a survey team visits a ward.

No feature in this module uses a building attribute from the damage survey.
"""

import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterstats import zonal_stats
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .paths import CRS_METRIC, EPICENTER_LONLAT, INTERIM, PROCESSED, RAW

WGS84 = "EPSG:4326"
SHAKEMAP_BANDS = ["MMI", "PGA", "PGV", "PSA03", "PSA10", "PSA30", "SVEL"]

# In the ALOS-2 Damage Proxy Map, 128 is the fill value outside the radar footprint.
# It is also the value of the masked pixels in the footprint.
DPM_NODATA = 128
# A ward needs this number of valid 30 m pixels (4.5 ha) to get a DPM statistic.
DPM_MIN_PIXELS = 50

LANDSLIDES = RAW / "landslides" / "Roback_Nepal_final_files"

FEATURES = [
    "mmi", "pga", "pgv", "vs30",
    "mmi_after", "pga_after", "pgv_after", "mmi_max",
    "elev_mean", "slope_mean",
    "dpm_cover", "dpm_valid_frac", "dpm_mean", "dpm_p90",
    "ls_mapped", "ls_frac",
    "dist_epi_km", "dist_epi2_km", "dist_road_km", "dist_hq_km",
    "area_km2",
]


def nodes() -> gpd.GeoDataFrame:
    """Matched wards with labels and polygons. The row position is the node index."""
    j = pd.read_parquet(INTERIM / "ward_join.parquet")
    j = j[j["matched"] & (j["n_buildings"] > 0)].copy()
    j["osm_id"] = j["osm_id"].astype("int64")
    lab = pd.read_parquet(INTERIM / "ward_labels.parquet")
    osm = gpd.read_file(INTERIM / "osm_wards.gpkg")[["osm_id", "area_km2", "geometry"]]
    cols = ["ward_id", "district_name", "vdcmun_id", "vdcmun_name", "ward_no", "osm_id"]
    n = j[cols].merge(lab, on="ward_id").merge(osm, on="osm_id")
    n = gpd.GeoDataFrame(n, geometry="geometry", crs=WGS84)
    return n.sort_values("ward_id").reset_index(drop=True)


def shakemap_raster(grid: Path | None = None, out: Path | None = None) -> Path:
    """Convert a ShakeMap grid.xml to a GeoTIFF with one band for each field.

    The default is the mainshock grid of the feature table.
    """
    grid = grid or RAW / "shakemap" / "grid.xml"
    out = out or INTERIM / "shakemap.tif"
    if out.exists():
        return out
    text = grid.read_text(encoding="utf-8")
    spec = dict(
        re.findall(r'(\w+)="([^"]*)"', re.search(r"<grid_specification([^>]*)>", text).group(1))
    )
    names = re.findall(r'<grid_field index="\d+" name="(\w+)"', text)
    start = text.index("<grid_data>") + len("<grid_data>")
    data = np.array(text[start : text.index("</grid_data>")].split(), dtype="float64")
    data = data.reshape(-1, len(names))
    nlon, nlat = int(spec["nlon"]), int(spec["nlat"])
    lon = data[:, names.index("LON")].reshape(nlat, nlon)
    lat = data[:, names.index("LAT")].reshape(nlat, nlon)
    # The rows go from north to south and the columns go from west to east.
    if not (lat[0, 0] > lat[-1, 0] and lon[0, 0] < lon[0, -1]):
        raise RuntimeError("the ShakeMap grid order is not north-up")
    dx = (float(spec["lon_max"]) - float(spec["lon_min"])) / (nlon - 1)
    dy = (float(spec["lat_max"]) - float(spec["lat_min"])) / (nlat - 1)
    # The grid points are cell centers.
    transform = from_origin(float(spec["lon_min"]) - dx / 2, float(spec["lat_max"]) + dy / 2, dx, dy)
    INTERIM.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out, "w", driver="GTiff", height=nlat, width=nlon, count=len(SHAKEMAP_BANDS),
        dtype="float32", crs=WGS84, transform=transform,
    ) as dst:
        for i, band in enumerate(SHAKEMAP_BANDS, 1):
            dst.write(data[:, names.index(band)].reshape(nlat, nlon).astype("float32"), i)
            dst.set_band_description(i, band)
    return out


def dem_rasters() -> tuple[Path, Path]:
    """Mosaic the DEM tiles and calculate the slope in degrees."""
    elev_path, slope_path = INTERIM / "dem.tif", INTERIM / "slope.tif"
    if elev_path.exists() and slope_path.exists():
        return elev_path, slope_path
    sources = [rasterio.open(p) for p in sorted((RAW / "dem").glob("*.tif"))]
    elev, transform = merge(sources)
    for s in sources:
        s.close()
    elev = elev[0].astype("float32")
    # Pixel size in meters. The east-west size changes with the latitude of the row.
    lat = transform.f + transform.e * (np.arange(elev.shape[0]) + 0.5)
    dy = abs(transform.e) * 110_574.0
    dx = transform.a * 111_320.0 * np.cos(np.radians(lat))
    gy = np.gradient(elev, axis=0) / dy
    gx = np.gradient(elev, axis=1) / dx[:, None]
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    profile = dict(driver="GTiff", height=elev.shape[0], width=elev.shape[1], count=1,
                   dtype="float32", crs=WGS84, transform=transform, compress="deflate")
    for path, array in ((elev_path, elev), (slope_path, slope)):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)
    return elev_path, slope_path


def dpm_footprint() -> Polygon:
    """Radar footprint of the ALOS-2 Damage Proxy Map."""
    text = (RAW / "dpm" / "ARIA_DPM_ALOS2_Radar_Footprint_all.kml").read_text(encoding="utf-8")
    coords = re.search(r"<coordinates>(.*?)</coordinates>", text, flags=re.S).group(1)
    return Polygon([tuple(map(float, p.split(",")[:2])) for p in coords.split()]).buffer(0)


def dpm_raster() -> Path:
    """Mosaic the three ALOS-2 frames. A valid pixel has priority over a fill pixel."""
    out = INTERIM / "dpm_alos2.tif"
    if out.exists():
        return out
    sources = [rasterio.open(p) for p in sorted((RAW / "dpm" / "alos2").glob("ARIA*/*dpmRaw.tif"))]
    data, transform = merge(sources, nodata=DPM_NODATA, method="first")
    for s in sources:
        s.close()
    with rasterio.open(
        out, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=1,
        dtype="uint8", crs=WGS84, transform=transform, nodata=DPM_NODATA, compress="deflate",
    ) as dst:
        dst.write(data[0], 1)
    return out


def zonal(geoms, path: Path, stats: list, band: int = 1, all_touched: bool = False,
          nodata=None) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = zonal_stats(list(geoms), str(path), band=band, stats=stats,
                        all_touched=all_touched, nodata=nodata)
    return pd.DataFrame(z)


def _major_roads() -> gpd.GeoSeries:
    elements = json.loads((RAW / "osm" / "major_roads.json").read_text(encoding="utf-8"))["elements"]
    lines = [
        LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
        for e in elements
        if e["type"] == "way" and len(e.get("geometry", [])) >= 2
    ]
    return gpd.GeoSeries(lines, crs=WGS84).to_crs(CRS_METRIC)


def build() -> gpd.GeoDataFrame:
    n = nodes()
    nm = n.to_crs(CRS_METRIC)
    area = nm.geometry.area
    cent = nm.geometry.centroid
    f = pd.DataFrame(index=n.index)

    # Ground motion and site condition. The ShakeMap cell is approximately 1.8 km, thus a
    # small ward can contain no cell center. `all_touched` includes each cell that the
    # ward touches.
    # Two earthquakes: the mainshock of 2015-04-25 and the aftershock of 2015-05-12. The
    # survey came after the two events. The grids are the USGS versions of July 2015.
    shake = shakemap_raster(RAW / "shakemap" / "grid_main_2015.xml",
                            INTERIM / "shakemap_main_2015.tif")
    for i, band in enumerate(SHAKEMAP_BANDS, 1):
        f[band.lower()] = zonal(n.geometry, shake, ["mean"], band=i, all_touched=True)["mean"]
    f = f.rename(columns={"svel": "vs30"})
    after = shakemap_raster(RAW / "shakemap" / "grid_after_2015.xml",
                            INTERIM / "shakemap_after_2015.tif")
    for band in ("MMI", "PGA", "PGV"):
        i = SHAKEMAP_BANDS.index(band) + 1
        f[f"{band.lower()}_after"] = zonal(n.geometry, after, ["mean"], band=i,
                                           all_touched=True)["mean"]
    f["mmi_max"] = f[["mmi", "mmi_after"]].max(axis=1)

    elev, slope = dem_rasters()
    f["elev_mean"] = zonal(n.geometry, elev, ["mean"])["mean"]
    f["slope_mean"] = zonal(n.geometry, slope, ["mean"])["mean"]

    # Damage Proxy Map. The values are (pixel - 128) / 127, thus 0 is "no change".
    footprint = gpd.GeoSeries([dpm_footprint()], crs=WGS84).to_crs(CRS_METRIC).iloc[0]
    f["dpm_cover"] = (nm.geometry.intersection(footprint).area / area).clip(0, 1)
    dpm = dpm_raster()
    z = zonal(n.geometry, dpm, ["count", "mean", "percentile_90"], nodata=DPM_NODATA)
    with rasterio.open(dpm) as src:
        res_x, res_y = src.res
    lat = n.to_crs(CRS_METRIC).geometry.centroid.to_crs(WGS84).y
    pixel_m2 = (res_x * 111_320.0 * np.cos(np.radians(lat))) * (res_y * 110_574.0)
    f["dpm_valid_frac"] = (z["count"].fillna(0).values * pixel_m2.values / area.values).clip(0, 1)
    enough = z["count"].fillna(0).values >= DPM_MIN_PIXELS
    f["dpm_mean"] = np.where(enough, (z["mean"].astype(float) - 128) / 127, 0.0)
    f["dpm_p90"] = np.where(enough, (z["percentile_90"].astype(float) - 128) / 127, 0.0)
    f["dpm_has_data"] = enough.astype(int)

    # Landslides. `ls_mapped` is the share of the ward that the inventory examined.
    full = gpd.read_file(LANDSLIDES / "Full20170209.shp").to_crs(CRS_METRIC)
    full["geometry"] = full.geometry.buffer(0)
    extent = unary_union(gpd.read_file(LANDSLIDES / "MappingExtent20170209.shp")
                         .to_crs(CRS_METRIC).geometry.buffer(0).values)
    obscured = unary_union(gpd.read_file(LANDSLIDES / "ObscuredAreas20170209.shp")
                           .to_crs(CRS_METRIC).geometry.buffer(0).values)
    mapped_area = nm.geometry.intersection(extent.difference(obscured)).area
    f["ls_mapped"] = (mapped_area / area).clip(0, 1)
    inter = gpd.overlay(nm[["ward_id", "geometry"]], full[["geometry"]],
                        how="intersection", keep_geom_type=True)
    ls_area = inter.dissolve("ward_id").geometry.area
    ls_area = ls_area.reindex(n["ward_id"]).fillna(0).values
    f["ls_frac"] = np.where(mapped_area.values > 0, ls_area / mapped_area.values.clip(1), 0.0)
    f["ls_frac"] = f["ls_frac"].clip(0, 1)

    # Distances from the ward centroid.
    epicenter = gpd.GeoSeries([Point(EPICENTER_LONLAT)], crs=WGS84).to_crs(CRS_METRIC).iloc[0]
    f["dist_epi_km"] = cent.distance(epicenter) / 1000
    # Epicenter of the second earthquake, from the header of its ShakeMap grid.
    head = (RAW / "shakemap" / "grid_after_2015.xml").read_text(encoding="utf-8")[:3000]
    event = dict(re.findall(r'(\w+)="([^"]*)"', re.search(r"<event ([^>]*)>", head).group(1)))
    second = gpd.GeoSeries([Point(float(event["lon"]), float(event["lat"]))], crs=WGS84)
    f["dist_epi2_km"] = cent.distance(second.to_crs(CRS_METRIC).iloc[0]) / 1000
    f["dist_road_km"] = cent.distance(unary_union(_major_roads().values)) / 1000
    d = gpd.read_file(INTERIM / "osm_districts.gpkg")
    hq = gpd.GeoSeries(gpd.points_from_xy(d["hq_lon"], d["hq_lat"]), crs=WGS84).to_crs(CRS_METRIC)
    hq = dict(zip(d["district"], hq))
    f["dist_hq_km"] = [c.distance(hq[k]) / 1000 for c, k in zip(cent, n["district_name"])]

    f["x_km"] = cent.x.values / 1000
    f["y_km"] = cent.y.values / 1000

    out = pd.concat([n, f], axis=1)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=WGS84)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_file(PROCESSED / "wards.gpkg", driver="GPKG")
    pd.DataFrame(out.drop(columns="geometry")).to_parquet(PROCESSED / "wards.parquet", index=False)
    return out

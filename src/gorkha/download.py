"""Download the raw data. Each function skips a file that is already on disk."""

import json
import time
import zipfile
from pathlib import Path

import requests

from .paths import RAW, STUDY_BBOX

USER_AGENT = "gorkha-cs230-project/0.1 (course project)"

KAGGLE_SURVEY_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "arashnic/earthquake-magnitude-damage-and-impact"
)
SHAKEMAP_GRID_URL = (
    "https://earthquake.usgs.gov/product/shakemap/us20002926/atlas/"
    "1594162031303/download/grid.xml"
)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
LANDSLIDES_URL = (
    "https://www.sciencebase.gov/catalog/file/get/582c74fbe4b04d580bd377e8"
    "?f=__disk__47%2F64%2F62%2F476462808bb07b2ea46b6ebe15d2214bbcfc5078"
)
GDIF_URL = (
    "https://stacks.stanford.edu/file/druid:gn368cq4893/"
    "GDIF-damageprediction-submission-12032020.zip"
)
DPM_BASE_URL = "https://d1z62tir4fw0q0.cloudfront.net/20150425-Nepal_EQ/DPM/"
DPM_FILES = [
    "ARIA_DPM_ALOS2_v0.5u_dpmRaw_GeoTiff.zip",
    "ARIA_DPM_ALOS2_Radar_Footprint_all.kml",
    "ARIA_DamageProxyMap_v0.5u_CSKd_20150430.tif",
    "ARIA_DPM_CSKd_Radar_Footprint.kml",
]
DEM_URL = (
    "https://copernicus-dem-90m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_30_{lat}_00_{lon}_00_DEM/"
    "Copernicus_DSM_COG_30_{lat}_00_{lon}_00_DEM.tif"
)


def fetch(url: str, dest: Path, timeout: int = 600) -> Path:
    """Download `url` to `dest`. Write to a temporary file first."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip   {dest.name} (on disk)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(
        url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
    ) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    print(f"done   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def survey() -> Path:
    """Damage survey CSV files from the Kaggle mirror of the NPC / KLL portal."""
    out = RAW / "survey"
    archive = fetch(KAGGLE_SURVEY_URL, RAW / "survey.zip", timeout=1800)
    if not any(out.glob("*.csv")):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    for p in sorted(out.rglob("*.csv")):
        print(f"       {p.relative_to(out)} ({p.stat().st_size / 1e6:.1f} MB)")
    return out


def shakemap() -> Path:
    return fetch(SHAKEMAP_GRID_URL, RAW / "shakemap" / "grid.xml")


# Other ShakeMap grids for the sensitivity checks (scripts/12_shakemap_checks.py).
# Mainshock us20002926 and aftershock us20002ejl (Mw 7.3, 2015-05-12).
_USGS = "https://earthquake.usgs.gov/product/shakemap/"
SHAKEMAP_VARIANTS = {
    "main_2015": _USGS + "us20002926/us/1435877532354/download/grid.xml",  # 2015-07-02
    "after_2015": _USGS + "us20002ejl/us/1436516848284/download/grid.xml",  # 2015-07-10
    "after_2020": _USGS + "us20002ejl/atlas/1594162075191/download/grid.xml",  # 2020-07-07
}


def shakemap_variants() -> dict:
    return {k: fetch(u, RAW / "shakemap" / f"grid_{k}.xml") for k, u in SHAKEMAP_VARIANTS.items()}


def overpass(query: str, dest: Path, tries: int = 3) -> Path:
    """Run an Overpass query. Try each endpoint in sequence."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip   {dest.name} (on disk)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(tries):
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                r = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=900,
                    headers={"User-Agent": USER_AGENT},
                )
                r.raise_for_status()
                data = r.json()
                if not data.get("elements"):
                    raise RuntimeError(f"no elements, remark={data.get('remark')}")
                dest.write_text(json.dumps(data), encoding="utf-8")
                print(f"done   {dest.name} ({len(data['elements'])} elements, {endpoint})")
                return dest
            except Exception as e:  # network errors and Overpass load errors
                last_error = e
                print(f"retry  {dest.name}: {endpoint}: {str(e)[:120]}")
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"all Overpass endpoints failed for {dest.name}") from last_error


def _bbox() -> str:
    s, w, n, e = STUDY_BBOX
    return f"{s},{w},{n},{e}"


def admin_boundaries() -> list[Path]:
    """District (6), municipality (7), and ward (9) relations in the study box."""
    out = []
    for level in (6, 7, 9):
        query = (
            "[out:json][timeout:600];"
            f'rel["boundary"="administrative"]["admin_level"="{level}"]({_bbox()});'
            "out geom;"
        )
        out.append(overpass(query, RAW / "osm" / f"admin_level_{level}.json"))
    return out


def roads() -> Path:
    query = (
        "[out:json][timeout:600];"
        f'way["highway"~"^(trunk|primary|secondary)$"]({_bbox()});'
        "out geom;"
    )
    return overpass(query, RAW / "osm" / "major_roads.json")


def _fetch_zip(url: str, name: str) -> Path:
    out = RAW / name
    archive = fetch(url, RAW / f"{name}.zip", timeout=1800)
    if not out.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    return out


def landslides() -> Path:
    """Roback et al. landslide inventory (USGS, doi:10.5066/F7DZ06F9)."""
    return _fetch_zip(LANDSLIDES_URL, "landslides")


def gdif() -> Path:
    """G-DIF code and data (Loos et al. 2020, CC BY-NC-SA 3.0)."""
    return _fetch_zip(GDIF_URL, "gdif")


def roads_full() -> list[Path]:
    """All roads that a vehicle can use, in 6 tiles of the study box (for the planner)."""
    s0, w0, n0, e0 = STUDY_BBOX
    classes = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track"
               "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|road")
    out = []
    lats = [s0, (s0 + n0) / 2, n0]
    lons = [w0 + k * (e0 - w0) / 3 for k in range(4)]
    for i in range(2):
        for j in range(3):
            box = f"{lats[i]},{lons[j]},{lats[i + 1]},{lons[j + 1]}"
            query = f'[out:json][timeout:900];way["highway"~"^({classes})$"]({box});out geom;'
            out.append(overpass(query, RAW / "osm" / f"roads_full_{i}{j}.json"))
    return out


def dpm() -> Path:
    """ARIA Damage Proxy Maps (ALOS-2 and COSMO-SkyMed) and their footprints.

    The file links come from https://aria-share.jpl.nasa.gov/20150425-Nepal_EQ/DPM/.
    """
    out = RAW / "dpm"
    for name in DPM_FILES:
        fetch(DPM_BASE_URL + name, out / name)
    archive = out / "ARIA_DPM_ALOS2_v0.5u_dpmRaw_GeoTiff.zip"
    if not any(out.glob("alos2/**/*.tif")):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out / "alos2")
    for p in sorted(out.rglob("*.tif")):
        print(f"       {p.relative_to(out)} ({p.stat().st_size / 1e6:.1f} MB)")
    return out


def dem() -> list[Path]:
    """Copernicus GLO-90 tiles that cover the study box."""
    s, w, n, e = STUDY_BBOX
    out = []
    for lat in range(int(s), int(n) + 1):
        for lon in range(int(w), int(e) + 1):
            url = DEM_URL.format(lat=f"N{lat:02d}", lon=f"E{lon:03d}")
            out.append(fetch(url, RAW / "dem" / f"N{lat:02d}_E{lon:03d}.tif"))
    return out

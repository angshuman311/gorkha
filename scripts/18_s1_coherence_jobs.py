"""Submit Sentinel-1 coherence jobs to the ASF HyP3 service, or show their status.

Usage: python scripts/18_s1_coherence_jobs.py [plan|submit|status|download]
The login comes from the file ~/.netrc (NASA Earthdata). The script does not read it directly.
Three pairs for each orbit: before the first earthquake ("pre"), across the first earthquake
("main", 2015-04-25), and across the second earthquake ("after", 2015-05-12).
"""

import sys

import asf_search as asf
import geopandas as gpd
import hyp3_sdk
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union

from gorkha.paths import CRS_METRIC, INTERIM, PROCESSED, RAW

JOB_NAME = "sherpa-gorkha"
LOOKS = "10x2"  # 40 m pixels, 15 credits for each job
# Two scenes of adjacent frames overlap by a small share. Such a pair fails at ASF
# ("no points available for determining average intensity"). A pair of the same frame
# overlaps by 0.3 or more.
MIN_OVERLAP = 0.3
PAIRS = {
    12: [("pre", "2015-04-04", "2015-04-16"), ("main", "2015-04-16", "2015-04-28"),
         ("after", "2015-04-28", "2015-05-22")],
    19: [("pre", "2015-04-05", "2015-04-17"), ("main", "2015-04-17", "2015-04-29"),
         ("after", "2015-05-11", "2015-05-23")],
    85: [("pre", "2015-04-09", "2015-04-21"), ("main", "2015-04-21", "2015-05-03"),
         ("after", "2015-05-03", "2015-05-15")],
    121: [("pre", "2015-04-12", "2015-04-24"), ("main", "2015-04-24", "2015-05-06"),
          ("after", "2015-05-06", "2015-05-18")],
}
# More pairs from before the first earthquake, because one pair for each orbit does not
# cover all wards. Some have 24 days between the two dates.
for _path, _d1, _d2 in [(12, "2015-03-23", "2015-04-16"), (12, "2015-03-23", "2015-04-04"),
                        (19, "2015-03-24", "2015-04-17"), (19, "2015-03-24", "2015-04-05"),
                        (85, "2015-03-28", "2015-04-21"), (85, "2015-03-28", "2015-04-09"),
                        (121, "2015-03-31", "2015-04-24"), (121, "2015-03-31", "2015-04-12")]:
    PAIRS[_path].append(("pre", _d1, _d2))


def plan() -> pd.DataFrame:
    wards = gpd.read_file(PROCESSED / "wards.gpkg")
    area = unary_union(wards.geometry.values)
    x0, y0, x1, y1 = wards.total_bounds
    box = f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"
    found = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1A], beamMode=[asf.BEAMMODE.IW],
                           processingLevel=[asf.PRODUCT_TYPE.SLC], intersectsWith=box,
                           start="2015-03-15", end="2015-06-05")
    scenes = [{"name": p.properties["sceneName"], "date": p.properties["startTime"][:10],
               "path": p.properties["pathNumber"], "geom": shape(p.geometry)} for p in found]
    jobs, cover = [], {}
    for path, pairs in PAIRS.items():
        for kind, d1, d2 in pairs:
            first = [s for s in scenes if s["path"] == path and s["date"] == d1]
            second = [s for s in scenes if s["path"] == path and s["date"] == d2]
            for a in first:
                for b in second:
                    common = a["geom"].intersection(b["geom"])
                    if common.area / a["geom"].area > MIN_OVERLAP and common.intersects(area):
                        jobs.append({"path": path, "kind": kind, "reference": a["name"],
                                     "secondary": b["name"], "d1": d1, "d2": d2})
                        cover.setdefault(kind, []).append(common)
    points = wards.to_crs(CRS_METRIC).geometry.centroid.to_crs(wards.crs)
    for kind, shapes in cover.items():
        inside = int(points.within(unary_union(shapes)).sum())
        print(f"{kind}: {sum(j['kind'] == kind for j in jobs)} jobs, {inside} of {len(wards)} wards covered")
    out = pd.DataFrame(jobs)
    out.to_csv(INTERIM / "s1_insar_jobs.csv", index=False)
    print(f"{len(out)} jobs, {15 * len(out)} credits")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if mode == "plan":
        plan()
    else:
        hyp3 = hyp3_sdk.HyP3()
        if mode == "submit":
            jobs = plan()
            for j in jobs.itertuples():
                hyp3.submit_insar_job(j.reference, j.secondary, name=JOB_NAME, looks=LOOKS,
                                      include_inc_map=True, include_look_vectors=False,
                                      include_dem=False, apply_water_mask=False)
            print("submitted. credits that remain:", hyp3.my_info()["remaining_credits"])
        batch = hyp3.find_jobs(name=JOB_NAME)
        print("jobs by status:", pd.Series([j.status_code for j in batch]).value_counts().to_dict())
        if mode == "download":
            done = hyp3_sdk.Batch([j for j in batch if j.succeeded()])
            done.download_files(RAW / "s1_coherence")

"""Check of the landslide-to-road-to-access model against observed access.

Observed: MapAction, Sindhupalchok, 6 May 2015. 79 VDC polygons with a vehicular access
status from the district disaster committee: "Accessible" or "pre-EQ" (access before the
earthquake only, read here as cut off).
Model: a VDC has access if a road route exists from the district headquarters (Chautara)
to the nearest road node of the VDC centroid, with the road pieces of each blockage level
closed.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from gorkha import roads
from gorkha.paths import CRS_METRIC, PROCESSED, RAW

HQ_LONLAT = (85.715, 27.775)  # Chautara
LEVELS = {"0": 1, "obs": 1, "30": 3, "60": 3, "100": 1}

if __name__ == "__main__":
    vdc = gpd.read_file(next((RAW / "mapaction").rglob("*.shp"))).to_crs(CRS_METRIC)
    observed = (vdc["access"] == "Accessible").to_numpy()
    r = roads.load()
    xy, a, b, hours = r["node_xy"], r["edge_a"], r["edge_b"], r["edge_hours"]
    n = len(xy)
    full = coo_matrix((hours, (a, b)), shape=(n, n))
    _, label = connected_components(full, directed=False)
    main = np.flatnonzero(label == np.bincount(label).argmax())
    tree = cKDTree(xy[main])
    hq = main[tree.query(gpd.GeoSeries.from_xy([HQ_LONLAT[0]], [HQ_LONLAT[1]], crs="EPSG:4326")
                         .to_crs(CRS_METRIC).iloc[0].coords[0])[1]]
    pts = np.column_stack([vdc.geometry.centroid.x, vdc.geometry.centroid.y])
    dist, near = tree.query(pts)
    nodes = main[near]
    crossed = np.flatnonzero(r["edge_blocked"])
    rows = []
    for level, draws in LEVELS.items():
        for d in range(draws):
            open_edge = np.ones(len(hours), dtype=bool)
            if level == "obs":
                open_edge[np.load(PROCESSED / "roads_obs_blocked_edges.npy")] = False
            elif level != "0":
                rng = np.random.default_rng([23, int(level), d])
                open_edge[rng.choice(crossed, size=int(round(len(crossed) * int(level) / 100)), replace=False)] = False
            g = coo_matrix((hours[open_edge], (a[open_edge], b[open_edge])), shape=(n, n)).tocsr()
            t = dijkstra(g, directed=False, indices=hq)[nodes]
            model = np.isfinite(t)
            hit = (model == observed)
            rows.append({"level": level, "draw": d, "VDCs with access, model": int(model.sum()),
                         "VDCs with access, observed": int(observed.sum()),
                         "agreement": hit.mean(),
                         "cut off and model agrees": float((~observed & ~model).sum() / max((~observed).sum(), 1)),
                         "accessible and model agrees": float((observed & model).sum() / observed.sum())})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    t.to_csv(PROCESSED.parent / "interim" / "access_validation.csv", index=False)
    cut = vdc.loc[~observed, ["admin4Name", "Dist_hub"]]
    print("\nVDCs observed as cut off on 6 May 2015:", ", ".join(cut["admin4Name"]))

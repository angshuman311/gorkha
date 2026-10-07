"""Build the 2015 road and trail graph, the travel times, and the blockage levels, then check
the access model against the observed access of the Sindhupalchok VDCs (6 May 2015)."""

import time

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from gorkha import roads2015
from gorkha.paths import CRS_METRIC, RAW

HQ_LONLAT = (85.715, 27.775)  # Chautara

if __name__ == "__main__":
    t0 = time.time()
    r = roads2015.build()
    g, blocked = r["graph"], r["blocked"]
    km = np.array([line.length for line in g["lines"]]) / 1000
    print(f"graph of 2015: {len(g['xy'])} nodes, {len(km)} edges, {km[g['vehicle']].sum():.0f} km of vehicle road, "
          f"{km[~g['vehicle']].sum():.0f} km of trail, {time.time() - t0:.0f} s")
    print(f"edges that a landslide crosses: {int(blocked.sum())} (vehicle {int((blocked & g['vehicle']).sum())}, "
          f"trail {int((blocked & ~g['vehicle']).sum())})")
    for k, t in r["levels"].items():
        off = ~np.eye(len(t), dtype=bool)
        fin = np.isfinite(t[off])
        print(f"{k}: ward pairs with a route {fin.mean():.3f}, median travel time {np.median(t[off][fin]):.1f} h")

    # Access check against the observed VDC status.
    vdc = gpd.read_file(next((RAW / "mapaction").rglob("*.shp"))).to_crs(CRS_METRIC)
    observed = (vdc["access"] == "Accessible").to_numpy()
    xy, a, b, hours = g["xy"], g["a"], g["b"], g["hours"]
    n = len(xy)
    _, label = connected_components(coo_matrix((hours, (a, b)), shape=(n, n)), directed=False)
    main = np.flatnonzero(label == np.bincount(label).argmax())
    tree = cKDTree(xy[main])
    hq_pt = gpd.GeoSeries.from_xy([HQ_LONLAT[0]], [HQ_LONLAT[1]], crs="EPSG:4326").to_crs(CRS_METRIC).iloc[0]
    hq = main[tree.query(hq_pt.coords[0])[1]]
    pts = np.column_stack([vdc.geometry.centroid.x, vdc.geometry.centroid.y])
    nodes = main[tree.query(pts)[1]]
    crossed = np.flatnonzero(blocked)
    print("\naccess check, VDCs of Sindhupalchok (12 observed as cut off, 67 accessible):")
    for level, draws in roads2015.BLOCK_LEVELS.items():
        for d in range(draws):
            for mode in ("vehicle roads only", "roads and trails"):
                open_edge = np.ones(len(hours), dtype=bool)
                if level != "0":
                    rng = np.random.default_rng([29, int(level), d])
                    open_edge[rng.choice(crossed, size=int(round(len(crossed) * int(level) / 100)), replace=False)] = False
                if mode == "vehicle roads only":
                    open_edge &= g["vehicle"]
                gm = coo_matrix((hours[open_edge], (a[open_edge], b[open_edge])), shape=(n, n)).tocsr()
                t = dijkstra(gm, directed=False, indices=hq)[nodes]
                # With trails, a VDC counts as accessible if the route takes less than 12 hours.
                model = np.isfinite(t) if mode == "vehicle roads only" else (t < 12)
                print(f"  level {level:>3} draw {d} {mode:19s}: cut off agreed {int((~observed & ~model).sum())} of 12, "
                      f"accessible agreed {int((observed & model).sum())} of 67, agreement {(model == observed).mean():.2f}")

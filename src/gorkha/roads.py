"""Road graph of the study area and travel times between wards.

Source: OpenStreetMap roads of 2026 (data/raw/osm/roads_full_*.json).
The graph keeps only the junctions and the ends of roads as nodes. An edge is the piece of
road between two of them, with a travel time from the road class.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString

from .features import LANDSLIDES
from .paths import CRS_METRIC, PROCESSED, RAW

# Speed for each road class in km/h (mountain roads). These are assumptions.
SPEED = {"motorway": 50, "trunk": 40, "primary": 35, "secondary": 30, "tertiary": 25,
         "unclassified": 15, "residential": 15, "service": 15, "road": 15, "track": 10}
WALK_KMH = 3.0            # from the ward centroid to the nearest road
SURVEY_HOURS = 8.0        # fixed time for the survey of one ward
HELICOPTER_HOURS = 12.0   # cost of a visit to a ward that no open road reaches
LANDSLIDE_BUFFER_M = 15.0


def build_graph() -> dict:
    """Read the OSM tiles and return the simplified road graph."""
    ways = {}
    for path in sorted((RAW / "osm").glob("roads_full_*.json")):
        for e in json.loads(path.read_text(encoding="utf-8"))["elements"]:
            if e["type"] == "way" and len(e.get("nodes", [])) >= 2:
                ways[e["id"]] = e
    count = {}
    for w in ways.values():
        for k, n in enumerate(w["nodes"]):
            count[n] = count.get(n, 0) + (2 if k in (0, len(w["nodes"]) - 1) else 1)
    to_xy = Transformer.from_crs("EPSG:4326", CRS_METRIC, always_xy=True)
    index, coords, a, b, hours, lines = {}, [], [], [], [], []

    def node(osm_id, pt):
        if osm_id not in index:
            index[osm_id] = len(coords)
            coords.append(pt)
        return index[osm_id]

    for w in ways.values():
        kind = w["tags"]["highway"].replace("_link", "")
        speed = SPEED.get(kind, 15)
        lon = [p["lon"] for p in w["geometry"]]
        lat = [p["lat"] for p in w["geometry"]]
        x, y = to_xy.transform(lon, lat)
        pts = np.column_stack([x, y])
        start = 0
        for k in range(1, len(w["nodes"])):
            # Cut the way at each junction (a node that more than one way uses) and at its end.
            if count[w["nodes"][k]] > 1 or k == len(w["nodes"]) - 1:
                seg = pts[start:k + 1]
                length = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
                if length > 0:
                    a.append(node(w["nodes"][start], seg[0]))
                    b.append(node(w["nodes"][k], seg[-1]))
                    hours.append(length / 1000 / speed)
                    lines.append(LineString(seg))
                start = k
    return {"xy": np.array(coords), "a": np.array(a), "b": np.array(b),
            "hours": np.array(hours), "lines": lines}


def blocked_edges(graph: dict) -> np.ndarray:
    """Boolean for each edge: the road crosses a landslide of the USGS inventory."""
    slides = gpd.read_file(LANDSLIDES / "Full20170209.shp").to_crs(CRS_METRIC)
    shapes = slides.geometry.buffer(LANDSLIDE_BUFFER_M).values
    tree = STRtree(graph["lines"])
    hit = tree.query(shapes, predicate="intersects")[1]
    out = np.zeros(len(graph["lines"]), dtype=bool)
    out[np.unique(hit)] = True
    return out


def ward_travel_hours(graph: dict, wards_xy_m: np.ndarray, open_edge: np.ndarray | None = None) -> np.ndarray:
    """Travel time in hours between all pairs of wards. Infinite if no open road connects them.

    A ward connects to the nearest node of the largest connected part of the full network,
    with a walk from its centroid.
    """
    n = len(graph["xy"])
    full = coo_matrix((graph["hours"], (graph["a"], graph["b"])), shape=(n, n))
    _, label = connected_components(full, directed=False)
    main = np.flatnonzero(label == np.bincount(label).argmax())
    dist, near = cKDTree(graph["xy"][main]).query(wards_xy_m)
    access = main[near]
    walk = dist / 1000 / WALK_KMH
    keep = np.ones(len(graph["hours"]), dtype=bool) if open_edge is None else open_edge
    g = coo_matrix((graph["hours"][keep], (graph["a"][keep], graph["b"][keep])), shape=(n, n)).tocsr()
    uniq, inverse = np.unique(access, return_inverse=True)
    d = dijkstra(g, directed=False, indices=uniq)[:, uniq]
    t = d[inverse][:, inverse] + walk[:, None] + walk[None, :]
    np.fill_diagonal(t, 0.0)
    return t


def build() -> dict:
    wards = pd.read_parquet(PROCESSED / "wards.parquet", columns=["ward_id", "x_km", "y_km"])
    xy = wards[["x_km", "y_km"]].to_numpy() * 1000
    graph = build_graph()
    blocked = blocked_edges(graph)
    t_open = ward_travel_hours(graph, xy)
    t_blocked = ward_travel_hours(graph, xy, ~blocked)
    np.savez_compressed(PROCESSED / "roads.npz", ward_id=wards["ward_id"].to_numpy(),
                        t_open=t_open.astype("float32"), t_blocked=t_blocked.astype("float32"),
                        edge_a=graph["a"], edge_b=graph["b"], edge_hours=graph["hours"],
                        edge_blocked=blocked, node_xy=graph["xy"])
    return {"graph": graph, "blocked": blocked, "t_open": t_open, "t_blocked": t_blocked}


def load() -> dict:
    with np.load(PROCESSED / "roads.npz") as f:
        return {k: f[k] for k in f.files}


BLOCK_LEVELS = {0: 1, 30: 3, 60: 3, 100: 1}   # share of crossed road pieces in %, and draws


def level_matrices() -> dict:
    """Travel times for each blockage level. A level blocks a random share of the road
    pieces that a landslide crosses. Each draw uses a different random selection."""
    r = load()
    wards = pd.read_parquet(PROCESSED / "wards.parquet", columns=["x_km", "y_km"])
    xy = wards.to_numpy() * 1000
    graph = {"xy": r["node_xy"], "a": r["edge_a"], "b": r["edge_b"], "hours": r["edge_hours"]}
    crossed = np.flatnonzero(r["edge_blocked"])
    out = {}
    for level, draws in BLOCK_LEVELS.items():
        for d in range(draws):
            rng = np.random.default_rng([23, level, d])
            closed = rng.choice(crossed, size=int(round(len(crossed) * level / 100)), replace=False)
            open_edge = np.ones(len(r["edge_hours"]), dtype=bool)
            open_edge[closed] = False
            out[f"t_{level}_{d}"] = ward_travel_hours(graph, xy, open_edge).astype("float32")
    np.savez_compressed(PROCESSED / "roads_levels.npz", **out)
    return out

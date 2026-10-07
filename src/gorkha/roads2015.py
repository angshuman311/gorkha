"""Road and trail graph of 2015 (OCHA ROAP, Survey Department 25K/50K transport layer).

The layer has vehicle roads and foot trails as arcs between numbered nodes (FNODE_, TNODE_).
A team drives on the roads and walks on the trails. The output has the same form as the
OpenStreetMap graph of roads.py, so the campaigns can use either.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree

from .features import LANDSLIDES
from .paths import CRS_METRIC, PROCESSED, RAW
from .roads import LANDSLIDE_BUFFER_M, ward_travel_hours

SHAPE = RAW / "roads_2015" / "road_network_2015"

# Speed in km/h for each class (assumptions). Walking classes are 3 to 4 km/h.
SPEED = {"Highway": 40, "Feeder Road": 30, "District Road": 25, "Other Road": 15, "Cart track": 10,
         "Bridge Road": 25, "Road Tunnel 1": 25, "Causeway": 15, "Built Up": 15, "Built up Area": 15,
         "Crossing Ford": 5, "Crossing Ferry": 5, "Main Trail": 4, "Local Trails": 3,
         "Bridge Trails &Tracks": 3}
VEHICLE = {"Highway", "Feeder Road", "District Road", "Other Road", "Cart track", "Bridge Road",
           "Road Tunnel 1", "Causeway", "Built Up", "Built up Area"}
BLOCK_LEVELS = {"0": 1, "60": 3, "100": 1}


def build_graph(margin_m: float = 20_000) -> dict:
    wards = gpd.read_file(PROCESSED / "wards.gpkg").to_crs(CRS_METRIC)
    x0, y0, x1, y1 = wards.total_bounds
    g = gpd.read_file(next(SHAPE.rglob("*.shp")), bbox=(x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m)
                      if False else None)
    g = g.to_crs(CRS_METRIC)
    g = g.cx[x0 - margin_m:x1 + margin_m, y0 - margin_m:y1 + margin_m]
    g = g[g["FEATURES"].isin(SPEED)].reset_index(drop=True)
    # Node ids of the layer. Check that one id has one position.
    starts = np.array([(line.coords[0]) for line in g.geometry])
    ends = np.array([(line.coords[-1]) for line in g.geometry])
    ids = pd.concat([pd.Series(g["FNODE_"].to_numpy()), pd.Series(g["TNODE_"].to_numpy())])
    pts = np.vstack([starts, ends])
    frame = pd.DataFrame({"id": ids.to_numpy(), "x": pts[:, 0], "y": pts[:, 1]})
    spread = frame.groupby("id")[["x", "y"]].agg(lambda s: s.max() - s.min()).max(axis=1)
    if (spread > 1.0).any():
        # The ids are not unique across map sheets. Use rounded positions as node keys.
        key = np.round(pts, 0)
        _, inverse = np.unique(key, axis=0, return_inverse=True)
        a, b = inverse[: len(g)], inverse[len(g):]
        xy = np.zeros((inverse.max() + 1, 2))
        xy[inverse] = pts
    else:
        uniq, inverse = np.unique(frame["id"].to_numpy(), return_inverse=True)
        a, b = inverse[: len(g)], inverse[len(g):]
        xy = np.zeros((len(uniq), 2))
        xy[inverse] = pts
    length = g.geometry.length.to_numpy()
    speed = g["FEATURES"].map(SPEED).to_numpy(dtype=float)
    hours = length / 1000 / speed
    vehicle = g["FEATURES"].isin(VEHICLE).to_numpy()
    return {"xy": xy, "a": a, "b": b, "hours": hours, "lines": list(g.geometry), "vehicle": vehicle,
            "feature": g["FEATURES"].to_numpy()}


def blocked_edges(graph: dict) -> np.ndarray:
    slides = gpd.read_file(LANDSLIDES / "Full20170209.shp").to_crs(CRS_METRIC)
    tree = STRtree(graph["lines"])
    hit = tree.query(slides.geometry.buffer(LANDSLIDE_BUFFER_M).values, predicate="intersects")[1]
    out = np.zeros(len(graph["lines"]), dtype=bool)
    out[np.unique(hit)] = True
    return out


def build() -> dict:
    wards = pd.read_parquet(PROCESSED / "wards.parquet", columns=["ward_id", "x_km", "y_km"])
    xy_w = wards[["x_km", "y_km"]].to_numpy() * 1000
    graph = build_graph()
    blocked = blocked_edges(graph)
    crossed = np.flatnonzero(blocked)
    levels = {}
    for level, draws in BLOCK_LEVELS.items():
        for d in range(draws):
            open_edge = np.ones(len(graph["hours"]), dtype=bool)
            if level != "0":
                rng = np.random.default_rng([29, int(level), d])
                open_edge[rng.choice(crossed, size=int(round(len(crossed) * int(level) / 100)), replace=False)] = False
            levels[f"t_{level}_{d}"] = ward_travel_hours(graph, xy_w, open_edge).astype("float32")
    np.savez_compressed(PROCESSED / "roads_2015_levels.npz", **levels)
    np.savez_compressed(PROCESSED / "roads_2015.npz", ward_id=wards["ward_id"].to_numpy(),
                        t_open=levels["t_0_0"], edge_a=graph["a"], edge_b=graph["b"],
                        edge_hours=graph["hours"], edge_blocked=blocked, edge_vehicle=graph["vehicle"],
                        node_xy=graph["xy"])
    return {"graph": graph, "blocked": blocked, "levels": levels}

"""Ward graph: polygon adjacency edges and nearest neighbor edges."""

import geopandas as gpd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from .paths import CRS_METRIC, PROCESSED

# Two wards are adjacent if the distance between the polygons is not more than this.
# A value of 0 gives strict queen adjacency (a shared point or a shared edge).
TOLERANCE_M = 10.0
KNN = 8


def adjacency_pairs(gm: gpd.GeoDataFrame, tolerance: float = TOLERANCE_M) -> np.ndarray:
    """Undirected adjacency edges as an (m, 2) array of node indices with i < j."""
    idx = np.arange(len(gm))
    left = gpd.GeoDataFrame({"i": idx}, geometry=gm.geometry.buffer(tolerance).values, crs=gm.crs)
    right = gpd.GeoDataFrame({"j": idx}, geometry=gm.geometry.values, crs=gm.crs)
    hit = gpd.sjoin(left, right, predicate="intersects")
    pairs = hit.loc[hit["i"] < hit["j"], ["i", "j"]].to_numpy()
    return np.unique(pairs, axis=0)


def sparse_adjacency(pairs: np.ndarray, n: int):
    data = np.ones(len(pairs))
    a = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return (a + a.T).tocsr()


def n_components(pairs: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    return connected_components(sparse_adjacency(pairs, n), directed=False)


def connect_components(pairs: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add the shortest centroid edge from each small component to the other nodes.

    Return the new edge array and the added edges.
    """
    n = len(xy)
    added = []
    while True:
        k, label = n_components(pairs, n)
        if k == 1:
            break
        small = np.argmin(np.bincount(label))
        inside, outside = np.where(label == small)[0], np.where(label != small)[0]
        d = np.linalg.norm(xy[inside][:, None, :] - xy[outside][None, :, :], axis=2)
        a, b = np.unravel_index(np.argmin(d), d.shape)
        edge = sorted((int(inside[a]), int(outside[b])))
        added.append(edge)
        pairs = np.unique(np.vstack([pairs, edge]), axis=0)
    return pairs, np.array(added, dtype=int).reshape(-1, 2)


def knn_pairs(xy: np.ndarray, k: int = KNN) -> np.ndarray:
    _, nbr = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    i = np.repeat(np.arange(len(xy)), k)
    j = nbr[:, 1:].ravel()
    pairs = np.sort(np.column_stack([i, j]), axis=1)
    return np.unique(pairs, axis=0)


def _dist(pairs: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)


def build() -> dict:
    w = gpd.read_file(PROCESSED / "wards.gpkg")
    gm = w.to_crs(CRS_METRIC)
    xy = w[["x_km", "y_km"]].to_numpy()
    n = len(w)

    strict = adjacency_pairs(gm, 0.0)
    raw = adjacency_pairs(gm, TOLERANCE_M)
    k_raw, _ = n_components(raw, n)
    degree_raw = np.asarray(sparse_adjacency(raw, n).sum(axis=1)).ravel()
    adj, added = connect_components(raw, xy)
    knn = knn_pairs(xy)

    np.savez(
        PROCESSED / "graph.npz",
        ward_id=w["ward_id"].to_numpy(),
        adj_pairs=adj,
        adj_dist_km=_dist(adj, xy),
        knn_pairs=knn,
        knn_dist_km=_dist(knn, xy),
        added_pairs=added,
    )
    degree = np.asarray(sparse_adjacency(adj, n).sum(axis=1)).ravel()
    return {
        "nodes": n,
        "strict_edges": len(strict),
        "tolerance_edges": len(raw),
        "components_before": k_raw,
        "islands_before": int((degree_raw == 0).sum()),
        "added_edges": added,
        "edges": len(adj),
        "degree": degree,
        "adj_dist_km": _dist(adj, xy),
        "knn_edges": len(knn),
        "knn_dist_km": _dist(knn, xy),
        "ward_id": w["ward_id"].to_numpy(),
    }


def load() -> dict:
    with np.load(PROCESSED / "graph.npz") as g:
        return {k: g[k] for k in g.files}

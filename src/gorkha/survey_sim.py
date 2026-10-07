"""Simulated early surveys: which wards have a revealed label."""

import numpy as np
from scipy.sparse.csgraph import breadth_first_order

from .graph import sparse_adjacency

BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
SCHEMES = ["random", "clustered", "access"]

# Wards in one cluster of the clustered scheme (a seed ward plus its nearest graph neighbors).
CLUSTER_SIZE = 5
# The access weight is exp(-distance to a major road / ACCESS_SCALE_KM).
ACCESS_SCALE_KM = 3.0


def n_surveyed(n: int, budget: float) -> int:
    return max(2, int(round(n * budget)))


def rng_for(seed: int, scheme: str, budget: float, realization: int) -> np.random.Generator:
    """Independent, repeatable random stream for one realization."""
    return np.random.default_rng(
        [seed, SCHEMES.index(scheme), int(round(budget * 10_000)), realization]
    )


def random_mask(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=m, replace=False)] = True
    return mask


def clustered_mask(adjacency, m: int, rng: np.random.Generator,
                   cluster_size: int = CLUSTER_SIZE) -> np.ndarray:
    """Random seed wards. Each seed adds wards in breadth-first order on the graph."""
    n = adjacency.shape[0]
    mask = np.zeros(n, dtype=bool)
    for seed in rng.permutation(n):
        if mask.sum() >= m:
            break
        if mask[seed]:
            continue
        order = breadth_first_order(adjacency, seed, directed=False, return_predecessors=False)
        take = order[~mask[order]][: min(cluster_size, m - mask.sum())]
        mask[take] = True
    return mask


def access_mask(dist_road_km: np.ndarray, m: int, rng: np.random.Generator,
                scale_km: float = ACCESS_SCALE_KM) -> np.ndarray:
    """Weighted sampling without replacement. A ward near a major road has a larger weight."""
    weight = np.exp(-np.asarray(dist_road_km, dtype=float) / scale_km)
    mask = np.zeros(len(weight), dtype=bool)
    mask[rng.choice(len(weight), size=m, replace=False, p=weight / weight.sum())] = True
    return mask


def survey_mask(scheme: str, budget: float, realization: int, pairs: np.ndarray,
                dist_road_km: np.ndarray, seed: int = 0) -> np.ndarray:
    n = len(dist_road_km)
    m = n_surveyed(n, budget)
    rng = rng_for(seed, scheme, budget, realization)
    if scheme == "random":
        return random_mask(n, m, rng)
    if scheme == "clustered":
        return clustered_mask(sparse_adjacency(pairs, n), m, rng)
    if scheme == "access":
        return access_mask(dist_road_km, m, rng)
    raise ValueError(f"unknown scheme: {scheme}")

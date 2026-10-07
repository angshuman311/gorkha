"""Build the ward graph."""

import numpy as np

from gorkha import graph

if __name__ == "__main__":
    r = graph.build()
    d = r["degree"]
    print(f"nodes: {r['nodes']}")
    print(f"adjacency edges, strict: {r['strict_edges']}, "
          f"with a {graph.TOLERANCE_M:.0f} m tolerance: {r['tolerance_edges']}")
    print(f"components before the correction: {r['components_before']}, "
          f"wards without a neighbor: {r['islands_before']}")
    for i, j in r["added_edges"]:
        print(f"  added edge: ward {r['ward_id'][i]} to ward {r['ward_id'][j]}")
    print(f"adjacency edges in the graph: {r['edges']}")
    print(f"degree: mean {d.mean():.2f}, min {d.min():.0f}, median {np.median(d):.0f}, max {d.max():.0f}")
    print("edge length (km): " + ", ".join(
        f"{q}%={np.percentile(r['adj_dist_km'], q):.1f}" for q in (5, 50, 95, 100)))
    print(f"kNN edges (k = {graph.KNN}): {r['knn_edges']}, "
          f"median length {np.median(r['knn_dist_km']):.1f} km")

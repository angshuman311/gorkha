"""Build the road graph and the travel times between wards, with open and blocked roads."""

import time

import numpy as np

from gorkha import roads

if __name__ == "__main__":
    start = time.time()
    r = roads.build()
    g, blocked = r["graph"], r["blocked"]
    km = np.array([line.length for line in g["lines"]]) / 1000
    print(f"road graph: {len(g['xy'])} nodes, {len(km)} edges, {km.sum():.0f} km, {time.time() - start:.0f} s")
    print(f"edges that a landslide crosses: {int(blocked.sum())} ({km[blocked].sum():.0f} km)")
    for name in ("t_open", "t_blocked"):
        t = r[name]
        finite = np.isfinite(t)
        off = ~np.eye(len(t), dtype=bool)
        alone = int((~finite & off).all(axis=1).sum())
        print(f"{name}: ward pairs with a road route {finite[off].mean():.3f}, "
              f"median travel time {np.median(t[finite & off]):.1f} h, "
              f"wards that no road reaches: {alone}")
    both = np.isfinite(r["t_open"]) & np.isfinite(r["t_blocked"])
    longer = r["t_blocked"][both] - r["t_open"][both]
    print(f"pairs with a longer route after the blockage: {(longer > 0.1).mean():.3f}, "
          f"mean increase for these pairs: {longer[longer > 0.1].mean():.1f} h")
    cut = np.isfinite(r["t_open"]) & ~np.isfinite(r["t_blocked"])
    print(f"pairs that lose their road route: {cut.mean():.3f}")

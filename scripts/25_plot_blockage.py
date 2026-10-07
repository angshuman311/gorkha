"""Map of the road network, the landslides, and the road pieces that a landslide crosses."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gorkha import plots, roads
from gorkha.features import LANDSLIDES
from gorkha.paths import CRS_METRIC, FIGURES, PROCESSED

if __name__ == "__main__":
    plots.style()
    r = roads.load()
    wards = gpd.read_file(PROCESSED / "wards.gpkg").to_crs(CRS_METRIC)
    slides = gpd.read_file(LANDSLIDES / "Full20170209.shp").to_crs(CRS_METRIC)
    extent = gpd.read_file(LANDSLIDES / "MappingExtent20170209.shp").to_crs(CRS_METRIC)
    xy, a, b, blocked = r["node_xy"], r["edge_a"], r["edge_b"], r["edge_blocked"]
    seg = np.stack([xy[a], xy[b]], axis=1)   # straight line between two junctions
    x0, y0, x1, y1 = wards.total_bounds
    inside = ((seg[:, :, 0] > x0 - 5000) & (seg[:, :, 0] < x1 + 5000) &
              (seg[:, :, 1] > y0 - 5000) & (seg[:, :, 1] < y1 + 5000)).all(axis=1)

    fig, ax = plt.subplots(figsize=(13, 8.4), layout="constrained")
    wards.dissolve().plot(ax=ax, color="#f0efec", edgecolor="none")
    plots.districts().boundary.plot(ax=ax, color=plots.INK_2, linewidth=0.6)
    extent.boundary.plot(ax=ax, color=plots.SERIES[2], linewidth=1.2)
    ax.add_collection(LineCollection(seg[inside & ~blocked], colors=plots.MUTED, linewidths=0.25))
    slides.plot(ax=ax, color=plots.SERIES[7], edgecolor=plots.SERIES[7], linewidth=0.6)
    ax.add_collection(LineCollection(seg[inside & blocked], colors=plots.SERIES[0], linewidths=1.6))
    ax.set_xlim(x0 - 5000, x1 + 5000)
    ax.set_ylim(y0 - 5000, y1 + 5000)
    ax.set_aspect("equal")
    ax.set_axis_off()
    t_open, t_blocked = r["t_open"], r["t_blocked"]
    lost = int((np.isfinite(t_open) & ~np.isfinite(t_blocked)).sum() / 2)
    ax.set_title(f"Roads and landslides: {int((inside & blocked).sum())} road pieces cross a landslide "
                 f"({len(slides):,} landslides in the USGS inventory)")
    ax.legend(handles=[
        Line2D([], [], color=plots.MUTED, linewidth=0.8, label="road (OpenStreetMap, 2026)"),
        Patch(color=plots.SERIES[7], label="landslide (Roback et al.)"),
        Line2D([], [], color=plots.SERIES[0], linewidth=2, label="road piece that a landslide crosses"),
        Line2D([], [], color=plots.SERIES[2], linewidth=1.2, label="area that the inventory examined"),
    ], loc="lower left", fontsize=9)
    fig.savefig(FIGURES / "roads_landslides.png")

    # Wards by the share of the other wards that they can reach on open roads.
    reach = np.isfinite(t_blocked).mean(axis=1)
    d = pd.DataFrame({"district": wards["district_name"].values, "reach": reach})
    print("share of the other wards with an open road route, after the blockage, by district:")
    print(d.groupby("district")["reach"].agg(["mean", "min"]).round(2).to_string())
    print(f"ward pairs that lose the road route: {lost}")
    km = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1) / 1000
    print("length of the crossed road pieces (km): median %.1f, largest %.1f" %
          (np.median(km[blocked]), km[blocked].max()))

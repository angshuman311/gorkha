"""Shared plot style: one sequential hue, one diverging pair, fixed series colors."""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

from .paths import CRS_METRIC, INTERIM

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NO_DATA = "#e1e0d9"

# Magnitude: one hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
     "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)
# Polarity: blue and red poles with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3aaa9", "#e34948", "#9c2322"],
)
# Identity: the color follows the model, in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })


def districts() -> gpd.GeoDataFrame:
    return gpd.read_file(INTERIM / "osm_districts.gpkg").to_crs(CRS_METRIC)


def choropleth(ax, wards_m, values, cmap, vmin, vmax, label, title, mask=None,
               mask_label=None, district_names=False, colorbar=True):
    """Ward map. `wards_m` is in the metric CRS. `mask` marks the wards without data."""
    g = wards_m.assign(_v=values)
    has = g if mask is None else g[~mask]
    if mask is not None and mask.any():
        g[mask].plot(ax=ax, color=NO_DATA, linewidth=0.15, edgecolor=SURFACE)
    has.plot(ax=ax, column="_v", cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.15,
             edgecolor=SURFACE)
    d = districts()
    d.boundary.plot(ax=ax, color=INK_2, linewidth=0.5)
    if district_names:
        for name, point in zip(d["district"], d.geometry.representative_point()):
            ax.annotate(name, (point.x, point.y), ha="center", va="center", fontsize=7.5,
                        color=INK, path_effects=[pe.withStroke(linewidth=2, foreground=SURFACE)])
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if colorbar:
        scale_bar(ax.figure, ax, cmap, vmin, vmax, label)
    if mask is not None and mask.any() and mask_label:
        ax.legend(handles=[Patch(facecolor=NO_DATA, edgecolor="none", label=mask_label)],
                  loc="lower left", fontsize=8, handlelength=1.2)


def scale_bar(fig, axes, cmap, vmin, vmax, label, shrink=0.62):
    """Color scale legend for one axis or for a row of axes with a shared scale."""
    bar = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), ax=axes,
                       shrink=shrink, pad=0.01, aspect=22)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, labelcolor=INK_2)
    bar.set_label(label, color=INK_2, fontsize=8)
    return bar

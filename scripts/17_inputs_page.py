"""Write docs/SHERPA_inputs.html: the inputs of the CNN and GNN model, with real data."""

import base64
import io

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gorkha import features, graph, plots, survey_sim
from gorkha.design import FEATURE_GROUPS, TARGET
from gorkha.paths import CRS_METRIC, FIGURES, PROCESSED, ROOT

PATCHES = ROOT / "data" / "patches"
TILE = "45RUL"


def b64(fig=None, path=None) -> str:
    if path is not None:
        data = path.read_bytes()
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def rgb(patch: np.ndarray) -> np.ndarray:
    """True-color picture from the bands B04, B03, B02 of one patch."""
    return np.clip(patch[[2, 1, 0]].transpose(1, 2, 0) / 2500.0, 0, 1)


def survey_figure(w, wm, pairs):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
    names = {"random": "Random", "clustered": "Clustered", "access": "Near major roads"}
    for ax, scheme in zip(axes, names):
        s = survey_sim.survey_mask(scheme, 0.05, 0, pairs, w["dist_road_km"].to_numpy())
        wm[~s].plot(ax=ax, color=plots.NO_DATA, edgecolor=plots.SURFACE, linewidth=0.15)
        wm[s].plot(ax=ax, column=TARGET, cmap=plots.SEQUENTIAL, vmin=0, vmax=1,
                   edgecolor=plots.INK, linewidth=0.4)
        plots.districts().boundary.plot(ax=ax, color=plots.INK_2, linewidth=0.5)
        if scheme == "access":
            roads = features._major_roads()
            roads.plot(ax=ax, color="#eb6834", linewidth=0.9)
            ax.plot([], [], color="#eb6834", linewidth=1.5, label="major road (OSM)")
            ax.legend(loc="lower left", fontsize=8)
            b = wm.total_bounds
            ax.set_xlim(b[0] - 3000, b[2] + 3000)
            ax.set_ylim(b[1] - 3000, b[3] + 3000)
        ax.set_title(f"{names[scheme]} survey: {int(s.sum())} of {len(w)} wards")
        ax.set_axis_off()
    plots.scale_bar(fig, axes, plots.SEQUENTIAL, 0, 1, "visible label (fraction with grade 4 or 5)", 0.7)
    return fig


def ward_patch_figure(w, wm):
    index = pd.read_parquet(PATCHES / "s2_index.parquet")
    count = index.groupby("node").size()
    in_tile = index[index["tile"] == TILE].groupby("node").size()
    # A ward of medium size with all its patches in one map tile and heavy damage.
    ok = [n for n in in_tile.index if in_tile[n] == count[n] and 30 <= count[n] <= 45
          and w[TARGET].iat[n] > 0.8]
    node = int(ok[0])
    rows = index[index["node"] == node]
    data = np.load(PATCHES / f"s2_{TILE}.npy", mmap_mode="r")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), layout="constrained",
                             width_ratios=[1.25, 1])
    ax = axes[0]
    for r in rows.itertuples():
        ax.imshow(rgb(np.asarray(data[r.i])), extent=(r.x - 320, r.x + 320, r.y - 320, r.y + 320))
    wm.iloc[[node]].boundary.plot(ax=ax, color="#eda100", linewidth=2)
    b = wm.iloc[[node]].total_bounds
    ax.set_xlim(b[0] - 700, b[2] + 700)
    ax.set_ylim(b[1] - 700, b[3] + 700)
    ax.set_aspect("equal")
    ax.set_axis_off()
    name = f"{w['vdcmun_name'].iat[node]}, ward {w['ward_no'].iat[node]} ({w['district_name'].iat[node]})"
    ax.set_title(f"{name}: {len(rows)} patches of 640 m, image of {rows['date'].iat[0]}")
    one = np.asarray(data[rows["i"].iat[len(rows) // 2]]).astype("float32")
    sub = axes[1].inset_axes([0, 0, 1, 1])
    axes[1].set_axis_off()
    axes[1].set_title("One patch: the 6 bands that the CNN gets (64 x 64 pixels)")
    grid = sub.inset_axes([0, 0, 1, 0.94])
    sub.set_axis_off()
    grid.set_axis_off()
    labels = ["Blue", "Green", "Red", "Near infrared", "Short-wave infrared 1", "Short-wave infrared 2"]
    for k in range(6):
        a = grid.inset_axes([(k % 3) / 3 + 0.01, 0.5 - (k // 3) * 0.5 + 0.02, 0.31, 0.42])
        lo, hi = np.percentile(one[k], [2, 98])
        a.imshow(one[k], cmap="gray", vmin=lo, vmax=hi)
        a.set_title(labels[k], fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    info = {"name": name, "patches": len(rows), "buildings": int(w["n_buildings"].iat[node]),
            "label": float(w[TARGET].iat[node]), "area": float(w["area_km2"].iat[node])}
    return fig, info, count


CSS = """
:root { --ink:#1f1f1e; --muted:#5f5e5a; --teal:#103d3e; --orange:#c05a26; --bg:#ffffff; --panel:#f5f4f0; }
body { background:var(--bg); color:var(--ink); font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       max-width:1000px; margin:0 auto; padding:24px 16px 80px; }
h1 { color:var(--teal); font-size:27px; margin:0 0 4px; }
h2 { color:var(--teal); border-bottom:2px solid var(--orange); padding-bottom:4px; margin-top:44px; }
.goal { background:var(--teal); color:#fff; padding:14px 18px; border-radius:8px; }
.box { background:var(--panel); padding:12px 16px; border-radius:8px; margin:12px 0; }
.honest { border-left:4px solid var(--orange); background:#fbf1ea; padding:10px 14px; border-radius:4px; margin:12px 0; }
.why { border-left:4px solid var(--teal); background:#eef4f4; padding:10px 14px; border-radius:4px; margin:12px 0; }
figure { margin:18px 0; } figure img, figure svg { width:100%; height:auto; border:1px solid #e4e3de; border-radius:6px; }
figcaption { color:var(--muted); font-size:14px; margin-top:4px; }
table { border-collapse:collapse; margin:8px 0; } td,th { border:1px solid #d9d8d2; padding:6px 12px; text-align:left; vertical-align:top; }
th { background:var(--panel); }
"""

DIAGRAM = """
<svg viewBox="0 0 980 330" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,Segoe UI,sans-serif" font-size="13">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#5f5e5a"/></marker></defs>
<rect width="980" height="330" fill="#ffffff"/>
<g fill="#f5f4f0" stroke="#103d3e" stroke-width="1.2">
<rect x="15" y="20" width="190" height="62" rx="6"/><rect x="15" y="97" width="190" height="62" rx="6"/>
<rect x="15" y="174" width="190" height="62" rx="6"/><rect x="15" y="251" width="190" height="62" rx="6"/>
</g>
<g fill="#eef4f4" stroke="#103d3e" stroke-width="1.2">
<rect x="270" y="20" width="150" height="62" rx="6"/><rect x="480" y="20" width="150" height="62" rx="6"/>
<rect x="480" y="135" width="150" height="110" rx="6"/><rect x="690" y="135" width="130" height="110" rx="6"/>
</g>
<rect x="870" y="135" width="100" height="110" rx="6" fill="#103d3e"/>
<g fill="#1f1f1e">
<text x="27" y="44" font-weight="700">Input 4: image patches</text><text x="27" y="64">of one ward (N patches)</text>
<text x="27" y="121" font-weight="700">Input 3: feature table</text><text x="27" y="141">20 values for the ward</text>
<text x="27" y="198" font-weight="700">Input 2: the survey</text><text x="27" y="218">label (or 0) and a flag</text>
<text x="27" y="275" font-weight="700">Input 1: the ward graph</text><text x="27" y="295">edges and edge lengths</text>
<text x="285" y="46" font-weight="700">CNN</text><text x="285" y="66">one vector for a patch</text>
<text x="495" y="46" font-weight="700">Pooling</text><text x="495" y="66">one vector for a ward</text>
<text x="495" y="165" font-weight="700">Node input</text><text x="495" y="187">image vector</text><text x="495" y="205">+ 20 features</text><text x="495" y="223">+ label and flag</text>
<text x="705" y="165" font-weight="700">GNN</text><text x="705" y="187">passes information</text><text x="705" y="205">between adjacent</text><text x="705" y="223">wards</text>
</g>
<g fill="#ffffff"><text x="883" y="170" font-weight="700">Output</text><text x="883" y="192">damage</text><text x="883" y="210">fraction and</text><text x="883" y="228">uncertainty</text></g>
<g stroke="#5f5e5a" stroke-width="1.5" fill="none" marker-end="url(#a)">
<path d="M205,51 H268"/><path d="M420,51 H478"/><path d="M555,82 V133"/>
<path d="M205,128 H340 V170 H478"/><path d="M205,205 H478"/>
<path d="M630,190 H688"/><path d="M205,282 H755 V247"/><path d="M820,190 H868"/>
</g></svg>
"""


if __name__ == "__main__":
    plots.style()
    w = gpd.read_file(PROCESSED / "wards.gpkg")
    wm = w.to_crs(CRS_METRIC)
    g = graph.load()
    fig_survey = survey_figure(w, wm, g["adj_pairs"])
    fig_survey.savefig(FIGURES / "survey_schemes.png")
    fig_ward, info, count = ward_patch_figure(w, wm)
    fig_ward.savefig(FIGURES / "s2_ward_example.png")
    parts = [pd.read_parquet(p) for p in sorted(PATCHES.glob("s2_index_part_*.parquet"))]
    n_patches = sum(len(p) for p in parts)
    n_wards_with = pd.concat(parts)["node"].nunique()
    count = pd.concat(parts).groupby("node").size()
    feature_rows = "".join(f"<tr><td>{k}</td><td>{', '.join(v)}</td><td>{len(v)}</td></tr>"
                           for k, v in FEATURE_GROUPS.items())
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SHERPA inputs</title><style>{CSS}</style></head><body>
<header><h1>SHERPA: the inputs of the model</h1><p style="color:var(--muted);margin:0">What the CNN and the GNN get, with real data from the Gorkha test. Status of 2026-10-07.</p></header>

<div class="goal"><b>The goal.</b> After an earthquake, survey teams visit a small number of wards. SHERPA estimates
the damage in all the other wards and shows where the estimate is uncertain. The test case is the 2015 Gorkha
sequence: two earthquakes (2015-04-25 and 2015-05-12), and then the survey.</div>

<p><b>The whole model in one sentence.</b> For each ward, a CNN changes the image patches into numbers, and a GNN
combines these numbers with the feature table and with the labels of the surveyed wards near it.</p>

<figure>{DIAGRAM}<figcaption>The four inputs and the path to the output. The model has one output for each ward:
the fraction of buildings with damage grade 4 or 5, and its uncertainty.</figcaption></figure>

<div class="why"><b>The unit.</b> One ward is one node. The study area has {len(w)} wards in 11 districts. A ward has
approximately {int(w['n_buildings'].median())} buildings (median).</div>

<h2>Input 1: the ward graph</h2>
<p><b>Why.</b> Damage is similar in adjacent wards (Moran's I = 0.76). The graph tells the model which wards are
adjacent, so that information from a surveyed ward can go to the wards near it.</p>
<p><b>What.</b> Two wards have an edge if they share a boundary ({len(g['adj_pairs'])} edges). For message passing,
the model also uses edges to the 8 nearest wards. Each edge has one value: its length in km.</p>
<figure><img src="{b64(path=FIGURES / 'graph.png')}" alt="Ward graph"><figcaption>The ward graph. The hole in the
middle is the Kathmandu Valley, which is not in the survey.</figcaption></figure>

<h2>Input 2: the survey (the human input)</h2>
<p><b>Why.</b> The field teams give the only direct measurement of damage. This is the "human-informed" part of
SHERPA.</p>
<p><b>What.</b> Each ward gets two numbers:</p>
<ul><li><b>A flag:</b> 1 if a team surveyed the ward, 0 if not.</li>
<li><b>A label:</b> the damage fraction if the ward is surveyed, and 0 if not.</li></ul>
<p>In the test, a survey is simulated: the true labels of a small number of wards become visible. The budgets are 1%,
2%, 5%, 10%, and 20% of the wards. Three survey schemes imitate the behavior of real teams.</p>
<figure><img src="{b64(fig_survey)}" alt="Three survey schemes"><figcaption>One example of each survey scheme at a 5%
budget. Colored wards are surveyed, and the model sees their labels. Gray wards have no label. The model must
estimate them.</figcaption></figure>
<div class="box"><b>One protocol.</b> In each run, the model learns only from the surveyed wards of that run. It
never sees the label of a gray ward. The gray wards are used only for the score.</div>

<h2>Input 3: the feature table</h2>
<p><b>Why.</b> These data are available for all wards without a visit. They give the regional pattern of the damage.</p>
<p><b>What.</b> 20 values for each ward, in 5 groups:</p>
<table><tr><th>Group</th><th>Values</th><th>Count</th></tr>{feature_rows}</table>
<p>The shaking values come from the USGS ShakeMaps of the two earthquakes (suffix "after" is the second earthquake).
"dpm" is the NASA Damage Proxy Map, "ls" is the landslide inventory, and "dist" is a distance.</p>
<figure><img src="{b64(path=FIGURES / 'feature_maps.png')}" alt="Feature maps"><figcaption>Four of the features as
maps. The Damage Proxy Map has data for 423 of {len(w)} wards only.</figcaption></figure>
<div class="honest"><b>Limit.</b> These features explain the difference between regions, but little of the difference
between two adjacent wards. The image patches of input 4 must give that information.</div>

<h2>Input 4: the image patches (the CNN input)</h2>
<p><b>Why.</b> An image shows what is in a ward and what changed there. A table value cannot show this.</p>
<p><b>What.</b> Each ward is cut into square patches. The CNN changes each patch into a vector of numbers. A pooling
step combines the vectors of all patches of a ward into one vector for the ward. A ward can have a different number
of patches, and the pooling step makes this possible.</p>
<p>SHERPA has three image modes:</p>
<table><tr><th>Mode</th><th>Images</th><th>Information</th><th>Status</th></tr>
<tr><td>1</td><td>Sentinel-1 radar, before and after the earthquake</td><td>Damage evidence: what changed</td><td>Example read. Patches are not made yet.</td></tr>
<tr><td>2</td><td>Sentinel-2 optical, one image</td><td>Land context: what is in the ward (settlements, fields, forest)</td><td>{n_patches:,} patches on disk for {n_wards_with} of {len(w)} wards.</td></tr>
<tr><td>3</td><td>Sentinel-2 optical, before and after</td><td>Damage evidence, optical</td><td>Not testable with Gorkha. For future earthquakes.</td></tr></table>

<h3>Mode 2: Sentinel-2 context</h3>
<p>A patch is 64 x 64 pixels of 10 m (640 m x 640 m) with 6 bands. Sentinel-2 started after the earthquakes, so the
test uses images from late 2015 and early 2016 as a proxy for an image from before the event.</p>
<figure><img src="{b64(fig_ward)}" alt="Sentinel-2 patches of one ward"><figcaption>Real data. Left: the patches of
{info['name']} in true color, with the ward boundary. This ward has {info['area']:.1f} km2, {info['buildings']:,}
buildings in the survey, and a damage fraction of {info['label']:.2f}. Right: the 6 bands of one patch.</figcaption></figure>

<h3>Mode 1: Sentinel-1 before and after</h3>
<p>A patch will have these channels: echo strength before, echo strength after, and coherence (a sensitive comparison
of two dates) before and across the earthquake.</p>
<figure><img src="{b64(path=FIGURES / 'sentinel1_example_chautara.png')}" alt="Sentinel-1 example"><figcaption>Real
data near Chautara (Sindhupalchok). Left and middle: echo strength 1 day before and 11 days after the first
earthquake. Right: the change. The damage is not visible to the eye in the echo strength.</figcaption></figure>
<div class="honest"><b>Limit.</b> The coherence channel needs processing at the Alaska Satellite Facility, which
needs a free NASA Earthdata account. Without it, mode 1 has only the echo strength.</div>

<h2>How the inputs come together for one ward</h2>
<table><tr><th>Part</th><th>Size for one ward</th><th>Source</th></tr>
<tr><td>Image patches</td><td>N x channels x 64 x 64 (N is 1 to {int(count.max())}, median {int(count.median())} in the patches on disk)</td><td>Input 4</td></tr>
<tr><td>Image vector after the CNN and the pooling</td><td>32 numbers (planned)</td><td>Calculated</td></tr>
<tr><td>Feature table</td><td>20 numbers</td><td>Input 3</td></tr>
<tr><td>Label and flag</td><td>2 numbers</td><td>Input 2</td></tr>
<tr><td>Node input of the GNN</td><td>32 + 20 + 2 = 54 numbers</td><td>Joined</td></tr></table>
<p>The GNN then passes these numbers along the edges of input 1, several times. After that, each ward has
information from the surveyed wards near it.</p>

<h2>What is not an input</h2>
<ul>
<li><b>The label of a ward without a survey.</b> It is only for the score.</li>
<li><b>Building data from the damage survey</b> (material, age, floors) for a ward without a survey. A team knows
them only after the visit.</li>
<li><b>A kriging estimate.</b> Kriging is a reference method in the result tables, not an input.</li>
</ul>

<h2>Status of each part</h2>
<table><tr><th>Part</th><th>Status</th></tr>
<tr><td>Ward graph, labels, feature table</td><td>Complete ({len(w)} wards)</td></tr>
<tr><td>Survey simulation (3 schemes, 5 budgets)</td><td>Complete</td></tr>
<tr><td>Reference methods (kriging, regression kriging, and others)</td><td>Complete</td></tr>
<tr><td>GNN on inputs 1 to 3 (no images)</td><td>First version runs. It is in the same range as kriging.</td></tr>
<tr><td>Sentinel-2 patches (mode 2)</td><td>Complete: {n_patches:,} patches</td></tr>
<tr><td>Sentinel-1 patches (mode 1)</td><td>Not started</td></tr>
<tr><td>CNN with pooling, joined to the GNN</td><td>Not started</td></tr></table>
<div class="honest"><b>Honest note.</b> A first test with 6 simple Sentinel-2 values for each ward (mean vegetation,
built-up index, brightness) gave no benefit. This does not test a CNN on patches, but it gives no evidence for a
benefit of mode 2. The CNN is the real test.</div>
</body></html>"""
    out = ROOT / "docs" / "SHERPA_inputs.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) / 1e6:.1f} MB), example ward: {info}")

"""Write docs/SHERPA_results.html from the campaign results and the figures."""

import base64

import pandas as pd

from gorkha.paths import FIGURES, RESULTS, ROOT

CSS = """
:root { --ink:#1f1f1e; --muted:#5f5e5a; --teal:#103d3e; --orange:#c05a26; --bg:#ffffff; --panel:#f5f4f0; }
body { background:var(--bg); color:var(--ink); font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       max-width:1000px; margin:0 auto; padding:24px 16px 80px; }
h1 { color:var(--teal); font-size:27px; margin:0 0 4px; }
h2 { color:var(--teal); border-bottom:2px solid var(--orange); padding-bottom:4px; margin-top:44px; }
.goal { background:var(--teal); color:#fff; padding:14px 18px; border-radius:8px; }
.box { background:var(--panel); padding:12px 16px; border-radius:8px; margin:12px 0; }
.honest { border-left:4px solid var(--orange); background:#fbf1ea; padding:10px 14px; border-radius:4px; margin:12px 0; }
figure { margin:18px 0; } figure img { width:100%; height:auto; border:1px solid #e4e3de; border-radius:6px; }
figcaption { color:var(--muted); font-size:14px; margin-top:4px; }
table { border-collapse:collapse; margin:8px 0; font-size:14px; } td,th { border:1px solid #d9d8d2; padding:5px 10px; text-align:left; }
th { background:var(--panel); }
"""
LABEL = {"random": "Kriging, random surveys",
         "nearest": "Kriging, nearest ward (no guidance)",
         "kriging_var": "Kriging, largest variance",
         "kriging_ivr": "Kriging, largest decrease of the total variance",
         "gorkha_mode2": "GORKHA, mode 2 (Sentinel-2 context)",
         "gorkha_mode1": "GORKHA, mode 1 (Sentinel-1 before and after)"}


def img(name: str) -> str:
    p = FIGURES / name
    if not p.exists():
        return f"<p><i>Figure {name} is missing.</i></p>"
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{name}">'


def table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{fmt.format(v) if isinstance(v, float) else v}</td>" for v in r)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


if __name__ == "__main__":
    final = pd.read_csv(RESULTS / "campaign_final.csv")
    final["method"] = final["method"].map(LABEL).fillna(final["method"])
    final = final.rename(columns={"level": "blocked %", "mae": "final error", "mae_sd": "sd over runs",
                                  "cost_days": "cost (team days)", "coverage_90": "coverage of the 90% interval"})
    reach = pd.read_csv(RESULTS / "campaign_cost_to_reach.csv")
    reach["cost"] = reach["cost"] / 24
    reach["method"] = reach["method"].map(LABEL).fillna(reach["method"])
    reach = reach.rename(columns={"level": "blocked %", "cost": "team days to reach the error",
                                  "reached": "share of runs that reach it", "target": "error target"})
    ab_path = RESULTS / "ablation_2x2.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path).pivot_table(index=["level", "round"], columns=["selection", "estimator"], values="mae")
        ab.columns = [f"{s} wards, {e} estimate" for s, e in ab.columns]
        ablation = ("<p>The same labels, two estimators and two selections. The rows are the blockage level and the "
                    "round.</p>" + table(ab.reset_index(), "{:.3f}"))
    else:
        ablation = "<p><i>The ablation did not run.</i></p>"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GORKHA results</title><style>{CSS}</style></head><body>
<header><h1>GORKHA: survey campaigns on the 2015 Nepal test case</h1>
<p style="color:var(--muted);margin:0">Error of the damage estimate against the survey cost, with roads blocked by landslides. Written by scripts/29_results_page.py.</p></header>

<div class="goal"><b>The question.</b> After the two earthquakes of 2015, 11 survey teams start at the district
headquarters. Each round, each team surveys one ward. Which method reaches an accurate damage map for all 945 wards
at the lowest cost, when landslides block a part of the roads?</div>

<h2>The survey methods</h2>
<table><tr><th>Method</th><th>Inputs of the estimate</th><th>Where the teams go</th></tr>
<tr><td>Kriging, random surveys</td><td>Survey, feature table</td><td>Random wards</td></tr>
<tr><td>Kriging, nearest ward</td><td>Survey, feature table</td><td>Each team goes to the nearest ward without a survey (no guidance, a clustered survey)</td></tr>
<tr><td>Kriging, largest variance</td><td>Survey, feature table</td><td>The wards with the largest kriging variance (the corners of the region)</td></tr>
<tr><td>Kriging, largest decrease of the total variance</td><td>Survey, feature table</td><td>The wards whose survey decreases the kriging variance of all wards the most</td></tr>
<tr><td>GORKHA, mode 2</td><td>Ward graph, survey, feature table, Sentinel-2 context images</td><td>A learned visit score that sees the road state. The estimate is a GNN group blended with regression kriging by a cross-fit weight.</td></tr>
<tr><td>GORKHA, mode 1</td><td>Ward graph, survey, feature table, Sentinel-1 coherence before and after</td><td>Same</td></tr></table>
<div class="box"><b>Cost.</b> Travel time on the open roads (OpenStreetMap, 2026) plus 8 hours for each ward survey.
A ward that no open road reaches costs a 12 hour helicopter trip. <b>Blockage.</b> Two kinds. Observed: the 13 road
segments that the NGA marked impassable on 28 April 2015 (HDX), 0.7 km near the Kathmandu valley, matched to 14 road
pieces. Simulated: a random share of the 656 road pieces that a landslide of the USGS inventory crosses is closed
(30%, 60%, or 100%), as a sensitivity case, because the observed data cover a small area.</div>

<h2>Error against cost</h2>
<figure>{img("campaign_error_cost.png")}<figcaption>Each line is the mean of the runs at one blockage level. Lower and
more to the left is better.</figcaption></figure>

<h2>Error against the share of surveyed wards</h2>
<figure>{img("campaign_error_share.png")}<figcaption>The same campaigns, with the share of surveyed wards on the
horizontal axis. This shows the value of the selection for each surveyed ward, without the cost.</figcaption></figure>

<h2>Cost against the blockage</h2>
<figure>{img("campaign_cost_blockage.png")}<figcaption>Team days to reach an error target, for each method, against the
share of blocked road pieces. A run that never reaches the target counts with its full cost.</figcaption></figure>

<h2>Benefit of GORKHA against the blockage</h2>
<figure>{img("campaign_benefit_blockage.png")}<figcaption>Team days saved against the strongest kriging method (largest
decrease of the total variance) to reach an error of 0.17. A value above zero means GORKHA is cheaper.</figcaption></figure>

<h2>Estimator against selection</h2>
{ablation}

<h2>Final round</h2>
{table(final)}

<h2>Cost to reach an error target</h2>
{table(reach, "{:.2f}")}

<div class="honest"><b>Honest notes.</b> The roads are those of 2026, not 2015. The travel speeds are assumptions.
The Sentinel-2 images are from late 2015, used as a proxy for an image from before the earthquakes. The number of
runs is small (1 to 6 for each level and method), so differences below the standard deviation over the runs are not
established. The GORKHA estimate blends a GNN group with regression kriging; the column "blend_gnn" of the final
table gives the mean weight of the GNN, selected by a cross-fit on the surveyed wards.</div>
</body></html>"""
    out = ROOT / "docs" / "SHERPA_results.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")

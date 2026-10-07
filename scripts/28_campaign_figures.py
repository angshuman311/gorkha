"""Figures and tables of the survey campaigns: error against cost, error against the share
of surveyed wards, and the cost to reach an error target against the blockage level."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gorkha import plots
from gorkha.paths import FIGURES, RESULTS

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="campaign2", help="campaign2 (OSM roads of 2026) or campaign3 (roads of 2015)")
parser.add_argument("--suffix", default="", help="suffix of the figure and table names, for example _2015")
ARGS = parser.parse_args()

# The second study (four survey methods) replaces the first one when its files exist.
FILES_NEW = {f"{ARGS.prefix}_mode2.parquet": {"gorkha": "gorkha_mode2"},
             f"{ARGS.prefix}_mode1.parquet": {"gorkha": "gorkha_mode1"}}
FILES_OLD = {"campaign_mode2.parquet": {"sherpa": "gorkha_mode2", "kriging": "kriging_var"},
             "campaign_mode1.parquet": {"sherpa": "gorkha_mode1"}}
LABEL = {"random": "Kriging, random surveys",
         "nearest": "Kriging, nearest ward (no guidance)",
         "kriging_var": "Kriging, largest variance",
         "kriging_ivr": "Kriging, largest decrease of the total variance",
         "gorkha_mode2": "GORKHA, mode 2 (Sentinel-2 context)",
         "gorkha_mode1": "GORKHA, mode 1 (Sentinel-1 before and after)"}
COLOR = {"random": plots.MUTED, "nearest": plots.SERIES[3], "kriging_var": plots.SERIES[0],
         "kriging_ivr": plots.SERIES[6], "gorkha_mode2": plots.SERIES[1], "gorkha_mode1": plots.SERIES[2]}
STYLE = {"kriging_var": "--"}
TARGETS = (0.20, 0.17, 0.15)
REFERENCE = "kriging_ivr"
LEVEL_ORDER = ["0", "obs", "30", "60", "100"]
LEVEL_TITLE = {"0": "all roads open", "obs": "the 13 road segments impassable on 28 April 2015 (NGA)",
               "60": "60% blocked: calibrated to the observed access of 6 May 2015"}


def level_title(level: str) -> str:
    return LEVEL_TITLE.get(level, f"{level}% of the crossed road pieces blocked")


def load() -> pd.DataFrame:
    files = FILES_NEW if any((RESULTS / f).exists() for f in FILES_NEW) else FILES_OLD
    parts = []
    for file, rename in files.items():
        path = RESULTS / file
        if path.exists():
            d = pd.read_parquet(path)
            d["method"] = d["method"].replace(rename)
            d["level"] = d["level"].astype(str)
            parts.append(d.drop(columns=[c for c in ("estimate", "surveyed") if c in d.columns]))
    res = pd.concat(parts, ignore_index=True)
    return res.drop_duplicates(["level", "draw", "method", "seed", "round"])


def curves(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby(["level", "method", "round"])[["cost_hours", "mae", "coverage_90", "flights",
                                                      "n_surveyed"]].mean().reset_index()


def cost_to_reach(res: pd.DataFrame, target: float) -> pd.DataFrame:
    """Cost (hours) of the first round where the error of a campaign is at or below `target`.
    A campaign that never reaches it counts with the cost of its last round."""
    rows = []
    for (level, method, draw, seed), d in res.groupby(["level", "method", "draw", "seed"]):
        d = d.sort_values("round")
        hit = d[d["mae"] <= target]
        rows.append({"level": level, "method": method, "draw": draw, "seed": seed,
                     "cost": hit["cost_hours"].iat[0] if len(hit) else d["cost_hours"].iat[-1],
                     "reached": len(hit) > 0})
    t = pd.DataFrame(rows)
    return t.groupby(["level", "method"]).agg(cost=("cost", "mean"), reached=("reached", "mean"))


def panel_figure(c, levels, methods, xcol, xlabel, title, path):
    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.4), sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        for m in methods:
            d = c[(c["level"] == level) & (c["method"] == m)].sort_values("round")
            if d.empty:
                continue
            xs = d[xcol] / 24 if xcol == "cost_hours" else 100 * d[xcol] / 945
            ax.plot(xs, d["mae"], color=COLOR[m], linestyle=STYLE.get(m, "-"), marker="o", markersize=3.5,
                    markeredgecolor=plots.SURFACE, markeredgewidth=0.8, label=LABEL[m])
        ax.set_title(level_title(level), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean absolute error, wards without a survey")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    plots.style()
    res = load()
    levels = [v for v in LEVEL_ORDER if v in set(res["level"])]
    methods = [m for m in LABEL if m in set(res["method"])]
    c = curves(res)
    panel_figure(c, levels, methods, "cost_hours", "survey cost (team days)",
                 "Survey campaigns: error against cost (11 teams, 16 rounds, mean of the runs)",
                 FIGURES / f"campaign_error_cost{ARGS.suffix}.png")
    panel_figure(c, levels, methods, "n_surveyed", "wards with a survey (% of 945)",
                 "Survey campaigns: error against the share of surveyed wards",
                 FIGURES / f"campaign_error_share{ARGS.suffix}.png")

    pd.set_option("display.width", 220)
    last = res["round"].max()
    final = res[res["round"] == last].groupby(["level", "method"]).agg(
        mae=("mae", "mean"), mae_sd=("mae", "std"), cost_days=("cost_hours", lambda s: s.mean() / 24),
        flights=("flights", "mean"), coverage_90=("coverage_90", "mean"), runs=("mae", "size"))
    if "blend_gnn" in res.columns:
        final["blend_gnn"] = res[res["round"] == last].groupby(["level", "method"])["blend_gnn"].mean()
    print("final round, mean of the runs:")
    print(final.round(3).to_string())
    tables = {t: cost_to_reach(res, t) for t in TARGETS}
    for t, tab in tables.items():
        print(f"\nteam days to reach an error of {t} (share of runs that reach it):")
        tab = tab.copy()
        tab["cost"] = tab["cost"] / 24
        print(tab.round(2).unstack("method").to_string())
    final.to_csv(RESULTS / f"campaign_final{ARGS.suffix}.csv")
    pd.concat({t: tab for t, tab in tables.items()}, names=["target"]).to_csv(RESULTS / f"campaign_cost_to_reach{ARGS.suffix}.csv")

    # Cost to reach the error target against the blockage level, all methods.
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.4 * len(TARGETS), 4.2), layout="constrained")
    for ax, t in zip(np.atleast_1d(axes), TARGETS):
        tab = tables[t].reset_index()
        tab = tab[tab["level"].str.isdigit()].assign(level=lambda q: q["level"].astype(int))
        for m in methods:
            d = tab[tab["method"] == m].sort_values("level")
            ax.plot(d["level"], d["cost"] / 24, color=COLOR[m], linestyle=STYLE.get(m, "-"), marker="o",
                    markersize=4, label=LABEL[m])
        ax.set_title(f"error target {t}")
        ax.set_xlabel("share of the crossed road pieces blocked (%)")
        ax.set_ylabel("team days to reach the target")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].legend(fontsize=7.5)
    fig.suptitle("Cost of a survey campaign against the road blockage (a run that never reaches the "
                 "target counts with its full cost)", x=0.01, ha="left", fontsize=11)
    fig.savefig(FIGURES / f"campaign_cost_blockage{ARGS.suffix}.png")
    plt.close(fig)

    # Benefit of GORKHA against the strongest kriging reference.
    rows = []
    for t, tab in tables.items():
        for level in levels:
            for m in methods:
                if m.startswith("gorkha") and (level, REFERENCE) in tab.index and (level, m) in tab.index:
                    rows.append({"target": t, "level": level, "method": m,
                                 "days_saved": (tab.loc[(level, REFERENCE), "cost"] - tab.loc[(level, m), "cost"]) / 24})
    if rows:
        b = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
        for m in [m for m in methods if m.startswith("gorkha")]:
            d = b[(b["method"] == m) & (b["target"] == 0.17) & b["level"].str.isdigit()]
            d = d.assign(level=d["level"].astype(int)).sort_values("level")
            ax.plot(d["level"], d["days_saved"], color=COLOR[m], marker="o", label=LABEL[m])
        ax.axhline(0, color=plots.AXIS, linewidth=1)
        ax.set_xlabel("share of the crossed road pieces blocked (%)")
        ax.set_ylabel(f"team days saved against {LABEL[REFERENCE].lower()} (error 0.17)")
        ax.set_title("Benefit of GORKHA against the road blockage")
        ax.legend(fontsize=8)
        fig.savefig(FIGURES / f"campaign_benefit_blockage{ARGS.suffix}.png")
        print(f"\nteam days saved against {REFERENCE}:")
        print(b.pivot_table(index=["target", "level"], columns="method", values="days_saved").round(2).to_string())

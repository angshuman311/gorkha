"""Figures and tables of the survey campaigns: error against cost at each blockage level."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gorkha import plots
from gorkha.paths import FIGURES, RESULTS

FILES = {"campaign_mode2.parquet": {"sherpa": "sherpa_mode2"},
         "campaign_mode1.parquet": {"sherpa": "sherpa_mode1"}}
LABEL = {"random": "Regression kriging, random surveys",
         "kriging": "Regression kriging, its uncertainty map",
         "sherpa_mode2": "SHERPA, mode 2 (Sentinel-2 context)",
         "sherpa_mode1": "SHERPA, mode 1 (Sentinel-1 before and after)"}
COLOR = {"random": plots.MUTED, "kriging": plots.SERIES[0], "sherpa_mode2": plots.SERIES[1],
         "sherpa_mode1": plots.SERIES[2]}
TARGETS = (0.20, 0.17, 0.15)


def load() -> pd.DataFrame:
    parts = []
    for file, rename in FILES.items():
        path = RESULTS / file
        if path.exists():
            d = pd.read_parquet(path)
            d["method"] = d["method"].replace(rename)
            parts.append(d)
    res = pd.concat(parts, ignore_index=True)
    return res.drop_duplicates(["level", "draw", "method", "seed", "round"])


def curves(res: pd.DataFrame) -> pd.DataFrame:
    """Mean error and mean cost at each round, over the draws and seeds."""
    return res.groupby(["level", "method", "round"])[["cost_hours", "mae", "coverage_90", "flights",
                                                      "n_surveyed"]].mean().reset_index()


def cost_to_reach(res: pd.DataFrame, target: float) -> pd.DataFrame:
    """Cost (hours) of the first round where the error of a campaign is at or below `target`."""
    rows = []
    for (level, method, draw, seed), d in res.groupby(["level", "method", "draw", "seed"]):
        d = d.sort_values("round")
        hit = d[d["mae"] <= target]
        rows.append({"level": level, "method": method, "draw": draw, "seed": seed,
                     "cost": hit["cost_hours"].iat[0] if len(hit) else np.nan,
                     "reached": len(hit) > 0})
    t = pd.DataFrame(rows)
    return t.groupby(["level", "method"]).agg(cost=("cost", "mean"), reached=("reached", "mean"))


if __name__ == "__main__":
    plots.style()
    res = load()
    levels = sorted(res["level"].unique())
    methods = [m for m in LABEL if m in set(res["method"])]
    c = curves(res)

    fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.3), sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        for m in methods:
            d = c[(c["level"] == level) & (c["method"] == m)].sort_values("round")
            if d.empty:
                continue
            ax.plot(d["cost_hours"] / 24, d["mae"], color=COLOR[m], marker="o", markersize=4,
                    markeredgecolor=plots.SURFACE, markeredgewidth=1, label=LABEL[m])
        ax.set_title(f"{level}% of the crossed road pieces blocked")
        ax.set_xlabel("survey cost (team days)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean absolute error, wards without a survey")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Survey campaigns: error against cost (11 teams, 16 rounds, mean of the runs)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIGURES / "campaign_error_cost.png")
    plt.close(fig)

    pd.set_option("display.width", 220)
    final = res[res["round"] == res["round"].max()].groupby(["level", "method"]).agg(
        mae=("mae", "mean"), mae_sd=("mae", "std"), cost_days=("cost_hours", lambda s: s.mean() / 24),
        flights=("flights", "mean"), coverage_90=("coverage_90", "mean"), runs=("mae", "size"))
    print("final round, mean of the runs:")
    print(final.round(3).to_string())
    tables = {t: cost_to_reach(res, t) for t in TARGETS}
    for t, tab in tables.items():
        print(f"\nteam days to reach an error of {t} (share of runs that reach it):")
        tab = tab.copy()
        tab["cost"] = tab["cost"] / 24
        print(tab.round(2).unstack("method").to_string())
    final.to_csv(RESULTS / "campaign_final.csv")
    pd.concat({t: tab for t, tab in tables.items()}, names=["target"]).to_csv(RESULTS / "campaign_cost_to_reach.csv")

    # Benefit of SHERPA against the blockage level: team days saved to reach the error.
    rows = []
    for t, tab in tables.items():
        for level in levels:
            for m in methods:
                if m.startswith("sherpa") and (level, "kriging") in tab.index and (level, m) in tab.index:
                    rows.append({"target": t, "level": level, "method": m,
                                 "days_saved": (tab.loc[(level, "kriging"), "cost"] - tab.loc[(level, m), "cost"]) / 24})
    if rows:
        b = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
        for m in [m for m in methods if m.startswith("sherpa")]:
            d = b[(b["method"] == m) & (b["target"] == 0.17)].sort_values("level")
            ax.plot(d["level"], d["days_saved"], color=COLOR[m], marker="o", label=LABEL[m])
        ax.axhline(0, color=plots.AXIS, linewidth=1)
        ax.set_xlabel("share of the crossed road pieces blocked (%)")
        ax.set_ylabel("team days saved against kriging (error 0.17)")
        ax.set_title("Benefit of SHERPA against the road blockage")
        ax.legend(fontsize=8)
        fig.savefig(FIGURES / "campaign_benefit_blockage.png")
        print("\nteam days saved against kriging:")
        print(b.pivot_table(index=["target", "level"], columns="method", values="days_saved").round(2).to_string())

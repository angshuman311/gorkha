"""Figures and the summary table for the baseline results."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gorkha import baselines, graph, plots, survey_sim
from gorkha.design import TARGET, design_matrix
from gorkha.paths import CRS_METRIC, FIGURES, PROCESSED, RESULTS

# Models in the figures. The color follows the model. The mean is a gray reference.
SHOWN = {"ok": plots.SERIES[0], "rk_gdif": plots.SERIES[1], "forest": plots.SERIES[2],
         "mean": plots.MUTED}
SHORT = {"ok": "Ordinary kriging", "rk_gdif": "Regression kriging", "forest": "Random forest",
         "mean": "Mean"}
SCHEME_TITLES = {"random": "Random survey", "clustered": "Clustered survey",
                 "access": "Survey near major roads"}
METRICS = ["mae", "mse", "bias", "mae_weighted", "total_rel_err", "district_mape",
           "coverage_90", "mean_sd"]


def budget_curves(res: pd.DataFrame, metric: str, ylabel: str, title: str, path) -> None:
    n_wards = len(pd.read_parquet(PROCESSED / "wards.parquet", columns=["ward_id"]))
    mean = res.groupby(["scheme", "budget", "model"], observed=True)[metric].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True, layout="constrained")
    for ax, scheme in zip(axes, survey_sim.SCHEMES):
        ends = []
        for model, color in SHOWN.items():
            v = mean.loc[scheme, :, model]
            x = 100 * v.index.to_numpy()
            ax.plot(x, v.to_numpy(), color=color, marker="o", markersize=6.5,
                    markeredgecolor=plots.SURFACE, markeredgewidth=1.5,
                    linewidth=1.2 if model == "mean" else 2.0, label=SHORT[model])
            ends.append((v.to_numpy()[-1], SHORT[model]))
        # Direct labels at the line ends, moved apart when two ends are near.
        ends.sort()
        low, high = ax.get_ylim()
        gap, last = 0.07 * (high - low), -np.inf
        for value, text in ends:
            at = max(value, last + gap)
            ax.annotate(text, (20, at), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=plots.INK_2, annotation_clip=False)
            last = at
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_xlim(0.85, 23.5)
        ax.set_title(SCHEME_TITLES[scheme])
        ax.set_xlabel(f"wards with a survey (% of {n_wards})")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def example_maps(path, scheme="random", budget=0.05, realization=0) -> None:
    w = gpd.read_file(PROCESSED / "wards.gpkg")
    wm = w.to_crs(CRS_METRIC)
    design = design_matrix(pd.DataFrame(w.drop(columns="geometry")))
    x = StandardScaler().fit_transform(design)
    xy = w[["x_km", "y_km"]].to_numpy()
    y = w[TARGET].to_numpy(dtype=float)
    s = survey_sim.survey_mask(scheme, budget, realization, graph.load()["adj_pairs"],
                               w["dist_road_km"].to_numpy())
    pred = baselines.predict_all(x, xy, y, s, seed=realization)["ok"][0]
    pred = np.where(s, y, pred)
    err = pred - y

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    plots.choropleth(axes[0], wm, y, plots.SEQUENTIAL, 0, 1, "", "Survey result (all wards)",
                     colorbar=False)
    plots.choropleth(axes[1], wm, pred, plots.SEQUENTIAL, 0, 1, "",
                     f"Ordinary kriging from {int(s.sum())} surveyed wards", colorbar=False)
    plots.scale_bar(fig, axes[:2], plots.SEQUENTIAL, 0, 1, "fraction with grade 4 or 5", 0.7)
    lim = 0.5
    plots.choropleth(axes[2], wm, err, plots.DIVERGING, -lim, lim, "predicted minus observed",
                     f"Error (mean absolute error {np.abs(err[~s]).mean():.3f})")
    for ax in axes[1:]:
        ax.scatter(xy[s, 0] * 1000, xy[s, 1] * 1000, s=16, color=plots.INK, zorder=4,
                   edgecolors=plots.SURFACE, linewidths=0.9, label="surveyed ward")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    plots.style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    res = pd.read_parquet(RESULTS / "baselines.parquet")
    res["model"] = pd.Categorical(res["model"], baselines.MODELS)

    g = res.groupby(["scheme", "budget", "model"], observed=True)
    summary = g[METRICS].mean()
    summary["rmse"] = np.sqrt(summary["mse"])
    summary["mae_sd_over_realizations"] = g["mae"].std()
    summary["n_realizations"] = g.size()
    summary["n_surveyed"] = g["n_surveyed"].first()
    summary.reset_index().to_csv(RESULTS / "baselines_summary.csv", index=False)

    budget_curves(res, "mae", "mean absolute error of the fraction",
                  "Error on the wards without a survey (grade 4 or 5 fraction, mean of 200 survey realizations)",
                  FIGURES / "baseline_mae.png")
    budget_curves(res, "district_mape", "mean relative error of the district count",
                  "Error of the count of grade 4 or 5 buildings in each district (mean of 200 survey realizations)",
                  FIGURES / "baseline_district_count.png")
    example_maps(FIGURES / "baseline_example_maps.png")

    pd.set_option("display.width", 250)
    for metric in ("rmse", "district_mape", "bias"):
        print(f"\nmean {metric}:")
        print(summary[metric].unstack("model").round(4).to_string())
    print("\nstandard deviation of the MAE over the realizations:")
    print(summary["mae_sd_over_realizations"].unstack("model").round(4).to_string())

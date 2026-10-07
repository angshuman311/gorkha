"""Simulated survey campaigns: where do the teams go next, and what does it cost?

A campaign has rounds. In each round, a method selects one ward for each team. The teams
travel on the open roads (shortest travel time), survey the wards, and the labels become
visible. The cost of a round is the travel time plus a fixed survey time for each ward.
A ward that no open road reaches costs a helicopter trip.

Methods (decisions of 2026-10-07):
- "random": regression kriging estimate (survey and feature table), random wards.
- "nearest": regression kriging estimate, each team goes to the nearest ward without a
  survey (teams without guidance, a clustered survey).
- "kriging_var": regression kriging estimate, wards with the largest kriging variance.
- "kriging_ivr": regression kriging estimate, wards whose survey decreases the total
  kriging variance the most.
- "gorkha": GNN estimate with uncertainty, blended with regression kriging by a weight that
  a cross-fit on the surveyed wards selects. A second network gives a visit score for each
  ward. It sees the road state and learns from damage maps that the model draws from its
  own uncertainty. The old name "sherpa" is accepted.
"""

import os

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeCV
from torch import nn
from torch_geometric.nn import GATv2Conv

from . import baselines, gnn
from .roads import HELICOPTER_HOURS, SURVEY_HOURS

# A ground route that takes more than this many hours is replaced by a helicopter trip.
# The default (infinite) keeps the ground route whatever its length. The 2015 graph has
# trails with walks of a day or more, so the third study sets GORKHA_FLY_ABOVE=12.
FLY_ABOVE_HOURS = float(os.environ.get("GORKHA_FLY_ABOVE", "inf"))

# ---------------------------------------------------------------- kriging


def kriging_variance(xy_s: np.ndarray, xy_all: np.ndarray, p: dict) -> np.ndarray:
    """Ordinary kriging variance at all wards for surveyed positions and a fixed variogram."""
    def gamma(h):
        return np.where(h > 0, p["nugget"] + p["psill"] * (1 - np.exp(-3 * h / p["range"])), 0.0)
    m = len(xy_s)
    a = np.ones((m + 1, m + 1))
    a[:m, :m] = gamma(np.linalg.norm(xy_s[:, None] - xy_s[None], axis=2))
    a[m, m] = 0.0
    b = np.ones((m + 1, len(xy_all)))
    b[:m] = gamma(np.linalg.norm(xy_s[:, None] - xy_all[None], axis=2))
    w = np.linalg.solve(a + 1e-10 * np.eye(m + 1), b)
    return np.maximum((w * b).sum(axis=0), 0.0)


def regression_kriging(x, xy, y, surveyed):
    """Ridge trend on the features and kriging of the leave-one-out residuals.

    Return the estimate, the standard deviation, and the residuals at the surveyed wards.
    """
    alpha = RidgeCV(alphas=baselines.RIDGE_ALPHAS).fit(x[surveyed], y[surveyed]).alpha_
    trend = Ridge(alpha=alpha).fit(x[surveyed], y[surveyed]).predict(x)
    resid = baselines.ridge_loo_residuals(x[surveyed], y[surveyed], alpha)
    r, var = baselines.krige(xy[surveyed], resid, xy)
    return np.clip(trend + r, 0, 1), np.sqrt(var), resid


def select_kriging(xy, resid, surveyed, n_pick: int) -> list:
    """Wards with the largest kriging variance of the residuals, one after the other.

    After each selection, the variance is calculated again as if that ward were surveyed.
    """
    p = baselines.fit_variogram(xy[surveyed], resid)
    if p["psill"] <= 1e-9:
        p = {"psill": max(np.var(resid), 1e-4), "range": 30.0, "nugget": 0.0}
    have = surveyed.copy()
    picks = []
    for _ in range(n_pick):
        var = kriging_variance(xy[have], xy, p)
        var[have] = -1.0
        i = int(np.argmax(var))
        picks.append(i)
        have[i] = True
    return picks


def select_kriging_ivr(xy, resid, surveyed, n_pick: int, candidates=None) -> list:
    """Wards whose survey decreases the total kriging variance the most (integrated variance
    reduction), one after the other.

    With a covariance C between wards, the kriging variance after the surveyed set is the
    diagonal of the conditional covariance P. A survey of ward i decreases the variance of
    ward j by P[i, j]^2 / P[i, i]. The ward with the largest sum over j is selected, and P
    is updated by the rank-one formula.
    """
    p = baselines.fit_variogram(xy[surveyed], resid)
    if p["psill"] <= 1e-9:
        p = {"psill": max(np.var(resid), 1e-4), "range": 30.0, "nugget": 0.0}
    h = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    cov = p["psill"] * np.exp(-3 * h / p["range"])
    n = len(xy)
    nugget = max(p["nugget"], 1e-6)
    # Condition on the surveyed wards.
    s = np.flatnonzero(surveyed)
    k_ss = cov[np.ix_(s, s)] + nugget * np.eye(len(s))
    k_s = cov[:, s]
    post = cov - k_s @ np.linalg.solve(k_ss, k_s.T)
    have = surveyed.copy()
    picks = []
    for _ in range(n_pick):
        gain = (post ** 2).sum(axis=1) / (np.diag(post) + nugget)
        gain[have] = -1.0
        if candidates is not None:
            gain[~candidates] = -1.0
        i = int(np.argmax(gain))
        picks.append(i)
        have[i] = True
        col = post[:, i].copy()
        post -= np.outer(col, col) / (post[i, i] + nugget)
    return picks


def select_nearest(travel: np.ndarray, teams: list, surveyed: np.ndarray) -> list:
    """Each team goes to the nearest ward without a survey (teams without guidance)."""
    have = surveyed.copy()
    picks = []
    for t in teams:
        cost = np.where(np.isfinite(travel[t]), travel[t], HELICOPTER_HOURS + 1e3).astype(float)
        cost[have] = np.inf
        i = int(np.argmin(cost))
        picks.append(i)
        have[i] = True
    return picks


# ---------------------------------------------------------------- travel and cost


def nearest_team_hours(travel: np.ndarray, teams: list) -> np.ndarray:
    """Cost in hours for the nearest team to reach and survey each ward."""
    t = travel[teams].min(axis=0)
    return np.where(np.isfinite(t) & (t <= FLY_ABOVE_HOURS), t, HELICOPTER_HOURS) + SURVEY_HOURS


def assign_and_move(picks: list, teams: list, travel: np.ndarray) -> tuple[float, list, int]:
    """Give each selected ward to a team (shortest travel first). Return cost, positions, flights."""
    teams, left, cost, flights = list(teams), list(picks), 0.0, 0
    free = list(range(len(teams)))
    while left and free:
        t = travel[np.ix_([teams[k] for k in free], left)]
        # A flight if no ground route exists, or if the ground route is too long.
        t = np.where(np.isfinite(t) & (t <= FLY_ABOVE_HOURS), t, HELICOPTER_HOURS + 1e3)
        a, b = np.unravel_index(np.argmin(t), t.shape)
        hours = t[a, b]
        if hours >= 1e3:
            hours, flights = HELICOPTER_HOURS, flights + 1
        cost += hours + SURVEY_HOURS
        teams[free[a]] = left[b]
        free.pop(a)
        left.pop(b)
    return cost, teams, flights


# ---------------------------------------------------------------- SHERPA visit score


class VisitScorer(nn.Module):
    """Network for the visit score (output 3). Edges carry the distance and the road time."""

    def __init__(self, n_in: int = 6, hidden: int = 32):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.convs = nn.ModuleList(GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=3) for _ in range(3))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.inp(x))
        for conv in self.convs:
            h = h + torch.relu(conv(h, edge_index, edge_attr))
        return self.out(h).squeeze(-1)


def kernel_matrix(xy: np.ndarray, range_km: float) -> np.ndarray:
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    return np.exp(-3 * d / range_km)


def imagined_benefit(sd, kern, cost, travel, surveyed, rng, n_maps: int = 24, lookahead: float = 0.5,
                     chol=None):
    """Benefit for each hour of a visit to each ward, on damage maps drawn from the model.

    A map is the model estimate plus an error with the model standard deviation and a
    spatial correlation. A visit to ward i shows its error r_i. The estimate of ward j then
    moves by kern_ij * r_i. The benefit is the decrease of the absolute error over all
    wards. The second term is the best benefit for each hour that is reachable after i.
    """
    n = len(sd)
    if chol is None:
        chol = np.linalg.cholesky(kern + 1e-6 * np.eye(n))
    benefit = np.zeros(n)
    for _ in range(n_maps):
        r = sd * (chol @ rng.standard_normal(n))
        after = np.abs(r[None, :] - kern * r[:, None])          # row i: errors after a visit to i
        gain = (np.abs(r)[None, :] - after)
        gain[:, surveyed] = 0.0
        benefit += gain.sum(axis=1)
    benefit /= n_maps
    benefit[surveyed] = 0.0
    rate = benefit / cost
    step = np.where(np.isfinite(travel) & (travel <= FLY_ABOVE_HOURS), travel, HELICOPTER_HOURS) + SURVEY_HOURS
    follow = (benefit[None, :] / step)
    np.fill_diagonal(follow, 0.0)
    return rate + lookahead * follow.max(axis=1) * (benefit > 0)


def learned_scores(sd, mean, surveyed, xy, pairs, travel, teams, range_km, seed) -> np.ndarray:
    """Train the visit scorer on imagined maps and return its score for each ward."""
    rng = np.random.default_rng([11, seed])
    torch.manual_seed(seed)
    kern = kernel_matrix(xy, range_km)
    chol = np.linalg.cholesky(kern + 1e-6 * np.eye(len(sd)))
    cost = nearest_team_hours(travel, teams)
    reach = np.isfinite(travel[teams].min(axis=0)).astype(float)
    x = np.column_stack([sd / sd.mean(), mean, surveyed.astype(float), np.log(cost), reach,
                         np.log1p(np.isfinite(travel).sum(axis=1))])
    t_edge = travel[pairs[:, 0], pairs[:, 1]]
    open_edge = np.isfinite(t_edge)
    attr = np.column_stack([np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1) / 10,
                            np.where(open_edge, np.log1p(t_edge), 0.0), open_edge.astype(float)])
    dev = gnn.DEVICE
    to = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device=dev)
    xt = to(x)
    edges = to(np.concatenate([pairs, pairs[:, ::-1]]).T, torch.long)
    attr_t = to(np.concatenate([attr, attr]))
    free = torch.as_tensor(~surveyed, device=dev)
    model = VisitScorer(x.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(120):
        target = imagined_benefit(sd, kern, cost, travel, surveyed, rng, n_maps=4, chol=chol)
        target = to(target / (target[~surveyed].mean() + 1e-9))
        loss = ((model(xt, edges, attr_t) - target)[free] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        score = model(xt, edges, attr_t).cpu().numpy()
    score[surveyed] = -np.inf
    return score


def select_sherpa(sd, mean, surveyed, xy, pairs, travel, teams, n_pick, range_km, seed,
                  learned: bool = True) -> list:
    """Select wards one after the other. After each selection, the team moves there in the
    plan, and the uncertainty near the selected ward decreases."""
    sd, teams, have = sd.copy(), list(teams), surveyed.copy()
    kern = kernel_matrix(xy, range_km)
    rng = np.random.default_rng([13, seed])
    picks = []
    for k in range(n_pick):
        if learned and k == 0:
            base = learned_scores(sd, mean, have, xy, pairs, travel, teams, range_km, seed)
            base_cost = nearest_team_hours(travel, teams)
        if learned:
            # The score is a benefit for each hour. Update it for the moved teams and for the
            # decreased uncertainty without a new training in the same round.
            cost = nearest_team_hours(travel, teams)
            with np.errstate(invalid="ignore"):
                score = base * (base_cost / cost) * (sd / sd0 if k else 1.0)
        else:
            cost = nearest_team_hours(travel, teams)
            score = imagined_benefit(sd, kern, cost, travel, have, rng)
        score = np.where(have, -np.inf, score)
        i = int(np.argmax(score))
        picks.append(i)
        if k == 0:
            sd0 = sd.copy()
        t = travel[teams, i]
        teams[int(np.argmin(np.where(np.isfinite(t), t, 1e6)))] = i
        have[i] = True
        sd = sd * (1 - kern[i])
        sd0 = np.maximum(sd0, 1e-9)
    return picks


# ---------------------------------------------------------------- GORKHA estimate


def gorkha_estimate(x, xy, y, surveyed, edges, seed: int, members: int = 3):
    """GNN group blended with regression kriging. The blend weight comes from a two-fold
    cross-fit on the surveyed wards: each fold is estimated by models that did not see it,
    and the weight with the smallest error on the held-out labels is selected.

    Return the estimate, the standard deviation, and the weight of the GNN.
    """
    n = len(y)
    rng = np.random.default_rng([19, seed])
    fold = rng.integers(0, 2, size=n)
    oof_g, oof_k = np.full(n, np.nan), np.full(n, np.nan)
    for f in (0, 1):
        vis = surveyed & (fold != f)
        hold = surveyed & (fold == f)
        if vis.sum() < 4 or hold.sum() == 0:
            continue
        mg, _, _ = gnn.fit_predict(x, xy, y, vis, edges, seed=seed * 7 + f, members=1)
        mk, _, _ = regression_kriging(x, xy, y, vis)
        oof_g[hold], oof_k[hold] = mg[hold], mk[hold]
    ok = ~np.isnan(oof_g)
    weight = 0.0
    if ok.sum() >= 4:
        grid = np.linspace(0.0, 1.0, 11)
        errs = [np.mean((w * oof_g[ok] + (1 - w) * oof_k[ok] - y[ok]) ** 2) for w in grid]
        weight = float(grid[int(np.argmin(errs))])
    mg, sg, _ = gnn.fit_predict(x, xy, y, surveyed, edges, seed=seed, members=members)
    mk, sk, _ = regression_kriging(x, xy, y, surveyed)
    mean = np.clip(weight * mg + (1 - weight) * mk, 0, 1)
    sd = weight * sg + (1 - weight) * sk
    return mean, sd, weight


# ---------------------------------------------------------------- campaign


def run(method: str, x, xy, y, pairs, edges, travel, start_wards: list, rounds: int, seed: int = 0,
        members: int = 3, learned: bool = True, x_sherpa=None, keep_estimates: bool = False) -> list:
    """Run one campaign. Return one record for each round (round 0 is the start).

    With `keep_estimates`, each record also has the estimate of each ward and the survey mask.
    `x` is the feature table (input 3) for the kriging cases. `x_sherpa` is the input of
    SHERPA: the feature table with the image vectors (inputs 3 and 4). If it is None,
    SHERPA uses `x`.
    """
    x_sherpa = x if x_sherpa is None else x_sherpa
    if method == "sherpa":
        method = "gorkha"
    n = len(y)
    rng = np.random.default_rng([17, seed])
    surveyed = np.zeros(n, dtype=bool)
    surveyed[start_wards] = True
    teams, cost, flights, records = list(start_wards), 0.0, 0, []
    for rnd in range(rounds + 1):
        weight = np.nan
        if method == "gorkha":
            mean, sd, weight = gorkha_estimate(x_sherpa, xy, y, surveyed, edges, seed=seed * 100 + rnd,
                                               members=members)
        else:
            mean, sd, resid = regression_kriging(x, xy, y, surveyed)
        err = np.abs(mean - y)[~surveyed]
        records.append({"method": method, "seed": seed, "round": rnd, "n_surveyed": int(surveyed.sum()),
                        "cost_hours": cost, "flights": flights, "mae": float(err.mean()),
                        "coverage_90": float((err <= 1.6449 * sd[~surveyed]).mean()),
                        "blend_gnn": weight})
        if keep_estimates:
            records[-1].update(estimate=mean.astype("float32").tolist(), surveyed=surveyed.tolist())
        if rnd == rounds:
            break
        if method == "random":
            picks = list(rng.choice(np.flatnonzero(~surveyed), size=len(teams), replace=False))
        elif method == "nearest":
            picks = select_nearest(travel, teams, surveyed)
        elif method in ("kriging", "kriging_var"):
            picks = select_kriging(xy, resid, surveyed, len(teams))
        elif method == "kriging_ivr":
            picks = select_kriging_ivr(xy, resid, surveyed, len(teams))
        else:
            p = baselines.fit_variogram(xy[surveyed], y[surveyed] - mean[surveyed])
            range_km = float(np.clip(p["range"], 10.0, 80.0))
            picks = select_sherpa(sd, mean, surveyed, xy, pairs, travel, teams, len(teams), range_km,
                                  seed * 100 + rnd, learned=learned)
        step_cost, teams, f = assign_and_move(picks, teams, travel)
        cost += step_cost
        flights += f
        surveyed[picks] = True
    return records

import numpy as np

from gorkha import baselines, metrics, survey_sim


def _grid(side=12):
    """Square grid graph with a smooth target."""
    ix, iy = np.meshgrid(np.arange(side), np.arange(side))
    xy = np.column_stack([ix.ravel(), iy.ravel()]).astype(float)
    node = np.arange(side * side).reshape(side, side)
    pairs = np.vstack([
        np.column_stack([node[:, :-1].ravel(), node[:, 1:].ravel()]),
        np.column_stack([node[:-1, :].ravel(), node[1:, :].ravel()]),
    ])
    y = 0.5 + 0.4 * np.sin(xy[:, 0] / 3.0) * np.cos(xy[:, 1] / 4.0)
    return xy, pairs, y


def test_masks_have_the_budget_size_and_repeat_with_the_seed():
    xy, pairs, _ = _grid()
    dist = np.linspace(0, 20, len(xy))
    for scheme in survey_sim.SCHEMES:
        for budget in survey_sim.BUDGETS:
            a = survey_sim.survey_mask(scheme, budget, 3, pairs, dist)
            b = survey_sim.survey_mask(scheme, budget, 3, pairs, dist)
            c = survey_sim.survey_mask(scheme, budget, 4, pairs, dist)
            assert a.sum() == survey_sim.n_surveyed(len(xy), budget)
            assert (a == b).all()
            assert budget < 0.05 or (a != c).any()


def test_access_mask_prefers_wards_near_a_road():
    dist = np.linspace(0, 30, 400)
    picks = np.zeros(400)
    for r in range(200):
        picks += survey_sim.access_mask(dist, 40, np.random.default_rng(r))
    assert picks[:100].sum() > 3 * picks[300:].sum()


def test_clustered_mask_is_more_compact_than_a_random_mask():
    xy, pairs, _ = _grid(20)
    adjacency = survey_sim.sparse_adjacency(pairs, len(xy))
    inside = lambda m: adjacency[m][:, m].sum() / 2
    clustered = np.mean([inside(survey_sim.clustered_mask(adjacency, 40, np.random.default_rng(r)))
                         for r in range(20)])
    random = np.mean([inside(survey_sim.random_mask(len(xy), 40, np.random.default_rng(r)))
                      for r in range(20)])
    assert clustered > 3 * random


def test_kriging_is_exact_at_the_surveyed_wards():
    xy, _, y = _grid()
    z, var = baselines.krige(xy, y, xy)
    assert np.allclose(z, y, atol=1e-6)
    assert np.allclose(var, 0, atol=1e-6)


def test_kriging_error_decreases_with_the_budget():
    xy, _, y = _grid()
    errors = []
    for m in (10, 40, 100):
        s = survey_sim.random_mask(len(xy), m, np.random.default_rng(0))
        z, _ = baselines.krige(xy[s], y[s], xy[~s])
        errors.append(np.mean(np.abs(z - y[~s])))
    assert errors[0] > errors[1] > errors[2]


def test_variogram_fit_finds_a_short_range():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, size=(300, 2))
    # Field with a practical range of approximately 30 km and a nugget.
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    cov = 0.03 * np.exp(-3 * d / 30.0) + 0.01 * np.eye(300)
    z = np.linalg.cholesky(cov) @ rng.standard_normal(300)
    p = baselines.fit_variogram(xy, z)
    assert 10 < p["range"] < 90
    assert 0.02 < p["psill"] + p["nugget"] < 0.07
    assert p["psill"] > 0.01


def test_ridge_loo_residuals_agree_with_a_refit():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(1)
    xs, ys = rng.standard_normal((12, 5)), rng.standard_normal(12)
    loo = baselines.ridge_loo_residuals(xs, ys, alpha=2.0)
    for i in range(12):
        keep = np.arange(12) != i
        pred = Ridge(alpha=2.0).fit(xs[keep], ys[keep]).predict(xs[i : i + 1])[0]
        assert np.isclose(loo[i], ys[i] - pred, atol=1e-8)


def test_metrics_on_a_known_case():
    y = np.array([0.2, 0.4, 0.6, 0.8])
    pred = np.array([0.9, 0.5, 0.5, 0.8])
    surveyed = np.array([True, False, False, False])
    n = np.array([100, 100, 300, 100])
    m = metrics.evaluate(y, pred, surveyed, n, np.array(["a", "a", "b", "b"]),
                         sd=np.full(4, 0.07))
    assert np.isclose(m["mae"], (0.1 + 0.1 + 0.0) / 3)
    assert np.isclose(m["bias"], (0.1 - 0.1 + 0.0) / 3)
    assert np.isclose(m["mae_weighted"], (100 * 0.1 + 300 * 0.1) / 500)
    # Estimate: 20 (known) + 50 + 150 + 80 = 300. Truth: 20 + 40 + 180 + 80 = 320.
    assert np.isclose(m["total_rel_err"], 20 / 320)
    assert np.isclose(m["district_mape"], (10 / 60 + 30 / 260) / 2)
    assert np.isclose(m["coverage_90"], 3 / 3)

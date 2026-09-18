import numpy as np
import pytest

from firesim.fitting import (bootstrap_x50, fit_crossing_curve, fit_logistic,
                             logistic, logistic_derivative, rmse)


def test_logistic_fit_recovers_parameters():
    x = np.linspace(0.3, 0.9, 25)
    y = logistic(x, 0.92, 35.0, 0.61)
    noise = np.random.default_rng(0).normal(0, 0.01, x.size)
    fit = fit_logistic(x, y + noise)
    assert fit.A == pytest.approx(0.92, abs=0.02)
    assert fit.k == pytest.approx(35.0, rel=0.15)
    assert fit.Dc == pytest.approx(0.61, abs=0.005)
    assert fit.r2 > 0.99
    assert "Dc" in fit.summary()


def test_logistic_confidence_intervals_have_nominal_coverage():
    x = np.linspace(0.3, 0.9, 25)
    y = logistic(x, 0.92, 35.0, 0.61)
    trials, covered = 300, 0
    for seed in range(trials):
        noise = np.random.default_rng(seed).normal(0, 0.01, x.size)
        lo, hi = fit_logistic(x, y + noise).confidence_intervals(0.95)["Dc"]
        covered += lo < 0.61 < hi
    assert 0.91 <= covered / trials <= 0.98


def test_logistic_fit_with_weights():
    x = np.linspace(0.3, 0.9, 25)
    y = logistic(x, 1.0, 25.0, 0.55)
    fit = fit_logistic(x, y, sigma=np.full(x.size, 0.02))
    assert fit.Dc == pytest.approx(0.55, abs=1e-3)


def test_steepest_point_and_max_gradient():
    x = np.linspace(0, 1, 100001)
    d = logistic_derivative(x, 0.8, 40.0, 0.6)
    assert x[np.argmax(d)] == pytest.approx(0.6, abs=1e-4)
    assert d.max() == pytest.approx(0.8 * 40.0 / 4.0, rel=1e-6)


def test_fit_logistic_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_logistic([0.1, 0.2], [0.0, 1.0])


def _binomial_data(rng, x50=0.6, k=60.0, n=500):
    x = np.linspace(x50 - 0.06, x50 + 0.06, 11)
    p = 1 / (1 + np.exp(-k * (x - x50)))
    return x, rng.binomial(n, p), np.full(x.size, n)


def test_crossing_fit_recovers_midpoint():
    x, s, n = _binomial_data(np.random.default_rng(1))
    fit = fit_crossing_curve(x, s, n)
    assert abs(fit.x50 - 0.6) < 4 * fit.x50_se
    assert fit.k == pytest.approx(60.0, rel=0.15)
    assert fit.goodness_of_fit_p > 0.001
    assert fit.predict(fit.x50) == pytest.approx(0.5)


def test_crossing_fit_error_bars_have_roughly_nominal_coverage():
    rng = np.random.default_rng(2)
    covered = 0
    trials = 300
    for _ in range(trials):
        fit = fit_crossing_curve(*_binomial_data(rng, n=200))
        lo, hi = fit.ci(0.95)
        covered += lo <= 0.6 <= hi
    assert 0.90 <= covered / trials <= 0.99


def test_bootstrap_spread_matches_delta_method():
    x, s, n = _binomial_data(np.random.default_rng(3))
    fit = fit_crossing_curve(x, s, n)
    boot = bootstrap_x50(x, s, n, n_boot=300, rng=4)
    assert np.std(boot) == pytest.approx(fit.x50_se, rel=0.3)


@pytest.mark.parametrize("s", [np.zeros(5), np.full(5, 10)])
def test_crossing_fit_needs_a_transition(s):
    with pytest.raises(ValueError):
        fit_crossing_curve(np.linspace(0, 1, 5), s, np.full(5, 10))


def test_crossing_fit_rejects_decreasing_curve():
    x = np.linspace(0, 1, 6)
    with pytest.raises(ValueError):
        fit_crossing_curve(x, [10, 9, 7, 3, 1, 0], np.full(6, 10))


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))

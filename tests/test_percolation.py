"""
Does the model reproduce percolation theory?

Sections 3, 5 and 6 of the original validation script (test_firesim.py),
plus a cross-check of rule B against an independent cluster-labelling
implementation of site-bond percolation.
"""

import numpy as np
import pytest

from firesim import (count_crossings, crossing_curve, crossing_probability,
                     estimate_threshold, find_threshold_bisection, theory)

SITE_PC = theory.SITE_PC


# ------------------------------------------- 3. the rules are distinct
def test_rule_b_spreads_more_than_rule_a_between_thresholds():
    pa = crossing_probability(64, 0.74, 0.75, "A", 400, 11)
    pb = crossing_probability(64, 0.74, 0.75, "B", 400, 11)
    assert pb > pa + 0.10, f"A={pa:.3f} B={pb:.3f}"


def test_rules_coincide_when_spread_is_certain():
    pa = crossing_probability(60, 0.62, 1.0, "A", 200, 12)
    pb = crossing_probability(60, 0.62, 1.0, "B", 200, 12)
    assert abs(pa - pb) < 0.10


@pytest.mark.parametrize("p", [0.4, 0.7])
def test_exact_two_by_two_case(p):
    # On a 2x2 all-tree lattice both left cells ignite at t=0 and each right
    # cell has exactly one burning neighbour, so for both rules
    # P(cross) = 1 - (1 - p)^2 exactly.
    target = 1 - (1 - p) ** 2
    for rule in "AB":
        measured = crossing_probability(2, 1.0, p, rule, 20000, 13)
        assert abs(measured - target) < 0.015, f"rule {rule}: {measured:.4f} vs {target:.4f}"


# ------------------------------- rule B *is* site-bond percolation
@pytest.mark.parametrize("L, density, p", [
    (48, 0.74, 0.75),     # inside the transition window
    (48, 0.60, 1.00),     # pure site percolation
    (48, 1.00, 0.52),     # pure bond percolation
    (32, 0.80, 0.70),
])
def test_dynamic_rule_b_matches_static_site_bond_percolation(L, density, p):
    n = 3000
    dynamic = count_crossings(L, density, p, "B", n, np.random.default_rng(21)) / n
    static = count_crossings(L, density, p, "B", n, np.random.default_rng(22),
                             method="static") / n
    pooled = (dynamic + static) / 2
    se = np.sqrt(2 * pooled * (1 - pooled) / n)
    assert abs(dynamic - static) < 4 * se + 1e-9, f"{dynamic:.4f} vs {static:.4f}"


def test_static_method_refuses_rule_a():
    with pytest.raises(ValueError):
        count_crossings(16, 0.7, 0.7, "A", 10, np.random.default_rng(0), method="static")


# ------------------------------------- 5. threshold ordering and bounds
@pytest.fixture(scope="module")
def thresholds_p075():
    return (find_threshold_bisection(64, 0.75, "A"),
            find_threshold_bisection(64, 0.75, "B"))


def test_rule_a_needs_more_fuel_than_rule_b(thresholds_p075):
    dA, dB = thresholds_p075
    assert dA > dB, f"A={dA:.4f} B={dB:.4f}"


def test_rule_a_respects_upper_bound(thresholds_p075):
    dA, _ = thresholds_p075
    assert dA <= theory.rule_a_upper_bound(0.75) + 0.02


def test_crossing_curve_is_monotone_and_reproducible():
    kw = dict(p_burn=1.0, seed=5)
    c1 = crossing_curve(32, [0.45, 0.55, 0.65, 0.75], "B", 300, **kw)
    c2 = crossing_curve(32, [0.45, 0.55, 0.65, 0.75], "B", 300, n_jobs=2, **kw)
    assert c1.equals(c2)
    assert np.all(np.diff(c1.P) >= -0.05)
    assert c1.P.iloc[0] < 0.1 and c1.P.iloc[-1] > 0.9


def test_estimate_threshold_scanning_density():
    est = estimate_threshold(32, "B", p_burn=1.0, n=400, seed=1, method="static")
    assert est.scanned == "density" and est.p_burn == 1.0
    assert abs(est.density - SITE_PC) < 0.02
    assert est.ci_low < est.density < est.ci_high


def test_estimate_threshold_scanning_spread_probability():
    # at full density rule B is bond percolation: p_c = 1/2
    est = estimate_threshold(32, "B", density=1.0, n=400, seed=2, method="static")
    assert est.scanned == "p_burn" and est.density == 1.0
    assert abs(est.p_burn - theory.BOND_PC) < 0.02


def test_estimate_threshold_reports_impossible_spread():
    with pytest.raises(ValueError):
        estimate_threshold(24, "B", p_burn=0.3, n=100, seed=3)


# ------------------------------------------ 6. literature recovery (slow)
@pytest.mark.slow
@pytest.mark.parametrize("L", [48, 64])
def test_rule_b_recovers_site_percolation_threshold(L):
    d = find_threshold_bisection(L, 1.0, "B", n=300, lo=0.45, hi=0.75, iters=9, seed=200)
    assert abs(d - SITE_PC) < 0.010, f"measured {d:.4f}"


@pytest.mark.slow
@pytest.mark.parametrize("L", [48, 64])
@pytest.mark.parametrize("p", [0.80, 0.70])
def test_rule_b_recovers_tarasevich_van_der_marck(L, p):
    d = find_threshold_bisection(L, p, "B", n=300, lo=0.55, hi=0.95, iters=9, seed=300)
    expected = theory.TVDM_SITE_BOND[p]
    assert abs(d - expected) < 0.010, f"measured {d:.4f} vs {expected}"


@pytest.mark.slow
def test_fitted_threshold_agrees_with_site_percolation_within_error():
    est = estimate_threshold(128, "B", p_burn=1.0, n=1000, seed=4, method="static", n_jobs=2)
    assert abs(est.density - SITE_PC) < 4 * est.se + 0.003

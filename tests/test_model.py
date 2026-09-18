"""
Correctness of the cellular automaton itself.

Sections 1, 2 and 4 of the original validation script (test_firesim.py),
plus unit tests of the update rule and the public API.
"""

import numpy as np
import pytest

from firesim import ASH, BURNING, EMPTY, TREE, ignition_probability, run, simulate
from firesim.model import _spread


def gen(seed):
    return np.random.default_rng(seed)


# ------------------------------------------------------------ 1. basics
def test_empty_lattice_never_burns():
    r = run(40, 0.0, 0.8, "A", gen(1))
    assert r["burned"] == 0 and not r["crossed"]


@pytest.mark.parametrize("rule, seed", [("A", 2), ("B", 3)])
def test_full_lattice_certain_spread_burns_everything(rule, seed):
    r = run(40, 1.0, 1.0, rule, gen(seed))
    assert r["burned"] == 1600 and r["crossed"]


def test_front_moves_one_column_per_step():
    # left column ignites at t=0, reaches column 39 at t=39, burns out at t=40
    r = run(40, 1.0, 1.0, "A", gen(4))
    assert r["steps"] in (39, 40)


def test_initial_trees_counted_before_ignition():
    r = run(30, 0.6, 0.7, "A", gen(5))
    assert abs(r["initial_trees"] - 0.6 * 900) < 60


def test_burned_never_exceeds_initial_trees():
    r = run(30, 0.5, 0.6, "A", gen(6))
    assert r["burned"] <= r["initial_trees"]


@pytest.mark.parametrize("rule", ["A", "B"])
def test_centre_ignition_always_starts_a_fire(rule):
    for seed in range(20):
        r = run(21, 0.1, 0.5, rule, gen(seed), ignition="center")
        assert 1 <= r["burned"] <= r["initial_trees"]
        assert 0.0 < r["burn_fraction"] <= 1.0


# ------------------------------------------------------ 2. determinism
def test_same_seed_gives_identical_result():
    assert run(50, 0.6, 0.7, "A", gen(42)) == run(50, 0.6, 0.7, "A", gen(42))


def test_different_seeds_give_different_results():
    # would fail if the model secretly used numpy's global random state
    assert run(50, 0.6, 0.7, "A", gen(1)) != run(50, 0.6, 0.7, "A", gen(2))


def test_integer_seed_is_accepted():
    assert run(30, 0.6, 0.7, "B", 7) == run(30, 0.6, 0.7, "B", gen(7))


# -------------------------------------------------- 4. fire dynamics
def test_cells_burn_for_exactly_one_step():
    # a full lattice crosses in ~L steps, not more
    r = run(200, 1.0, 1.0, "A", gen(15))
    assert r["steps"] <= 201


def test_well_below_threshold_fire_dies():
    assert not run(60, 0.30, 0.9, "B", gen(16))["crossed"]


def test_well_above_threshold_fire_crosses():
    assert run(60, 0.95, 0.95, "B", gen(17))["crossed"]


# -------------------------------------------------- the update rule
def _single_tree(k):
    """Padded 3x3 lattice: a tree in the middle with k burning neighbours."""
    L, W = 3, 5
    state = np.zeros((W, W), dtype=np.int8)
    state[2, 2] = TREE
    around = [(1, 2), (3, 2), (2, 1), (2, 3)][:k]
    for r, c in around:
        state[r, c] = BURNING
    flat = state.reshape(-1)
    burning = np.array([r * W + c for r, c in around])
    return flat, burning, np.array([-W, W, -1, 1])


@pytest.mark.parametrize("rule", ["A", "B"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_ignition_probability_matches_rule(rule, k):
    p, n = 0.3, 20000
    rng = gen(100 + k)
    hits = 0
    for _ in range(n):
        flat, burning, offsets = _single_tree(k)
        new = _spread(flat, burning, offsets, p, rule, rng)
        hits += new.size
        assert np.all(flat[burning] == ASH)          # burning cells burn out
    expected = ignition_probability(k, p, rule)
    se = np.sqrt(expected * (1 - expected) / n)
    assert abs(hits / n - expected) < 5 * se


def test_ignition_probability_formula():
    assert ignition_probability(0, 0.5, "A") == 0.0
    assert ignition_probability(3, 0.5, "A") == 0.5
    assert ignition_probability(3, 0.5, "B") == pytest.approx(0.875)


# ------------------------------------------------------- public API
def test_simulate_agrees_with_run():
    res, frames = simulate(40, 0.65, 0.8, "B", gen(9))
    assert res == run(40, 0.65, 0.8, "B", gen(9))
    assert len(frames) == res["steps"] + 1
    first, last = frames[0], frames[-1]
    assert np.all(np.isin(first[:, 1:], [EMPTY, TREE]))       # only column 0 is lit
    assert not np.any(last == BURNING)
    assert np.count_nonzero(last == ASH) == res["burned"]


def test_simulate_max_frames_keeps_final_state():
    res, frames = simulate(40, 1.0, 1.0, "A", gen(1), max_frames=5)
    assert len(frames) == 5
    assert np.count_nonzero(frames[-1] == ASH) == 1600


@pytest.mark.parametrize("kwargs", [
    dict(L=0, density=0.5, p_burn=0.5),
    dict(L=10, density=1.5, p_burn=0.5),
    dict(L=10, density=0.5, p_burn=-0.1),
    dict(L=10, density=0.5, p_burn=0.5, rule="C"),
    dict(L=10, density=0.5, p_burn=0.5, ignition="top"),
])
def test_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)


def test_lowercase_rule_and_british_spelling_accepted():
    a = run(20, 0.7, 0.7, "b", gen(3), ignition="centre")
    b = run(20, 0.7, 0.7, "B", gen(3), ignition="center")
    assert a == b

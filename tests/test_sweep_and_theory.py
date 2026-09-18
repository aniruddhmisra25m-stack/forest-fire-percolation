import numpy as np
import pandas as pd
import pytest

from firesim import burn_sweep, load_sweep, summarise_sweep, theory, validate_sweep


@pytest.fixture(scope="module")
def small_sweep():
    return burn_sweep([0.3, 0.85], [0.5, 1.0], rules=("A", "B"), runs=8, L=32, seed=3)


def test_sweep_shape_and_validity(small_sweep):
    assert len(small_sweep) == 2 * 2 * 2 * 8
    validate_sweep(small_sweep)


def test_sweep_does_not_depend_on_worker_count(small_sweep):
    parallel = burn_sweep([0.3, 0.85], [0.5, 1.0], rules=("A", "B"), runs=8, L=32,
                          seed=3, n_jobs=2)
    pd.testing.assert_frame_equal(small_sweep, parallel)


def test_summary_shows_the_transition(small_sweep):
    s = summarise_sweep(small_sweep).set_index(["rule", "density", "p_burn"])
    assert s.loc[("B", 0.3, 1.0), "mean"] < 0.1
    assert s.loc[("B", 0.85, 1.0), "mean"] > 0.8
    assert (s["runs"] == 8).all()


def test_load_sweep_round_trip(tmp_path, small_sweep):
    path = tmp_path / "sweep.csv"
    small_sweep.to_csv(path, index=False)
    assert len(load_sweep(path)) == len(small_sweep)


@pytest.mark.parametrize("corrupt", [
    lambda df: df.drop(columns="burn_fraction"),
    lambda df: df.iloc[:0],
    lambda df: df.assign(burn_fraction=df.burn_fraction.where(df.index != 0, np.nan)),
    lambda df: df.assign(burn_fraction=df.burn_fraction + 2),
    lambda df: df.assign(burned=df.initial_trees + 1),
    lambda df: df.iloc[1:],
])
def test_validation_catches_problems(small_sweep, corrupt):
    with pytest.raises(ValueError):
        validate_sweep(corrupt(small_sweep.copy()))


def test_theory_limits():
    assert theory.rule_a_upper_bound(1.0) == pytest.approx(theory.SITE_PC)
    assert theory.rule_a_upper_bound(0.4) == 1.0
    assert theory.yanuka_englman(1.0) == pytest.approx(theory.SITE_PC)
    assert theory.yanuka_englman(0.5) == pytest.approx(1.0)
    assert np.isnan(theory.yanuka_englman(0.4))
    # the approximation sits just above the tabulated critical line
    for pb, ps in theory.TVDM_SITE_GIVEN_BOND.items():
        assert -0.001 < theory.yanuka_englman(pb) - ps < 0.006
    pts = theory.tvdm_points()
    assert pts.shape == (16, 2) and np.all(np.diff(pts[:, 0]) >= 0)
    # the table is monotone: more open bonds -> fewer sites needed
    assert np.all(np.diff(pts[:, 1]) <= 0)

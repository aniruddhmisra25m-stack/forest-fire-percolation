"""
Landscapes: reading cover rasters, turning them into fuel maps, and the
controls that separate spatial structure from tree density.
"""

import numpy as np
import pytest

from firesim.landscape import (LandscapeSource, block_shuffle, coarsen, correlation_length,
                               cover_fraction, crossing_curve_landscape, load_cover,
                               radial_autocorrelation, shuffle, synthetic_cover, to_lattice)
from firesim.model import run_lattice


# ------------------------------------------------------------------ reading
def test_coarsen_averages_blocks_and_trims():
    cover = np.arange(36, dtype=float).reshape(6, 6)
    out = coarsen(cover, 3)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(cover[:3, :3].mean())
    assert coarsen(np.ones((7, 5)), 2).shape == (3, 2)      # remainder dropped
    assert coarsen(cover, 1) is not cover and np.array_equal(coarsen(cover, 1), cover)
    with pytest.raises(ValueError):
        coarsen(np.ones((2, 2)), 5)


@pytest.mark.parametrize("scale, top", [(1.0, 1.0), (100.0, 100.0), (255.0, 255.0)])
def test_load_cover_rescales_to_fractions(tmp_path, scale, top):
    raw = np.array([[0.0, 0.5], [0.25, 1.0]]) * top
    path = tmp_path / "cover.npy"
    np.save(path, raw)
    out = load_cover(path)
    assert out.max() == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(0.5, abs=0.01)


def test_load_cover_window(tmp_path):
    path = tmp_path / "cover.npy"
    np.save(path, np.arange(100, dtype=float).reshape(10, 10) / 99)
    assert load_cover(path, window=(2, 3, 4, 5)).shape == (4, 5)


def test_load_cover_geotiff_round_trip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    cover = (np.linspace(0, 100, 64 * 64).reshape(64, 64)).astype("float32")
    path = tmp_path / "cover.tif"
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=1,
                       dtype="float32", nodata=-9999) as dst:
        dst.write(cover, 1)
    out = load_cover(path)
    assert out.shape == (64, 64)
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)
    assert load_cover(path, window=(10, 10, 8, 8)).shape == (8, 8)


# --------------------------------------------------------- cover -> lattice
def test_quantile_lattice_hits_the_requested_density():
    cover = synthetic_cover((64, 64), 0, correlation=5)
    for level in (0.1, 0.45, 0.8):
        assert cover_fraction(to_lattice(cover, "quantile", level, 1)) == pytest.approx(level, abs=0.001)


def test_threshold_and_bernoulli_methods():
    cover = np.array([[0.0, 0.4], [0.6, 1.0]])
    assert to_lattice(cover, "threshold", 0.5).tolist() == [[False, False], [True, True]]
    drawn = to_lattice(np.full((200, 200), 0.3), "bernoulli", 1.0, rng=0)
    assert cover_fraction(drawn) == pytest.approx(0.3, abs=0.02)
    with pytest.raises(ValueError):
        to_lattice(cover, "nonsense")


def test_quantile_extremes():
    cover = synthetic_cover((16, 16), 0)
    assert not to_lattice(cover, "quantile", 0.0).any()
    assert to_lattice(cover, "quantile", 1.0).all()


# ------------------------------------------------------------ the controls
@pytest.fixture(scope="module")
def clumped():
    return to_lattice(synthetic_cover((128, 128), 1, correlation=8), "quantile", 0.5, 1)


def test_shuffle_keeps_density_and_destroys_structure(clumped):
    flat = shuffle(clumped, 2)
    assert flat.sum() == clumped.sum()
    assert correlation_length(clumped.astype(float)) > 4
    assert correlation_length(flat.astype(float)) < 1.5


def test_block_shuffle_keeps_short_range_structure(clumped):
    blocks = block_shuffle(clumped, 16, 3)
    assert blocks.sum() == clumped.sum()
    xi_blocks = correlation_length(blocks.astype(float))
    assert 1.5 < xi_blocks < correlation_length(clumped.astype(float))
    with pytest.raises(ValueError):
        block_shuffle(clumped, 200)


def test_autocorrelation_starts_at_one_and_decays():
    r, c = radial_autocorrelation(synthetic_cover((128, 128), 4, correlation=6))
    assert r[0] == 0 and c[0] == pytest.approx(1.0)
    assert c[8] < c[2] < c[0]


def test_correlation_length_grows_with_clumping():
    lengths = [correlation_length(synthetic_cover((128, 128), 5, correlation=c))
               for c in (0, 4, 12)]
    assert lengths[0] < 1.5 < lengths[1] < lengths[2]


def test_fractal_landscape_is_correlated():
    assert correlation_length(synthetic_cover((128, 128), 6, hurst=0.7)) > 5


# --------------------------------------------------------------- burning
def test_run_lattice_on_a_hand_made_landscape():
    trees = np.zeros((7, 12), dtype=bool)
    trees[3, :] = True                       # a single corridor from left to right
    result = run_lattice(trees, 1.0, "B", rng=0)
    assert result["crossed"] and result["burned"] == 12 and result["steps"] == 12
    trees[3, 6] = False                      # break the corridor
    assert not run_lattice(trees, 1.0, "B", rng=0)["crossed"]


def test_landscape_source_labels_and_shapes():
    src = LandscapeSource("gaussian", (32, 48), correlation=4)
    assert "clumped" in src.label() and src.build(0.5, 0).shape == (32, 48)
    assert "shuffled" in LandscapeSource("iid", control="shuffle").label()
    cover = synthetic_cover((20, 30), 0)
    assert LandscapeSource("raster", cover=cover).build(0.4, 0).shape == (20, 30)
    with pytest.raises(ValueError):
        LandscapeSource("raster").build(0.4, 0)


def test_independent_planting_matches_the_usual_model():
    # the "iid" landscape is the model's own lattice, so its threshold is p_c
    curve = crossing_curve_landscape(LandscapeSource("iid", (64, 64)),
                                     [0.45, 0.60, 0.75], n=200, seed=7)
    assert curve.P.iloc[0] < 0.05 and curve.P.iloc[1] > 0.4 and curve.P.iloc[2] > 0.99


@pytest.mark.parametrize("correlation", [4, 16])
def test_clumped_landscapes_cross_below_the_random_threshold(correlation):
    at = 0.55                                 # just below p_c = 0.5927
    clumped = crossing_curve_landscape(LandscapeSource("gaussian", (64, 64),
                                                       correlation=correlation),
                                       [at], n=300, seed=11).P.iloc[0]
    control = crossing_curve_landscape(LandscapeSource("gaussian", (64, 64),
                                                       correlation=correlation,
                                                       control="shuffle"),
                                       [at], n=300, seed=11).P.iloc[0]
    assert clumped > control + 0.2, f"clumped {clumped:.2f} vs shuffled {control:.2f}"

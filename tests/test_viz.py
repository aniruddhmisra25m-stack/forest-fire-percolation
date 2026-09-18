import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from firesim import simulate  # noqa: E402
from firesim.viz import THEMES, lattice_rgb, save, state_cmap, themed, write_gif  # noqa: E402


def test_lattice_rgb_uses_state_colours():
    grid = np.array([[0, 1], [2, 3]])
    img = lattice_rgb(grid, "light", scale=3)
    assert img.shape == (6, 6, 3) and img.dtype == np.uint8
    assert tuple(img[0, 3]) == (0, 131, 0)          # tree green, #008300


def test_write_gif(tmp_path):
    runs = [simulate(20, d, 1.0, "B", np.random.default_rng(1))[1] for d in (0.5, 0.8)]
    path = write_gif(runs, tmp_path / "fire.gif", every=1)
    from PIL import Image
    with Image.open(path) as im:
        assert im.n_frames == max(len(f) for f in runs)
        assert im.size == (20 * 2 * 2 + 12, 20 * 2)


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_themed_figure_saves_both_variants(tmp_path, theme):
    import matplotlib.pyplot as plt

    with themed(theme) as t:
        fig, ax = plt.subplots()
        cmap, norm = state_cmap(theme)
        ax.imshow(np.arange(4).reshape(2, 2), cmap=cmap, norm=norm)
        ax.plot([0, 1], [0, 1], color=t["rule"]["A"])
        path = save(fig, tmp_path / "figure", theme, dpi=50)
        plt.close(fig)
    assert path.exists()
    assert path.name == ("figure.png" if theme == "light" else "figure-dark.png")

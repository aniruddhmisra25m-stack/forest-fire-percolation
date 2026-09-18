"""
Plot styling and animation helpers.

Every figure is rendered twice -- ``name.png`` on a light surface and
``name-dark.png`` on a dark one -- so the README can serve the right one for
the reader's GitHub theme. Colours come from one validated palette:

* rule A / rule B: blue / orange (colour-vision-deficiency safe pair)
* lattice size (ordered): a single blue ramp, light -> dark
* lattice states: empty = surface, tree = green, burning = amber,
  ash = neutral grey (tree/ash/burning stay distinct under protanopia and
  deuteranopia simulation)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Union

import numpy as np

THEMES: Dict[str, dict] = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "rule": {"A": "#2a78d6", "B": "#eb6834"},
        "cat": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ramp": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
        "seq": ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#256abf", "#184f95", "#0d366b"],
        "states": ["#fcfcfb", "#008300", "#eda100", "#898781"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "rule": {"A": "#3987e5", "B": "#d95926"},
        "cat": ["#3987e5", "#d95926", "#199e70"],
        "ramp": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
        "seq": ["#1a1a19", "#0d366b", "#104281", "#184f95", "#256abf",
                "#3987e5", "#6da7ec", "#b7d3f6"],
        "states": ["#1a1a19", "#008300", "#eda100", "#898781"],
    },
}

STATE_LABELS = ["empty", "tree", "burning", "ash"]


@contextmanager
def themed(theme: str = "light") -> Iterator[dict]:
    """Context manager applying the matplotlib style for ``theme``; yields the palette."""
    import matplotlib as mpl

    t = THEMES[theme]
    rc = {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "savefig.edgecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.titlecolor": t["ink"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": t["ink2"],
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 6, "patch.edgecolor": t["surface"],
        "image.interpolation": "nearest",
    }
    with mpl.rc_context(rc):
        yield t


def state_cmap(theme: str = "light"):
    """Colormap and norm for drawing lattice states 0..3."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(THEMES[theme]["states"])
    return cmap, BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)


def seq_cmap(theme: str = "light"):
    """Single-hue sequential colormap (near-zero recedes into the surface)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(f"seq-{theme}", THEMES[theme]["seq"])


def save(fig, stem: Union[str, Path], theme: str, dpi: int = 200) -> Path:
    """Save ``fig`` as ``stem.png`` (light) or ``stem-dark.png`` (dark)."""
    stem = Path(stem)
    path = stem.with_name(stem.name + ("" if theme == "light" else f"-{theme}") + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    return path


def lattice_rgb(grid: np.ndarray, theme: str = "dark", scale: int = 1) -> np.ndarray:
    """Convert a lattice of states to an RGB uint8 image, each cell ``scale`` px wide."""
    from matplotlib.colors import to_rgb

    lut = (np.array([to_rgb(c) for c in THEMES[theme]["states"]]) * 255).astype(np.uint8)
    img = lut[grid]
    if scale > 1:
        img = img.repeat(scale, axis=0).repeat(scale, axis=1)
    return img


def write_gif(panels: Sequence[List[np.ndarray]], path: Union[str, Path],
              theme: str = "dark", scale: int = 2, gap: int = 12,
              every: int = 2, frame_ms: int = 40, hold_ms: int = 2500) -> Path:
    """Write side-by-side lattice animations to an animated GIF.

    ``panels`` is a list of frame lists (one per panel, from
    :func:`firesim.simulate`). Shorter runs hold their final frame while the
    others finish. Every ``every``-th step is drawn; the last frame is held for
    ``hold_ms`` before the loop restarts.
    """
    from matplotlib.colors import to_rgb
    from PIL import Image

    n = max(len(f) for f in panels)
    steps = list(range(0, n, every))
    if steps[-1] != n - 1:
        steps.append(n - 1)
    bg = (np.array(to_rgb(THEMES[theme]["surface"])) * 255).astype(np.uint8)

    lut = [tuple(int(255 * v) for v in to_rgb(c)) for c in THEMES[theme]["states"]]
    palette = [v for rgb in lut for v in rgb]
    palette += [0] * (768 - len(palette))

    images = []
    for s in steps:
        tiles = [lattice_rgb(f[min(s, len(f) - 1)], theme, scale) for f in panels]
        h = tiles[0].shape[0]
        spacer = np.broadcast_to(bg, (h, gap, 3))
        row = [tiles[0]]
        for tile in tiles[1:]:
            row += [spacer, tile]
        frame = np.concatenate(row, axis=1)
        im = Image.fromarray(np.ascontiguousarray(frame))
        pal = Image.new("P", (1, 1))
        pal.putpalette(palette)
        images.append(im.quantize(palette=pal, dither=getattr(Image, "Dither", Image).NONE))

    durations = [frame_ms] * (len(images) - 1) + [hold_ms]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    return path

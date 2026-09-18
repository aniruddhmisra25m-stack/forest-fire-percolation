"""
Real landscapes instead of random ones.

The model in :mod:`firesim.model` plants trees independently, which is the
assumption percolation theory needs but not how forests look: canopy cover is
patchy and correlated over hundreds of metres. This module turns a canopy-cover
raster (or a synthetic landscape with a tunable amount of clumping) into the
boolean map of fuel cells that :func:`firesim.run_lattice` burns, and provides
the two controls that make such a comparison meaningful:

``shuffle``
    the same cells in random positions -- identical tree density, no spatial
    structure at all. This is the independent-planting model of the rest of the
    package, drawn from the same pixels.
``block_shuffle``
    blocks of cells moved around at random -- keeps structure below the block
    size, destroys it above.

Comparing a landscape with its own shuffles isolates what the *pattern*
does, separately from how much fuel there is.

Cover rasters come in as fractions in [0, 1]. Where a raster gives percent
cover (Hansen Global Forest Change `treecover2000`, MODIS VCF) or a 0-255
image, :func:`load_cover` rescales it. Which cells count as fuel is a choice,
so it is a parameter: :func:`to_lattice` either thresholds at a cover level or
keeps the densest fraction of cells (``quantile``), which is what makes a
landscape directly comparable with a random lattice of the same density.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .model import run_lattice

PathLike = Union[str, Path]


# ------------------------------------------------------------------ loading
def load_cover(path: PathLike, window: Optional[Tuple[int, int, int, int]] = None,
               band: int = 1) -> np.ndarray:
    """Read a canopy-cover map as a float array of fractions in [0, 1].

    Supports GeoTIFF (needs ``rasterio``), ``.npy``, and ordinary images.
    ``window`` is ``(row0, col0, height, width)`` and, for GeoTIFFs, is read
    without loading the whole file -- useful for the 40000 x 40000 tiles of
    the Hansen Global Forest Change dataset.

    Values are rescaled to fractions: 0-100 (percent cover) and 0-255 (image)
    are detected from the data range and divided accordingly. A nodata value
    declared by the file becomes 0 (no fuel).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("reading GeoTIFFs needs rasterio: pip install "
                              "'firesim[geo]'") from exc
        with rasterio.open(path) as src:
            win = Window(window[1], window[0], window[3], window[2]) if window else None
            data = src.read(band, window=win).astype(np.float64)
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = 0.0
    elif suffix == ".npy":
        data = np.load(path).astype(np.float64)
        if window:
            r, c, h, w = window
            data = data[r:r + h, c:c + w]
    else:
        from PIL import Image
        with Image.open(path) as im:
            data = np.asarray(im.convert("F"), dtype=np.float64)
        if window:
            r, c, h, w = window
            data = data[r:r + h, c:c + w]

    if data.ndim != 2:
        raise ValueError(f"expected a 2-D raster, got shape {data.shape}")
    top = float(np.nanmax(data)) if data.size else 0.0
    scale = 1.0 if top <= 1.0 else (100.0 if top <= 100.0 else 255.0)
    return np.clip(np.nan_to_num(data / scale), 0.0, 1.0)


def coarsen(cover: np.ndarray, factor: int) -> np.ndarray:
    """Average ``factor`` x ``factor`` blocks -- e.g. 30 m pixels to 120 m cells.

    Rows and columns that do not fill a whole block are dropped.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return np.array(cover, dtype=float)          # a copy: never alias the caller's data
    rows = (cover.shape[0] // factor) * factor
    cols = (cover.shape[1] // factor) * factor
    if rows == 0 or cols == 0:
        raise ValueError("raster is smaller than one block")
    trimmed = np.asarray(cover[:rows, :cols], dtype=float)
    return trimmed.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


# ------------------------------------------------------- cover -> fuel cells
def to_lattice(cover: np.ndarray, method: str = "quantile", level: float = 0.5,
               rng=None) -> np.ndarray:
    """Decide which cells hold fuel.

    ``method="quantile"``
        keep the densest ``level`` fraction of cells, so the map has exactly
        that tree density and can be compared with a random lattice of the
        same density (ties are broken at random when ``rng`` is given).
    ``method="threshold"``
        every cell with cover >= ``level`` holds fuel.
    ``method="bernoulli"``
        each cell holds fuel with probability equal to its cover (times
        ``level``), which keeps partial cover as a probability rather than a
        yes/no decision.
    """
    cover = np.asarray(cover, dtype=float)
    rng = np.random.default_rng(rng)
    if method == "threshold":
        return cover >= level
    if method == "bernoulli":
        return rng.random(cover.shape) < np.clip(cover * level, 0, 1)
    if method != "quantile":
        raise ValueError(f"unknown method {method!r}")

    if not 0.0 <= level <= 1.0:
        raise ValueError("level must be a fraction in [0, 1] for method='quantile'")
    n_keep = int(round(level * cover.size))
    if n_keep <= 0:
        return np.zeros(cover.shape, dtype=bool)
    if n_keep >= cover.size:
        return np.ones(cover.shape, dtype=bool)
    # rank by cover, breaking ties at random so flat areas do not become stripes
    jitter = rng.random(cover.shape) * 1e-9
    order = np.argsort((cover + jitter).ravel(), kind="stable")
    keep = np.zeros(cover.size, dtype=bool)
    keep[order[-n_keep:]] = True
    return keep.reshape(cover.shape)


def cover_fraction(trees: np.ndarray) -> float:
    """Fraction of cells holding fuel."""
    return float(np.count_nonzero(trees) / trees.size)


# ------------------------------------------------------------------ controls
def shuffle(trees: np.ndarray, rng=None) -> np.ndarray:
    """Same cells, random positions: same density, no spatial structure."""
    rng = np.random.default_rng(rng)
    flat = np.asarray(trees, dtype=bool).ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(trees.shape)


def block_shuffle(trees: np.ndarray, block: int, rng=None) -> np.ndarray:
    """Move whole ``block`` x ``block`` tiles at random.

    Structure smaller than ``block`` survives; anything larger does not. The
    density changes only through the rows and columns that do not fill a whole
    block, which are trimmed.
    """
    rng = np.random.default_rng(rng)
    trees = np.asarray(trees, dtype=bool)
    rows = (trees.shape[0] // block) * block
    cols = (trees.shape[1] // block) * block
    if rows == 0 or cols == 0:
        raise ValueError("landscape is smaller than one block")
    tiles = trees[:rows, :cols].reshape(rows // block, block, cols // block, block)
    tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, block, block)
    tiles = tiles[rng.permutation(len(tiles))]
    return (tiles.reshape(rows // block, cols // block, block, block)
            .transpose(0, 2, 1, 3).reshape(rows, cols))


# ------------------------------------------------ measuring spatial structure
def radial_autocorrelation(field: np.ndarray, max_r: Optional[int] = None):
    """Autocorrelation of a field averaged over direction.

    Returns ``(r, C)`` with ``C[0] = 1``. Computed with FFTs on the
    mean-subtracted field, so it is cheap even for large maps.
    """
    x = np.asarray(field, dtype=float)
    x = x - x.mean()
    power = np.abs(np.fft.fft2(x)) ** 2
    corr = np.fft.ifft2(power).real / (x.size * max(x.var(), 1e-30))
    corr = np.fft.fftshift(corr)
    centre = np.array(corr.shape) // 2
    yy, xx = np.indices(corr.shape)
    r = np.hypot(yy - centre[0], xx - centre[1]).ravel()
    limit = max_r or int(min(corr.shape) // 4)
    bins = np.arange(limit + 2) - 0.5
    idx = np.digitize(r, bins) - 1
    valid = (idx >= 0) & (idx <= limit)
    sums = np.bincount(idx[valid], weights=corr.ravel()[valid], minlength=limit + 1)
    counts = np.bincount(idx[valid], minlength=limit + 1)
    return np.arange(limit + 1), sums[:limit + 1] / np.maximum(counts[:limit + 1], 1)


def correlation_length(field: np.ndarray, max_r: Optional[int] = None) -> float:
    """Distance (in cells) where the autocorrelation first falls below 1/e."""
    r, c = radial_autocorrelation(field, max_r)
    below = np.nonzero(c < np.exp(-1.0))[0]
    if not below.size:
        return float(r[-1])
    i = below[0]
    if i == 0:
        return 0.0
    c0, c1 = c[i - 1], c[i]
    return float(r[i - 1] + (c0 - np.exp(-1.0)) / max(c0 - c1, 1e-12))


# --------------------------------------------------------- synthetic patches
def synthetic_cover(shape: Tuple[int, int], rng, correlation: float = 0.0,
                    hurst: Optional[float] = None) -> np.ndarray:
    """A random cover field with a controllable amount of clumping.

    ``correlation`` is the width (in cells) of a Gaussian smoothing kernel
    applied to white noise: 0 gives an uncorrelated field, larger values give
    patchier landscapes with a correlation length of roughly that size.
    ``hurst`` instead produces a fractal field with a power-law spectrum
    (H = 0.5 is Brownian; larger H is smoother), the "fractal neutral
    landscape" used in landscape ecology, whose correlations decay slowly with
    distance rather than exponentially.

    The result is a field of *ranks*, uniform on [0, 1], so that thresholding
    it at a level gives exactly that fraction of cells.
    """
    rng = np.random.default_rng(rng)
    rows, cols = shape
    noise = rng.normal(size=(rows, cols))
    if hurst is not None:
        ky = np.fft.fftfreq(rows)[:, None]
        kx = np.fft.fftfreq(cols)[None, :]
        k = np.hypot(ky, kx)
        k[0, 0] = 1.0
        spectrum = k ** (-(hurst + 1.0))
        spectrum[0, 0] = 0.0
        field = np.fft.ifft2(np.fft.fft2(noise) * spectrum).real
    elif correlation > 0:
        y = np.fft.fftfreq(rows) * rows
        x = np.fft.fftfreq(cols) * cols
        kernel = np.exp(-0.5 * ((y[:, None] / correlation) ** 2 + (x[None, :] / correlation) ** 2))
        kernel /= kernel.sum()
        field = np.fft.ifft2(np.fft.fft2(noise) * np.fft.fft2(kernel)).real
    else:
        field = noise
    ranks = np.empty(field.size)
    ranks[np.argsort(field.ravel(), kind="stable")] = np.arange(field.size)
    return (ranks / max(field.size - 1, 1)).reshape(shape)


# ------------------------------------------------- crossing on a landscape
@dataclass
class LandscapeSource:
    """A recipe for making fuel maps at a chosen density.

    ``kind`` is one of ``"iid"`` (independent planting -- the model's usual
    lattice), ``"gaussian"`` / ``"fractal"`` (synthetic clumping, see
    :func:`synthetic_cover`), or ``"raster"`` (a cover map supplied in
    ``cover``). ``control`` optionally randomises the result:
    ``"shuffle"`` or ``"block:<n>"``.
    """

    kind: str
    shape: Tuple[int, int] = (128, 128)
    correlation: float = 0.0
    hurst: Optional[float] = None
    cover: Optional[np.ndarray] = field(default=None, repr=False)
    control: Optional[str] = None
    tile: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.tile is not None:
            self.shape = tuple(self.tile)
        elif self.kind == "raster" and self.cover is not None:
            self.shape = tuple(np.asarray(self.cover).shape)

    def label(self) -> str:
        names = {"iid": "independent planting", "raster": "landscape",
                 "gaussian": f"clumped (correlation {self.correlation:g} cells)",
                 "fractal": f"fractal (H = {self.hurst})"}
        base = names.get(self.kind, self.kind)
        if self.control == "shuffle":
            return f"{base}, shuffled"
        if self.control and self.control.startswith("block:"):
            return f"{base}, {self.control.split(':')[1]}-cell blocks shuffled"
        return base

    def _window(self, rng) -> np.ndarray:
        """The whole raster, or a random window of ``tile`` cells cut from it.

        Sampling windows turns one big map into an ensemble of landscapes, so
        a crossing probability -- and an error bar -- means something. Without
        it a fixed map has a single sharp critical density instead of a curve.
        """
        cover = np.asarray(self.cover, dtype=float)
        if self.tile is None:
            return cover
        rows, cols = self.tile
        if cover.shape[0] < rows or cover.shape[1] < cols:
            raise ValueError(f"raster {cover.shape} is smaller than the tile {self.tile}")
        r = int(rng.integers(0, cover.shape[0] - rows + 1))
        c = int(rng.integers(0, cover.shape[1] - cols + 1))
        return cover[r:r + rows, c:c + cols]

    def build(self, density: float, rng) -> np.ndarray:
        rng = np.random.default_rng(rng)
        if self.kind == "iid":
            trees = rng.random(self.shape) < density
        else:
            if self.kind == "raster":
                if self.cover is None:
                    raise ValueError("kind='raster' needs a cover array")
                cover = self._window(rng)
            else:
                cover = synthetic_cover(self.shape, rng, self.correlation, self.hurst)
            trees = to_lattice(cover, "quantile", density, rng)
        if self.control == "shuffle":
            trees = shuffle(trees, rng)
        elif self.control and self.control.startswith("block:"):
            trees = block_shuffle(trees, int(self.control.split(":")[1]), rng)
        return trees


def _landscape_worker(args):
    source, density, p_burn, rule, n, seed_seq, ignition = args
    rng = np.random.default_rng(seed_seq)
    crossed = 0
    for _ in range(n):
        trees = source.build(density, rng)
        crossed += run_lattice(trees, p_burn, rule, rng, ignition)["crossed"]
    return crossed


def crossing_curve_landscape(source: LandscapeSource, densities: Sequence[float],
                             p_burn: float = 1.0, rule: str = "B", n: int = 200,
                             seed: int = 0, n_jobs: int = 1,
                             ignition: str = "left") -> pd.DataFrame:
    """Crossing probability against tree density for one kind of landscape.

    A fresh landscape is drawn for every run (for ``kind="raster"`` the map is
    fixed, so only the threshold, the tie-breaking and the fire are random).
    """
    densities = np.asarray(densities, dtype=float)
    streams = np.random.SeedSequence(seed).spawn(len(densities))
    tasks = [(source, float(d), float(p_burn), str(rule).upper(), int(n), ss, ignition)
             for d, ss in zip(densities, streams)]
    if n_jobs == 1:
        counts = [_landscape_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            counts = list(pool.map(_landscape_worker, tasks))
    counts = np.asarray(counts)
    P = counts / n
    return pd.DataFrame({
        "landscape": source.label(), "kind": source.kind,
        "correlation": source.correlation, "hurst": source.hurst,
        "control": source.control or "", "rows": source.shape[0], "cols": source.shape[1],
        "p_burn": float(p_burn), "rule": str(rule).upper(),
        "density": densities, "crossed": counts, "runs": int(n),
        "P": P, "se": np.sqrt(P * (1 - P) / n),
    })

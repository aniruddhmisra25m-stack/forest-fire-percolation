"""
Measuring the percolation threshold of the fire model.

A fire is lit along the left edge; it *crosses* if it reaches the right edge.
For a large lattice the crossing probability jumps from ~0 to ~1 in a narrow
window around the critical line D_c(p). On a finite lattice we define the
threshold as the point where half of all fires cross.

Two ways to get that point are provided:

* :func:`find_threshold_bisection` -- quick bisection (the method used by the
  original validation script).
* :func:`estimate_threshold` -- measure a crossing-probability curve on a grid
  and fit it with binomial logistic regression, which also gives a standard
  error. It can scan tree density at fixed ``p_burn`` or ``p_burn`` at fixed
  density, so the whole critical line -- including its full-density end --
  can be traced.

:func:`site_bond_crossing` is an *independent* implementation of rule B: it
never simulates a fire, it just labels connected clusters of a random
site-bond lattice. Agreement between the two is a strong check that the
dynamic model really is site-bond percolation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

from .fitting import CrossingFit, fit_crossing_curve
from .model import _check_args, run

SCAN_AXES = ("density", "p_burn")

# ------------------------------------------------------- static reference
_CROSS = ndimage.generate_binary_structure(2, 1)   # von Neumann connectivity


def site_bond_crossing(L: int, density: float, p_bond: float,
                       rng: np.random.Generator) -> bool:
    """Does a random site-bond lattice connect its left and right edges?

    Sites are occupied with probability ``density`` and bonds are open with
    probability ``p_bond``. The lattice is drawn on a (2L-1) x (2L-1) grid in
    which even-even cells are sites and even-odd / odd-even cells are bonds,
    so ordinary 4-connected cluster labelling finds site-bond clusters.
    """
    sites = rng.random((L, L)) < density
    horiz = rng.random((L, L - 1)) < p_bond
    vert = rng.random((L - 1, L)) < p_bond

    grid = np.zeros((2 * L - 1, 2 * L - 1), dtype=bool)
    grid[::2, ::2] = sites
    grid[::2, 1::2] = horiz & sites[:, :-1] & sites[:, 1:]
    grid[1::2, ::2] = vert & sites[:-1, :] & sites[1:, :]

    labels, _ = ndimage.label(grid, structure=_CROSS)
    left = labels[::2, 0]
    right = labels[::2, -1]
    return bool(np.intersect1d(left[left > 0], right[right > 0]).size)


# ---------------------------------------------------------- counting runs
def count_crossings(L: int, density: float, p_burn: float, rule: str, n: int,
                    rng: np.random.Generator, method: str = "ca") -> int:
    """How many of ``n`` independent fires cross the lattice.

    ``method="ca"`` runs the cellular automaton; ``method="static"`` uses
    :func:`site_bond_crossing` (exact only for rule B).
    """
    if method == "ca":
        return sum(run(L, density, p_burn, rule, rng)["crossed"] for _ in range(n))
    if method == "static":
        if str(rule).upper() != "B":
            raise ValueError("the static method is exact only for rule B")
        return sum(site_bond_crossing(L, density, p_burn, rng) for _ in range(n))
    raise ValueError(f"unknown method {method!r}")


def crossing_probability(L: int, density: float, p_burn: float, rule: str,
                         n: int, seed=None, method: str = "ca") -> float:
    """Fraction of ``n`` independent fires that cross the lattice."""
    rng = np.random.default_rng(seed)
    return count_crossings(L, density, p_burn, rule, n, rng, method) / n


def _count_worker(args):
    L, density, p_burn, rule, n, seed_seq, method = args
    return count_crossings(L, density, p_burn, rule, n,
                           np.random.default_rng(seed_seq), method)


def crossing_curve(L: int, values: Sequence[float], rule: str, n: int, *,
                   p_burn: Optional[float] = None, density: Optional[float] = None,
                   seed: int = 0, n_jobs: int = 1, method: str = "ca") -> pd.DataFrame:
    """Crossing probability along a line in the (density, p_burn) plane.

    Give ``p_burn`` to scan ``values`` as densities, or ``density`` to scan
    ``values`` as spread probabilities. Each point gets its own random stream
    spawned from ``seed`` (``numpy.random.SeedSequence``), so the result is
    reproducible and does not depend on ``n_jobs``.
    """
    if (p_burn is None) == (density is None):
        raise ValueError("give exactly one of p_burn= or density=")
    values = np.asarray(values, dtype=float)
    if p_burn is not None:
        points = [(v, float(p_burn)) for v in values]
    else:
        points = [(float(density), v) for v in values]
    for d, p in points:
        _check_args(L, d, p, rule, "left")

    streams = np.random.SeedSequence(seed).spawn(len(points))
    tasks = [(int(L), d, p, str(rule).upper(), int(n), ss, method)
             for (d, p), ss in zip(points, streams)]
    if n_jobs == 1:
        counts = [_count_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            counts = list(pool.map(_count_worker, tasks))

    counts = np.asarray(counts)
    P = counts / n
    return pd.DataFrame({
        "L": int(L), "rule": str(rule).upper(),
        "density": [d for d, _ in points], "p_burn": [p for _, p in points],
        "crossed": counts, "runs": int(n),
        "P": P, "se": np.sqrt(P * (1 - P) / n),
    })


# ------------------------------------------------------------- thresholds
def find_threshold_bisection(L: int, p_burn: float, rule: str, n: int = 200,
                             lo: float = 0.45, hi: float = 1.0, iters: int = 9,
                             seed: int = 100, method: str = "ca") -> float:
    """Bisect on density for the point where P(cross) = 1/2 at fixed ``p_burn``.

    Each iteration uses a fresh seed (``seed + i``), exactly as in the
    original validation script. Resolution is ``(hi - lo) / 2**iters``, but
    the statistical error from ``n`` runs per step is usually larger.
    """
    for i in range(iters):
        mid = (lo + hi) / 2
        if crossing_probability(L, mid, p_burn, rule, n, seed + i, method) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class ThresholdEstimate:
    """One point on the critical line, from a fitted crossing curve.

    ``scanned`` says which parameter was varied; that parameter's field holds
    the critical value and ``se``/``ci_*`` refer to it. The other field is
    the value that was held fixed.
    """

    rule: str
    L: int
    scanned: str
    density: float
    p_burn: float
    se: float
    ci_low: float
    ci_high: float
    slope: float            #: logistic steepness k (per unit of the scanned axis)
    runs_per_point: int
    n_points: int
    fit_p_value: float      #: deviance goodness-of-fit p-value

    @property
    def critical(self) -> float:
        return self.density if self.scanned == "density" else self.p_burn

    def as_dict(self) -> dict:
        return asdict(self)


def _p_cross(L, value, fixed, scanned, rule, n, seed, method):
    d, p = (value, fixed) if scanned == "density" else (fixed, value)
    return crossing_probability(L, d, p, rule, n, seed, method)


def _bisect(L, fixed, scanned, rule, n, lo, hi, iters, seed, method):
    for i in range(iters):
        mid = (lo + hi) / 2
        if _p_cross(L, mid, fixed, scanned, rule, n, seed + i, method) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _window(fit: CrossingFit, lo: float, hi: float, n_points: int) -> np.ndarray:
    """Grid spanning roughly P = 5% .. 95% of a previous fit, clipped to [lo, hi]."""
    half = np.log(19.0) / fit.k              # logit(0.95) / k
    a, b = max(lo, fit.x50 - half), min(hi, fit.x50 + half)
    if b - a < 1e-3:
        a, b = max(lo, fit.x50 - 5e-4), min(hi, fit.x50 + 5e-4)
    return np.linspace(a, b, n_points)


def estimate_threshold(L: int, rule: str, *, p_burn: Optional[float] = None,
                       density: Optional[float] = None, n: int = 400,
                       n_points: int = 13, seed: int = 0, n_jobs: int = 1,
                       lo: float = 0.3, hi: float = 1.0,
                       values: Optional[Sequence[float]] = None,
                       method: str = "ca") -> ThresholdEstimate:
    """Locate the critical point on a line through the phase diagram.

    Give ``p_burn`` to find the critical *density* at that spread
    probability, or ``density`` to find the critical *spread probability*
    at that density.

    1. (unless ``values`` is given) a coarse pass -- bisection plus a small
       crossing curve -- locates the transition and its width;
    2. a fine crossing curve with ``n`` runs at ``n_points`` values spanning
       P ~ 5%..95%;
    3. binomial logistic regression gives the 50% point and its error.

    Raises ``ValueError`` if fire does not cross even at ``hi`` -- e.g. rule B
    with ``p_burn < 0.5``, where no density is enough.
    """
    if (p_burn is None) == (density is None):
        raise ValueError("give exactly one of p_burn= or density=")
    scanned = "density" if p_burn is not None else "p_burn"
    fixed = float(p_burn if p_burn is not None else density)
    rule = str(rule).upper()

    s_check, s_bisect, s_coarse, s_fine = (
        int(c.generate_state(1)[0]) for c in np.random.SeedSequence(seed).spawn(4))
    kw = {"p_burn": fixed} if scanned == "density" else {"density": fixed}

    if values is None:
        n0 = max(60, n // 4)
        if _p_cross(L, hi, fixed, scanned, rule, n0, s_check, method) < 0.5:
            raise ValueError(f"fire rarely crosses even at {scanned}={hi} "
                             f"({'p_burn' if scanned == 'density' else 'density'}={fixed}, rule {rule})")
        guess = _bisect(L, fixed, scanned, rule, n0, lo, hi, 7, s_bisect, method)
        width = 3.0 * L ** -0.75 / max(fixed, 0.3)
        coarse_vals = np.linspace(max(lo, guess - width), min(hi, guess + width), 9)
        coarse = crossing_curve(L, coarse_vals, rule, n0, seed=s_coarse,
                                n_jobs=n_jobs, method=method, **kw)
        try:
            fit0 = fit_crossing_curve(coarse[scanned], coarse.crossed, coarse.runs)
            values = _window(fit0, lo, hi, n_points)
        except ValueError:
            values = np.linspace(coarse_vals[0], coarse_vals[-1], n_points)

    curve = crossing_curve(L, values, rule, n, seed=s_fine, n_jobs=n_jobs,
                           method=method, **kw)
    fit = fit_crossing_curve(curve[scanned], curve.crossed, curve.runs)
    low, high = fit.ci()
    crit = {"density": fit.x50, "p_burn": fixed} if scanned == "density" \
        else {"density": fixed, "p_burn": fit.x50}
    return ThresholdEstimate(
        rule=rule, L=int(L), scanned=scanned, **crit,
        se=fit.x50_se, ci_low=low, ci_high=high, slope=fit.k,
        runs_per_point=int(n), n_points=len(curve),
        fit_p_value=fit.goodness_of_fit_p,
    )


def thresholds_table(estimates: List[ThresholdEstimate]) -> pd.DataFrame:
    """Collect estimates into a tidy table (one row per critical point)."""
    return pd.DataFrame([e.as_dict() for e in estimates])

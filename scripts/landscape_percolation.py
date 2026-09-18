"""
What changes when the trees are not planted independently?

Real canopy cover is clumped: cover in one cell predicts cover in its
neighbours for hundreds of metres. This experiment measures how much that
changes the tipping point, using three matched landscapes at every density:

* the landscape itself,
* the same cells shuffled (identical density, no spatial structure),
* the same cells shuffled in blocks (structure kept below the block size).

With synthetic landscapes the amount of clumping is a knob (a Gaussian
correlation length in cells). With ``--raster`` the landscape is a real
canopy-cover map -- see docs/landscapes.md for where to get one:

    python scripts/landscape_percolation.py
    python scripts/landscape_percolation.py --raster cover.tif --window 0 0 2048 2048 --coarsen 4

Writes results/landscape_thresholds.csv, results/landscape_curves.csv
and results/landscape_burn.csv.
"""

from _common import RESULTS, Timer, parser

import numpy as np
import pandas as pd

from firesim import theory
from firesim.fitting import fit_crossing_curve
from firesim.landscape import (LandscapeSource, coarsen, correlation_length,
                               crossing_curve_landscape, load_cover, to_lattice)
from firesim.model import run_lattice

CORRELATIONS = [0, 1, 2, 4, 8, 16, 32]


def threshold_of(source, p_burn, rule, n, n_points, seed, n_jobs, lo=0.15, hi=0.97):
    """Bisect for the 50% crossing density, then fit a curve around it for an error bar."""
    for i in range(8):
        mid = (lo + hi) / 2
        df = crossing_curve_landscape(source, [mid], p_burn, rule, max(60, n // 4),
                                      seed + i, n_jobs)
        if df.P.iloc[0] < 0.5:
            lo = mid
        else:
            hi = mid
    guess = (lo + hi) / 2
    width = 0.06
    for attempt in range(3):
        grid = np.clip(np.linspace(guess - width, guess + width, n_points), 0.01, 0.99)
        curve = crossing_curve_landscape(source, grid, p_burn, rule, n, seed + 100, n_jobs)
        try:
            fit = fit_crossing_curve(curve.density, curve.crossed, curve.runs)
        except ValueError:
            width *= 2
            continue
        if fit.x50_se < 0.02:
            return fit.x50, fit.x50_se, curve
        # a single fixed map with p = 1 has a sharp threshold, not a curve:
        # the logistic fit degenerates, so keep bisecting instead
        for i in range(8, 20):
            mid = (lo + hi) / 2
            df = crossing_curve_landscape(source, [mid], p_burn, rule, 40, seed + i, n_jobs)
            if df.P.iloc[0] < 0.5:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2, float("nan"), curve
    raise RuntimeError(f"could not bracket the transition for {source.label()}")


def burn_curve(source, densities, p_burn, rule, n, seed, ignition="center"):
    """Mean burned fraction at each density, for a fire lit at one central tree."""
    rows = []
    for k, d in enumerate(densities):
        rng = np.random.default_rng(np.random.SeedSequence(seed + k))
        vals = []
        for _ in range(n):
            trees = source.build(float(d), rng)
            vals.append(run_lattice(trees, p_burn, rule, rng, ignition)["burn_fraction"])
        vals = np.asarray(vals)
        rows.append({"landscape": source.label(), "density": float(d),
                     "mean": vals.mean(), "sem": vals.std(ddof=1) / np.sqrt(n), "runs": n})
    return pd.DataFrame(rows)


def main():
    p = parser(__doc__)
    p.add_argument("--L", type=int, default=128, help="lattice size for the synthetic runs")
    p.add_argument("--raster", type=str, default=None, help="canopy-cover raster to add")
    p.add_argument("--window", type=int, nargs=4, default=None,
                   metavar=("ROW", "COL", "HEIGHT", "WIDTH"), help="window to read")
    p.add_argument("--coarsen", type=int, default=1, help="average NxN pixels into one cell")
    p.add_argument("--p-burn", type=float, default=1.0)
    p.add_argument("--rule", default="B")
    args = p.parse_args()

    L, n, n_points = (48, 120, 7) if args.quick else (args.L, 400, 11)
    correlations = [0, 4] if args.quick else CORRELATIONS
    shape = (L, L)

    sources = []
    for c in correlations:
        kind = "iid" if c == 0 else "gaussian"
        sources.append(LandscapeSource(kind, shape, correlation=c))
        if c:
            sources.append(LandscapeSource(kind, shape, correlation=c, control="shuffle"))
    if not args.quick:
        for h in (0.2, 0.5, 0.8):
            sources.append(LandscapeSource("fractal", shape, hurst=h))

    if args.raster:
        with Timer(f"reading {args.raster}"):
            cover = coarsen(load_cover(args.raster, tuple(args.window) if args.window else None),
                            args.coarsen)
        print(f"    {cover.shape[0]} x {cover.shape[1]} cells, mean cover {cover.mean():.3f}, "
              f"correlation length {correlation_length(cover):.1f} cells")
        # sample windows of the same size as the synthetic lattices when the
        # raster is big enough: one map then gives an ensemble of landscapes
        tile = shape if min(cover.shape) >= 2 * L else None
        if tile is None:
            print(f"    raster smaller than 2 x {L} cells: using it whole, so its "
                  "threshold is a single sharp number rather than a distribution")
        sources += [LandscapeSource("raster", cover=cover, tile=tile),
                    LandscapeSource("raster", cover=cover, tile=tile, control="shuffle"),
                    LandscapeSource("raster", cover=cover, tile=tile,
                                    control=f"block:{max(4, L // 8)}")]

    rows, curves = [], []
    for k, source in enumerate(sources):
        with Timer(source.label()):
            x50, se, curve = threshold_of(source, args.p_burn, args.rule, n, n_points,
                                          args.seed + 1000 * k, args.jobs)
        # how clumped is this landscape, measured on a sample at its own threshold
        rng = np.random.default_rng(args.seed + k)
        sample = source.build(x50, rng)
        xi = correlation_length(sample.astype(float))
        rows.append({"landscape": source.label(), "kind": source.kind,
                     "correlation": source.correlation, "hurst": source.hurst,
                     "control": source.control or "", "rows": source.shape[0],
                     "cols": source.shape[1], "p_burn": args.p_burn, "rule": args.rule.upper(),
                     "density_c": x50, "se": se, "xi_cells": xi, "runs_per_point": n})
        curves.append(curve)
        print(f"    D_c = {x50:.4f} ± {se:.4f}   (correlation length of the fuel map "
              f"{xi:.1f} cells)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"landscape_thresholds{tag}.csv", index=False, float_format="%.6g")
    pd.concat(curves).to_csv(RESULTS / f"landscape_curves{tag}.csv", index=False)

    # how much burns, once a fire starts, above the tipping point
    densities = np.round(np.arange(0.45, 0.901, 0.05 if args.quick else 0.025), 4)
    burn = []
    for source in [LandscapeSource("iid", shape),
                   LandscapeSource("gaussian", shape, correlation=8),
                   LandscapeSource("gaussian", shape, correlation=8, control="shuffle")]:
        with Timer(f"burned fraction, {source.label()}"):
            burn.append(burn_curve(source, densities, args.p_burn, args.rule,
                                   max(30, n // 4), args.seed + 7))
    pd.concat(burn).to_csv(RESULTS / f"landscape_burn{tag}.csv", index=False, float_format="%.6g")

    print(f"\nindependent planting gives D_c = {table.density_c.iloc[0]:.4f} "
          f"(percolation theory: {theory.SITE_PC:.4f})")
    print(table[["landscape", "density_c", "se", "xi_cells"]].to_string(index=False))


if __name__ == "__main__":
    main()

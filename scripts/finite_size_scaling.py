"""
Finite-size scaling: does rule A belong to the 2-D percolation universality class?

Near a critical point the crossing probability of an L x L lattice depends on
density only through (D - D_c) * L**(1/nu): the jump sharpens like L**(1/nu)
and curves for different L collapse onto one master curve. In two dimensions
ordinary percolation has nu = 4/3 exactly.

We measure crossing curves for L = 16 ... 256 for two cases:

* rule B at p = 1 -- ordinary site percolation, where every answer is known.
  This is the *control*: it shows how well the analysis recovers the truth
  from lattices of this size.
* rule A at p = 0.8 -- no known answer.

and estimate nu three ways:

1. the steepness k(L) of a logistic fit to each curve grows like L**(1/nu);
2. a free collapse: the (D_c, nu) that best superimpose all curves, using a
   pairwise variant of the Houdayer-Hartmann measure (Phys. Rev. B 70, 014418
   (2004)), with parametric-bootstrap errors;
3. a collapse with nu fixed at 4/3, whose quality S (about 1 when curves agree
   within their errors, larger when they do not) says whether the exact
   percolation exponent describes the data.

Writes results/fss_curves.csv, results/fss_sizes.csv and results/fss_collapse.csv.

    python scripts/finite_size_scaling.py              # ~4 min on 2 cores
    python scripts/finite_size_scaling.py --analyse    # reuse saved curves
    python scripts/finite_size_scaling.py --quick
"""

from _common import RESULTS, Timer, parser

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from firesim import crossing_curve, estimate_threshold, theory
from firesim.fitting import fit_crossing_curve


def collapse_quality(params, curves):
    """Collapse measure S, ~1 when the curves agree within their errors.

    A pairwise variant of Houdayer & Hartmann (2004): for every point, each
    other curve is linearly interpolated at the same scaled abscissa, and S
    is the mean squared deviation in units of the combined error. Only pairs
    where the other curve brackets the point count.
    """
    Dc, inv_nu = params
    scaled = [((c.density - Dc) * c.L ** inv_nu, c.P, c.err) for c in curves]
    total, count = 0.0, 0
    for i, (xi, yi, ei) in enumerate(scaled):
        for xj, yj, ej in (s for j, s in enumerate(scaled) if j != i):
            inside = (xi >= xj[0]) & (xi <= xj[-1])
            if not inside.any():
                continue
            y_int = np.interp(xi[inside], xj, yj)
            e_int = np.interp(xi[inside], xj, ej)
            total += np.sum((yi[inside] - y_int) ** 2 / (ei[inside] ** 2 + e_int ** 2))
            count += inside.sum()
    return total / count if count else np.inf


class Curve:
    def __init__(self, L, density, crossed, runs):
        order = np.argsort(np.asarray(density, float))
        self.L = int(L)
        self.density = np.asarray(density, float)[order]
        self.crossed = np.asarray(crossed, float)[order]
        self.runs = np.asarray(runs, float)[order]
        self.P = self.crossed / self.runs
        # add-one smoothing keeps the error bar positive at P = 0 or 1
        p_s = (self.crossed + 1) / (self.runs + 2)
        self.err = np.sqrt(p_s * (1 - p_s) / self.runs)

    def resampled(self, rng):
        return Curve(self.L, self.density, rng.binomial(self.runs.astype(int), self.P), self.runs)


def fit_free(curves, start):
    res = minimize(collapse_quality, x0=start, args=(curves,), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-5, "maxiter": 4000})
    return res.x, res.fun


def fit_fixed_nu(curves, centre, nu=theory.NU):
    res = minimize_scalar(lambda d: collapse_quality([d, 1 / nu], curves),
                          bounds=(centre - 0.02, centre + 0.02), method="bounded",
                          options={"xatol": 1e-6})
    return res.x, res.fun


def per_size(curves):
    """Logistic fit of each curve's central part: 50% point and steepness."""
    rows = []
    for c in curves:
        keep = (c.P > 0.05) & (c.P < 0.95)
        f = fit_crossing_curve(c.density[keep], c.crossed[keep], c.runs[keep])
        rows.append({"L": c.L, "x50": f.x50, "x50_se": f.x50_se, "k": f.k, "k_se": f.k_se})
    return pd.DataFrame(rows)


def slope_nu(sizes, L_min=32):
    """1/nu from a weighted straight-line fit of log k against log L."""
    s = sizes[sizes.L >= L_min]
    x, y = np.log(s.L.to_numpy(float)), np.log(s.k.to_numpy())
    w = (s.k / s.k_se).to_numpy() ** 2
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv((X * w[:, None]).T @ X)
    beta = cov @ (X * w[:, None]).T @ y
    inv_nu, inv_nu_se = beta[1], np.sqrt(cov[1, 1])
    return 1 / inv_nu, inv_nu_se / inv_nu ** 2


def analyse(frames, n_boot, rng):
    curves_all = pd.concat(frames)
    size_rows, summary = [], []
    for (rule, p), sub in curves_all.groupby(["rule", "p_burn"], sort=False):
        curves = [Curve(L, g.density, g.crossed, g.runs) for L, g in sub.groupby("L")]
        sizes = per_size(curves).assign(rule=rule, p_burn=p)
        size_rows.append(sizes)
        centre = float(sizes.x50.iloc[-1])

        (Dc, inv_nu), S = fit_free(curves, [centre, 0.75])
        boots = np.array([fit_free([c.resampled(rng) for c in curves], [Dc, inv_nu])[0]
                          for _ in range(n_boot)])
        Dc_fixed, S_fixed = fit_fixed_nu(curves, centre)
        # drop L = 16 when enough larger sizes remain (it carries the largest corrections)
        L_min = 32 if (sizes.L >= 32).sum() >= 3 else int(sizes.L.min())
        nu_k, nu_k_se = slope_nu(sizes, L_min=L_min)
        biggest = sizes.iloc[-1]
        summary.append({
            "rule": rule, "p_burn": p, "sizes": " ".join(str(c.L) for c in curves),
            "runs_per_point": int(curves[0].runs[0]),
            "x50_largest_L": biggest.x50, "x50_largest_L_se": biggest.x50_se,
            "nu_slope": nu_k, "nu_slope_se": nu_k_se,
            "Dc_free": Dc, "Dc_free_se": boots[:, 0].std(ddof=1),
            "nu_free": 1 / inv_nu, "nu_free_se": (1 / boots[:, 1]).std(ddof=1), "S_free": S,
            "Dc_nu43": Dc_fixed, "S_nu43": S_fixed,
        })
    return pd.concat(size_rows), pd.DataFrame(summary)


def main():
    p = parser(__doc__)
    p.add_argument("--analyse", action="store_true",
                   help="skip the simulations and re-analyse results/fss_curves.csv")
    args = p.parse_args()
    tag = "_quick" if args.quick else ""
    if args.quick:
        sizes, n, n_x, n_boot = [16, 32], 150, 9, 5
    else:
        sizes, n, n_x, n_boot = [16, 32, 64, 128, 256], 800, 21, 100
    rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(1)[0])

    if args.analyse:
        frames = [pd.read_csv(RESULTS / f"fss_curves{tag}.csv")]
    else:
        x_grid = np.linspace(-2.5, 2.5, n_x)          # scaled distance from D_c
        with Timer("locating rule A threshold at p = 0.8"):
            centre_a = estimate_threshold(32 if args.quick else 128, "A", p_burn=0.8,
                                          n=100 if args.quick else 400,
                                          seed=args.seed + 7, n_jobs=args.jobs).density
        print(f"    D_c(L=128) ~ {centre_a:.4f}")
        frames = []
        for k, (rule, pb, centre) in enumerate([("B", 1.0, theory.SITE_PC),
                                                 ("A", 0.8, centre_a)]):
            for L in sizes:
                dens = np.clip(centre + x_grid * L ** -0.75, 0.0, 1.0)
                with Timer(f"rule {rule}, p = {pb}, L = {L}"):
                    frames.append(crossing_curve(L, dens, rule, n, p_burn=pb,
                                                 seed=args.seed + 100 * k + L, n_jobs=args.jobs))
        RESULTS.mkdir(exist_ok=True)
        pd.concat(frames).to_csv(RESULTS / f"fss_curves{tag}.csv", index=False)

    with Timer("collapse analysis"):
        size_table, summary = analyse(frames, n_boot, rng)
    size_table.to_csv(RESULTS / f"fss_sizes{tag}.csv", index=False, float_format="%.6g")
    summary.to_csv(RESULTS / f"fss_collapse{tag}.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(size_table.to_string(index=False))
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

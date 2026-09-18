"""
Exact and published results the simulator is checked against.

Rule B is site-bond percolation on the square lattice (site probability D,
bond probability p), so its critical line D_c^B(p) is known:

* p = 1   -> pure site percolation,  D_c = 0.592746...   (Jacobsen 2015)
* D = 1   -> pure bond percolation,  p_c = 1/2 exactly   (Kesten 1980)
* in between -> the site-bond critical line tabulated by
  Tarasevich & van der Marck (1999).

Rule A has no textbook answer, but it is squeezed between two bounds that
follow from coupling arguments (proofs in ``docs/model.md``):

    D_c^B(p)  <=  D_c^A(p)  <=  p_c^site / p

References
----------
J. L. Jacobsen, J. Phys. A: Math. Theor. 48, 454003 (2015), arXiv:1507.03027.
H. Kesten, Commun. Math. Phys. 74, 41 (1980).
Yu. Yu. Tarasevich and S. C. van der Marck, Int. J. Mod. Phys. C 10, 1193
(1999), doi:10.1142/S0129183199000978, arXiv:cond-mat/9906078.
M. Yanuka and R. Englman, J. Phys. A: Math. Gen. 23, L339 (1990).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

#: Site percolation threshold of the square lattice, 0.59274605079210(2).
SITE_PC: float = 0.59274605079210

#: Bond percolation threshold of the square lattice, exact.
BOND_PC: float = 0.5

#: Correlation-length exponent of 2-D percolation (exact).
NU: float = 4.0 / 3.0

#: Fractal dimension of the incipient infinite cluster in 2-D (exact).
D_FRACTAL: float = 91.0 / 48.0

#: Site-bond critical line, square lattice (Tarasevich & van der Marck 1999,
#: Table 1): bond probability p_b -> critical site probability p_s.
#: Quoted uncertainties are 3-4 in the last digit.
TVDM_SITE_GIVEN_BOND: Dict[float, float] = {
    0.95: 0.6149, 0.90: 0.6396, 0.85: 0.6673, 0.80: 0.6980,
    0.75: 0.7322, 0.70: 0.7709, 0.65: 0.8157, 0.60: 0.8671,
}

#: Same table, other column: site probability p_s -> critical bond probability p_b.
TVDM_BOND_GIVEN_SITE: Dict[float, float] = {
    0.95: 0.5334, 0.90: 0.5716, 0.85: 0.6154, 0.80: 0.6663,
    0.75: 0.7257, 0.70: 0.7961, 0.65: 0.8807, 0.60: 0.9840,
}

#: Backwards-compatible name used by the original validation script.
TVDM_SITE_BOND = {k: TVDM_SITE_GIVEN_BOND[k] for k in (0.70, 0.75, 0.80, 0.85, 0.90)}


def tvdm_points() -> np.ndarray:
    """All tabulated critical points as an ``(n, 2)`` array of ``(p_bond, p_site)``."""
    pts = [(pb, ps) for pb, ps in TVDM_SITE_GIVEN_BOND.items()]
    pts += [(pb, ps) for ps, pb in TVDM_BOND_GIVEN_SITE.items()]
    return np.array(sorted(pts))


def rule_a_upper_bound(p_burn):
    """Upper bound on rule A's critical density: ``SITE_PC / p`` (capped at 1).

    Every tree gets at least one ignition attempt once a neighbour burns, so
    the trees whose *first* attempt would succeed form an ordinary site
    percolation configuration of density D*p -- and all of them that connect
    to the fire do burn. Hence rule A spreads whenever D*p > SITE_PC.
    """
    p = np.asarray(p_burn, dtype=float)
    with np.errstate(divide="ignore"):
        bound = np.where(p > 0, SITE_PC / np.where(p > 0, p, 1.0), np.inf)
    bound = np.minimum(bound, 1.0)
    return float(bound) if bound.ndim == 0 else bound


def yanuka_englman(p_bond):
    """Yanuka-Englman approximation to the site-bond critical line.

    ``ln(p_s)/ln(p_s^c) + ln(p_b)/ln(p_b^c) = 1``. Exact at both ends; in
    between it lies slightly *above* the true line (by up to ~0.005 in p_s
    on the square lattice, per Tarasevich & van der Marck). Used only to draw
    a smooth reference curve.
    """
    pb = np.asarray(p_bond, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 1.0 - np.log(pb) / np.log(BOND_PC)
        ps = np.exp(np.log(SITE_PC) * frac)
    ps = np.where(pb < BOND_PC, np.nan, ps)
    return float(ps) if ps.ndim == 0 else ps

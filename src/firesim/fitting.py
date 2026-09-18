"""
Curve fitting for phase-transition data.

Two tools, for two kinds of data:

``fit_logistic``
    Least-squares fit of a three-parameter logistic
    ``A / (1 + exp(-k (x - Dc)))`` to *averaged* data such as the mean burn
    fraction at each density. ``Dc`` is the steepest point of the curve.

``fit_crossing_curve``
    Maximum-likelihood (binomial) fit of ``1 / (1 + exp(-k (x - x50)))`` to
    *counts*: "fire crossed in 173 of 400 runs at density 0.60". Each point is
    weighted by how many runs it contains, and ``x50`` -- the density at
    which a fire crosses half the time -- comes with a standard error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import expit


# =============================================================================
# Three-parameter logistic (least squares)
# =============================================================================
def logistic(x, A: float, k: float, Dc: float):
    """Three-parameter logistic: upper asymptote ``A``, steepness ``k``, midpoint ``Dc``."""
    return A * expit(k * (np.asarray(x, dtype=float) - Dc))


def logistic_derivative(x, A: float, k: float, Dc: float):
    """First derivative of :func:`logistic`; largest at ``x = Dc``."""
    s = expit(k * (np.asarray(x, dtype=float) - Dc))
    return A * k * s * (1.0 - s)


@dataclass
class FitResult:
    """Parameters and goodness of fit for :func:`fit_logistic`."""

    A: float
    k: float
    Dc: float
    r2: float
    rmse: float
    covariance: np.ndarray
    stderr: np.ndarray
    fitted_y: np.ndarray
    dof: int

    @property
    def max_gradient(self) -> float:
        """Slope of the curve at its steepest point, ``A k / 4``."""
        return self.A * self.k / 4.0

    def confidence_intervals(self, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Parameter confidence intervals using Student's t with the fit's degrees of freedom."""
        t = stats.t.ppf(0.5 + confidence / 2.0, max(self.dof, 1))
        values = {"A": self.A, "k": self.k, "Dc": self.Dc}
        return {name: (v - t * se, v + t * se)
                for (name, v), se in zip(values.items(), self.stderr)}

    def curve(self, x_min: float, x_max: float, resolution: int = 500):
        """Smooth fitted curve for plotting."""
        x = np.linspace(x_min, x_max, resolution)
        return x, logistic(x, self.A, self.k, self.Dc)

    def summary(self) -> str:
        ci = self.confidence_intervals()
        lines = [
            "Logistic fit  y = A / (1 + exp(-k (x - Dc)))",
            f"  A   = {self.A:.4f}   95% CI [{ci['A'][0]:.4f}, {ci['A'][1]:.4f}]",
            f"  k   = {self.k:.3f}   95% CI [{ci['k'][0]:.3f}, {ci['k'][1]:.3f}]",
            f"  Dc  = {self.Dc:.4f}   95% CI [{ci['Dc'][0]:.4f}, {ci['Dc'][1]:.4f}]",
            f"  R^2 = {self.r2:.4f}   RMSE = {self.rmse:.5f}",
        ]
        return "\n".join(lines)


def _initial_guess(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    A0 = float(np.max(ys))
    Dc0 = float(xs[np.argmax(np.gradient(ys, xs))]) if len(xs) > 2 else float(np.mean(xs))
    return A0, 20.0, Dc0


def fit_logistic(x, y, sigma=None,
                 bounds=((0.0, 0.1, 0.0), (1.2, 500.0, 1.0))) -> FitResult:
    """Fit :func:`logistic` to ``(x, y)`` by (optionally weighted) least squares.

    Parameters
    ----------
    x, y : array-like
        E.g. tree density and mean burn fraction.
    sigma : array-like, optional
        Standard error of each ``y`` (e.g. std / sqrt(runs)); points with
        smaller errors get more weight.
    bounds : pair of 3-tuples
        Lower and upper bounds for ``(A, k, Dc)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if len(x) < 4:
        raise ValueError("need at least 4 points to fit 3 parameters")
    if sigma is not None:
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)

    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    p0 = np.clip(_initial_guess(x, y), lo + 1e-9, hi - 1e-9)
    try:
        params, cov = curve_fit(logistic, x, y, p0=p0, sigma=sigma,
                                absolute_sigma=False, bounds=(lo, hi), maxfev=20000)
    except RuntimeError as exc:  # pragma: no cover - depends on data
        raise RuntimeError("logistic optimisation did not converge") from exc

    fitted = logistic(x, *params)
    resid = y - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return FitResult(
        A=float(params[0]), k=float(params[1]), Dc=float(params[2]),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        covariance=cov, stderr=np.sqrt(np.diag(cov)), fitted_y=fitted,
        dof=len(x) - 3,
    )


# =============================================================================
# Binomial logistic regression for crossing probabilities
# =============================================================================
@dataclass
class CrossingFit:
    """Result of :func:`fit_crossing_curve`."""

    x50: float          #: x at which the success probability is 1/2
    x50_se: float       #: standard error of x50 (delta method)
    k: float            #: steepness (logit slope)
    k_se: float
    deviance: float     #: residual deviance (~ chi-square with dof if the model fits)
    dof: int

    def ci(self, confidence: float = 0.95) -> Tuple[float, float]:
        z = stats.norm.ppf(0.5 + confidence / 2.0)
        return self.x50 - z * self.x50_se, self.x50 + z * self.x50_se

    def predict(self, x):
        return expit(self.k * (np.asarray(x, dtype=float) - self.x50))

    @property
    def goodness_of_fit_p(self) -> float:
        """p-value of the deviance test (small values mean the logistic shape fits poorly)."""
        return float(stats.chi2.sf(self.deviance, self.dof)) if self.dof > 0 else float("nan")


def fit_crossing_curve(x, successes, trials, max_iter: int = 100,
                       tol: float = 1e-10) -> CrossingFit:
    """Fit ``P(x) = 1 / (1 + exp(-k (x - x50)))`` to binomial counts.

    Uses iteratively reweighted least squares (the standard algorithm for
    logistic regression). The standard error of ``x50 = -a/b`` comes from the
    delta method applied to the fitted intercept ``a`` and slope ``b``.
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if not (x.shape == s.shape == n.shape) or x.ndim != 1:
        raise ValueError("x, successes and trials must be 1-D arrays of equal length")
    if np.any(s < 0) or np.any(s > n):
        raise ValueError("successes must lie between 0 and trials")
    if len(x) < 3:
        raise ValueError("need at least 3 points")
    if s.sum() == 0 or s.sum() == n.sum():
        raise ValueError("all outcomes identical -- the window misses the transition")

    xm = float(np.average(x, weights=n))
    X = np.column_stack([np.ones_like(x), x - xm])
    beta = np.zeros(2)
    beta[0] = np.log((s.sum() + 0.5) / (n.sum() - s.sum() + 0.5))

    def nll(b):
        eta = X @ b
        return float(np.sum(n * np.logaddexp(0.0, eta) - s * eta))

    current = nll(beta)
    for _ in range(max_iter):
        mu = expit(X @ beta)
        w = np.maximum(n * mu * (1.0 - mu), 1e-12)
        grad = X.T @ (s - n * mu)
        H = (X * w[:, None]).T @ X
        newton = np.linalg.solve(H, grad)
        t = 1.0
        while True:                            # step halving keeps the fit stable
            trial = beta + t * newton
            val = nll(trial)
            if val <= current + 1e-12 or t < 1e-8:
                break
            t *= 0.5
        improvement = current - val
        beta, current = trial, val
        if abs(improvement) < tol * (1.0 + abs(current)):
            break
    else:  # pragma: no cover
        raise RuntimeError("logistic regression did not converge")

    mu = expit(X @ beta)
    w = np.maximum(n * mu * (1.0 - mu), 1e-12)
    cov = np.linalg.inv((X * w[:, None]).T @ X)

    a, b = beta
    if b <= 0:
        raise ValueError("fitted curve is decreasing -- check the input")
    x50 = xm - a / b
    grad = np.array([-1.0 / b, a / b ** 2])     # d(x50)/d(a, b)
    x50_se = float(np.sqrt(grad @ cov @ grad))

    with np.errstate(divide="ignore", invalid="ignore"):
        dev_terms = np.where(s > 0, s * np.log(s / (n * mu)), 0.0) + \
            np.where(n - s > 0, (n - s) * np.log((n - s) / (n * (1.0 - mu))), 0.0)
    return CrossingFit(x50=float(x50), x50_se=x50_se, k=float(b),
                       k_se=float(np.sqrt(cov[1, 1])),
                       deviance=float(2.0 * np.sum(dev_terms)), dof=len(x) - 2)


def rmse(y_true, y_pred) -> float:
    """Root-mean-square error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bootstrap_x50(x, successes, trials, n_boot: int = 500,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Parametric bootstrap of ``x50``: resample each count from its binomial and refit."""
    rng = np.random.default_rng(rng)
    x = np.asarray(x, float)
    s = np.asarray(successes, float)
    n = np.asarray(trials).astype(int)
    p_hat = s / n
    out = []
    for _ in range(n_boot):
        sb = rng.binomial(n, p_hat)
        try:
            out.append(fit_crossing_curve(x, sb, n).x50)
        except (ValueError, np.linalg.LinAlgError):
            continue
    return np.asarray(out)

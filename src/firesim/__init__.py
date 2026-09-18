"""
firesim -- a forest-fire cellular automaton, and the tools to measure its
percolation threshold.

Quick start::

    import numpy as np
    from firesim import run

    result = run(L=64, density=0.62, p_burn=1.0, rule="B",
                 rng=np.random.default_rng(0))
    result["crossed"], result["burned"]
"""

from . import theory
from .fitting import CrossingFit, FitResult, fit_crossing_curve, fit_logistic, logistic
from .model import (ASH, BURNING, EMPTY, IGNITIONS, RULES, STATE_NAMES, TREE,
                    ignition_probability, run, simulate)
from .percolation import (ThresholdEstimate, count_crossings, crossing_curve,
                          crossing_probability, estimate_threshold,
                          find_threshold_bisection, site_bond_crossing,
                          thresholds_table)
from .sweep import burn_sweep, summarise_sweep
from .datasets import load_sweep, validate_sweep

__version__ = "0.1.0"

__all__ = [
    "ASH", "BURNING", "EMPTY", "TREE", "IGNITIONS", "RULES", "STATE_NAMES",
    "run", "simulate", "ignition_probability",
    "CrossingFit", "FitResult", "fit_crossing_curve", "fit_logistic", "logistic",
    "ThresholdEstimate", "count_crossings", "crossing_curve", "crossing_probability",
    "estimate_threshold", "find_threshold_bisection", "site_bond_crossing",
    "thresholds_table", "burn_sweep", "summarise_sweep", "load_sweep",
    "validate_sweep", "theory", "__version__",
]

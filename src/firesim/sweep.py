"""
Parameter sweeps: how much of the forest burns?

This is the experiment the project started with: light a single tree at the
centre of the lattice and record the fraction of trees that burn, over a grid
of tree densities and spread probabilities. Averaged over many runs, the
burn fraction jumps from ~0 to ~1 across the critical line -- the same phase
transition the crossing experiments in :mod:`firesim.percolation` measure
more precisely.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .model import run

SWEEP_COLUMNS = ["rule", "L", "density", "p_burn", "run",
                 "burn_fraction", "burned", "initial_trees", "steps"]


def _sweep_worker(args):
    rule, L, density, p_values, runs, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    rows = []
    for p in p_values:
        for r in range(runs):
            res = run(L, density, p, rule, rng, ignition="center")
            rows.append((rule, L, density, p, r, res["burn_fraction"],
                         res["burned"], res["initial_trees"], res["steps"]))
    return rows


def burn_sweep(densities: Sequence[float], p_burns: Sequence[float],
               rules: Iterable[str] = ("A",), runs: int = 20, L: int = 64,
               seed: int = 0, n_jobs: int = 1) -> pd.DataFrame:
    """Run ``runs`` centre-ignited fires at every (rule, density, p_burn).

    Returns one row per fire (see ``SWEEP_COLUMNS``). Random streams are
    spawned per (rule, density) from ``seed``, so results do not depend on
    ``n_jobs``.
    """
    densities = np.round(np.asarray(densities, dtype=float), 6)
    p_burns = [float(p) for p in np.round(np.asarray(p_burns, dtype=float), 6)]
    rules = [str(r).upper() for r in rules]
    jobs = [(rule, int(L), float(d)) for rule in rules for d in densities]
    streams = np.random.SeedSequence(seed).spawn(len(jobs))
    tasks = [(rule, L_, d, p_burns, int(runs), ss) for (rule, L_, d), ss in zip(jobs, streams)]

    if n_jobs == 1:
        chunks = [_sweep_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            chunks = list(pool.map(_sweep_worker, tasks))
    return pd.DataFrame([row for chunk in chunks for row in chunk], columns=SWEEP_COLUMNS)


def summarise_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and standard error of the burn fraction per setting."""
    g = df.groupby(["rule", "L", "density", "p_burn"])["burn_fraction"]
    out = g.agg(mean="mean", std="std", runs="count").reset_index()
    out["sem"] = out["std"] / np.sqrt(out["runs"])
    return out

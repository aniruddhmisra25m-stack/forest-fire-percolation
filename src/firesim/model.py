"""
The forest-fire cellular automaton.

An L x L square lattice. Every cell is in one of four states:

    EMPTY    (0)  bare ground, never changes
    TREE     (1)  unburned fuel
    BURNING  (2)  on fire for exactly one time step
    ASH      (3)  burned out, inert

Each run:

1.  Plant a tree in every cell independently with probability ``density`` (D).
2.  Ignite: either every tree in the left-hand column (``ignition="left"``,
    used to measure whether fire *crosses* the lattice) or a single tree at
    the centre (``ignition="center"``, used to measure how much *burns*).
3.  Update all cells synchronously until nothing is burning:

    * BURNING -> ASH
    * TREE with k >= 1 burning von Neumann neighbours (N, S, E, W) ignites:

      - rule ``"A"`` (single attempt): with probability p, whatever k is.
        This is the rule described in the original project write-up.
      - rule ``"B"`` (independent attempts): each burning neighbour gets its
        own attempt, so the tree ignites with probability 1 - (1 - p)**k.

    The lattice has closed ("invisible wall") boundaries.

Because a cell burns for exactly one step, every (burning cell -> tree)
link is tried at most once under rule B. That makes rule B *exactly*
site-bond percolation with site probability D and bond probability p,
which is why it can be checked against known results (see ``theory.py``
and ``docs/model.md``). Rule A is a different, less contagious process.

Implementation note
-------------------
The lattice is stored with a one-cell EMPTY border and flattened, so the
four neighbours of flat index ``i`` are ``i - W, i + W, i - 1, i + 1``
with no bounds checks. Each step only touches the burning front and its
neighbours, so the cost per step is proportional to the size of the front,
not to L**2. This is ~100x faster than looping over every cell in Python.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

EMPTY: int = 0
TREE: int = 1
BURNING: int = 2
ASH: int = 3

STATE_NAMES = {EMPTY: "empty", TREE: "tree", BURNING: "burning", ASH: "ash"}

RULES = ("A", "B")
IGNITIONS = ("left", "center")

RngLike = Union[np.random.Generator, int, None]
Result = Dict[str, Union[bool, int, float]]


# ---------------------------------------------------------------- validation
def _check_args(L: int, density: float, p_burn: float, rule: str, ignition: str) -> Tuple[str, str]:
    if int(L) != L or L < 1:
        raise ValueError(f"L must be a positive integer, got {L!r}")
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density!r}")
    if not 0.0 <= p_burn <= 1.0:
        raise ValueError(f"p_burn must be in [0, 1], got {p_burn!r}")
    rule = str(rule).upper()
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    ignition = "center" if ignition == "centre" else ignition
    if ignition not in IGNITIONS:
        raise ValueError(f"ignition must be one of {IGNITIONS}, got {ignition!r}")
    return rule, ignition


# ------------------------------------------------------------------- engine
def _initialise(trees: np.ndarray, ignition: str):
    """Build the padded lattice from a map of trees and light the initial fire.

    ``trees`` is a boolean array (rows x cols; it need not be square -- a real
    landscape rarely is). Returns ``(state, flat, burning, initial_trees)``
    where ``state`` is the padded array, ``flat`` is a 1-D view of it, and
    ``burning`` holds the flat indices of the cells currently on fire.
    """
    rows, cols = trees.shape
    W = cols + 2
    state = np.zeros((rows + 2, W), dtype=np.int8)
    interior = state[1:-1, 1:-1]
    interior[trees] = TREE

    if ignition == "left":
        seeds = np.arange(1, rows + 1) * W + 1       # column 0 of the interior
    else:
        r, c = rows // 2, cols // 2
        interior[r, c] = TREE                        # guarantee a fire starts
        seeds = np.array([(r + 1) * W + (c + 1)])

    # counted BEFORE ignition, so burned <= initial_trees always holds
    initial_trees = int(np.count_nonzero(interior == TREE))

    flat = state.reshape(-1)
    burning = seeds[flat[seeds] == TREE]
    flat[burning] = BURNING
    return state, flat, burning, initial_trees


def _spread(flat: np.ndarray, burning: np.ndarray, offsets: np.ndarray,
            p_burn: float, rule: str, rng: np.random.Generator) -> np.ndarray:
    """Advance the fire by one synchronous time step (in place).

    Returns the flat indices of the cells that are burning after the step.
    """
    neighbours = (burning[:, None] + offsets).ravel()
    # one entry per (burning cell, tree neighbour) link
    links = neighbours[flat[neighbours] == TREE]

    if p_burn >= 1.0:                    # every attempt succeeds: no coins needed
        new = np.unique(links)
    elif p_burn <= 0.0:
        new = links[:0]
    elif rule == "A":
        exposed = np.unique(links)                       # one coin per tree
        new = exposed[rng.random(exposed.size) < p_burn]
    else:
        hits = links[rng.random(links.size) < p_burn]    # one coin per link
        new = np.unique(hits)

    flat[burning] = ASH
    flat[new] = BURNING
    return new


def _evolve(trees: np.ndarray, p_burn: float, rule: str, rng: np.random.Generator,
            ignition: str) -> Iterator[Tuple[np.ndarray, int, int]]:
    """Yield ``(interior_view, step, initial_trees)`` after ignition and after every step."""
    state, flat, burning, initial_trees = _initialise(trees, ignition)
    W = trees.shape[1] + 2
    offsets = np.array([-W, W, -1, 1])
    interior = state[1:-1, 1:-1]
    step = 0
    yield interior, step, initial_trees
    while burning.size:
        burning = _spread(flat, burning, offsets, p_burn, rule, rng)
        step += 1
        yield interior, step, initial_trees


def _summarise(interior: np.ndarray, steps: int, initial_trees: int) -> Result:
    burned = int(np.count_nonzero(interior == ASH))
    return {
        "crossed": bool(np.any(interior[:, -1] == ASH)),
        "burned": burned,
        "initial_trees": initial_trees,
        "steps": steps,
        "burn_fraction": burned / initial_trees if initial_trees else 0.0,
    }


# --------------------------------------------------------------- public API
def run(L: int, density: float, p_burn: float, rule: str = "A",
        rng: RngLike = None, ignition: str = "left") -> Result:
    """Run one fire to completion and summarise it.

    Parameters
    ----------
    L : int
        Lattice side length (the lattice is L x L).
    density : float
        Probability that each cell starts as a tree (D).
    p_burn : float
        Spread probability (p).
    rule : {"A", "B"}
        "A": one ignition attempt per tree per time step.
        "B": one ignition attempt per burning neighbour per time step.
    rng : numpy.random.Generator, int or None
        Source of *all* randomness. Pass a Generator (or a seed) for
        reproducible runs; the same generator state always gives the
        same result.
    ignition : {"left", "center"}
        Ignite the whole left column, or a single tree at the centre.

    Returns
    -------
    dict
        ``crossed``        did any cell in the right-hand column burn?
        ``burned``         number of ASH cells at the end
        ``initial_trees``  number of trees before ignition
        ``steps``          time steps until the fire died out
        ``burn_fraction``  burned / initial_trees (0 if there were no trees)

    Examples
    --------
    >>> import numpy as np
    >>> from firesim import run
    >>> run(64, 0.62, 1.0, rule="B", rng=np.random.default_rng(0))["crossed"]
    True
    """
    rule, ignition = _check_args(L, density, p_burn, rule, ignition)
    rng = np.random.default_rng(rng)
    trees = rng.random((int(L), int(L))) < density
    return run_lattice(trees, p_burn, rule, rng, ignition)


def run_lattice(trees, p_burn: float, rule: str = "A", rng: RngLike = None,
                ignition: str = "left") -> Result:
    """Run one fire on a *given* map of trees, instead of a random one.

    ``trees`` is a boolean array (True where a cell holds fuel); it need not be
    square. This is the entry point for real landscapes -- see
    :mod:`firesim.landscape`, which turns a canopy-cover raster into such a
    map. Everything else works exactly as in :func:`run`.
    """
    trees = np.ascontiguousarray(trees, dtype=bool)
    if trees.ndim != 2 or min(trees.shape) < 1:
        raise ValueError(f"trees must be a non-empty 2-D array, got shape {trees.shape}")
    rule, ignition = _check_args(max(trees.shape), 0.0, p_burn, rule, ignition)
    rng = np.random.default_rng(rng)
    interior, steps, initial_trees = None, 0, 0
    for interior, steps, initial_trees in _evolve(trees, p_burn, rule, rng, ignition):
        pass
    return _summarise(interior, steps, initial_trees)


def simulate(L: int, density: float, p_burn: float, rule: str = "A",
             rng: RngLike = None, ignition: str = "left",
             max_frames: Optional[int] = None) -> Tuple[Result, List[np.ndarray]]:
    """Like :func:`run`, but also return a copy of the lattice at every step.

    Useful for animations and for looking at burn patterns. ``frames[0]`` is
    the lattice just after ignition and ``frames[-1]`` is the final state.
    ``max_frames`` caps memory use for very long fires (the final state is
    always kept).
    """
    rule, ignition = _check_args(L, density, p_burn, rule, ignition)
    rng = np.random.default_rng(rng)
    trees = rng.random((int(L), int(L))) < density
    return simulate_lattice(trees, p_burn, rule, rng, ignition, max_frames)


def simulate_lattice(trees, p_burn: float, rule: str = "A", rng: RngLike = None,
                     ignition: str = "left",
                     max_frames: Optional[int] = None) -> Tuple[Result, List[np.ndarray]]:
    """Like :func:`run_lattice`, but also return the lattice at every step."""
    trees = np.ascontiguousarray(trees, dtype=bool)
    if trees.ndim != 2 or min(trees.shape) < 1:
        raise ValueError(f"trees must be a non-empty 2-D array, got shape {trees.shape}")
    rule, ignition = _check_args(max(trees.shape), 0.0, p_burn, rule, ignition)
    if max_frames is not None and max_frames < 2:
        raise ValueError("max_frames must be at least 2 (first and last frame)")
    rng = np.random.default_rng(rng)
    frames: List[np.ndarray] = []
    interior, steps, initial_trees = None, 0, 0
    for interior, steps, initial_trees in _evolve(trees, p_burn, rule, rng, ignition):
        if max_frames is None or len(frames) < max_frames - 1:
            frames.append(interior.copy())
    if not np.array_equal(frames[-1], interior):
        frames.append(interior.copy())
    return _summarise(interior, steps, initial_trees), frames


def ignition_probability(k: int, p_burn: float, rule: str) -> float:
    """Probability that a tree with ``k`` burning neighbours ignites this step."""
    if k <= 0:
        return 0.0
    return p_burn if str(rule).upper() == "A" else 1.0 - (1.0 - p_burn) ** k

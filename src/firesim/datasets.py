"""
Loading and checking sweep datasets written by ``scripts/burn_sweep.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .sweep import SWEEP_COLUMNS


def validate_sweep(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if a sweep table is malformed.

    Checks: required columns present, no missing values, fractions in
    [0, 1], burned never exceeds the initial number of trees, and every
    setting has the same number of runs.
    """
    missing = set(SWEEP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("dataset is empty")
    if df[SWEEP_COLUMNS].isna().any().any():
        raise ValueError("dataset contains missing values")
    if ((df["burn_fraction"] < 0) | (df["burn_fraction"] > 1)).any():
        raise ValueError("burn_fraction outside [0, 1]")
    if (df["burned"] > df["initial_trees"]).any():
        raise ValueError("a run burned more trees than it started with")
    counts = df.groupby(["rule", "L", "density", "p_burn"]).size()
    if counts.nunique() != 1:
        raise ValueError("settings have different numbers of runs")


def load_sweep(path: Union[str, Path]) -> pd.DataFrame:
    """Read a sweep CSV and validate it."""
    df = pd.read_csv(path)
    validate_sweep(df)
    return df

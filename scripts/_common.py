"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"

# let the scripts run from a fresh clone without `pip install -e .`
sys.path.insert(0, str(ROOT / "src"))


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                   help="worker processes (default: all cores)")
    p.add_argument("--quick", action="store_true",
                   help="tiny version of the experiment, for smoke-testing")
    p.add_argument("--seed", type=int, default=2026, help="master random seed")
    return p


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"{self.label} ...", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"{self.label} done in {time.perf_counter() - self.t0:.1f} s", flush=True)

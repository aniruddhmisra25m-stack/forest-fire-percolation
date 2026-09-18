"""
The original experiment, rebuilt: how much of the forest burns?

Lights one tree at the centre of a 64 x 64 lattice and records the fraction
of trees that burn, for both ignition rules over a grid of tree densities
and spread probabilities.

Writes data/burn_sweep.csv (one row per fire, ~2 MB, not tracked by git)
and results/burn_sweep_summary.csv (means and standard errors).

    python scripts/burn_sweep.py            # ~10 s on 2 cores
    python scripts/burn_sweep.py --quick
"""

from _common import DATA, RESULTS, Timer, parser

import numpy as np

from firesim import burn_sweep, summarise_sweep, validate_sweep


def main():
    args = parser(__doc__).parse_args()
    if args.quick:
        densities, p_values, runs = np.arange(0.2, 1.0, 0.2), [0.5, 1.0], 5
    else:
        densities = np.arange(0.15, 1.0001, 0.025)
        p_values = np.arange(0.15, 1.0001, 0.05)
        runs = 30

    with Timer("sweep"):
        df = burn_sweep(densities, p_values, rules=("A", "B"), runs=runs, L=64,
                        seed=args.seed, n_jobs=args.jobs)
    validate_sweep(df)

    tag = "_quick" if args.quick else ""
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(DATA / f"burn_sweep{tag}.csv", index=False)
    summary = summarise_sweep(df)
    summary.to_csv(RESULTS / f"burn_sweep_summary{tag}.csv", index=False, float_format="%.6g")
    print(f"{len(df)} fires -> {DATA / f'burn_sweep{tag}.csv'}")


if __name__ == "__main__":
    main()

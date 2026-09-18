"""
Phase diagram: the critical line D_c(p) for both ignition rules.

Scans tree density at fixed spread probability (the flat part of the line)
and spread probability at fixed density (the steep part near D = 1), fitting
each crossing curve for its 50% point.

Writes results/phase_diagram.csv.

    python scripts/phase_diagram.py            # ~7 min on 2 cores
    python scripts/phase_diagram.py --quick
"""

from _common import RESULTS, Timer, parser

import numpy as np

from firesim import estimate_threshold, thresholds_table


def main():
    p = parser(__doc__)
    p.add_argument("--L", type=int, default=128)
    args = p.parse_args()

    if args.quick:
        L, n, n_points = 32, 120, 7
        p_values = [0.7, 1.0]
        d_values = [1.0]
    else:
        L, n, n_points = args.L, 500, 11
        p_values = [round(x, 2) for x in np.arange(0.55, 1.001, 0.05)]
        d_values = [1.0, 0.95, 0.90, 0.85]

    plan = [(rule, "p_burn", v) for rule in "AB" for v in p_values] + \
           [(rule, "density", v) for rule in "AB" for v in d_values]
    seeds = np.random.SeedSequence(args.seed).spawn(len(plan))

    estimates, skipped = [], []
    for (rule, fixed_name, value), ss in zip(plan, seeds):
        label = f"rule {rule}, {fixed_name} = {value:.2f}"
        try:
            with Timer(label):
                est = estimate_threshold(L, rule, n=n, n_points=n_points,
                                         seed=int(ss.generate_state(1)[0]),
                                         n_jobs=args.jobs, **{fixed_name: value})
        except ValueError as exc:
            skipped.append(label)
            print(f"    skipped: {exc}")
            continue
        estimates.append(est)
        print(f"    critical {est.scanned} = {est.critical:.4f} ± {est.se:.4f}", flush=True)

    df = thresholds_table(estimates)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("phase_diagram_quick.csv" if args.quick else "phase_diagram.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} points)")
    if skipped:
        print("no spread possible for:", "; ".join(skipped))


if __name__ == "__main__":
    main()

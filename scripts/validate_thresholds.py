"""
Validation: does rule B reproduce the known percolation thresholds?

For every exactly known or tabulated point on the site-bond critical line,
measure the 50% crossing point of the cellular automaton and compare.
Rule B is also measured with the independent cluster-labelling method, so
the table doubles as a check that the two implementations agree.

Writes results/validation.csv and results/validation.md.

    python scripts/validate_thresholds.py            # ~7 min on 2 cores
    python scripts/validate_thresholds.py --quick    # smoke test
"""

from _common import RESULTS, Timer, parser

import numpy as np
import pandas as pd

from firesim import estimate_threshold, theory


def cases():
    yield ("site percolation (p = 1)", "Jacobsen 2015", "density", 1.0, theory.SITE_PC)
    yield ("bond percolation (D = 1)", "Kesten 1980 (exact)", "p_burn", 1.0, theory.BOND_PC)
    for pb, ps in sorted(theory.TVDM_SITE_GIVEN_BOND.items(), reverse=True):
        yield (f"site-bond, p = {pb:.2f}", "Tarasevich & van der Marck 1999", "density", pb, ps)
    for ps, pb in sorted(theory.TVDM_BOND_GIVEN_SITE.items(), reverse=True):
        yield (f"site-bond, D = {ps:.2f}", "Tarasevich & van der Marck 1999", "p_burn", ps, pb)


def main():
    args = parser(__doc__).parse_args()
    L, n, n_points = (32, 150, 7) if args.quick else (128, 600, 11)
    all_cases = list(cases())[: 4 if args.quick else None]
    seeds = np.random.SeedSequence(args.seed).spawn(len(all_cases))

    rows = []
    for (name, source, scanned, fixed, literature), ss in zip(all_cases, seeds):
        seed = int(ss.generate_state(1)[0])
        kw = {"p_burn": fixed} if scanned == "density" else {"density": fixed}
        with Timer(f"{name:28s}"):
            ca = estimate_threshold(L, "B", n=n, n_points=n_points, seed=seed,
                                    n_jobs=args.jobs, **kw)
            st = estimate_threshold(L, "B", n=n, n_points=n_points, seed=seed + 1,
                                    n_jobs=args.jobs, method="static", **kw)
        rows.append({
            "case": name, "source": source, "scanned": scanned, "fixed": fixed,
            "literature": literature, "L": L, "runs_per_point": n,
            "ca": ca.critical, "ca_se": ca.se,
            "static": st.critical, "static_se": st.se,
            "ca_minus_literature": ca.critical - literature,
            "ca_minus_static": ca.critical - st.critical,
            "z_ca_static": (ca.critical - st.critical) / np.hypot(ca.se, st.se),
        })
        r = rows[-1]
        print(f"    literature {literature:.4f}   CA {r['ca']:.4f} ± {r['ca_se']:.4f}"
              f"   static {r['static']:.4f} ± {r['static_se']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    df.to_csv(RESULTS / f"validation{suffix}.csv", index=False)

    sym = {"density": "D_c", "p_burn": "p_c"}
    lines = [
        f"| Critical point | Source | Literature | Fire model, rule B (L = {L}) "
        "| Difference | Cluster labelling (independent) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.case}: {sym[r.scanned]} | {r.source} | {r.literature:.4f} | "
            f"{r.ca:.4f} ± {r.ca_se:.4f} | {r.ca_minus_literature:+.4f} | "
            f"{r.static:.4f} ± {r.static_se:.4f} |")
    worst = df["ca_minus_literature"].abs().max()
    rms = float(np.sqrt(np.mean(df["ca_minus_literature"] ** 2)))
    lines += ["", f"Largest |difference|: {worst:.4f}; RMS difference: {rms:.4f} "
              f"over {len(df)} points. Uncertainties are 1 standard error "
              f"({n} fires at each of {n_points} values per point)."]
    (RESULTS / f"validation{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

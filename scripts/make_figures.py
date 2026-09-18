"""
Draw every figure from the files in results/ (light and dark versions).

    python scripts/make_figures.py
"""

from _common import FIGURES, RESULTS, parser

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from firesim import theory  # noqa: E402
from firesim.fitting import fit_logistic  # noqa: E402
from firesim.viz import save, seq_cmap, themed  # noqa: E402

RULE_NAMES = {"A": "rule A: single attempt", "B": "rule B: independent attempts"}


def critical_line(df, rule):
    d = df[df.rule == rule].copy()
    d["err_x"] = np.where(d.scanned == "p_burn", d.se, 0.0)
    d["err_y"] = np.where(d.scanned == "density", d.se, 0.0)
    return d.sort_values(["p_burn", "density"], ascending=[True, False])


# ------------------------------------------------------------------ figure 1
def phase_diagram(theme):
    df = pd.read_csv(RESULTS / "phase_diagram.csv")
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        p_grid = np.linspace(0.45, 1.0, 400)

        bound = theory.rule_a_upper_bound(p_grid)
        ax.plot(p_grid[bound < 1], bound[bound < 1], color=t["muted"], lw=1.2, ls=(0, (4, 3)),
                label=r"upper bound for rule A (proved): $p_c^{site}/p$")

        lit = theory.tvdm_points()
        ax.plot(lit[:, 0], lit[:, 1], "o", ms=7, mfc="none", mec=t["ink2"], mew=1.1,
                label="published thresholds (theory for rule B)", zorder=4)
        ax.plot([1.0, 0.5], [theory.SITE_PC, 1.0], "s", ms=7, mfc="none", mec=t["ink2"],
                mew=1.1, label="exact: site / bond percolation", zorder=4)

        ends = {}
        for rule in "BA":
            d = critical_line(df, rule)
            c = t["rule"][rule]
            ax.errorbar(d.p_burn, d.density, xerr=d.err_x, yerr=d.err_y, fmt="none",
                        ecolor=c, elinewidth=1, zorder=5)
            ax.plot(d.p_burn, d.density, "-o", color=c, ms=5.5, mec=t["surface"], mew=1.2,
                    label=f"rule {rule}, measured", zorder=6)
            ends[rule] = d

        # region labels
        ax.text(0.97, 0.985, "fire crosses", ha="right", va="top", color=t["ink2"], fontsize=10,
                transform=ax.transAxes)
        ax.text(0.03, 0.03, "fire dies out", ha="left", va="bottom", color=t["ink2"],
                fontsize=10, transform=ax.transAxes)

        # annotate the gap between the rules at p = 0.75
        a = ends["A"].set_index("p_burn").query("scanned == 'density'")
        b = ends["B"].set_index("p_burn").query("scanned == 'density'")
        if 0.75 in a.index and 0.75 in b.index:
            ya, yb = a.loc[0.75, "density"], b.loc[0.75, "density"]
            ax.annotate("", xy=(0.75, ya + 0.004), xytext=(0.75, yb - 0.004),
                        arrowprops=dict(arrowstyle="<->", color=t["ink"], lw=1,
                                        shrinkA=0, shrinkB=0))
            ax.annotate(f"at p = 0.75, rule A needs a tree\ndensity {ya - yb:.3f} higher "
                        f"({ya:.3f} vs {yb:.3f})",
                        xy=(0.745, (ya + yb) / 2), xytext=(0.515, 0.655), fontsize=8.5,
                        color=t["ink2"], va="center",
                        arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8,
                                        connectionstyle="arc3,rad=-0.15"))

        ax.set_xlim(0.45, 1.01)
        ax.set_ylim(0.55, 1.015)
        ax.set_xlabel("spread probability p")
        ax.set_ylabel("critical tree density $D_c$")
        ax.set_title("The critical line depends on how fire spreads, not just how likely it is")
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=8.5)
        L = int(df.L.iloc[0])
        fig.text(0.01, -0.01, f"Each point: 50% crossing point on a {L} x {L} lattice, from "
                 f"{int(df.runs_per_point.iloc[0])} fires at each of "
                 f"{int(df.n_points.iloc[0])} settings; error bars are 1 s.e. "
                 "(mostly smaller than the markers).", fontsize=8, color=t["muted"])
        return save(fig, FIGURES / "phase_diagram", theme)


# ------------------------------------------------------------------ figure 2
def validation(theme):
    df = pd.read_csv(RESULTS / "validation.csv")
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        y = np.arange(len(df))[::-1]
        ax.axvspan(-0.4, 0.4, color=t["grid"], lw=0, zorder=0,
                   label="typical uncertainty of the tabulated values (±0.0004)")
        ax.axvline(0, color=t["axis"], lw=1)
        ax.errorbar(1e3 * (df.static - df.literature), y - 0.15, xerr=1e3 * df.static_se,
                    fmt="o", ms=6, mfc="none", mec=t["ink2"], ecolor=t["ink2"], elinewidth=1,
                    label="cluster labelling (no fire, independent code)")
        ax.errorbar(1e3 * df.ca_minus_literature, y + 0.15, xerr=1e3 * df.ca_se, fmt="o",
                    ms=6, color=t["rule"]["B"], mec=t["surface"], mew=1, elinewidth=1,
                    label="fire simulation, rule B")
        labels = [(c.replace("site-bond, ", "site-bond at ")
                   + (": find $D_c$" if s == "density" else ": find $p_c$"))
                  if c.startswith("site-bond") else c for c, s in zip(df.case, df.scanned)]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        lim = max(1.5, 1e3 * np.max(np.abs(df.ca_minus_literature) + df.ca_se) * 1.3)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("measured minus published threshold  (× 10⁻³)")
        ax.set_title(f"Rule B reproduces {len(df)} known critical points")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=8.5)
        return save(fig, FIGURES / "validation", theme)


# ------------------------------------------------------------------ figure 3
def finite_size(theme):
    curves = pd.read_csv(RESULTS / "fss_curves.csv")
    fits = pd.read_csv(RESULTS / "fss_collapse.csv")
    with themed(theme) as t:
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.4), sharey=True)
        for row, fit in enumerate(fits.itertuples()):
            sub = curves[(curves.rule == fit.rule) & np.isclose(curves.p_burn, fit.p_burn)]
            sizes = sorted(sub.L.unique())
            ramp = t["ramp"][-len(sizes):]
            raw, col = axes[row]
            for L, colour in zip(sizes, ramp):
                s = sub[sub.L == L].sort_values("density")
                raw.plot(s.density, s.P, "-o", color=colour, ms=3.5, lw=1.4, label=f"L = {L}")
                x = (s.density - fit.Dc_nu43) * L ** (1 / theory.NU)
                col.plot(x, s.P, "o", color=colour, ms=4.2, mec=t["surface"], mew=0.6)
            raw.set_xlabel("tree density D")
            raw.set_ylabel("P(fire crosses)")
            if fit.rule == "B":
                name = "Control: site percolation (rule B, p = 1)"
                truth = "exact: ν = 4/3 ≈ 1.333"
            else:
                name = f"Rule A, p = {fit.p_burn:g} (no known answer)"
                truth = "same exponent as percolation?"
            raw.set_title(name)
            raw.legend(loc="lower right", fontsize=8)
            col.set_xlabel(r"scaled density $(D - D_c)\,L^{3/4}$")
            col.set_title("collapse with the percolation exponent ν = 4/3")
            col.text(0.03, 0.97,
                     f"{truth}\n"
                     f"free collapse:  ν = {fit.nu_free:.2f} ± {fit.nu_free_se:.2f}\n"
                     f"slope of k(L): ν = {fit.nu_slope:.2f} ± {fit.nu_slope_se:.2f}\n"
                     f"fit quality with ν = 4/3: S = {fit.S_nu43:.2f}",
                     transform=col.transAxes, va="top", fontsize=8.5, color=t["ink2"],
                     linespacing=1.45)
            col.set_xlim(-2.6, 2.6)
        fig.tight_layout(h_pad=2.2)
        return save(fig, FIGURES / "finite_size_scaling", theme)


# ------------------------------------------------------------------ figure 4
def burn_map(theme):
    s = pd.read_csv(RESULTS / "burn_sweep_summary.csv")
    crit = pd.read_csv(RESULTS / "phase_diagram.csv")
    with themed(theme) as t:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharey=True)
        fig.subplots_adjust(top=0.80, wspace=0.08)
        cmap = seq_cmap(theme)
        for ax, rule in zip(axes, "AB"):
            d = s[s.rule == rule].pivot(index="p_burn", columns="density", values="mean")
            im = ax.pcolormesh(d.columns, d.index, d.values, cmap=cmap, vmin=0, vmax=1,
                               shading="nearest", rasterized=True)
            c = critical_line(crit, rule)
            ax.plot(c.density, c.p_burn, color=t["ink"], lw=1.6,
                    label="critical line from crossing runs")
            ax.set_title(RULE_NAMES[rule])
            ax.set_xlabel("tree density D")
            ax.grid(False)
            ax.set_xlim(d.columns.min() - 0.0125, d.columns.max() + 0.0125)
            ax.set_ylim(d.index.min() - 0.025, d.index.max() + 0.025)
        axes[0].set_ylabel("spread probability p")
        axes[1].legend(loc="lower left", fontsize=8.5)
        cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
        cb.set_label("mean fraction of trees burned")
        cb.outline.set_visible(False)
        runs = int(s.runs.iloc[0])
        fig.text(0.125, 0.975, "How much burns when one central tree is lit",
                 ha="left", va="top", fontsize=12, weight="semibold", color=t["ink"])
        fig.text(0.125, 0.925, f"Mean over {runs} fires per cell on a {int(s.L.iloc[0])} x "
                 f"{int(s.L.iloc[0])} lattice. The burned fraction switches on along the "
                 "critical line measured from crossing runs.",
                 ha="left", va="top", fontsize=9, color=t["ink2"])
        return save(fig, FIGURES / "burn_fraction_map", theme)


# ------------------------------------------------------------------ figure 5
def burn_fits():
    """Student's original analysis, redone: logistic fits of burn fraction vs density."""
    s = pd.read_csv(RESULTS / "burn_sweep_summary.csv")
    crit = pd.read_csv(RESULTS / "phase_diagram.csv")
    rows = []
    for (rule, p), g in s.groupby(["rule", "p_burn"]):
        g = g.sort_values("density")
        if g["mean"].max() < 0.5:            # no transition inside the sweep
            rows.append({"rule": rule, "p_burn": p, "transition": False})
            continue
        fit = fit_logistic(g.density, g["mean"], sigma=np.maximum(g["sem"], 0.005))
        lo, hi = fit.confidence_intervals()["Dc"]
        c = crit[(crit.rule == rule) & (crit.scanned == "density") & np.isclose(crit.p_burn, p)]
        rows.append({"rule": rule, "p_burn": p, "transition": True, "Dc_burn": fit.Dc,
                     "Dc_low": lo, "Dc_high": hi, "A": fit.A, "k": fit.k, "r2": fit.r2,
                     "Dc_crossing": c.density.iloc[0] if len(c) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "burn_fraction_fits.csv", index=False, float_format="%.5g")
    return out


def burn_curves(theme, fits):
    s = pd.read_csv(RESULTS / "burn_sweep_summary.csv")
    scenarios = [(0.40, "p = 0.40 (damp)"), (0.70, "p = 0.70 (average)"), (1.00, "p = 1.00 (dry)")]
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ramp = [t["ramp"][0], t["ramp"][2], t["ramp"][4]] if theme == "light" else \
               [t["ramp"][0], t["ramp"][2], t["ramp"][4]]
        for (p, label), colour in zip(scenarios, ramp):
            g = s[(s.rule == "A") & np.isclose(s.p_burn, p)].sort_values("density")
            ax.errorbar(g.density, g["mean"], yerr=g["sem"], fmt="o", ms=4.5, color=colour,
                        mec=t["surface"], mew=0.8, elinewidth=0.8)
            f = fits[(fits.rule == "A") & np.isclose(fits.p_burn, p)].iloc[0]
            if f.transition:
                x = np.linspace(g.density.min(), g.density.max(), 400)
                y = f.A / (1 + np.exp(-f.k * (x - f.Dc_burn)))
                ax.plot(x, y, color=colour, lw=1.8,
                        label=f"{label}: steepest at D = {f.Dc_burn:.3f} "
                              f"[{f.Dc_low:.3f}, {f.Dc_high:.3f}]")
            else:
                ax.plot(g.density, g["mean"], color=colour, lw=1.8,
                        label=f"{label}: no transition at any density")
        ax.set_xlabel("tree density D")
        ax.set_ylabel("mean fraction of trees burned")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title("The original analysis, redone with logistic fits (rule A)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=8.5)
        ax.text(0.0, -0.36, "Brackets: 95% confidence interval of the fitted midpoint. "
                "Error bars: 1 s.e. over fires.", fontsize=8, color=t["muted"],
                transform=ax.transAxes, va="top")
        return save(fig, FIGURES / "burn_fraction_curves", theme)


# ------------------------------------------------------------------ figure 6
LANDSCAPE_DEMO = [("iid", 0.0, None, "independent planting"),
                  ("gaussian", 8.0, None, "clumped, correlation 8 cells"),
                  ("gaussian", 8.0, "shuffle", "the same cells, shuffled")]


def landscape(theme):
    from firesim.landscape import LandscapeSource, crossing_curve_landscape  # noqa: F401
    from firesim.model import simulate_lattice
    from firesim.viz import state_cmap

    thresholds = pd.read_csv(RESULTS / "landscape_thresholds.csv", keep_default_na=False,
                             na_values=[""]).fillna({"control": ""})
    thresholds["control"] = thresholds["control"].astype(str).replace("nan", "")
    thresholds[["density_c", "se", "xi_cells"]] = thresholds[
        ["density_c", "se", "xi_cells"]].apply(pd.to_numeric)
    curves = pd.read_csv(RESULTS / "landscape_curves.csv")
    burn = pd.read_csv(RESULTS / "landscape_burn.csv")
    demo_density = 0.55

    with themed(theme) as t:
        fig = plt.figure(figsize=(10.4, 8.2))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0], hspace=0.55, wspace=0.28)
        cmap, norm = state_cmap(theme)

        # (a) three landscapes burned at the same tree density
        for col, (kind, corr, control, name) in enumerate(LANDSCAPE_DEMO):
            src = LandscapeSource(kind, (160, 160), correlation=corr, control=control)
            rng = np.random.default_rng(4)
            trees = src.build(demo_density, rng)
            res, frames = simulate_lattice(trees, 1.0, "B", rng)
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(frames[-1], cmap=cmap, norm=norm)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(name, fontsize=10, pad=18)
            ax.text(0, 1.02, ("fire crossed" if res["crossed"] else "fire died out")
                    + f", {res['burn_fraction']:.0%} of trees burned",
                    transform=ax.transAxes, fontsize=8.5, color=t["ink2"], va="bottom")

        # (b) crossing curves for the same three
        ax = fig.add_subplot(gs[1, :2])
        wanted = ["independent planting", "clumped (correlation 8 cells)",
                  "clumped (correlation 8 cells), shuffled"]
        for name, colour, label in zip(wanted, t["cat"],
                                       ["independent planting", "clumped (correlation 8)",
                                        "clumped, then shuffled"]):
            sub = curves[curves.landscape == name].sort_values("density")
            if sub.empty:
                continue
            ax.errorbar(sub.density, sub.P, yerr=sub.se, fmt="-o", color=colour, ms=4,
                        lw=1.6, mec=t["surface"], mew=0.8, elinewidth=0.8, label=label)
        ax.axvline(theory.SITE_PC, color=t["muted"], lw=1, ls=(0, (4, 3)))
        ax.text(theory.SITE_PC - 0.004, 0.5, "$p_c$ = 0.593", rotation=90, ha="right",
                va="center", fontsize=8.5, color=t["muted"])
        ax.set_xlabel("tree density D")
        ax.set_ylabel("P(fire crosses)")
        ax.set_xlim(0.42, 0.67)
        ax.set_title("Clumped fuel crosses at a lower density")
        ax.legend(loc="upper left", fontsize=8.5)

        # (c) threshold against measured correlation length
        ax = fig.add_subplot(gs[1, 2])
        g = thresholds[(thresholds.kind.isin(["iid", "gaussian"])) & (thresholds.control == "")]
        s = thresholds[(thresholds.kind.isin(["iid", "gaussian"])) & (thresholds.control == "shuffle")]
        ax.axhline(theory.SITE_PC, color=t["muted"], lw=1, ls=(0, (4, 3)))
        ax.errorbar(s.xi_cells, s.density_c, yerr=s.se, fmt="o", ms=5, mfc="none",
                    mec=t["ink2"], ecolor=t["ink2"], elinewidth=0.8, label="shuffled control")
        ax.errorbar(g.xi_cells, g.density_c, yerr=g.se, fmt="-o", color=t["cat"][0], ms=5,
                    lw=1.6, mec=t["surface"], mew=0.8, elinewidth=0.8, label="landscape")
        ax.set_xscale("log")
        ax.set_xlabel("correlation length of the fuel map (cells)")
        ax.set_ylabel("critical density $D_c$")
        ax.set_ylim(0.47, 0.62)
        ax.set_title("…and saturates near 1/2")
        ax.legend(loc="center right", fontsize=8)

        # (d) how much burns once a fire starts
        ax = fig.add_subplot(gs[2, :])
        for name, colour, label in zip(wanted, t["cat"],
                                       ["independent planting", "clumped (correlation 8)",
                                        "clumped, then shuffled"]):
            sub = burn[burn.landscape == name].sort_values("density")
            if sub.empty:
                continue
            ax.errorbar(sub.density, sub["mean"], yerr=sub["sem"], fmt="-o", color=colour,
                        ms=4, lw=1.6, mec=t["surface"], mew=0.8, elinewidth=0.8, label=label)
        ax.set_xlabel("tree density D")
        ax.set_ylabel("mean fraction of trees burned")
        ax.set_title("A fire lit at one tree: clumping helps it start spreading, "
                     "but leaves more of the forest unburned")
        ax.legend(loc="upper left", fontsize=8.5)

        fig.text(0.0, -0.02, f"Rule B, p = 1, {int(thresholds.rows.iloc[0])} x "
                 f"{int(thresholds.cols.iloc[0])} lattices; panel (a) at D = {demo_density}. "
                 "Shuffling keeps the tree density and destroys the spatial pattern. "
                 "Each crossing curve was measured around its own threshold.",
                 fontsize=8, color=t["muted"])
        return save(fig, FIGURES / "landscape_percolation", theme)


def main():
    parser(__doc__).parse_args()
    fits = burn_fits()
    for theme in ("light", "dark"):
        for make in (phase_diagram, validation, finite_size, burn_map,
                     lambda th: burn_curves(th, fits), landscape):
            try:
                print("wrote", make(theme))
            except FileNotFoundError as exc:
                print(f"skipped ({exc.filename} not found; run the experiment scripts first)")
            plt.close("all")


if __name__ == "__main__":
    main()

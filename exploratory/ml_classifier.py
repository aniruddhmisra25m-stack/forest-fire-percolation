"""
Exploratory (superseded): a random-forest "catastrophe" classifier.

This is where the project started. Fires from the burn-fraction sweep are
labelled "catastrophic" if more than 70% of the trees burned, and a random
forest learns to predict that label from (tree density, spread probability).

It works -- the test AUC is close to 1 -- but the figure shows why the
project moved on: the boundary the forest learns is just the critical line
D_c(p), which percolation theory predicts directly (overlaid in black) and
which the main analysis measures with error bars. The 70% cut-off is also
arbitrary. See exploratory/README.md.

Needs scikit-learn (pip install -e ".[ml]") and the sweep data:

    python scripts/burn_sweep.py
    python exploratory/ml_classifier.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firesim.viz import save, seq_cmap, themed  # noqa: E402

CATASTROPHE = 0.70   # burn fraction above which a fire is labelled catastrophic


def main():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        sys.exit("scikit-learn is needed: pip install -e '.[ml]'")
    import matplotlib.pyplot as plt

    data_path = ROOT / "data" / "burn_sweep.csv"
    if not data_path.exists():
        sys.exit("run `python scripts/burn_sweep.py` first to create data/burn_sweep.csv")
    df = pd.read_csv(data_path)
    df = df[df.rule == "A"]                       # the original model used rule A
    X = df[["density", "p_burn"]].to_numpy()
    y = (df.burn_fraction > CATASTROPHE).astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"test ROC AUC = {auc:.3f}  ({len(X_test)} held-out fires)")

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 300),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), 300))
    zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    critical = None
    pd_path = ROOT / "results" / "phase_diagram.csv"
    if pd_path.exists():
        crit = pd.read_csv(pd_path)
        critical = crit[crit.rule == "A"].sort_values("p_burn")

    for theme in ("light", "dark"):
        with themed(theme) as t:
            cmap = seq_cmap(theme)
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            im = ax.pcolormesh(xx, yy, zz, cmap=cmap, vmin=0, vmax=1, shading="auto",
                               rasterized=True)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("P(catastrophic), random forest")
            cb.outline.set_visible(False)
            if critical is not None:
                ax.plot(critical.density, critical.p_burn, color=t["ink"], lw=1.6,
                        label="critical line D_c(p), measured (rule A)")
                ax.legend(loc="lower right")
            ax.set_xlabel("tree density D")
            ax.set_ylabel("spread probability p")
            ax.set_title(f"What the classifier learned (test AUC = {auc:.3f})")
            ax.grid(False)
            save(fig, ROOT / "exploratory" / "ml_decision_map", theme)
            plt.close(fig)
    print("wrote exploratory/ml_decision_map.png")


if __name__ == "__main__":
    main()

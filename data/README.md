# data/

Raw simulation output lives here and is not tracked by git.

`python scripts/burn_sweep.py` writes `burn_sweep.csv`: one row per fire
(37,800 fires, about 2 MB) with the columns `rule, L, density, p_burn, run,
burn_fraction, burned, initial_trees, steps`. The averaged version used by the
figures is committed as `results/burn_sweep_summary.csv`.

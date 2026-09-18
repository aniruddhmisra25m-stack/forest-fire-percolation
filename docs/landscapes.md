# Running the model on a real forest

The model plants trees independently: every cell is fuel with probability D,
regardless of its neighbours. Real canopy cover is not like that — it is
patchy, and cover in one cell predicts cover in the next for hundreds of
metres. This note explains how to feed a real canopy-cover map into the same
fire engine, what the comparison can show, and what it cannot.

## 1. Where the maps come from

All of these are free and need no account. Pick one, download a tile, and cut
out a window.

| Dataset | Resolution | What a pixel holds | Notes |
|---|---|---|---|
| Hansen / GLAD **Global Forest Change** | 30 m | `treecover2000`: canopy cover in percent | 10° × 10° GeoTIFF tiles; also has annual loss, so you can age a landscape |
| **ESA WorldCover** | 10 m | land-cover class (10 = tree cover) | 3° × 3° tiles; classes, not a fraction — convert with `cover = (map == 10)` |
| **Copernicus Global Land Cover** | 100 m | tree-cover *fraction* layer | already a fraction, coarse but easy |
| **MODIS VCF** (MOD44B) | 250 m | percent tree cover, yearly | good for continental-scale questions |
| **LANDFIRE** (United States only) | 30 m | fuel models, canopy cover, canopy bulk density | built for fire modelling, so it carries fuel type as well as cover |

Take the download URL from the dataset's own page rather than from this file —
Hansen and WorldCover re-issue tiles with new version numbers every year, and
the tile naming follows the top-left corner of the tile (for example
`50N_120W`).

A 30 m dataset gives 0.09 ha per pixel. A 2048 × 2048 window is about 61 km
across, which is a realistic size for a large fire.

## 2. From raster to lattice

```python
from firesim.landscape import load_cover, coarsen, to_lattice
from firesim import run_lattice

cover = load_cover("Hansen_treecover2000_50N_120W.tif",
                   window=(12000, 8000, 2048, 2048))   # row, col, height, width
cover = coarsen(cover, 4)          # 30 m pixels -> 120 m cells, 512 x 512
trees = to_lattice(cover, method="threshold", level=0.3)   # >= 30 % cover is fuel
run_lattice(trees, p_burn=0.9, rule="B", rng=0)
```

`load_cover` reads GeoTIFF (through `rasterio`, installed with
`pip install -e ".[geo]"`), `.npy`, and ordinary images, rescales percent or
0–255 values to fractions, and reads only the window you ask for, so a 40000 ×
40000 tile costs nothing to sample.

Two choices decide what you are simulating, and both should be varied as a
sensitivity check rather than fixed silently:

* **Cell size** (`coarsen`). The model has one length scale: a cell. Making a
  cell 30 m or 120 m changes what "neighbouring" means, and any threshold you
  measure moves with it.
* **What counts as fuel** (`to_lattice`). `threshold` keeps cells above a cover
  level. `quantile` keeps the densest fraction of cells, which is the useful
  one for comparisons because it fixes the tree density exactly. `bernoulli`
  treats partial cover as a probability.

## 3. The controls are the point

A single number from a real landscape — "fire crossed at 46 % cover" — means
little on its own, because it mixes up *how much* fuel there is with *how it
is arranged*. The module provides the two controls that separate them:

```python
from firesim.landscape import shuffle, block_shuffle
random_version = shuffle(trees)             # same density, no pattern at all
local_version  = block_shuffle(trees, 32)   # pattern kept below 32 cells only
```

Running the same fire on all three answers a question the raw number cannot:
how much of the landscape's behaviour comes from its structure?

`scripts/landscape_percolation.py` does this automatically for synthetic
landscapes with a tunable amount of clumping, and for a real raster:

```bash
python scripts/landscape_percolation.py --raster cover.tif \
       --window 12000 8000 2048 2048 --coarsen 4 --jobs 4
```

## 4. What the answer looks like

On synthetic landscapes, clumping moves the tipping point a long way. With
independent planting the fire needs 59.3 % of cells to be fuel before it
crosses; once the fuel is clumped over a few cells the threshold falls to
about 50 %, and the *shuffled* version of each landscape goes straight back to
59.3 %. Above the threshold the picture reverses: a fire in a clumped
landscape follows its patch and leaves more of the forest unburned than a
fire in a random landscape of the same density.

So spatial structure does not just perturb the percolation answer, it
replaces it: the critical density of a correlated landscape is a different
number from *p*<sub>c</sub> = 0.5927, and near 1/2 it is governed by the
symmetry between fuel and gaps rather than by lattice percolation. Long-range
correlations can also change the critical exponents themselves (the
Weinrib–Halperin criterion), so a finite-size-scaling analysis on real
landscapes should not assume ν = 4/3 — it should measure it.

## 5. What this is not

This is a connectivity experiment, not a fire forecast. The model has no wind,
no slope, no fuel moisture, no fuel types, no spotting and no suppression, and
its spread probability *p* is a free parameter with no measured value. A
landscape that percolates in this model is one where fuel is connected — a
necessary condition for a large fire, not a sufficient one.

Models built for prediction do carry that physics: FARSITE and FlamMap
(Rothermel spread equations), ELMFIRE, Cell2Fire, Prometheus, SPARK. If the
question is "where will this fire go", those are the tools. If the question is
"how does fuel arrangement change where the tipping point sits", this model
answers it with a control attached — and that is a question the operational
models do not isolate.

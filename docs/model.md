# The model, and why it can be checked exactly

This note states the model precisely, proves the results the code is tested
against, and explains how thresholds are measured. Notation: an L × L square
lattice, tree density D, spread probability p, von Neumann neighbourhood
(north, south, east, west), closed boundaries.

## 1. Definition

Every cell is `EMPTY`, `TREE`, `BURNING` or `ASH`. A run:

1. **Plant.** Each cell independently becomes a tree with probability D.
2. **Ignite.** Either every tree in the left column is set burning (*edge
   ignition*, used to ask "does the fire cross?"), or one tree at the centre
   is (*point ignition*, used to ask "how much burns?"). For point ignition
   the centre cell is planted first, so every run has a fire.
3. **Update synchronously** until nothing is burning:
   - `BURNING → ASH` (a cell burns for exactly one step);
   - a `TREE` with k ≥ 1 burning neighbours ignites with probability
     - **rule A** (single attempt): p, whatever k is;
     - **rule B** (independent attempts): 1 − (1 − p)^k, i.e. each burning
       neighbour gets its own attempt.

Rule A is the rule described in the project's original write-up ("a tree
adjacent to at least one burning neighbour undergoes a single attempt to
ignite"). Rule B is what you get if every burning tree independently throws
sparks at each neighbour.

The observable for edge ignition is whether any cell of the right column
burns. For a large lattice, the probability of that jumps from 0 to 1 across
a *critical line* D = D_c(p) in the (p, D) plane.

## 2. Rule B is exactly site-bond percolation

**Site-bond percolation**: every site is occupied with probability D, every
edge is open with probability p, all independently. Two occupied sites are
connected if a path of open edges through occupied sites joins them.

**Claim.** With edge ignition, the set of cells that burn under rule B has the
same distribution as the set of occupied sites connected to the left column
in site-bond percolation with parameters (D, p). In particular the fire
crosses exactly when the site-bond configuration has a left-right crossing,
so D_c^B(p) is the site-bond critical line.

**Proof.** Under rule B the random attempt made across an edge {x, y} happens
in the step after x starts burning, and only if y is then still a tree.

* *Each edge is tried at most once.* After that step x is ash, so x never
  tries again. If the attempt failed and y later ignites through another
  neighbour, x is already ash, so y makes no attempt back. If x and y burn in
  the same step, neither is a tree and nothing is tried.

So we may draw an independent coin ω(e) ~ Bernoulli(p) for every edge in
advance and let the dynamics read ω(e) the one time it tries e; this
produces exactly the law of rule B. With the coins fixed, let C be the set of
trees joined to the left column by open edges through trees.

* *Everything that burns is in C*: each ignition happens across an open edge
  from a cell that burned earlier; induct on the time of ignition.
* *Everything in C burns*: take a path x₀, x₁, …, x_k of trees joined by open
  edges, x₀ in the left column (burning at time 0). If x_i burns at time t,
  then at time t the tree x_{i+1} has either already caught fire, or is still
  a tree with a burning neighbour, in which case the open edge {x_i, x_{i+1}}
  is tried and succeeds. By induction x_k burns. ∎

This is why the tests can compare rule B with tabulated site-bond thresholds,
and why `firesim.site_bond_crossing` (which only labels clusters, with no
fire at all) must agree with the simulation.

Special cases: p = 1 is **site percolation**, D_c = 0.592746…
(Jacobsen 2015); D = 1 is **bond percolation**, p_c = 1/2 exactly (Kesten
1980).

## 3. Rule A: two bounds

Rule A has no textbook answer, but it is trapped between two curves:

  **D_c^B(p) ≤ D_c^A(p) ≤ p_c^site / p.**

**Lower bound (rule A never spreads further than rule B).** Draw the edge
coins ω(e) as above. Run rule A as follows: when a tree y is exposed (has at
least one burning neighbour), pick one of its burning neighbours x by a fixed
ordering and use ω({x, y}) as y's single attempt. That coin has never been
read before: x only started burning this step, and y has never burned. So the
coins are fresh and this is a faithful run of rule A. But now every ignition
under rule A crosses an open edge from a burned cell, so the burned set is
contained in C, the burned set of rule B under the same coins. Hence rule A
crosses less often, and its threshold is at least as high. ∎

**Upper bound (rule A still spreads once D·p > p_c^site).** Give each tree y
an independent sequence of coins c_y(1), c_y(2), … ~ Bernoulli(p) and let the
m-th exposure of y use c_y(m); again this is exactly rule A. Call a tree
*good* if c_y(1) = 1. Good trees are an ordinary site-percolation
configuration of density D·p. Every good tree joined to the left column
through good trees burns: when its predecessor on the path burns, either it
is already alight, or it is exposed for the first time — it cannot have been
exposed before, because its first attempt would have succeeded — and its
first coin succeeds. So rule A crosses at least as often as site
percolation at density D·p, which crosses with probability → 1 once
D·p > p_c^site. ∎

Consequences:

* at p = 1 both bounds equal p_c^site, and the two rules coincide exactly;
* at full density, rule A's critical spread probability lies in
  [1/2, 0.5927]. The simulation locates it inside that interval
  (see the README).

## 4. Measuring a threshold on a finite lattice

On an L × L lattice the crossing probability P_L(D) is a smooth S-curve that
steepens as L grows. Near the critical point it obeys the finite-size
scaling form

  P_L(D) ≈ Φ((D − D_c) · L^{1/ν}),   ν = 4/3 in two dimensions,

so the width of the transition shrinks like L^{−3/4}. For a square the
crossing probability at criticality tends to 1/2 (Cardy 1992), which makes
**the density where half of all fires cross** a natural finite-size
estimate of D_c. Its bias shrinks quickly with L; at L = 128 the measured
values agree with the literature to a few parts in 10⁴.

`firesim.estimate_threshold` measures P_L on a grid of densities (or of
spread probabilities, where the critical line is steep) chosen to span
roughly P = 5 % … 95 %, then fits

  P(x) = 1 / (1 + exp(−k (x − x₅₀)))

by maximum-likelihood binomial (logistic) regression. The standard error of
x₅₀ = −a/b, where a and b are the fitted intercept and slope, comes from the
delta method. The test suite checks that the resulting 95 % intervals cover
the truth about 95 % of the time on synthetic data.

The finite-size-scaling script goes one step further: it measures P_L for
L = 16 … 256 and searches for the (D_c, ν) that make all curves fall onto a
single master curve, using a pairwise version of the collapse measure of
Houdayer & Hartmann (2004), with parametric-bootstrap error bars. Site
percolation, where ν = 4/3 is known, is the control.

## 5. Implementation notes

* The lattice is stored with a one-cell empty border and flattened, so the
  neighbours of flat index i are i ± 1 and i ± (L + 2) with no bounds
  checks. Each step touches only the burning front and its neighbours:
  rule A draws one random number per exposed tree (`np.unique` over the
  front's tree neighbours), rule B one per (burning cell, tree) link.
  A 256 × 256 fire takes about 10 ms.
* All randomness comes from a `numpy.random.Generator` passed in by the
  caller. Parallel experiments give every task its own stream spawned from
  one `SeedSequence`, so results do not depend on the number of workers.
* When p = 1 no random numbers are drawn during spreading, so the two rules
  produce *identical* runs from the same seed, as the theory says they
  should.

## 6. Limitations

This is a deliberately minimal model. It has no wind, slope, fuel moisture,
fuel types, spotting, regrowth or suppression; all trees are identical; the
lattice is square with nearest-neighbour spread; burning lasts exactly one
step. Real fire scars live on irregular landscapes with correlated fuel. The
model's value is that it is *exactly* connected to percolation theory, which
makes it a calibrated reference point, not a forecast.

## References

* J. L. Cardy, *Critical percolation in finite geometries*, J. Phys. A 25,
  L201 (1992).
* J. Houdayer and A. K. Hartmann, *Low-temperature behavior of
  two-dimensional Gaussian Ising spin glasses*, Phys. Rev. B 70, 014418
  (2004), arXiv:cond-mat/0402036.
* J. L. Jacobsen, *Critical points of Potts and O(N) models from eigenvalue
  identities in periodic Temperley–Lieb algebras*, J. Phys. A 48, 454003
  (2015), arXiv:1507.03027.
* H. Kesten, *The critical probability of bond percolation on the square
  lattice equals 1/2*, Commun. Math. Phys. 74, 41 (1980).
* D. Stauffer and A. Aharony, *Introduction to Percolation Theory*, 2nd ed.,
  Taylor & Francis (1994).
* Yu. Yu. Tarasevich and S. C. van der Marck, *An investigation of site-bond
  percolation on many lattices*, Int. J. Mod. Phys. C 10, 1193 (1999),
  doi:10.1142/S0129183199000978.
* M. Yanuka and R. Englman, J. Phys. A 23, L339 (1990).

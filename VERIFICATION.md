# Reproduction review

This report distinguishes recomputation from comparisons of archived results.
The numerical targets are the supplied manuscript's Figure 1, its two
synthetic plots (Figures 2 and 3), and Tables 2 and 4.

## Installation and scientific checks

- A fresh CPython 3.9 environment installed `requirements.txt` successfully;
  `pip check` found no dependency conflicts. The complete installed package
  set is recorded in `requirements-lock.txt`.
- All 19 tests passed in the pinned environment. They cover independent
  principal-angle formulas, rank completion, diagonal covariance-distance
  formulas, rejection of singular PCA inputs, absence of held-out leakage,
  participant weighting, the strict unscaled Naive rank threshold, archived
  simulation draws, bit-identical serial/parallel distances, and verification
  failures for missing or different draws.
- Input metadata are frozen for all 54 dataset/participant units.
  `reference/data_checksums.json` fingerprints the numerical inputs.
- `check_reproduction.py --quick` completed successfully, including all
  tests and independent comparison of two freshly generated simulation draws.

## Figure 1

Recomputed from the original prepared RICE SCMs. The pooled covariance agrees
exactly with the archived pooled SCM. All 35 histogram bars agree with the
published vector PDF: maximum coordinate/density error is
`5.044e-7`, below the `3e-6` PDF-coordinate tolerance.

The ratio is `256/396 = 0.6464646464646465`; the MP upper edge is
`3.254525150879386`. The three largest eigenvalues are
`3.247972111770852`, `4.951216069507620`, and `7.735779947345739`.
Exactly two exceed the edge; the histogram integrates to one.

## Synthetic experiment

All 160 draws were recomputed (eight dimensions, twenty repetitions each).
All 800 classification accuracies match exactly. Comparing the 480 distance
records (population target, estimated mean, MAE and MSE) gives a maximum
absolute error of `9.95e-14`. The 40 accuracy/interval summaries agree within
`1.12e-16`, and the 24 MAE/interval summaries within `5.34e-15`.
Both generated plots were visually inspected.

The scalar formulas for AI-CORR and LE-CORR were also checked against the
original implementations on all 256 pairs of the first draw; both matrices
were bit-identical. ORACLE uses known class-specific alignment coefficients
and is a simulation benchmark only.

Memory optimizations left all 24 test distance matrices bit-identical
(1,152 pairs, dimensions 32/96/256, ranks from zero to the full dimension,
single- and double-precision input bases). The full
simulation reference checks and all regression tests still pass.
Shared-memory parallel row calculations were also bit-identical for both
corrected and naive distances, including orthogonal complements, dimensions
96 and 300, zero/full ranks, and reference groups of 8/17/32/64 objects.
A four-thread CapgMyo subject/split replay reproduced all six scores exactly;
resuming it with one thread reused the verified checkpoint successfully.

## Tables 2 and 4

The 3,240 archived scores have unique dataset/participant/split/method keys,
and every row is valid. Independent reaggregation (equal participant weights,
then ten-split sample standard deviation) reproduces all 36 displayed
mean/standard-deviation pairs. The largest difference from the archived
unrounded summary is `2.23e-16`.

A fresh complete CapgMyo validation grid was evaluated for seed 42: 1,700
candidates for each of 18 participants, or 30,600 evaluations. All six
selected parameter settings, their mean validation balanced accuracies and
the subsequent 108 test scores agree exactly with the references.

Fresh selected-parameter recalculation completed 1,386 of 3,240 test scores:
all ten RICE splits (60 scores), CORN seed 42 (6), all ten CapgMyo splits
(1,080), and Hyser seeds 42 and 123 (240). Every completed score matches
exactly. The remaining 1,854 scores have not been fully recalculated; full
reproduction of both real-data tables is therefore not yet claimed.
The independent driver `python check_reproduction.py` runs the remaining
checks and reuses compatible completed checkpoints when available.

Replaying frozen selected parameters does not by itself verify the
hyperparameter search; the commands and verification modes keep these checks separate.
The other 59 complete dataset/split hyperparameter grids have not been rerun.

## Raw-data preparation

Reconstruction from the public raw files was bit-identical to the prepared
paper inputs for all 2,392 CORN SCMs, all 3,056 WHEAT SCMs, all 18 CapgMyo
participants (1,440 windows), all 13 FlexWear participants (1,281 windows),
and Hyser participant 1 (202 windows). Labels, sample counts, physical groups
and contraction order were checked as well. These checks ran in the fresh
pinned environment.

The selected RICE raw acquisitions and Hyser participants 2--20 raw `.dat`
files were unavailable locally. Their prepared SCMs/windows are available
and used for numerical reproduction. Consequently, complete raw-to-result
reconstruction is not claimed for these inputs. For RICE, the bounding boxes
and 48 grain masks on a separate available image were identical between the
original preprocessing environment and the pinned environment. The public
preparation command checks each selected grain's label and sample count.

## Manuscript discrepancies and conventions

1. The plotted simulation uses four test SCMs per class; the manuscript
   instead describes `ceil(0.4*sqrt(p))`. This repository reproduces the
   plotted protocol.
2. Four-class pooling gives `p/n_pool = c/(4*m_p)`, rather than `c/m_p`.
3. Figure 1 is a descriptive all-grain pooled example with SCM denominator
   `n`. Tables 2 and 4 use training-only pooling and denominator `n-1`.
   The two conventions are explicit in the code and README.

## Anonymous artifact

Only source, documentation, tests, frozen partitions and numerical references
are versioned. The source manuscript, original repository history, notebooks,
machine paths, cluster scripts, data, logs and generated plots are excluded.
Text files and all string arrays inside the 54 NPZ references were inspected
for author identities and private paths. No matches were found. The new Git
identity is `Anonymous <anonymous@example.invalid>`; no remote is configured.
Only the listed paper distances and experimental protocols are implemented.

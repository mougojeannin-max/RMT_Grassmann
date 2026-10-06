# Anonymous reproduction package

Code for Figure 1, the two synthetic experiment plots (Figures 2 and 3 in
the manuscript), and Tables 2 and 4. All commands below run from this directory.

## Installation

Use **Python 3.9** and an isolated environment:

```sh
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The numerical package versions are fixed. No notebook, cluster scheduler,
network service or external project checkout is required.

## Independent review command

After installing the requirements, run:

```sh
python check_reproduction.py --quick
python check_reproduction.py
```

The first command checks the installation, all tests and two simulation draws;
it needs no dataset. After preparing the data, the second checks all inputs,
reproduces Figure 1, both complete simulation plots and all 3,240 test scores,
then compares them with the independent references. It resumes compatible
completed checkpoints and can take hours. The reviewer driver uses only the
standard library and imports none of the experimental implementations.

Errors stop the driver with a nonzero exit code. Its report is
`results/reproduction_check.json` (`quick_check.json` for the quick mode).
Only `python check_reproduction.py --full-grid` also repeats and checks every
hyperparameter search; it is substantially more expensive. Selected-parameter
replay and a complete parameter-search check are explicitly distinguished.

## Simulations: the two plots

No dataset is needed. A small installation check is:

```sh
python simulation.py --dimensions 32 --repetitions 2 --output results/smoke
python verify.py --simulation results/smoke --allow-partial
```

The complete experiment is:

```sh
python simulation.py --workers 2
python verify.py --simulation results/simulation
```

This generates both PDF/PNG plots and per-repetition CSVs under
`results/simulation/`. The 160 draws use seed 9236101, dimensions
32, 64, ..., 256, and 20 repetitions. The five methods are CORR, NAIVE,
ORACLE, AI-CORR and LE-CORR. The MAE plot uses the three Grassmann methods.
Every method is recomputed from simulated Gaussian observations. Re-running
the same command resumes completed draws after checking the code and configuration.

The covariance model has four classes, AR(1) correlation 0.3, sine-vector
signals of strengths 36/45, and an angle of 3 degrees between paired raw
directions. Each SCM uses 10p observations. There are `ceil(0.6*sqrt(p))`
training objects and **four test objects per class**. Whitening uses training
objects only. The finite-sample exponents are alpha=0.6, beta=0.45,
gamma=1/3, and the classifier is 3-NN. ORACLE uses known ranks and population
alignment coefficients with empirical eigenvectors and empirical whitening.
Its alignment coefficients use the known generating classes; ORACLE is a
synthetic benchmark, not a deployable classifier.

**Manuscript discrepancies:** the supplied plots use four test objects per
class, while the text says `ceil(0.4*sqrt(p))`. This package follows the
plotted experiment. Also, pooling the four classes gives
`p/n_pool = 0.1/(4*ceil(0.6*sqrt(p)))`; the text omits the factor four.

## Real data

Download the six public releases and prepare them as explained in
[data/README.md](data/README.md). Dataset files remain outside version control.
The exact object order and ten partitions are supplied in `reference/splits/`.
After preparation, `python verify.py --data data` checks all 54 numerical
inputs and their metadata against the frozen checksums.

Recompute the Figure 1 spectrum from the prepared RICE SCMs:

```sh
python figure1.py
python verify.py --figure1 results/figure1
```

This descriptive figure uses grain HT18-01_000, 256 bands and 396 pixels,
with all 961 RICE grains in its pooled covariance. It uses SCM denominator
`n` and ratio `256/396`, as in the original figure. The classification
experiments below use centered SCMs with denominator `n-1` and training-only
pooling. The plotting code computes the spectrum from the SCMs.

Reproduce validation, parameter selection and testing for Tables 2 and 4:

```sh
python real_data.py --distance-threads 4
python verify.py --tables results/real_data --selection results/real_data
```

The full grid is computationally expensive. An explicit, faster check of
the published selected parameters is available:

```sh
python real_data.py --selected reference/selected.csv --distance-threads 4 --output results/selected_check
python verify.py --tables results/selected_check
```

The second command path **does not repeat validation or verify parameter
selection**. It recomputes preprocessing, all six distance matrices,
predictions and test scores. Choose a new output directory for each real-data
run, or repeat an interrupted command with `--resume` to reuse completed
dataset/split tasks. Resumption checks input, source and output hashes and
requires the same protocol and dependency versions; `--workers` and
`--distance-threads` may change.
To test one complete dataset/split, add `--datasets capgmyo --seeds 42`;
`verify.py --allow-partial` explicitly reports omitted results.

`real_data.py` writes per-subject predictions, scores, selected parameters,
training indices used for preprocessing, numerical summaries and
`table2.csv` / `table4.csv`. All ten seeds and all participants are required
for a complete table. Multiple workers process dataset/split tasks;
each worker uses one BLAS thread by default. `--distance-threads 4` shares
the same subspace arrays across four threads, with bit-identical distance
calculations. Keep the default single worker on an 8 GB machine. Large
hyperspectral runs can take hours on a laptop; use `--workers 2` if more
memory is available. The default distance thread count is one when this
option is omitted.

The retained methods are Corr raw/whitened, Naive raw/whitened, AI-PCA and
LE-PCA. Only Corr estimates and divides by a bulk variance. Naive applies
the threshold `lambda > 1 + dof**(-gamma)` directly to the raw or whitened
eigenvalues. It has only gamma and q as hyperparameters.

Whitening, channel selection and PCA use training objects only. PCA retains
95% of training pooled variance, capped by the smallest SCM degrees of
freedom. Validation and test objects use training objects as neighbours;
there is no refit after validation. Hyperparameters maximize validation
balanced accuracy separately for each dataset/split/method. EMG participants
receive equal weight. Ties use the lexicographically smallest
`(q,tau,alpha,beta,gamma)` after rounding validation scores to 12 decimals.
Reported uncertainty is the sample standard deviation across the ten splits,
after averaging participants within each split.

## Files and verification

| File | Purpose |
|---|---|
| `prepare_raw.py` | Read public data, segment grains and extract the fixed EMG windows |
| `figure1.py` | RICE spectrum and histogram |
| `simulation.py`, `simulation_metrics.py` | Frozen synthetic protocol and covariance-distance corrections |
| `real_data.py`, `distances.py` | Validation, classification and the retained real-data distances |
| `verify.py` | Strict comparison with independent archived references |
| `check_reproduction.py` | Independent driver for installation, recomputation and comparison checks |
| `tests/` | Formula, leakage, tie-breaking and numerical regression tests |
| `reference/` | Frozen splits, selected parameters and numerical comparison targets |

`reference/scores.csv` contains the 3,240 archived subject/split/method scores;
`tables.csv` contains the 36 mean/standard-deviation pairs. Simulation CSVs
contain 800 accuracy and 480 distance-error records. These are comparison
targets, not inputs to distance computation. `figure1_histogram.csv` was
independently extracted from the supplied vector figure; its tolerance
accounts for PDF coordinate rounding. `verify.py` fails on duplicate,
missing, unexpected, nonfinite or numerically different results. A partial
check never certifies a complete table.

See [VERIFICATION.md](VERIFICATION.md) for the checks actually executed and
their limitations. The repository starts with an anonymous history and no
remote. The shareable archive contains only tracked source, documentation,
tests and numerical references; datasets and generated results are excluded.

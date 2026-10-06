# Reproduce the experiments

Use Python 3.9. Run all commands from this directory.

Install the dependencies in a virtual environment:

```sh
python -m pip install -r requirements.txt
```

For Figure 1 and the tables, first prepare the datasets in `data/`.
Follow [data/README.md](data/README.md).

| Experiment | Command |
|---|---|
| Figure 1: RICE spectrum | `python figure1.py` |
| Figures 2 and 3: simulation plots | `python simulation.py` |
| Tables 2 and 4: real-data experiments | `python real_data.py --distance-threads 4 --output results/tables` |

The simulations need no dataset. The table command includes hyperparameter
selection. Add `--resume` to continue an interrupted table run.
All outputs are saved under `results/`.

To reproduce and check all results with the published parameter settings, run:

```sh
python check_reproduction.py
```

This check does not repeat hyperparameter selection. Add `--full-grid` to
include it, or use `--quick` for a short check without datasets.

See [VERIFICATION.md](VERIFICATION.md) for completed checks and known limitations.

# Reproduce the experiments

Use Python 3.9. Run all commands from this directory.

Install the dependencies in a virtual environment:

```sh
python -m pip install -r requirements.txt
```

For Figure 1 and the tables, download the datasets into `data/raw/<dataset>`.
Keep the release folder structure.

| Dataset | Download | Required files |
|---|---|---|
| `rice` | [Zenodo](https://zenodo.org/records/3241923) | Extracted release with `index.csv` and acquisition folders |
| `corn` | [Mendeley](https://data.mendeley.com/datasets/4n4xbnx8sr/1) | `4n4xbnx8sr-1.zip`, kept zipped |
| `wheat` | [Mendeley](https://data.mendeley.com/datasets/j7jm7rbwxh/1) | Extracted `fanmai8`, `jinan17`, `xingmai13`, `yangmai6` folders |
| `capgmyo` | [Figshare](https://figshare.com/articles/dataset/7210397) | `dba-s1.zip` through `dba-s18.zip`, kept zipped |
| `hyser` | [PhysioNet](https://physionet.org/content/hd-semg/2.0.0/) | PR session 1 for subjects 01–20: `label_dynamic.txt` and `dynamic_preprocess_sample*.hea` / `.dat` |
| `flex` | [FlexWear-HD](https://huggingface.co/datasets/jehanyang/FlexWear-HD/tree/90d253c1fd31b8756e8a05c473cc550b7f8ba2c0/FlexWear-HD_Dataset) | `p001` through `p013`, each with `data_allchannels_initial.h5` |

Prepare each dataset with `prepare_raw.py`. For example:

```sh
python prepare_raw.py rice --source data/raw/rice
```

Replace `rice` with each dataset name above. Prepared files go into `data/`.

| Experiment | Command |
|---|---|
| Figure 1: RICE spectrum | `python figure1.py` |
| Figures 2 and 3: simulation plots | `python simulation.py` |
| Tables 2 and 4: real-data experiments | `python real_data.py --distance-threads 4 --output results/tables` |

The simulations need no dataset. The table command includes hyperparameter
selection. Add `--resume` to continue an interrupted table run.
All outputs are saved under `results/`.
Frozen partitions and expected results are bundled in `reference.zip`.
The scripts read this archive directly.

To reproduce and check all results with the published parameter settings, run:

```sh
python check_reproduction.py
```

This check does not repeat hyperparameter selection. Add `--full-grid` to
include it, or use `--quick` for a short check without datasets.

# Data directory

Place downloads under `data/raw/`, then run `prepare_raw.py` from the repository
root. Data files are ignored by Git. Keep the folder structure inside each
public release; no manual NPZ construction is needed.

| Dataset | Public release | Expected `--source` contents |
|---|---|---|
| RICE | [Zenodo v1](https://zenodo.org/records/3241923) | `index.csv` and acquisition directories containing `.hdr` / `.raw` files |
| CORN | [Mendeley v1](https://data.mendeley.com/datasets/4n4xbnx8sr/1) | `4n4xbnx8sr-1.zip`, kept zipped |
| WHEAT | [Mendeley v1](https://data.mendeley.com/datasets/j7jm7rbwxh/1) | `fanmai8`, `jinan17`, `xingmai13`, `yangmai6`, with their two-view `.mat` subdirectories |
| CapgMyo DB-a | [Figshare release](https://figshare.com/articles/dataset/7210397) | `dba-s1.zip` through `dba-s18.zip`, kept zipped |
| Hyser PR | [PhysioNet 2.0.0](https://physionet.org/content/hd-semg/2.0.0/) | `subject01_session1` through `subject20_session1` from `pr_dataset` |
| FlexWear-HD | [Fixed public revision](https://huggingface.co/datasets/jehanyang/FlexWear-HD/tree/90d253c1fd31b8756e8a05c473cc550b7f8ba2c0/FlexWear-HD_Dataset) | `p001` through `p013`, each containing `data_allchannels_initial.h5` |

For Hyser, obtain `label_dynamic.txt` and all
`dynamic_preprocess_sample*.hea` / `.dat` files in each participant's first
session. The other tasks and sessions are not used. FlexWear uses the initial
session only. The RICE ZIP is about 17.3 GB; extract the release before preparation.
Consult the linked releases for the datasets' original terms and citations.

Example commands (replace source folders if necessary):

```sh
python prepare_raw.py rice --source "data/raw/rice"
python prepare_raw.py corn --source "data/raw/corn"
python prepare_raw.py wheat --source "data/raw/wheat"
python prepare_raw.py capgmyo --source "data/raw/capgmyo"
python prepare_raw.py hyser --source "data/raw/hyser"
python prepare_raw.py flex --source "data/raw/flex"
```

Each prepared hyperspectral dataset has two files:

```text
data/rice/subject00.npz
data/rice/subject00_covariances.npy
data/corn/subject00.npz
data/corn/subject00_covariances.npy
data/wheat/subject00.npz
data/wheat/subject00_covariances.npy
```

EMG uses `data/capgmyo/subject01.npz` through `subject18.npz`,
`data/hyser/subject01.npz` through `subject20.npz`, and
`data/flex/subject01.npz` through `subject13.npz`. Participant IDs are local
to each dataset. An EMG file stores 64-observation windows. The classifier
centers each window and computes its covariance using 63 degrees of freedom.

Preparation checks object labels, sample counts, grouping and contraction
order against the frozen split metadata. It rejects a different selection.
The released subsets have 961 RICE SCMs, 2,392 CORN SCMs and 3,056 WHEAT SCMs.
WHEAT keeps 1,528 complete two-view pairs, with both views in the same partition.
RICE uses the ten varieties listed in `prepare_raw.py`; CORN uses all twelve.

For already prepared data, the safe NumPy format (`allow_pickle=False`) is:

| Field | Meaning |
|---|---|
| `labels` | Integer class labels, one per object |
| `dof` | Positive centered-SCM degrees of freedom, `n-1` |
| `source_denominator` | Scalar string `n` or `n-1` |
| `seeds`, `partitions` | The supplied 10 seeds and a `(10,N)` array: 0=train, 1=validation, 2=test |
| `groups` | Physical grain identifiers when grouping is needed |
| `covariances` | Optional `(N,p,p)` array; otherwise load the sibling `_covariances.npy` |
| `signals` | EMG alternative to covariance storage, shape `(N,64,p)`, denominator `n-1` |

Use exactly the supplied metadata in `reference/splits/` with the corresponding
object order. The loader does not generate new partitions. Numerical data
checksums are available in `reference/data_checksums.json`; metadata are
checked against `reference/splits/`.
Checksum verification expects the canonical arrays produced by `prepare_raw.py`:
hyperspectral SCMs and EMG windows. The classifier also accepts the alternative
covariance storage described above.

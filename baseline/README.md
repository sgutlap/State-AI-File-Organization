# Simple file organizer baseline

This baseline recreates the six classifiers compared in the paper: XGBoost,
LightGBM, Random Forest, KNN, Decision Tree, and Logistic Regression. It trains
one selected model and sorts the top-level files in a folder into `Wanted/` and
`Unwanted/`. LightGBM is the default because it performed best in the study.

## Setup

```powershell
python -m pip install -r baseline/requirements.txt
```

The labeled training CSV must contain these columns:

```text
filename,extension,size_bytes,created_time,modified_time,accessed_time,label
```

Times may be Unix timestamps or date strings. Labels must be `wanted` or
`unwanted`. Example:

```csv
filename,extension,size_bytes,created_time,modified_time,accessed_time,label
assignment.pdf,.pdf,245000,2026-07-01,2026-07-20,2026-07-21,wanted
cache.tmp,.tmp,800,2026-07-20,2026-07-20,2026-07-20,unwanted
```

## Run

Preview the organization without changing files:

```powershell
python baseline/organize.py --data labels.csv path/to/folder
```

Apply the planned moves:

```powershell
python baseline/organize.py --data labels.csv path/to/folder --apply
```

Choose another paper model with `--model xgboost`, `lightgbm`,
`random_forest`, `knn`, `decision_tree`, or `logistic_regression`.

Only files directly inside the selected folder are considered. Hidden files,
the training CSV, nested folders, and existing destination folders are skipped.
Existing destination files are never overwritten; a numeric suffix is added.
The paper does not publish its training dataset or exact tuned parameters, so
this is a small structural reproduction rather than an exact metric recreation.

## Clustering without labels

`clustering.py` does not use classification or a training CSV. It groups files
from their extension, size, filename length, and modified/accessed times.

Preview K-Means clustering:

```powershell
python baseline/clustering.py path/to/folder --algorithm kmeans --clusters 3
```

Move the files into `Cluster_1`, `Cluster_2`, and so on:

```powershell
python baseline/clustering.py path/to/folder --algorithm kmeans --clusters 3 --apply
```

Available algorithms are `kmeans`, `agglomerative`, and `dbscan`. DBSCAN places
outliers in `Noise/` and uses `--eps` and `--min-samples` instead of `--clusters`.

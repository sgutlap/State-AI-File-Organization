"""Train one paper classifier and organize files as Wanted or Unwanted."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


LABELS = {"wanted": 0, "unwanted": 1}
FOLDERS = {0: "Wanted", 1: "Unwanted"}
REQUIRED_COLUMNS = [
    "filename",
    "extension",
    "size_bytes",
    "created_time",
    "modified_time",
    "accessed_time",
    "label",
]
NUMERIC_FEATURES = [
    "size_bytes",
    "created_time",
    "modified_time",
    "accessed_time",
    "name_length",
]
FEATURES = ["extension", *NUMERIC_FEATURES]


def get_models(random_state=42):
    """Return the six classifiers used in the study."""
    return {
        "xgboost": XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=1),
        "lightgbm": LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=1),
        "random_forest": RandomForestClassifier(random_state=random_state, n_jobs=1),
        "knn": KNeighborsClassifier(n_neighbors=3),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "logistic_regression": LogisticRegression(random_state=random_state, max_iter=1000),
    }


MODEL_NAMES = tuple(get_models())


def prepare_training_data(data):
    """Validate the CSV data and return model features and binary labels."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(missing)}")

    data = data[REQUIRED_COLUMNS].drop_duplicates().copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    if not set(data["label"]).issubset(LABELS):
        raise ValueError("labels must be wanted or unwanted")

    data["filename"] = data["filename"].astype(str)
    data["extension"] = data["extension"].fillna("").astype(str).str.lower()
    for column in ["size_bytes", "created_time", "modified_time", "accessed_time"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    data = data[data["size_bytes"] >= 0]
    if data.empty:
        raise ValueError("no valid training rows")

    data["name_length"] = data["filename"].str.len()
    targets = data["label"].map(LABELS).astype(int)
    counts = targets.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("training data needs at least two wanted and two unwanted rows")
    return data[FEATURES], targets


def train_model(data, model_name="lightgbm", random_state=42):
    """Train one of the six classifiers on a metadata DataFrame."""
    models = get_models(random_state)
    if model_name not in models:
        raise ValueError(f"unknown model: {model_name}")

    features, targets = prepare_training_data(data)
    minority_count = int(targets.value_counts().min())
    preprocess = ColumnTransformer(
        [
            (
                "extension",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["extension"],
            ),
            ("numbers", MinMaxScaler(), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "smote",
                SMOTE(random_state=random_state, k_neighbors=min(5, minority_count - 1)),
            ),
            ("select", SelectKBest(f_classif, k=5)),
            ("model", models[model_name]),
        ]
    )
    model.fit(features, targets)
    return model


def train_from_csv(csv_path, model_name="lightgbm", random_state=42):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"training CSV not found: {csv_path}")
    return train_model(pd.read_csv(csv_path), model_name, random_state)


def file_features(path):
    """Extract the same metadata fields from one real file."""
    path = Path(path)
    info = path.stat()
    return pd.DataFrame(
        [
            {
                "extension": path.suffix.lower(),
                "size_bytes": info.st_size,
                "created_time": info.st_ctime,
                "modified_time": info.st_mtime,
                "accessed_time": info.st_atime,
                "name_length": len(path.name),
            }
        ]
    )


def unique_path(path):
    """Return a destination that does not overwrite an existing file."""
    path = Path(path)
    candidate = path
    number = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        number += 1
    return candidate


def organize_folder(folder_path, model, apply=False, skip_paths=()):
    """Preview or organize the top-level files in a folder."""
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise ValueError(f"folder not found: {folder}")

    skipped = {Path(path).resolve() for path in skip_paths}
    moves = []
    for source in sorted(folder.iterdir()):
        if not source.is_file() or source.name.startswith(".") or source.resolve() in skipped:
            continue
        prediction = int(model.predict(file_features(source))[0])
        confidence = float(max(model.predict_proba(file_features(source))[0]))
        destination = unique_path(folder / FOLDERS[prediction] / source.name)
        moves.append((source, destination, FOLDERS[prediction], confidence))

    if apply:
        for source, destination, _, _ in moves:
            destination.parent.mkdir(exist_ok=True)
            shutil.move(str(source), str(destination))
    return moves


def main():
    parser = argparse.ArgumentParser(description="Organize files as Wanted or Unwanted.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--data", required=True, type=Path, help="labeled metadata CSV")
    parser.add_argument("--model", choices=MODEL_NAMES, default="lightgbm")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    try:
        model = train_from_csv(args.data, args.model)
        moves = organize_folder(args.folder, model, args.apply, [args.data])
    except (OSError, ValueError) as error:
        parser.error(str(error))

    action = "Moved" if args.apply else "Would move"
    for source, destination, _, confidence in moves:
        print(f"{action}: {source.name} -> {destination.parent.name}/{destination.name} ({confidence:.1%})")
    if not args.apply:
        print("Dry run only. Add --apply to move files.")


if __name__ == "__main__":
    main()

"""Organize files with traditional, unsupervised clustering algorithms."""

from __future__ import annotations

import argparse
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


ALGORITHMS = ("kmeans", "agglomerative", "dbscan")


@dataclass(frozen=True)
class ClusterMove:
    source: Path
    destination: Path
    cluster: int


def _files_in(folder_path):
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise ValueError(f"folder not found: {folder}")
    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and not path.name.startswith(".")
    ]
    if not files:
        raise ValueError("folder has no files to cluster")
    return folder, files


def extract_features(folder_path):
    """Return top-level files and scaled metadata feature vectors."""
    _, files = _files_in(folder_path)
    extensions = sorted({path.suffix.lower() or "<none>" for path in files})
    now = time.time()
    rows = []
    for path in files:
        info = path.stat()
        extension = path.suffix.lower() or "<none>"
        numeric = [
            math.log1p(info.st_size),
            math.log1p(max(0.0, (now - info.st_mtime) / 86400)),
            math.log1p(max(0.0, (now - info.st_atime) / 86400)),
            len(path.name),
        ]
        one_hot_extension = [1.0 if extension == item else 0.0 for item in extensions]
        rows.append(numeric + one_hot_extension)
    return files, StandardScaler().fit_transform(np.asarray(rows, dtype=float))


def kmeans_clusters(folder_path, n_clusters=3, random_state=42):
    """Cluster files with K-Means and return {file_path: cluster_id}."""
    files, features = extract_features(folder_path)
    if not 1 <= n_clusters <= len(files):
        raise ValueError("n_clusters must be between 1 and the number of files")
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(features)
    return dict(zip(files, map(int, labels)))


def agglomerative_clusters(folder_path, n_clusters=3):
    """Cluster files with agglomerative hierarchical clustering."""
    files, features = extract_features(folder_path)
    if not 1 <= n_clusters <= len(files):
        raise ValueError("n_clusters must be between 1 and the number of files")
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
    return dict(zip(files, map(int, labels)))


def dbscan_clusters(folder_path, eps=1.5, min_samples=2):
    """Cluster files with DBSCAN; cluster -1 represents noise."""
    files, features = extract_features(folder_path)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
    return dict(zip(files, map(int, labels)))


def _unique_path(path):
    candidate = path
    number = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        number += 1
    return candidate


def organize_clusters(folder_path, assignments, apply=False):
    """Preview or move assigned files into Cluster_N folders."""
    folder = Path(folder_path).resolve()
    moves = []
    for source, cluster in assignments.items():
        cluster_name = "Noise" if cluster == -1 else f"Cluster_{cluster + 1}"
        destination = _unique_path(folder / cluster_name / source.name)
        moves.append(ClusterMove(source, destination, cluster))

    if apply:
        for move in moves:
            move.destination.parent.mkdir(exist_ok=True)
            shutil.move(str(move.source), str(move.destination))
    return moves


def organize_with_kmeans(folder_path, n_clusters=3, apply=False):
    """Organize a folder using K-Means clusters."""
    return organize_clusters(
        folder_path,
        kmeans_clusters(folder_path, n_clusters),
        apply,
    )


def organize_with_agglomerative(folder_path, n_clusters=3, apply=False):
    """Organize a folder using agglomerative clusters."""
    return organize_clusters(
        folder_path,
        agglomerative_clusters(folder_path, n_clusters),
        apply,
    )


def organize_with_dbscan(folder_path, eps=1.5, min_samples=2, apply=False):
    """Organize a folder using DBSCAN clusters."""
    return organize_clusters(
        folder_path,
        dbscan_clusters(folder_path, eps, min_samples),
        apply,
    )


def cluster_folder(
    folder_path,
    algorithm="kmeans",
    n_clusters=3,
    eps=1.5,
    min_samples=2,
    apply=False,
):
    """Cluster a folder with the chosen algorithm and optionally move its files."""
    if algorithm == "kmeans":
        return organize_with_kmeans(folder_path, n_clusters, apply)
    if algorithm == "agglomerative":
        return organize_with_agglomerative(folder_path, n_clusters, apply)
    if algorithm == "dbscan":
        return organize_with_dbscan(folder_path, eps, min_samples, apply)
    raise ValueError(f"unknown algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser(description="Organize files into metadata clusters.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="kmeans")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1.5)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    try:
        moves = cluster_folder(
            args.folder,
            algorithm=args.algorithm,
            n_clusters=args.clusters,
            eps=args.eps,
            min_samples=args.min_samples,
            apply=args.apply,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    action = "Moved" if args.apply else "Would move"
    for move in moves:
        print(f"{action}: {move.source.name} -> {move.destination.parent.name}/{move.destination.name}")
    if not args.apply:
        print("Dry run only. Add --apply to move files.")


if __name__ == "__main__":
    main()

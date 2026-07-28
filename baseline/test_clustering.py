from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clustering import (
    agglomerative_clusters,
    cluster_folder,
    dbscan_clusters,
    kmeans_clusters,
)


def make_files(folder):
    files = []
    for name, size in [
        ("small_1.txt", 10),
        ("small_2.txt", 12),
        ("large_1.bin", 10_000),
        ("large_2.bin", 12_000),
    ]:
        path = folder / name
        path.write_bytes(b"x" * size)
        files.append(path)
    return files


class ClusteringTests(unittest.TestCase):
    def test_all_algorithms_assign_every_file(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            files = make_files(folder)
            results = [
                kmeans_clusters(folder, n_clusters=2),
                agglomerative_clusters(folder, n_clusters=2),
                dbscan_clusters(folder, eps=2.0, min_samples=2),
            ]
            for assignments in results:
                self.assertEqual(set(assignments), set(files))

    def test_dry_run_does_not_move_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            files = make_files(folder)
            moves = cluster_folder(folder, algorithm="kmeans", n_clusters=2)
            self.assertEqual(len(moves), len(files))
            self.assertTrue(all(path.exists() for path in files))
            self.assertFalse(any(path.is_dir() for path in folder.iterdir()))

    def test_apply_moves_every_file(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            files = make_files(folder)
            moves = cluster_folder(folder, algorithm="agglomerative", n_clusters=2, apply=True)
            self.assertEqual(len(moves), len(files))
            self.assertTrue(all(move.destination.exists() for move in moves))
            self.assertTrue(all(not path.exists() for path in files))


if __name__ == "__main__":
    unittest.main()

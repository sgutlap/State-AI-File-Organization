"""Benchmark manifests must prevent source and content leakage."""

import unittest
import tempfile
from pathlib import Path

from core.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit, TaxonomyRankingTask
from core.taxonomy.spec import FolderSpec, TaxonomySpec


def _task(task_id: str, split: DatasetSplit, source: str = "source-a", content_hash: str = "hash-a"):
    taxonomy = TaxonomySpec((FolderSpec("papers", "Papers", "Research papers"),))
    return TaxonomyRankingTask(
        task_id=task_id,
        workspace_id=f"workspace-{task_id}",
        source_group_id=source,
        taxonomy_id="research-v1",
        split=split,
        file_id=f"file-{task_id}",
        content_hash=content_hash,
        file_state={"relative_path": f"{task_id}.pdf", "extension": ".pdf"},
        taxonomy=taxonomy,
        candidate_folder_ids=("papers",),
        acceptable_folder_ids=("papers",),
    )


class TaxonomyDatasetTests(unittest.TestCase):
    def test_source_group_cannot_cross_splits(self):
        manifest = BenchmarkManifest("test", [_task("one", DatasetSplit.TRAIN), _task("two", DatasetSplit.TEST)])
        with self.assertRaisesRegex(ValueError, "source group crosses splits"):
            manifest.validate()

    def test_content_hash_cannot_cross_splits(self):
        manifest = BenchmarkManifest(
            "test",
            [_task("one", DatasetSplit.TRAIN, source="a", content_hash="same"), _task("two", DatasetSplit.TEST, source="b", content_hash="same")],
        )
        with self.assertRaisesRegex(ValueError, "content hash crosses splits"):
            manifest.validate()

    def test_test_taxonomy_cannot_overlap_training_taxonomy(self):
        train = _task("one", DatasetSplit.TRAIN, source="a", content_hash="one")
        test = _task("two", DatasetSplit.TEST, source="b", content_hash="two")
        with self.assertRaisesRegex(ValueError, "test taxonomy overlaps"):
            BenchmarkManifest("test", [train, test]).validate_heldout_taxonomies()

    def test_valid_manifest_reports_split_counts(self):
        manifest = BenchmarkManifest(
            "test",
            [_task("one", DatasetSplit.TRAIN, source="a", content_hash="one"), _task("two", DatasetSplit.TEST, source="b", content_hash="two")],
        )
        manifest.validate()
        self.assertEqual(manifest.counts(), {"train": 1, "validation": 0, "test": 1})

    def test_jsonl_roundtrip(self):
        manifest = BenchmarkManifest("test", [_task("one", DatasetSplit.TRAIN, source="a", content_hash="one")])
        with tempfile.TemporaryDirectory() as td:
            path = manifest.write_jsonl(Path(td) / "pilot.jsonl")
            loaded = BenchmarkManifest.read_jsonl(path)
        self.assertEqual(loaded.tasks[0].taxonomy.folder("papers").name, "Papers")

    def test_teacher_ranking_must_cover_every_candidate(self):
        task = _task("one", DatasetSplit.TRAIN)
        with self.assertRaisesRegex(ValueError, "ranking"):
            TaxonomyRankingTask(
                **{**task.__dict__, "label_ranking": ("not-a-folder",)}
            )

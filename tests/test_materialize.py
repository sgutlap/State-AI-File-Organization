"""Fixture materialization must hide original destination paths from model records."""

import tempfile
import unittest
from pathlib import Path

from scripts.materialize import materialize
from core.research.taxonomy_dataset import DatasetSplit


class MaterializeTaxonomyFixtureTests(unittest.TestCase):
    def test_copies_flat_inbox_and_keeps_path_rule_label_separate(self):
        fixture = {
            "taxonomy": {"version": "fixture-v1", "source": "test", "folders": [
                {"id": "events", "name": "Events", "description": "Event files"},
                {"id": "review", "name": "Review", "description": "Other files"},
            ]},
            "rules": [{"glob": "Events/**", "folder_id": "events"}],
            "default_folder_id": "review",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "Events").mkdir(parents=True)
            (source / "Events" / "party.jpg").write_text("image")
            manifest = materialize(source, root / "flat", fixture, "fixture-source", DatasetSplit.TEST)
        task = manifest.tasks[0]
        self.assertTrue(task.file_state["relative_path"].startswith("inbox/"))
        self.assertNotIn("Events", task.file_state["relative_path"])
        self.assertEqual(task.acceptable_folder_ids, ("events",))

    def test_content_requires_explicit_local_flag(self):
        fixture = {
            "taxonomy": {"version": "fixture-v1", "source": "test", "folders": [
                {"id": "review", "name": "Review", "description": "Other files"},
            ]},
            "default_folder_id": "review",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "note.txt").write_text("semantic training signal")
            private = materialize(source, root / "private", fixture, "private", DatasetSplit.TRAIN)
            local = materialize(
                source, root / "local", fixture, "local", DatasetSplit.TRAIN, include_content=True
            )
        self.assertNotIn("content_sample", private.tasks[0].file_state)
        self.assertIn("semantic training signal", local.tasks[0].file_state["content_sample"])

    def test_exclude_globs_keep_duplicate_source_out_of_fixture(self):
        fixture = {
            "taxonomy": {"version": "fixture-v1", "source": "test", "folders": [
                {"id": "review", "name": "Review", "description": "Other files"},
            ]},
            "exclude_globs": ["duplicate/**"],
            "default_folder_id": "review",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "duplicate").mkdir(parents=True)
            (source / "duplicate" / "same.md").write_text("already in another split")
            (source / "new.md").write_text("independent")
            manifest = materialize(source, root / "flat", fixture, "fixture-source", DatasetSplit.TRAIN)
        self.assertEqual(len(manifest.tasks), 1)

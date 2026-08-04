import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.models.taxonomy_scorer import RankedFolder
from core.models.visual_taxonomy import CLIPVisualTaxonomyScorer, is_visual_candidate, visual_folder_text
from core.taxonomy.spec import FolderSpec, TaxonomySpec


class VisualTaxonomyTests(unittest.TestCase):
    def test_visual_text_excludes_folder_id_and_constraints(self):
        folder = FolderSpec("raw-files", "Camera imports", "Unedited camera images", constraints=("extension:.raw",))
        self.assertEqual("Camera imports. Unedited camera images", visual_folder_text(folder))
    def test_suffix_only_selects_decoder_not_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(b"not an image")
            image = SimpleNamespace(absolute_path=str(path), metadata=SimpleNamespace(extension=".jpg"))
            text = SimpleNamespace(absolute_path=str(path), metadata=SimpleNamespace(extension=".txt"))
            self.assertTrue(is_visual_candidate(image))
            self.assertFalse(is_visual_candidate(text))

    def test_visual_scores_only_replace_semantic_abstention(self):
        taxonomy = TaxonomySpec((FolderSpec("events", "Events", "Family events"), FolderSpec("edits", "Edits", "Edited images")))
        scorer = object.__new__(CLIPVisualTaxonomyScorer)
        scorer.score_threshold = 0.0
        scorer.margin_threshold = 0.10
        scorer.semantic_scorer = SimpleNamespace(rank=lambda state, _: RankedFolder(None, 0.2, 0.01, True, "semantic_dual"))
        scorer._visual_scores = lambda state, folders: [("events", 0.8), ("edits", 0.1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(b"x")
            state = SimpleNamespace(absolute_path=str(path), metadata=SimpleNamespace(extension=".jpg"))
            result = scorer.rank(state, taxonomy)
        self.assertEqual("events", result.folder_id)
        self.assertEqual("clip_visual", result.source)

    def test_visual_rank_many_replaces_only_visual_abstentions(self):
        taxonomy = TaxonomySpec((FolderSpec("events", "Events", "Family events"), FolderSpec("edits", "Edits", "Edited images")))
        scorer = object.__new__(CLIPVisualTaxonomyScorer)
        scorer.score_threshold = 0.0
        scorer.margin_threshold = 0.10
        scorer.semantic_scorer = SimpleNamespace(rank_many=lambda states, _: [RankedFolder(None, 0.2, 0.01, True, "semantic_dual"), RankedFolder("edits", 0.9, 0.4, False, "semantic_dual")])
        scorer._visual_scores_many = lambda states, folders: {0: [("events", 0.8), ("edits", 0.1)]}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "photo.jpg"
            text = Path(directory) / "note.txt"
            image.write_bytes(b"x")
            text.write_bytes(b"x")
            states = [
                SimpleNamespace(absolute_path=str(image), metadata=SimpleNamespace(extension=".jpg")),
                SimpleNamespace(absolute_path=str(text), metadata=SimpleNamespace(extension=".txt")),
            ]
            results = scorer.rank_many(states, taxonomy)
        self.assertEqual(["clip_visual", "semantic_dual"], [result.source for result in results])

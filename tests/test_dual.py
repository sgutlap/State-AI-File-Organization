import unittest
from types import SimpleNamespace
import torch

from core.discovery_text import format_semantic_file_record
from core.models.dual_taxonomy import (
    SEMANTIC_INPUT_FORMAT,
    VIRTUAL_ABSTAIN_FOLDER_ID,
    DualTaxonomyScorer,
    assert_semantic_dual_checkpoint,
    checkpoint_backend,
    hard_negative_ranking_loss,
    task_batches,
    virtual_abstain_prototype,
)
from core.taxonomy.spec import FolderSpec


class DualTaxonomyTests(unittest.TestCase):
    def test_virtual_abstain_candidate_is_semantic_and_not_a_real_folder_id(self):
        prototype = virtual_abstain_prototype()
        self.assertIn("does not belong", prototype)
        self.assertIn(VIRTUAL_ABSTAIN_FOLDER_ID, prototype)
        self.assertTrue(VIRTUAL_ABSTAIN_FOLDER_ID.startswith("__state_ai_"))

    def test_task_batches_preserve_order_and_size(self):
        self.assertEqual(list(task_batches([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_task_batches_reject_nonpositive_size(self):
        with self.assertRaises(ValueError):
            list(task_batches([1], 0))

    def test_checkpoint_backend_is_explicit(self):
        self.assertEqual("dual", checkpoint_backend({"state_dict": {"file_adapter.weight": 1, "folder_adapter.weight": 2}}))
        self.assertEqual("pair", checkpoint_backend({"state_dict": {"head.0.weight": 1}}))
        with self.assertRaises(ValueError):
            checkpoint_backend({"state_dict": {}})

    def test_dual_checkpoint_requires_recorded_semantic_input_format(self):
        assert_semantic_dual_checkpoint({"input_format": SEMANTIC_INPUT_FORMAT})
        with self.assertRaises(ValueError):
            assert_semantic_dual_checkpoint({})

    def test_semantic_file_format_removes_extension_path_and_metadata(self):
        text = format_semantic_file_record({
            "relative_path": "private/finance/secret.csv",
            "metadata": {
                "filename": "secret.csv", "extension": ".csv", "mime_type": "text/csv",
                "size_bytes": 42, "age_days": 12,
            },
            "content_sample": "Quarterly revenue forecast and approved budget.",
        })
        self.assertIn("Filename: secret", text)
        self.assertIn("Quarterly revenue", text)
        self.assertNotIn(".csv", text)
        self.assertNotIn("private/finance", text)
        self.assertNotIn("42", text)

    def test_folder_vectors_are_cached_by_folder_definition(self):
        scorer = object.__new__(DualTaxonomyScorer)
        scorer._folder_vector_cache = {}
        calls = []
        scorer._encode = lambda texts, role: calls.append((tuple(texts), role)) or torch.ones(len(texts), 2)
        folders = [FolderSpec("papers", "Papers", "Research papers")]
        self.assertTrue(torch.equal(scorer._folder_vectors(folders), scorer._folder_vectors(folders)))
        self.assertEqual(1, len(calls))
        scorer.clear_taxonomy_cache()
        scorer._folder_vectors(folders)
        self.assertEqual(2, len(calls))

    def test_rank_abstains_below_calibrated_score_or_margin(self):
        scorer = object.__new__(DualTaxonomyScorer)
        scorer.score_threshold = 0.70
        scorer.margin_threshold = 0.10
        scorer.score = lambda state, folders: [("papers", 0.69), ("notes", 0.20)]
        taxonomy = type("Taxonomy", (), {"folders": [FolderSpec("papers", "Papers", "Research"), FolderSpec("notes", "Notes", "Notes")]})()
        state = SimpleNamespace(content_sample="A semantic note.", metadata=SimpleNamespace(filename="note.txt", extension=".txt"))
        choice = scorer.rank(state, taxonomy)
        self.assertTrue(choice.abstained)
        self.assertIsNone(choice.folder_id)

    def test_virtual_none_candidate_abstains_before_any_real_folder(self):
        scorer = object.__new__(DualTaxonomyScorer)
        scorer.score_threshold = 0.0
        scorer.margin_threshold = 0.0
        scorer.virtual_abstain = True
        scorer.score = lambda state, folders: [(VIRTUAL_ABSTAIN_FOLDER_ID, .70), ("papers", .20), ("notes", .10)]
        taxonomy = type("Taxonomy", (), {"folders": [FolderSpec("papers", "Papers", "Research"), FolderSpec("notes", "Notes", "Notes")]})()
        state = SimpleNamespace(content_sample="Outside this taxonomy.", metadata=SimpleNamespace(filename="outside.txt", extension=".txt"))
        choice = scorer.rank(state, taxonomy)
        self.assertTrue(choice.abstained)
        self.assertIsNone(choice.folder_id)
        self.assertEqual("semantic_dual_virtual_abstain", choice.source)

    def test_opaque_file_uses_one_user_declared_extension_constraint(self):
        scorer = object.__new__(DualTaxonomyScorer)
        scorer.score_threshold = 0.0
        scorer.margin_threshold = 0.0
        taxonomy = type("Taxonomy", (), {"folders": [
            FolderSpec("photos", "Photos", "Personal images", constraints=("extension:.jpg",)),
            FolderSpec("notes", "Notes", "Text notes"),
        ]})()
        opaque = SimpleNamespace(content_sample="[Binary File: image/jpeg]", metadata=SimpleNamespace(filename="photo.jpg", extension=".jpg"))
        self.assertEqual("photos", scorer.rank(opaque, taxonomy).folder_id)
        semantic = SimpleNamespace(content_sample="A textual report saved under a misleading name.", metadata=opaque.metadata)
        scorer.score = lambda state, folders: [("notes", 0.9), ("photos", 0.1)]
        self.assertEqual("notes", scorer.rank(semantic, taxonomy).folder_id)

    def test_example_extension_fallback_requires_one_current_taxonomy_owner(self):
        scorer = object.__new__(DualTaxonomyScorer)
        taxonomy = type("Taxonomy", (), {"folders": [
            FolderSpec("cad", "CAD", "Models", examples=("villa.dwg",)),
            FolderSpec("plans", "Plans", "Exports", examples=("plan.pdf",)),
            FolderSpec("bills", "Bills", "Invoices", examples=("bill.pdf",)),
        ]})()
        dwg = SimpleNamespace(content_sample="[Binary File: image/vnd.dwg]", metadata=SimpleNamespace(extension=".dwg"))
        pdf = SimpleNamespace(content_sample="[Binary File: application/pdf]", metadata=SimpleNamespace(extension=".pdf"))
        self.assertEqual("cad", scorer._example_extension_folder(dwg, taxonomy).id)
        self.assertIsNone(scorer._example_extension_folder(pdf, taxonomy))

    def test_hard_negative_loss_prefers_correct_over_close_teacher_negative(self):
        candidates = ("tax", "financial", "receipts")
        ranking = ("tax", "financial", "receipts")
        correct = hard_negative_ranking_loss(torch.tensor([3.0, 1.0, 0.0]), candidates, ("tax",), ranking)
        incorrect = hard_negative_ranking_loss(torch.tensor([0.0, 3.0, 1.0]), candidates, ("tax",), ranking)
        self.assertLess(correct.item(), incorrect.item())

    def test_hard_negative_loss_weights_closer_teacher_negative_more(self):
        candidates = ("tax", "financial", "receipts")
        ranking = ("tax", "financial", "receipts")
        # Only the second-ranked alternative competes closely in this case.
        near_error = hard_negative_ranking_loss(torch.tensor([0.0, 2.0, -4.0]), candidates, ("tax",), ranking)
        far_error = hard_negative_ranking_loss(torch.tensor([0.0, -4.0, 2.0]), candidates, ("tax",), ranking)
        self.assertGreater(near_error.item(), far_error.item())

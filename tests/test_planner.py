"""Open-taxonomy plans must abstain safely instead of inventing a fixed label."""

import tempfile
import unittest
from pathlib import Path
import torch

from core.agent.open_taxonomy_planner import OpenTaxonomyPlanner
from core.extractors.content_extractor import FileMetadata
from core.extractors.file_scanner import FileState
from core.models.taxonomy_scorer import RankedFolder
from core.models.taxonomy_inducer import SlotTaxonomyInducer
from core.taxonomy.slot_discovery import SlotTaxonomyDiscoverer
from core.taxonomy.spec import FolderSpec, TaxonomySpec


class _StubScorer:
    def rank(self, state, taxonomy):
        if state.metadata.filename.startswith("unknown"):
            return RankedFolder(None, 0.2, 0.01, True)
        return RankedFolder("papers", 0.9, 0.3, False)


class _BatchStubScorer:
    def rank(self, state, taxonomy):
        raise AssertionError("planner should use rank_many when supplied")

    def rank_many(self, states, taxonomy):
        return [RankedFolder("papers", 0.9, 0.3, False) for _ in states]


def _state(root: Path, name: str) -> FileState:
    path = root / name
    path.write_text("x")
    return FileState(
        file_id=name,
        absolute_path=str(path),
        relative_path=name,
        metadata=FileMetadata(name, Path(name).suffix, 1, "", 0, 0, "text/plain", False),
        content_sample="Meaningful project research content for a semantic taxonomy proposal. " * 2,
    )


class OpenTaxonomyPlannerTests(unittest.TestCase):
    def test_uses_batch_ranker_when_available(self):
        taxonomy = TaxonomySpec((FolderSpec("papers", "Papers", "Research papers"),))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = OpenTaxonomyPlanner(_BatchStubScorer(), taxonomy).plan(str(root), [_state(root, "one.pdf"), _state(root, "two.pdf")])
        self.assertEqual(2, len(plan.actions))

    def test_abstention_is_ask_user_and_confident_choice_is_relative_move(self):
        taxonomy = TaxonomySpec((FolderSpec("papers", "Papers", "Research papers"),))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = OpenTaxonomyPlanner(_StubScorer(), taxonomy).plan(
                str(root), [_state(root, "paper.pdf"), _state(root, "unknown.bin")]
            )
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].target_category, "papers")
        self.assertEqual(len(plan.ask_user), 1)
        self.assertEqual(plan.ask_user[0].action_type.value, "ASK_USER")

    def test_slot_discovery_can_only_add_pending_proposals(self):
        taxonomy = TaxonomySpec((FolderSpec("papers", "Papers", "Research papers"),))
        model = SlotTaxonomyInducer(4, hidden_dim=8, max_slots=3, heads=2)
        for parameter in model.parameters(): parameter.data.zero_()
        discoverer = SlotTaxonomyDiscoverer(model, lambda texts: torch.ones(len(texts), 4), existence_threshold=0.4, max_cluster_fraction=1.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = OpenTaxonomyPlanner(_StubScorer(), taxonomy).plan(
                str(root), [_state(root, "unknown.bin"), _state(root, "unknown2.bin"), _state(root, "unknown3.bin")], discoverer=discoverer
            )
        self.assertEqual(1, len(plan.taxonomy_proposals))
        self.assertTrue(plan.taxonomy_proposals[0].requires_confirmation)

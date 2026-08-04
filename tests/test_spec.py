"""Taxonomy-neutral contracts and mandatory-confirmation invariant."""

import tempfile
import unittest
from pathlib import Path

from core.agent.action_planner import MovePlan
from core.agent.executor import PlanExecutor
from core.taxonomy.spec import FolderSpec, TaxonomyEditType, TaxonomyProposal, TaxonomySpec


class TaxonomySpecTests(unittest.TestCase):
    def test_legacy_taxonomy_is_an_editable_seed(self):
        from core.taxonomy.taxonomy_manager import TaxonomyManager

        spec = TaxonomySpec.from_legacy_manager(TaxonomyManager())
        self.assertEqual(spec.source, "legacy-seed")
        self.assertEqual(len(spec.folders), 8)
        self.assertEqual(spec.folder("code/projects").name, "Source Code & Scripts")

    def test_rejects_unknown_parent(self):
        with self.assertRaisesRegex(ValueError, "unknown parent"):
            TaxonomySpec((FolderSpec("papers", "Papers", "Research", parent_id="missing"),))

    def test_unapproved_taxonomy_edit_blocks_execution(self):
        proposal = TaxonomyProposal(
            operation=TaxonomyEditType.ADD,
            proposed_folders=(FolderSpec("papers", "Papers", "Research papers"),),
            confidence=0.8,
            utility=0.4,
            rationale="Representative files form a coherent group.",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = MovePlan(root_dir=str(root), taxonomy_proposals=[proposal])
            result = PlanExecutor(log_dir=str(root / "logs")).execute(plan, dry_run=False, confirm=False)
            self.assertEqual(result["status"], "taxonomy_confirmation_required")
            self.assertEqual(result["pending_taxonomy_proposal_ids"], [proposal.proposal_id])

    def test_approved_taxonomy_edit_can_proceed_to_dry_run(self):
        proposal = TaxonomyProposal(
            operation=TaxonomyEditType.KEEP,
            proposed_folders=(FolderSpec("papers", "Papers", "Research papers"),),
            confidence=0.8,
            utility=0.4,
            rationale="No edit needed after review.",
        ).approve()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = PlanExecutor(log_dir=str(root / "logs")).execute(
                MovePlan(root_dir=str(root), taxonomy_proposals=[proposal]), dry_run=True
            )
            self.assertEqual(result["status"], "dry_run")

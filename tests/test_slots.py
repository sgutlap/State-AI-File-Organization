import tempfile
import unittest
import torch

from core.models.taxonomy_inducer import SlotTaxonomyInducer
from core.taxonomy.slot_discovery import SlotTaxonomyDiscoverer, load_slot_discoverer, slot_model_config
from core.taxonomy.spec import FolderSpec, TaxonomySpec


class SlotDiscoveryTests(unittest.TestCase):
    def test_emits_only_confirmation_required_add_proposals(self):
        model = SlotTaxonomyInducer(4, hidden_dim=8, max_slots=3, heads=2)
        for parameter in model.parameters(): parameter.data.zero_()
        discoverer = SlotTaxonomyDiscoverer(model, lambda texts: torch.ones(len(texts), 4), existence_threshold=0.4, max_cluster_fraction=1.0)
        proposals = discoverer.discover_texts({"a":"solar panel study", "b":"solar panel analysis", "c":"solar panel notes"}, TaxonomySpec((FolderSpec("existing","Existing","Existing user folder"),)))
        self.assertEqual(1, len(proposals))
        self.assertTrue(proposals[0].requires_confirmation)
        self.assertFalse(proposals[0].approved)

    def test_semantic_evidence_gate_rejects_filename_only_cluster(self):
        model = SlotTaxonomyInducer(4, hidden_dim=8, max_slots=3, heads=2)
        for parameter in model.parameters(): parameter.data.zero_()
        discoverer = SlotTaxonomyDiscoverer(model, lambda texts: torch.ones(len(texts), 4), existence_threshold=0.4, max_cluster_fraction=1.0)
        proposals = discoverer.discover_texts({"a": "invoice", "b": "invoice", "c": "invoice"}, TaxonomySpec((FolderSpec("existing", "Existing", "Existing user folder"),)), semantic_ids=set())
        self.assertEqual([], proposals)

    def test_recovers_architecture_from_legacy_checkpoint_shapes(self):
        config = slot_model_config({"state_dict": {
            "file_projection.0.weight": torch.empty(384, 256),
            "slot_queries": torch.empty(7, 384),
        }})
        self.assertEqual({"input_dim": 256, "hidden_dim": 384, "max_slots": 7, "heads": 4}, config)

    def test_refuses_checkpoint_with_legacy_extension_input_format(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"state_dict": {}}, handle.name)
            with self.assertRaisesRegex(ValueError, "semantic-only"):
                load_slot_discoverer(handle.name)

"""Discovery is bounded ML clustering that only emits reviewable proposals."""

import unittest

from core.taxonomy.discovery import DiscoveryConfig, TaxonomyDiscoverer
from core.taxonomy.spec import FolderSpec, TaxonomyEditType, TaxonomySpec


class TaxonomyDiscoveryTests(unittest.TestCase):
    def test_coherent_unresolved_group_becomes_unapproved_add_proposal(self):
        taxonomy = TaxonomySpec((FolderSpec("code", "Code", "Programming projects"),))
        texts = {
            "one": "neural network experiment notebook results transformer embeddings",
            "two": "transformer embedding experiment results notebook neural network",
            "three": "experiment notes for neural embeddings transformer result analysis",
            "noise": "birthday photo picnic beach camera image",
        }
        proposals = TaxonomyDiscoverer(
            DiscoveryConfig(min_cluster_size=3, cosine_distance=0.9, min_coherence=0.1)
        ).discover_texts(texts, taxonomy)
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.operation, TaxonomyEditType.ADD)
        self.assertTrue(proposal.requires_confirmation)
        self.assertFalse(proposal.approved)
        self.assertEqual(set(proposal.affected_files), {"one", "two", "three"})

    def test_too_few_unresolved_files_makes_no_taxonomy_edit(self):
        taxonomy = TaxonomySpec((FolderSpec("code", "Code", "Programming projects"),))
        proposals = TaxonomyDiscoverer().discover_texts({"one": "only one file"}, taxonomy)
        self.assertEqual(proposals, [])

    def test_cluster_matching_existing_folder_is_not_proposed(self):
        taxonomy = TaxonomySpec((
            FolderSpec("code", "Code", "Programming scripts source code and build configuration"),
        ))
        texts = {
            "one": "python source code script build configuration",
            "two": "source code scripts and project build config",
            "three": "programming code configuration and software scripts",
        }
        proposals = TaxonomyDiscoverer(
            DiscoveryConfig(
                min_cluster_size=3,
                cosine_distance=0.9,
                min_coherence=0.1,
                max_existing_similarity=0.15,
            )
        ).discover_texts(texts, taxonomy)
        self.assertEqual(proposals, [])

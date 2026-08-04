"""Bounded ML taxonomy discovery for unresolved files.

Discovery proposes additive folders from coherent clusters.  It does not create
folders or modify a taxonomy; callers must explicitly approve each proposal.
Destructive edits (merge/split/remove) remain unavailable until supervised
proposal-utility data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from core.extractors.file_scanner import FileState
from core.discovery_text import format_discovery_file_record
from core.taxonomy.spec import FolderSpec, TaxonomyEditType, TaxonomyProposal, TaxonomySpec


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


@dataclass(frozen=True)
class DiscoveryConfig:
    min_cluster_size: int = 3
    max_proposals: int = 5
    cosine_distance: float = 0.65
    min_coherence: float = 0.25
    max_existing_similarity: float = 0.70


class TaxonomyDiscoverer:
    """Cluster unresolved file descriptions and emit review-only ADD proposals."""

    def __init__(self, config: DiscoveryConfig | None = None):
        self.config = config or DiscoveryConfig()

    @staticmethod
    def state_text(state: FileState) -> str:
        return format_discovery_file_record({
            "relative_path": state.relative_path,
            "metadata": {
                "filename": state.metadata.filename,
                "extension": state.metadata.extension,
                "size_bytes": state.metadata.size_bytes,
                "age_days": state.metadata.age_days,
                "mime_type": state.metadata.mime_type,
                "is_binary": state.metadata.is_binary,
            },
            "content_sample": state.content_sample,
        })

    def discover(
        self, unresolved_states: Sequence[FileState], taxonomy: TaxonomySpec
    ) -> List[TaxonomyProposal]:
        texts = {state.file_id: self.state_text(state) for state in unresolved_states}
        semantic_ids = {
            state.file_id for state in unresolved_states
            if len(state.content_sample.strip()) >= 80
            and not state.content_sample.startswith("[Binary File:")
            and not state.content_sample.startswith("[Empty File]")
        }
        return self.discover_texts(texts, taxonomy, semantic_ids=semantic_ids)

    def discover_texts(
        self, unresolved_texts: Mapping[str, str], taxonomy: TaxonomySpec, *, semantic_ids: set[str] | None = None
    ) -> List[TaxonomyProposal]:
        """ML clustering entry point suitable for redacted benchmark states."""
        if len(unresolved_texts) < self.config.min_cluster_size:
            return []
        file_ids = list(unresolved_texts)
        documents = [unresolved_texts[file_id] or "" for file_id in file_ids]
        existing_texts = [
            f"{folder.name}\n{folder.description}\n{' '.join(folder.constraints)}"
            for folder in taxonomy.folders
        ]
        proposal_stop_words = ENGLISH_STOP_WORDS | {
            "file", "filename", "path", "extension", "size", "bytes", "age", "mime", "binary", "content", "sample", "redacted", "inbox", "true", "false",
        }
        vectorizer = TfidfVectorizer(stop_words=list(proposal_stop_words), ngram_range=(1, 2), max_features=2048)
        try:
            all_vectors = vectorizer.fit_transform(documents + existing_texts)
        except ValueError:  # empty vocabulary
            return []
        vectors = all_vectors[:len(documents)]
        existing_vectors = all_vectors[len(documents):]
        labels = DBSCAN(
            eps=self.config.cosine_distance,
            min_samples=self.config.min_cluster_size,
            metric="cosine",
        ).fit_predict(vectors)
        terms = np.asarray(vectorizer.get_feature_names_out())
        proposals: List[TaxonomyProposal] = []
        existing = {folder.id for folder in taxonomy.folders}
        for cluster_id in sorted(set(labels)):
            if cluster_id < 0:
                continue  # noise belongs in ABSTAIN / user review, not a new folder
            members = np.flatnonzero(labels == cluster_id)
            if len(members) < self.config.min_cluster_size:
                continue
            if semantic_ids is not None and sum(file_ids[index] in semantic_ids for index in members) / len(members) < 0.60:
                continue
            centroid = np.asarray(vectors[members].mean(axis=0)).ravel()
            norm = np.linalg.norm(centroid)
            if not norm:
                continue
            member_vectors = vectors[members].toarray()
            member_norms = np.linalg.norm(member_vectors, axis=1)
            coherence = float(np.mean((member_vectors @ centroid) / np.maximum(member_norms * norm, 1e-12)))
            if coherence < self.config.min_coherence:
                continue
            if existing_vectors.shape[0]:
                existing_norms = np.sqrt(existing_vectors.multiply(existing_vectors).sum(axis=1)).A1
                similarities = np.asarray(existing_vectors @ centroid).ravel() / np.maximum(
                    existing_norms * norm, 1e-12
                )
                closest_existing = float(similarities.max())
            else:
                closest_existing = 0.0
            if closest_existing >= self.config.max_existing_similarity:
                continue  # a similar user-owned bin already exists; score it instead
            keywords = []
            for term in terms[centroid.argsort()[::-1]]:
                if not term or any(term in chosen or chosen in term for chosen in keywords):
                    continue
                keywords.append(term)
                if len(keywords) == 3:
                    break
            if not keywords:
                continue
            folder_id = self._unique_id(keywords, existing)
            existing.add(folder_id)
            folder_name = "Suggested: " + " / ".join(keyword.replace("_", " ").title() for keyword in keywords[:2])
            examples = tuple(file_ids[index] for index in members[:5])
            support = len(members)
            confidence = min(
                0.99,
                coherence * (1.0 - math.exp(-support / 5.0)) * (1.0 - closest_existing),
            )
            utility = confidence - 0.05  # explicit folder-complexity penalty
            proposed = FolderSpec(
                id=folder_id,
                name=folder_name,
                description=(
                    f"ML cluster of {support} unresolved files; dominant terms: "
                    + ", ".join(keywords)
                    + "."
                ),
                examples=examples,
            )
            proposals.append(
                TaxonomyProposal(
                    operation=TaxonomyEditType.ADD,
                    proposed_folders=(proposed,),
                    affected_files=tuple(file_ids[index] for index in members),
                    confidence=round(confidence, 4),
                    utility=round(utility, 4),
                    rationale=(
                        f"Unsupervised TF-IDF/DBSCAN cluster: n={support}, "
                        f"coherence={coherence:.3f}, nearest-existing={closest_existing:.3f}; "
                        "review before creating this folder."
                    ),
                )
            )
        proposals.sort(key=lambda proposal: (proposal.utility, proposal.confidence), reverse=True)
        return proposals[: self.config.max_proposals]

    @staticmethod
    def _unique_id(keywords: Iterable[str], existing: set[str]) -> str:
        base = "suggested/" + "-".join(
            re.sub(r"[^a-z0-9]+", "-", keyword.lower()).strip("-")
            for keyword in keywords
        )
        base = base.rstrip("-") or "suggested/group"
        candidate, suffix = base, 2
        while candidate in existing:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

"""Metrics for proposed folder sets before routing starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from core.research.taxonomy_episode import TaxonomyEpisode


@dataclass(frozen=True)
class TaxonomyInductionMetrics:
    proposal_count: int
    oracle_count: int
    count_absolute_error: int
    covered_files: int
    coverage: float
    purity: float
    b3_precision: float
    b3_recall: float
    b3_f1: float


def evaluate_induction(
    episode: TaxonomyEpisode, proposed_clusters: Mapping[str, Sequence[str]],
) -> TaxonomyInductionMetrics:
    """Measure coherent, non-overlapping proposals without requiring name equality."""
    oracle = episode.oracle_assignments
    seen, correct, predicted = set(), 0, {}
    for cluster_id, members in proposed_clusters.items():
        labels = [oracle[file_id] for file_id in members if file_id in oracle and file_id not in seen]
        new_members = [file_id for file_id in members if file_id in oracle and file_id not in seen]
        seen.update(new_members)
        predicted.update({file_id: cluster_id for file_id in new_members})
        if labels:
            correct += max(labels.count(label) for label in set(labels))
    covered = len(seen)
    # B-cubed penalizes both merging different oracle folders and splitting one
    # folder across many proposed clusters.  Coverage remains separate so an
    # inducer cannot improve this score by declining difficult files.
    b3_precision = b3_recall = 0.0
    if seen:
        precision_sum = recall_sum = 0.0
        for file_id in seen:
            predicted_peers = {other for other in seen if predicted[other] == predicted[file_id]}
            oracle_peers = {other for other in seen if oracle[other] == oracle[file_id]}
            overlap = len(predicted_peers & oracle_peers)
            precision_sum += overlap / len(predicted_peers)
            recall_sum += overlap / len(oracle_peers)
        b3_precision = precision_sum / covered
        b3_recall = recall_sum / covered
    b3_f1 = 2 * b3_precision * b3_recall / max(1e-12, b3_precision + b3_recall)
    oracle_count = len(set(oracle.values()))
    return TaxonomyInductionMetrics(
        proposal_count=len(proposed_clusters),
        oracle_count=oracle_count,
        count_absolute_error=abs(len(proposed_clusters) - oracle_count),
        covered_files=covered,
        coverage=covered / max(1, len(oracle)),
        purity=correct / max(1, covered),
        b3_precision=b3_precision,
        b3_recall=b3_recall,
        b3_f1=b3_f1,
    )

"""Validation-only threshold fitting for selective taxonomy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from core.research.taxonomy_dataset import TaxonomyRankingTask


@dataclass(frozen=True)
class AbstentionThresholds:
    score: float
    margin: float
    validation_accuracy: float
    validation_coverage: float

    def to_dict(self) -> dict[str, float]:
        return {
            "abstain_threshold": self.score,
            "margin_threshold": self.margin,
            "validation_action_accuracy": self.validation_accuracy,
            "validation_coverage": self.validation_coverage,
        }


def fit_abstention_thresholds(
    tasks: Sequence[TaxonomyRankingTask], scored_predictions: Iterable[Mapping[str, object]]
) -> AbstentionThresholds:
    """Maximize validation action accuracy, then coverage, on a finite score grid."""
    by_id = {str(row["task_id"]): row for row in scored_predictions}
    if len(by_id) != len(tasks) or set(by_id) != {task.task_id for task in tasks}:
        raise ValueError("calibration predictions must contain each validation task exactly once")
    scores = {0.0, 1.0}
    margins = {0.0}
    for row in by_id.values():
        try:
            scores.add(float(row["score"]))
            margins.add(float(row["margin"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("calibration predictions require numeric score and margin") from error
    best = None
    for score_threshold in sorted(scores):
        for margin_threshold in sorted(margins):
            correct = moved = 0
            for task in tasks:
                row = by_id[task.task_id]
                ranked = tuple(str(folder) for folder in row.get("ranked_folder_ids", []) or ())
                abstain = float(row["score"]) < score_threshold or float(row["margin"]) < margin_threshold
                predicted = None if abstain or not ranked else ranked[0]
                moved += int(predicted is not None)
                correct += int((task.abstain and predicted is None) or (
                    not task.abstain and predicted in set(task.acceptable_folder_ids)
                ))
            accuracy = correct / len(tasks)
            coverage = moved / len(tasks)
            candidate = (accuracy, coverage, -score_threshold, -margin_threshold)
            if best is None or candidate > best[0]:
                best = (candidate, score_threshold, margin_threshold)
    assert best is not None
    return AbstentionThresholds(best[1], best[2], best[0][0], best[0][1])


def fit_virtual_abstention_thresholds(
    tasks: Sequence[TaxonomyRankingTask], scored_predictions: Iterable[Mapping[str, object]]
) -> AbstentionThresholds:
    """Fit the virtual-none versus best-real margin on validation only."""
    by_id = {str(row["task_id"]): row for row in scored_predictions}
    if len(by_id) != len(tasks) or set(by_id) != {task.task_id for task in tasks}:
        raise ValueError("calibration predictions must contain each validation task exactly once")
    margins = {0.0}
    for row in by_id.values():
        try:
            margins.add(max(0.0, float(row["virtual_abstain_score"]) - float(row["score"])))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("virtual calibration requires virtual_abstain_score and score") from error
    best = None
    for threshold in sorted(margins):
        correct = moved = 0
        for task in tasks:
            row = by_id[task.task_id]
            abstain = float(row["virtual_abstain_score"]) - float(row["score"]) >= threshold
            ranked = tuple(str(folder) for folder in row.get("ranked_folder_ids", []) or ())
            predicted = None if abstain or not ranked else ranked[0]
            moved += int(predicted is not None)
            correct += int((task.abstain and predicted is None) or (
                not task.abstain and predicted in set(task.acceptable_folder_ids)
            ))
        accuracy, coverage = correct / len(tasks), moved / len(tasks)
        candidate = (accuracy, coverage, -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, threshold)
    assert best is not None
    return AbstentionThresholds(0.0, best[1], best[0][0], best[0][1])

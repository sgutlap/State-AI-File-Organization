"""Honest task-level metrics for taxonomy-conditioned organization.

Every manifest task is retained in denominators. Missing predictions, invalid
folder ids, abstentions, and unmatched destinations are observable outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from core.research.taxonomy_dataset import TaxonomyRankingTask


ABSTAIN = "__ABSTAIN__"


@dataclass(frozen=True)
class TaxonomyPrediction:
    task_id: str
    ranked_folder_ids: Tuple[str, ...] = ()
    abstained: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TaxonomyPrediction":
        return cls(
            task_id=str(data["task_id"]),
            ranked_folder_ids=tuple(str(value) for value in data.get("ranked_folder_ids", []) or []),
            abstained=bool(data.get("abstained", False)),
        )


@dataclass(frozen=True)
class OpenTaxonomyMetrics:
    n_tasks: int
    missing_predictions: int
    action_accuracy: float
    macro_destination_f1: float
    mrr: float
    recall_at_1: float
    recall_at_3: float
    coverage: float
    abstain_precision: float
    abstain_recall: float
    abstain_f1: float
    taxonomy_macro_accuracy: float
    action_accuracy_ci95: Tuple[float, float]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["action_accuracy_ci95"] = list(self.action_accuracy_ci95)
        return data


def evaluate_open_taxonomy(
    tasks: Sequence[TaxonomyRankingTask], predictions: Iterable[TaxonomyPrediction], *, bootstrap_samples: int = 1000
) -> OpenTaxonomyMetrics:
    if not tasks:
        raise ValueError("cannot evaluate an empty task list")
    by_id: Dict[str, TaxonomyPrediction] = {}
    for prediction in predictions:
        if prediction.task_id in by_id:
            raise ValueError(f"duplicate prediction task_id: {prediction.task_id}")
        by_id[prediction.task_id] = prediction
    known_task_ids = {task.task_id for task in tasks}
    extras = set(by_id) - known_task_ids
    if extras:
        raise ValueError(f"predictions contain unknown task ids: {sorted(extras)[:3]}")

    successes: List[float] = []
    source_groups: List[str] = []
    reciprocal_ranks: List[float] = []
    recall1: List[float] = []
    recall3: List[float] = []
    predicted_labels: List[str] = []
    true_labels: List[str] = []
    per_taxonomy: Dict[str, List[float]] = {}
    abstain_tp = abstain_fp = abstain_fn = 0
    missing = 0
    non_abstained = 0

    for task in tasks:
        prediction = by_id.get(task.task_id)
        if prediction is None:
            missing += 1
            prediction = TaxonomyPrediction(task_id=task.task_id, abstained=True)
        ranked = tuple(folder for folder in prediction.ranked_folder_ids if folder in task.candidate_folder_ids)
        pred_label = ABSTAIN if prediction.abstained or not ranked else ranked[0]
        true_is_abstain = task.abstain
        if pred_label != ABSTAIN:
            non_abstained += 1

        if true_is_abstain:
            correct = pred_label == ABSTAIN
            true_label = ABSTAIN
            reciprocal_ranks.append(0.0)
            recall1.append(0.0)
            recall3.append(0.0)
            if pred_label == ABSTAIN:
                abstain_tp += 1
            else:
                abstain_fn += 1
        else:
            accepted = set(task.acceptable_folder_ids)
            correct = pred_label in accepted
            # Multi-acceptable target: a correct selected folder is the target;
            # otherwise use a stable reference to avoid removing the error.
            true_label = pred_label if correct else task.acceptable_folder_ids[0]
            rank = next((index + 1 for index, folder in enumerate(ranked) if folder in accepted), None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            recall1.append(1.0 if rank == 1 else 0.0)
            recall3.append(1.0 if rank is not None and rank <= 3 else 0.0)
        if true_is_abstain and pred_label != ABSTAIN:
            pass
        elif not true_is_abstain and pred_label == ABSTAIN:
            abstain_fp += 1
        successes.append(float(correct))
        source_groups.append(task.source_group_id)
        predicted_labels.append(pred_label)
        true_labels.append(true_label)
        per_taxonomy.setdefault(task.taxonomy_id, []).append(float(correct))

    precision = abstain_tp / (abstain_tp + abstain_fp) if abstain_tp + abstain_fp else 0.0
    recall = abstain_tp / (abstain_tp + abstain_fn) if abstain_tp + abstain_fn else 0.0
    abstain_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return OpenTaxonomyMetrics(
        n_tasks=len(tasks),
        missing_predictions=missing,
        action_accuracy=float(mean(successes)),
        macro_destination_f1=_macro_f1(true_labels, predicted_labels),
        mrr=float(mean(reciprocal_ranks)),
        recall_at_1=float(mean(recall1)),
        recall_at_3=float(mean(recall3)),
        coverage=non_abstained / len(tasks),
        abstain_precision=precision,
        abstain_recall=recall,
        abstain_f1=abstain_f1,
        taxonomy_macro_accuracy=float(mean(mean(values) for values in per_taxonomy.values())),
        action_accuracy_ci95=_cluster_bootstrap_ci(successes, source_groups, bootstrap_samples),
    )


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(mean(scores)) if scores else 0.0


def _cluster_bootstrap_ci(
    values: Sequence[float], source_groups: Sequence[str], samples: int
) -> Tuple[float, float]:
    """Resample source groups, not correlated files, for headline accuracy CIs."""
    if len(values) != len(source_groups):
        raise ValueError("values and source groups must have equal length")
    grouped: Dict[str, List[float]] = {}
    for value, group in zip(values, source_groups):
        grouped.setdefault(group, []).append(value)
    if not values:
        raise ValueError("cannot bootstrap empty values")
    if len(grouped) == 1:
        value = float(mean(values))
        return (value, value)
    rng = np.random.default_rng(20260801)
    clusters = list(grouped.values())
    draws = []
    for _ in range(max(1, samples)):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sampled = [value for index in selected for value in clusters[index]]
        draws.append(float(mean(sampled)))
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))

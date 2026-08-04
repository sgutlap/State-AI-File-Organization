"""Metrics retain abstentions and missing predictions in their denominators."""

import unittest

from core.metrics.open_taxonomy import TaxonomyPrediction, _cluster_bootstrap_ci, evaluate_open_taxonomy
from core.research.taxonomy_dataset import DatasetSplit, TaxonomyRankingTask
from core.taxonomy.spec import FolderSpec, TaxonomySpec


def _task(task_id: str, *, abstain: bool = False) -> TaxonomyRankingTask:
    taxonomy = TaxonomySpec((
        FolderSpec("papers", "Papers", "Research papers"),
        FolderSpec("code", "Code", "Programming projects"),
    ))
    return TaxonomyRankingTask(
        task_id=task_id, workspace_id="workspace", source_group_id=task_id,
        taxonomy_id="taxonomy", split=DatasetSplit.TEST, file_id=task_id,
        content_hash=task_id, file_state={"relative_path": f"{task_id}.txt"}, taxonomy=taxonomy,
        candidate_folder_ids=("papers", "code"),
        acceptable_folder_ids=() if abstain else ("papers",), abstain=abstain,
    )


class OpenTaxonomyMetricsTests(unittest.TestCase):
    def test_missing_prediction_is_counted_as_failure(self):
        metrics = evaluate_open_taxonomy([_task("a"), _task("b")], [
            TaxonomyPrediction("a", ("papers",)),
        ])
        self.assertEqual(metrics.missing_predictions, 1)
        self.assertEqual(metrics.action_accuracy, 0.5)
        self.assertEqual(metrics.coverage, 0.5)

    def test_abstention_metrics_distinguish_correct_and_incorrect_abstain(self):
        metrics = evaluate_open_taxonomy([_task("a", abstain=True), _task("b")], [
            TaxonomyPrediction("a", abstained=True),
            TaxonomyPrediction("b", abstained=True),
        ])
        self.assertEqual(metrics.action_accuracy, 0.5)
        self.assertEqual(metrics.abstain_precision, 0.5)
        self.assertEqual(metrics.abstain_recall, 1.0)

    def test_unknown_prediction_fails_instead_of_being_normalized(self):
        metrics = evaluate_open_taxonomy([_task("a")], [TaxonomyPrediction("a", ("unknown",))])
        self.assertEqual(metrics.action_accuracy, 0.0)
        self.assertEqual(metrics.coverage, 0.0)

    def test_confidence_interval_resamples_source_groups(self):
        lower, upper = _cluster_bootstrap_ci([1.0, 1.0, 0.0], ["a", "a", "b"], 1000)
        self.assertLess(lower, upper)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)

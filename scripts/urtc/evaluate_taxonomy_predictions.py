from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from state_ai.metrics.open_taxonomy import TaxonomyPrediction, evaluate_open_taxonomy
from state_ai.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="locked TaxonomyRankingTask JSONL manifest")
    parser.add_argument("predictions", help="JSONL: task_id, ranked_folder_ids, abstained")
    parser.add_argument("--out", help="optional JSON metrics output")
    parser.add_argument(
        "--split", choices=[split.value for split in DatasetSplit], default="test",
        help="Evaluation split; use validation only for development/calibration, never model selection on test.",
    )
    args = parser.parse_args()

    manifest = BenchmarkManifest.read_jsonl(args.manifest)
    split = DatasetSplit(args.split)
    tasks = [task for task in manifest.tasks if task.split == split]
    if not tasks:
        if split == DatasetSplit.TEST:
            raise SystemExit("evaluation requires a manifest containing locked test tasks")
        raise SystemExit(f"manifest contains no {split.value} tasks")
    if split == DatasetSplit.TEST:
        try:
            manifest.validate_heldout_taxonomies()
        except ValueError as error:
            raise SystemExit(f"locked transfer evaluation rejected: {error}") from error
    predictions = [
        TaxonomyPrediction.from_dict(json.loads(line))
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = evaluate_open_taxonomy(tasks, predictions)
    rendered = json.dumps(metrics.to_dict(), indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Fit abstention thresholds on validation predictions; reject test labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics.abstention_calibration import fit_abstention_thresholds, fit_virtual_abstention_thresholds
from core.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("predictions", help="Scored JSONL from a validation split prediction command")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint to bind into the artifact when predictions are a late-fusion scorer without one.",
    )
    parser.add_argument(
        "--scorer", choices=["dual", "virtual_dual", "semantic_hybrid", "hybrid_topk_cross_encoder"], default="dual",
        help="Score scale used to fit thresholds; hybrid thresholds must never be applied to raw dual scores.",
    )
    parser.add_argument("--dual-weight", type=float, help="Required with --scorer semantic_hybrid.")
    parser.add_argument("--reranker-weight", type=float, help="Required with --scorer hybrid_topk_cross_encoder.")
    args = parser.parse_args()
    manifest = BenchmarkManifest.read_jsonl(args.manifest)
    tasks = [task for task in manifest.tasks if task.split == DatasetSplit.VALIDATION]
    if not tasks:
        raise SystemExit("calibration requires validation tasks; test labels are forbidden")
    predictions = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    declared_splits = {str(row.get("split")) for row in predictions if row.get("split") is not None}
    if declared_splits and declared_splits != {DatasetSplit.VALIDATION.value}:
        raise SystemExit("calibration predictions must be produced on the validation split")
    checkpoints = {str(row.get("checkpoint")) for row in predictions if row.get("checkpoint")}
    if len(checkpoints) > 1:
        raise SystemExit("calibration predictions must use one checkpoint")
    if args.scorer in {"semantic_hybrid", "hybrid_topk_cross_encoder"} and (
        args.dual_weight is None or not 0.0 <= args.dual_weight <= 1.0
    ):
        raise SystemExit(f"{args.scorer} calibration requires --dual-weight in [0, 1]")
    if args.scorer == "hybrid_topk_cross_encoder" and (
        args.reranker_weight is None or not 0.0 <= args.reranker_weight <= 1.0
    ):
        raise SystemExit("hybrid_topk_cross_encoder calibration requires --reranker-weight in [0, 1]")
    thresholds = fit_virtual_abstention_thresholds(tasks, predictions) if args.scorer == "virtual_dual" else fit_abstention_thresholds(tasks, predictions)
    payload = thresholds.to_dict()
    payload.update({
        "calibration_manifest": str(Path(args.manifest).resolve()),
        "calibration_split": DatasetSplit.VALIDATION.value,
        "calibration_scorer": args.scorer,
    })
    if args.scorer in {"semantic_hybrid", "hybrid_topk_cross_encoder"}:
        payload["calibration_dual_weight"] = args.dual_weight
    if args.scorer == "hybrid_topk_cross_encoder":
        payload["calibration_reranker_weight"] = args.reranker_weight
    if args.checkpoint:
        payload["calibration_checkpoint"] = str(Path(args.checkpoint).resolve())
    elif checkpoints:
        payload["calibration_checkpoint"] = next(iter(checkpoints))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

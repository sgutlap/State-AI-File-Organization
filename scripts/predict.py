"""Score taxonomy tasks with a saved dual-encoder checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.dual_taxonomy import (
    DualTaxonomyEncoder, DualTaxonomyScorer, VIRTUAL_ABSTAIN_FOLDER_ID,
    assert_semantic_dual_checkpoint, task_batches, virtual_abstain_prototype,
)
from core.models.hf_load import from_pretrained_cached
from core.discovery_text import format_semantic_file_record
from core.models.taxonomy_scorer import folder_prototype_texts, format_folder_spec
from core.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit
from scripts.train import device_for, encode


def calibration_thresholds(path: str | None) -> tuple[float, float]:
    """Read validation-only score/margin thresholds for selective predictions."""
    if not path:
        return 0.0, 0.0
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        score = float(data["abstain_threshold"])
        margin = float(data["margin_threshold"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid dual-scorer calibration artifact") from error
    if not 0.0 <= score <= 1.0 or margin < 0.0:
        raise ValueError("dual-scorer calibration thresholds are invalid")
    return score, margin


def calibration_scorer_name(path: str | None) -> str:
    if not path:
        return ""
    try:
        return str(json.loads(Path(path).read_text(encoding="utf-8")).get("calibration_scorer", "dual"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid dual-scorer calibration artifact") from error


def should_abstain(score: float, margin: float, thresholds: tuple[float, float]) -> bool:
    """Apply the same strict threshold policy used by the runtime scorer."""
    return score < thresholds[0] or margin < thresholds[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=[split.value for split in DatasetSplit], default="validation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration", help="Validation-only abstention threshold JSON")
    parser.add_argument(
        "--example-extension-fallback", action="store_true",
        help="For opaque files only, use a type when current folder examples make that extension uniquely user-defined.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        score_threshold, margin_threshold = calibration_thresholds(args.calibration)
        calibration_scorer = calibration_scorer_name(args.calibration)
    except ValueError as error:
        parser.error(str(error))
    requested = DatasetSplit(args.split)
    tasks = [task for task in BenchmarkManifest.read_jsonl(args.manifest).tasks if task.split == requested]
    if not tasks:
        raise SystemExit(f"manifest contains no {requested.value} tasks")
    device = device_for(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert_semantic_dual_checkpoint(checkpoint)
    tokenizer = from_pretrained_cached(AutoTokenizer.from_pretrained, checkpoint["base_model"])
    model = DualTaxonomyEncoder(checkpoint["base_model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    folder_prototypes = bool(checkpoint.get("folder_prototypes", False))
    virtual_abstain = bool(checkpoint.get("virtual_abstain", False))
    if virtual_abstain and args.calibration and calibration_scorer != "virtual_dual":
        parser.error("virtual-abstain checkpoint requires a virtual_dual calibration artifact")
    if not virtual_abstain and calibration_scorer == "virtual_dual":
        parser.error("virtual_dual calibration artifact requires a virtual-abstain checkpoint")
    fallback_scorer = DualTaxonomyScorer(
        model, base_model_name=checkpoint["base_model"], device=str(device),
        folder_prototypes=folder_prototypes, example_extension_fallback=args.example_extension_fallback,
    ) if args.example_extension_fallback else None
    rows = []
    with torch.no_grad():
        for batch in task_batches(tasks, args.batch_size):
            files = encode(model, tokenizer, [format_semantic_file_record(task.file_state) for task in batch], "file", device)
            groups = []
            for task in batch:
                groups.extend(
                    folder_prototype_texts(task.taxonomy.folder(folder_id)) if folder_prototypes
                    else (format_folder_spec(task.taxonomy.folder(folder_id)),)
                    for folder_id in task.candidate_folder_ids
                )
                if virtual_abstain:
                    groups.append((virtual_abstain_prototype(),))
            folders = encode(model, tokenizer, [text for group in groups for text in group], "folder", device)
            offset = group_offset = 0
            for task, vector in zip(batch, files):
                # Groups are materialized for the whole batch.  Advance their
                # cursors even when the bounded fallback supplies this task's
                # answer, otherwise subsequent tasks would score against the
                # previous taxonomy's prototypes.
                real_count = len(task.candidate_folder_ids)
                count = real_count + int(virtual_abstain)
                if fallback_scorer is not None:
                    # The policy is permitted only for opaque files. It never
                    # sees a destination label or changes semantic files.
                    state = type("ManifestState", (), {
                        "content_sample": task.file_state.get("content_sample", ""),
                        "metadata": type("Metadata", (), task.file_state.get("metadata") or {})(),
                    })()
                    inferred = fallback_scorer._example_extension_folder(state, task.taxonomy)
                    if inferred is not None:
                        ordered = [inferred.id] + [folder_id for folder_id in task.candidate_folder_ids if folder_id != inferred.id]
                        rows.append({
                            "task_id": task.task_id, "ranked_folder_ids": ordered, "abstained": False,
                            "score": 0.90, "margin": 1.0, "candidate_scores": {inferred.id: 0.90},
                            "model": "dual_taxonomy_user_example_extension", "checkpoint": str(Path(args.checkpoint).resolve()),
                            "split": requested.value,
                        })
                        for _ in range(count):
                            offset += len(groups[group_offset])
                            group_offset += 1
                        continue
                folder_groups = []
                for _ in range(count):
                    group = groups[group_offset]
                    folder_groups.append(folders[offset:offset + len(group)])
                    offset += len(group)
                    group_offset += 1
                scale = model.logit_scale.exp().clamp(max=100.0)
                logits = torch.stack([(scale * (vector @ group.T)).max() for group in folder_groups])
                probabilities = torch.softmax(logits, dim=0).cpu().tolist()
                candidate_ids = tuple(task.candidate_folder_ids) + ((VIRTUAL_ABSTAIN_FOLDER_ID,) if virtual_abstain else ())
                scored = sorted(zip(candidate_ids, probabilities), key=lambda item: item[1], reverse=True)
                virtual_score = next((score for folder_id, score in scored if folder_id == VIRTUAL_ABSTAIN_FOLDER_ID), None)
                ranked = [(folder_id, score) for folder_id, score in scored if folder_id != VIRTUAL_ABSTAIN_FOLDER_ID]
                score = ranked[0][1]
                margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
                abstained = (
                    virtual_score - score >= margin_threshold
                    if virtual_abstain else should_abstain(score, margin, (score_threshold, margin_threshold))
                )
                rows.append({
                    "task_id": task.task_id,
                    "ranked_folder_ids": [folder_id for folder_id, _ in ranked],
                    "abstained": abstained,
                    "score": score,
                    "margin": margin,
                    "candidate_scores": dict(ranked),
                    "model": "dual_taxonomy_virtual_abstain" if virtual_abstain else "dual_taxonomy",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "split": requested.value,
                    "virtual_abstain_score": virtual_score,
                })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} {requested.value} dual-encoder predictions to {output}")


if __name__ == "__main__":
    main()

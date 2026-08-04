"""Development-only, source-disjoint evaluation for taxonomy discovery.

This script evaluates only episode IDs held out inside a slot-model checkpoint.
It is not a locked-test evaluator and refuses test-split rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics.taxonomy_induction import evaluate_induction
from core.discovery_text import format_discovery_file_record
from core.models.taxonomy_inducer import SEMANTIC_SLOT_INPUT_FORMATS, SlotTaxonomyInducer, load_slot_inducer_state
from core.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit
from core.research.taxonomy_episode import episodes_from_tasks
from core.taxonomy.slot_discovery import MiniLMFileEmbedder, slot_model_config


def clusters_from_slots(episode, vectors, model, *, threshold: float, min_cluster_size: int = 3, max_cluster_fraction: float = 0.65) -> dict[str, list[str]]:
    with torch.no_grad():
        out = model(vectors.unsqueeze(0))
    assignments = out["assignment_logits"][0].argmax(dim=-1).cpu().tolist()
    active = torch.sigmoid(out["existence_logits"][0]).cpu().tolist()
    groups: dict[str, list[str]] = {}
    ids = [task.file_id for task in episode.non_abstain_tasks]
    for slot in sorted(set(assignments)):
        members = [file_id for file_id, assigned in zip(ids, assignments) if assigned == slot]
        if active[slot] >= threshold and min_cluster_size <= len(members) <= len(ids) * max_cluster_fraction:
            groups[f"slot-{slot}"] = members
    return groups


def clusters_from_dense_dbscan(episode, vectors, *, eps: float = 0.45, min_cluster_size: int = 3) -> dict[str, list[str]]:
    labels = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="cosine").fit_predict(vectors.detach().cpu().numpy())
    groups: dict[str, list[str]] = {}
    for task, label in zip(episode.non_abstain_tasks, labels):
        if label >= 0:
            groups.setdefault(f"dense-{label}", []).append(task.file_id)
    return groups


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    fields = ("proposal_count", "oracle_count", "count_absolute_error", "coverage", "purity", "b3_precision", "b3_recall", "b3_f1")
    return {field: sum(row[field] for row in rows) / len(rows) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = BenchmarkManifest.read_jsonl(args.manifest)
    if any(task.split == DatasetSplit.TEST for task in manifest.tasks):
        raise SystemExit("development induction evaluator refuses test-split rows")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("input_format") not in SEMANTIC_SLOT_INPUT_FORMATS:
        raise SystemExit("checkpoint is not trained on semantic-only discovery input")
    requested_ids = set(checkpoint.get("validation_episode_ids", ()))
    episodes = [episode for episode in episodes_from_tasks(manifest.tasks) if episode.episode_id in requested_ids]
    if not episodes or len(episodes) != len(requested_ids):
        raise SystemExit("checkpoint validation episodes do not match this manifest")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)
    model = SlotTaxonomyInducer(**slot_model_config(checkpoint)).to(device).eval()
    load_slot_inducer_state(model, checkpoint)
    embedder = MiniLMFileEmbedder(device=args.device)
    learned, dense = [], []
    for episode in episodes:
        texts = [format_discovery_file_record(task.file_state) for task in episode.non_abstain_tasks]
        vectors = embedder(texts)
        learned.append(asdict(evaluate_induction(episode, clusters_from_slots(episode, vectors, model, threshold=float(checkpoint["existence_threshold"])))))
        dense.append(asdict(evaluate_induction(episode, clusters_from_dense_dbscan(episode, vectors))))
    report = {
        "purpose": "development-only source-disjoint taxonomy-induction comparison",
        "episode_ids": [episode.episode_id for episode in episodes],
        "learned_slot": {"per_episode": learned, "macro_average": aggregate(learned)},
        "dense_dbscan": {"per_episode": dense, "macro_average": aggregate(dense)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "learned_slot": report["learned_slot"]["macro_average"], "dense_dbscan": report["dense_dbscan"]["macro_average"]}, sort_keys=True))


if __name__ == "__main__":
    main()

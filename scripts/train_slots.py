"""Train/evaluate collection-level latent-slot taxonomy induction on train-only episodes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.dual_taxonomy import mean_pool
from core.discovery_text import format_discovery_file_record
from core.models.hf_load import from_pretrained_cached
from core.models.taxonomy_inducer import (
    SlotTaxonomyInducer,
    calibrate_existence_threshold,
    permutation_invariant_assignment_accuracy,
    permutation_invariant_induction_loss,
)
from core.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit
from core.research.taxonomy_episode import episodes_from_tasks


def embed(encoder, tokenizer, episode, device):
    encoded = tokenizer([format_discovery_file_record(task.file_state) for task in episode.non_abstain_tasks], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        states = encoder(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device)).last_hidden_state
        vectors = mean_pool(states, encoded["attention_mask"].to(device))
    folders = {folder_id: index for index, folder_id in enumerate(sorted(set(episode.oracle_assignments.values())))}
    labels = torch.tensor([folders[task.acceptable_folder_ids[0]] for task in episode.non_abstain_tasks], device=device)
    return vectors.unsqueeze(0), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--validation-episode-id", action="append", default=[], help="Explicit held-out episode id; repeat to preserve a validation split across corpus expansions")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.validation_episodes <= 0:
        parser.error("epochs and validation-episodes must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tasks = [task for task in BenchmarkManifest.read_jsonl(args.manifest).tasks if task.split == DatasetSplit.TRAIN]
    episodes = [episode for episode in episodes_from_tasks(tasks) if len(episode.non_abstain_tasks) >= 4]
    if len(episodes) <= args.validation_episodes:
        raise SystemExit("need more source-disjoint episodes than validation episodes")
    if args.validation_episode_id:
        requested = set(args.validation_episode_id)
        validation = [episode for episode in episodes if episode.episode_id in requested]
        if len(validation) != len(requested):
            missing = requested - {episode.episode_id for episode in validation}
            raise SystemExit(f"unknown validation episode ids: {sorted(missing)}")
        train = [episode for episode in episodes if episode.episode_id not in requested]
    else:
        random.shuffle(episodes)
        validation, train = episodes[:args.validation_episodes], episodes[args.validation_episodes:]
    tokenizer = from_pretrained_cached(AutoTokenizer.from_pretrained, "sentence-transformers/all-MiniLM-L6-v2")
    encoder = from_pretrained_cached(AutoModel.from_pretrained, "sentence-transformers/all-MiniLM-L6-v2").to(device).eval()
    for parameter in encoder.parameters(): parameter.requires_grad = False
    cached = {episode.episode_id: embed(encoder, tokenizer, episode, device) for episode in episodes}
    model = SlotTaxonomyInducer(cached[train[0].episode_id][0].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    history, best, best_state = [], float("inf"), None
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train); model.train(); losses=[]
        for episode in train:
            vectors, labels = cached[episode.episode_id]; optimizer.zero_grad()
            out = model(vectors); loss = permutation_invariant_induction_loss(out["assignment_logits"][0], out["existence_logits"][0], labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_outputs = [(model(cached[e.episode_id][0]), cached[e.episode_id][1]) for e in validation]
            validation_loss = float(np.mean([permutation_invariant_induction_loss(out["assignment_logits"][0], out["existence_logits"][0], labels).item() for out, labels in validation_outputs]))
            validation_accuracy = float(np.mean([permutation_invariant_assignment_accuracy(out["assignment_logits"][0], labels) for out, labels in validation_outputs]))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":validation_loss,"validation_assignment_accuracy":validation_accuracy})
        if validation_loss < best:
            best=validation_loss; best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        threshold_examples = [
            (model(cached[episode.episode_id][0])["existence_logits"][0], cached[episode.episode_id][1])
            for episode in validation
        ]
    existence_threshold = calibrate_existence_threshold(threshold_examples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":best_state,"history":history,"existence_threshold":existence_threshold,"input_format":"discovery_semantic_only_v3_scaled","model_config":{"input_dim":cached[train[0].episode_id][0].shape[-1],"hidden_dim":model.file_projection[0].out_features,"max_slots":model.max_slots,"heads":4},"validation_episode_ids":[e.episode_id for e in validation],"train_episode_ids":[e.episode_id for e in train]},args.out)
    print(json.dumps({"episodes": len(episodes), "train": len(train), "validation": len(validation), "best_validation_loss": best, "existence_threshold": existence_threshold}))

if __name__ == "__main__": main()

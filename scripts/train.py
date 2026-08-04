"""Train a retrieval-native open-taxonomy dual encoder on train tasks only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.models.dual_taxonomy import (
    DualTaxonomyEncoder,
    SEMANTIC_INPUT_FORMAT,
    VIRTUAL_ABSTAIN_FOLDER_ID,
    hard_negative_ranking_loss,
    task_batches,
    virtual_abstain_prototype,
)
from core.models.hf_load import from_pretrained_cached
from core.discovery_text import format_semantic_file_record
from core.models.taxonomy_scorer import folder_prototype_texts, format_folder_spec
from core.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def encode(model, tokenizer, texts, role, device):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    return model.encode(encoded["input_ids"].to(device), encoded["attention_mask"].to(device), role=role)


def batch_loss(
    model, tokenizer, tasks, device, *, ranking_weight: float = 0.0,
    folder_prototypes: bool = False, virtual_abstain: bool = False,
):
    file_texts = [format_semantic_file_record(task.file_state) for task in tasks]
    folder_groups = []
    for task in tasks:
        folder_groups.extend(
            folder_prototype_texts(task.taxonomy.folder(folder_id)) if folder_prototypes
            else (format_folder_spec(task.taxonomy.folder(folder_id)),)
            for folder_id in task.candidate_folder_ids
        )
        if virtual_abstain:
            folder_groups.append((virtual_abstain_prototype(),))
    folder_texts = [text for group in folder_groups for text in group]
    files = encode(model, tokenizer, file_texts, "file", device)
    folders = encode(model, tokenizer, folder_texts, "folder", device)
    losses, offset, group_offset = [], 0, 0
    for task, file_vector in zip(tasks, files):
        real_count = len(task.candidate_folder_ids)
        count = real_count + int(virtual_abstain)
        groups = []
        for _ in range(count):
            group = folder_groups[group_offset]
            groups.append(folders[offset:offset + len(group)])
            offset += len(group)
            group_offset += 1
        if task.abstain and not virtual_abstain:
            continue
        scale = model.logit_scale.exp().clamp(max=100.0)
        logits = torch.stack([(scale * (file_vector @ group.T)).max() for group in groups])
        candidate_ids = tuple(task.candidate_folder_ids) + ((VIRTUAL_ABSTAIN_FOLDER_ID,) if virtual_abstain else ())
        positives = (VIRTUAL_ABSTAIN_FOLDER_ID,) if task.abstain else task.acceptable_folder_ids
        positive = torch.tensor(
            [index for index, folder_id in enumerate(candidate_ids) if folder_id in positives],
            dtype=torch.long, device=device,
        )
        destination_loss = -F.log_softmax(logits, dim=0)[positive].mean()
        ranking_loss = hard_negative_ranking_loss(
            logits[:real_count], task.candidate_folder_ids, task.acceptable_folder_ids, task.label_ranking,
        ) if not task.abstain else logits.sum() * 0.0
        losses.append(destination_loss + ranking_weight * ranking_loss)
    return torch.stack(losses).mean() if losses else files.sum() * 0.0


def evaluate(
    model, tokenizer, tasks, device, batch_size, *, ranking_weight: float = 0.0,
    folder_prototypes: bool = False, virtual_abstain: bool = False,
):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in task_batches(tasks, batch_size):
            values.append(float(batch_loss(
                model, tokenizer, batch, device, ranking_weight=ranking_weight,
                folder_prototypes=folder_prototypes, virtual_abstain=virtual_abstain,
            ).item()))
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--out", default="artifacts/dual_taxonomy.pt")
    parser.add_argument(
        "--init-checkpoint",
        help="Optional semantic dual checkpoint for local development fine-tuning; test tasks remain unread.",
    )
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4, help="Tasks per batch, not candidate pairs")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--adapter-learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--hard-negative-ranking-weight", type=float, default=0.0,
        help="Weight teacher-ranked close-negative loss; 0 reproduces the semantic-dual baseline.",
    )
    parser.add_argument(
        "--folder-prototypes", action="store_true",
        help="Pool a folder definition with each separate user-provided representative example.",
    )
    parser.add_argument(
        "--freeze-transformer", action="store_true",
        help="Train only role adapters and temperature; prevents backbone overfitting on small KD sets.",
    )
    parser.add_argument(
        "--virtual-abstain", action="store_true",
        help="Train an explicit taxonomy-relative none-of-these-folders candidate on abstain rows.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size) <= 0
        or min(args.learning_rate, args.adapter_learning_rate) <= 0
        or args.hard_negative_ranking_weight < 0
    ):
        parser.error("epochs, batch-size, and learning rates must be positive; ranking weight cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = BenchmarkManifest.read_jsonl(args.manifest)
    train = [task for task in manifest.tasks if task.split == DatasetSplit.TRAIN]
    validation = [task for task in manifest.tasks if task.split == DatasetSplit.VALIDATION]
    if not train or not validation:
        raise SystemExit("manifest must contain train and validation; locked test is never used here")
    if args.folder_prototypes and not any(folder.examples for task in train for folder in task.taxonomy.folders):
        raise SystemExit("--folder-prototypes requires train taxonomies with representative folder examples")
    device = device_for(args.device)
    tokenizer = from_pretrained_cached(AutoTokenizer.from_pretrained, args.base_model)
    model = DualTaxonomyEncoder(args.base_model).to(device)
    initialized_from = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("input_format") != SEMANTIC_INPUT_FORMAT:
            raise SystemExit("--init-checkpoint must be a semantic dual checkpoint")
        if checkpoint.get("base_model") != args.base_model:
            raise SystemExit("--init-checkpoint base model does not match --base-model")
        model.load_state_dict(checkpoint["state_dict"])
        initialized_from = str(Path(args.init_checkpoint).resolve())
    if args.freeze_transformer:
        for parameter in model.transformer.parameters():
            parameter.requires_grad = False
    adapter_ids = {id(parameter) for parameter in list(model.file_adapter.parameters()) + list(model.folder_adapter.parameters()) + [model.logit_scale]}
    encoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in adapter_ids and parameter.requires_grad
    ]
    adapter_parameters = [parameter for parameter in model.parameters() if id(parameter) in adapter_ids]
    parameter_groups = [{"params": adapter_parameters, "lr": args.adapter_learning_rate}]
    if encoder_parameters:
        parameter_groups.insert(0, {"params": encoder_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=0.01,
    )
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        model.train()
        if args.freeze_transformer:
            model.transformer.eval()
        losses = []
        for batch in task_batches(train, args.batch_size):
            optimizer.zero_grad()
            loss = batch_loss(
                model, tokenizer, batch, device,
                ranking_weight=args.hard_negative_ranking_weight,
                folder_prototypes=args.folder_prototypes,
                virtual_abstain=args.virtual_abstain,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation_loss = evaluate(
            model, tokenizer, validation, device, args.batch_size,
            ranking_weight=args.hard_negative_ranking_weight,
            folder_prototypes=args.folder_prototypes,
            virtual_abstain=args.virtual_abstain,
        )
        row = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)), "validation_loss": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"base_model": args.base_model, "state_dict": best_state, "history": history, "validation_loss": best_loss, "freeze_transformer": args.freeze_transformer, "hard_negative_ranking_weight": args.hard_negative_ranking_weight, "folder_prototypes": args.folder_prototypes, "virtual_abstain": args.virtual_abstain, "input_format": SEMANTIC_INPUT_FORMAT, "manifest": str(Path(args.manifest).resolve(),), "initialized_from": initialized_from}, args.out)
    print(f"saved best validation checkpoint to {args.out}")


if __name__ == "__main__":
    main()

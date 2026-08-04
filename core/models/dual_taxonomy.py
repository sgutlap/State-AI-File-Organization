"""Retrieval-native, taxonomy-conditioned dual encoder.

The model independently embeds a file state and a candidate folder definition.
It therefore has no fixed output taxonomy and starts from a usable zero-shot
retrieval representation rather than a random classification head.
"""

from __future__ import annotations

import math
from pathlib import PurePath
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from core.extractors.file_scanner import FileState
from core.models.hf_load import from_pretrained_cached
from core.discovery_text import format_semantic_file_record
from core.models.taxonomy_scorer import RankedFolder, folder_prototype_texts, format_folder_spec
from core.taxonomy.spec import FolderSpec, TaxonomySpec


VIRTUAL_ABSTAIN_FOLDER_ID = "__state_ai_abstain__"


def virtual_abstain_prototype() -> str:
    """Taxonomy-relative null option used only by explicit abstention experiments."""
    return (
        "Folder: None of these folders\n"
        "ID: __state_ai_abstain__\n"
        "Description: This file does not belong in any user-defined folder in this taxonomy."
    )


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weighted = hidden * attention_mask.unsqueeze(-1)
    return weighted.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)


class DualTaxonomyEncoder(nn.Module):
    """Shared semantic encoder with role-specific residual adapters."""

    def __init__(self, base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.transformer = from_pretrained_cached(AutoModel.from_pretrained, base_model_name)
        hidden = self.transformer.config.hidden_size
        self.file_adapter = nn.Linear(hidden, hidden, bias=False)
        self.folder_adapter = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.file_adapter.weight)
        nn.init.zeros_(self.folder_adapter.weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, role: str) -> torch.Tensor:
        pooled = mean_pool(self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state, attention_mask)
        adapter = self.file_adapter if role == "file" else self.folder_adapter
        return F.normalize(pooled + adapter(pooled), p=2, dim=1)

    def similarity(self, file_vectors: torch.Tensor, folder_vectors: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0) * (file_vectors * folder_vectors).sum(dim=-1)

class DualTaxonomyScorer:
    """Inference wrapper for the retrieval-native open-taxonomy checkpoint.

    It encodes a file once and compares it against every editable folder
    specification. It uses an ambiguity margin, rather than an absolute
    top-probability cutoff that would change meaning with taxonomy size.
    """

    def __init__(
        self,
        model: DualTaxonomyEncoder,
        *,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        score_threshold: float = 0.0,
        margin_threshold: float = 0.10,
        folder_prototypes: bool = False,
        example_extension_fallback: bool = False,
        virtual_abstain: bool = False,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= margin_threshold < 1.0:
            raise ValueError("margin_threshold must be in [0, 1)")
        self.device = torch.device(device)
        self.tokenizer = from_pretrained_cached(AutoTokenizer.from_pretrained, base_model_name)
        self.model = model.to(self.device).eval()
        self.score_threshold = score_threshold
        self.margin_threshold = margin_threshold
        self.folder_prototypes = folder_prototypes
        self.example_extension_fallback = example_extension_fallback
        self.virtual_abstain = virtual_abstain
        self._folder_vector_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self._folder_prototype_vector_cache: dict[tuple[tuple[str, ...], ...], tuple[torch.Tensor, ...]] = {}

    def _encode(self, texts: Sequence[str], *, role: str) -> torch.Tensor:
        encoded = self.tokenizer(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            return self.model.encode(
                encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device), role=role,
            )

    def _folder_vectors(self, folders: Sequence[FolderSpec]) -> torch.Tensor:
        texts = tuple(format_folder_spec(folder) for folder in folders)
        cached = self._folder_vector_cache.get(texts)
        if cached is None:
            cached = self._encode(texts, role="folder").detach()
            self._folder_vector_cache[texts] = cached
        return cached

    def clear_taxonomy_cache(self) -> None:
        """Use after a caller edits folder definitions in place."""
        self._folder_vector_cache.clear()
        getattr(self, "_folder_prototype_vector_cache", {}).clear()

    def _folder_vector_groups(self, folders: Sequence[FolderSpec]) -> tuple[torch.Tensor, ...]:
        """Encode definition/example alternatives once, preserving folder order."""
        groups = tuple(folder_prototype_texts(folder) for folder in folders)
        cached = self._folder_prototype_vector_cache.get(groups)
        if cached is None:
            flat = [text for group in groups for text in group]
            vectors = self._encode(flat, role="folder").detach()
            parts, offset = [], 0
            for group in groups:
                parts.append(vectors[offset:offset + len(group)])
                offset += len(group)
            cached = tuple(parts)
            self._folder_prototype_vector_cache[groups] = cached
        return cached

    def _candidate_logits(self, file_vectors: torch.Tensor, folders: Sequence[FolderSpec]) -> torch.Tensor:
        """Return one logit per folder, optionally max-pooling user examples."""
        scale = self.model.logit_scale.exp().clamp(max=100.0)
        if not self.folder_prototypes:
            logits = scale * (file_vectors @ self._folder_vectors(folders).T)
        else:
            logits = torch.stack([
            (scale * (file_vectors @ group.T)).max(dim=1).values
            for group in self._folder_vector_groups(folders)
            ], dim=1)
        if not getattr(self, "virtual_abstain", False):
            return logits
        null_vector = self._encode([virtual_abstain_prototype()], role="folder")
        return torch.cat((logits, scale * (file_vectors @ null_vector.T)), dim=1)

    def score_record(self, record: Mapping, folders: Sequence[FolderSpec]) -> list[tuple[str, float]]:
        if not folders:
            return []
        with torch.no_grad():
            file_vector = self._encode([format_semantic_file_record(record)], role="file")
            logits = self._candidate_logits(file_vector, folders).squeeze(0)
            probabilities = torch.softmax(logits, dim=0)
        ids = tuple(folder.id for folder in folders) + ((VIRTUAL_ABSTAIN_FOLDER_ID,) if getattr(self, "virtual_abstain", False) else ())
        return sorted(zip(ids, probabilities.cpu().tolist()), key=lambda pair: pair[1], reverse=True)

    def score(self, state: FileState, folders: Sequence[FolderSpec]) -> list[tuple[str, float]]:
        return self.score_record({
            "relative_path": state.relative_path,
            "metadata": {
                "filename": state.metadata.filename,
                "extension": state.metadata.extension,
                "size_bytes": state.metadata.size_bytes,
                "age_days": state.metadata.age_days,
                "mime_type": state.metadata.mime_type,
                "is_binary": state.metadata.is_binary,
            },
            "content_sample": state.content_sample,
        }, folders)

    def score_many(self, states: Sequence[FileState], folders: Sequence[FolderSpec]) -> list[list[tuple[str, float]]]:
        """Score a batch against one editable taxonomy without changing ranking semantics."""
        if not states or not folders:
            return [[] for _ in states]
        records = [{
            "relative_path": state.relative_path,
            "metadata": {
                "filename": state.metadata.filename,
                "extension": state.metadata.extension,
                "size_bytes": state.metadata.size_bytes,
                "age_days": state.metadata.age_days,
                "mime_type": state.metadata.mime_type,
                "is_binary": state.metadata.is_binary,
            },
            "content_sample": state.content_sample,
        } for state in states]
        with torch.no_grad():
            file_vectors = self._encode([format_semantic_file_record(record) for record in records], role="file")
            logits = self._candidate_logits(file_vectors, folders)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
        ids = tuple(folder.id for folder in folders) + ((VIRTUAL_ABSTAIN_FOLDER_ID,) if getattr(self, "virtual_abstain", False) else ())
        return [sorted(zip(ids, values), key=lambda pair: pair[1], reverse=True) for values in probabilities]

    @staticmethod
    def _declared_binary_folder(state: FileState, taxonomy: TaxonomySpec) -> FolderSpec | None:
        """Return one explicit user-declared type destination for opaque files.

        This is a bounded planner fallback, not a learned fixed-taxonomy rule:
        it is available only when the current editable taxonomy names exactly
        one matching ``extension:`` constraint, and it is never used when a
        file supplies meaningful semantic text.
        """
        sample = (state.content_sample or "").strip()
        opaque = (
            sample.startswith("[Binary File:")
            or sample.startswith("[Image File:")
            or sample.startswith("[Archive File:")
        )
        if not opaque:
            return None
        path = PurePath(state.metadata.filename or "")
        suffixes = []
        while path.suffix:
            suffixes.append(path.suffix.lower())
            path = PurePath(path.stem)
        candidates = {state.metadata.extension.lower(), "".join(reversed(suffixes))}
        matched = [
            folder for folder in taxonomy.folders
            if any(
                str(constraint).lower() == f"extension:{suffix}"
                for constraint in folder.constraints for suffix in candidates if suffix
            )
        ]
        return matched[0] if len(matched) == 1 else None

    @staticmethod
    def _example_extension_folder(state: FileState, taxonomy: TaxonomySpec) -> FolderSpec | None:
        """Use an opaque-file type only when user examples make it unambiguous.

        This is deliberately a separately reported policy fallback, not a
        global extension classifier: the current editable taxonomy supplies
        the evidence and an extension occurring in two folders abstains.
        """
        sample = (state.content_sample or "").strip()
        if not (sample.startswith("[Binary File:") or sample.startswith("[Image File:") or sample.startswith("[Archive File:")):
            return None
        extension = state.metadata.extension.lower()
        if not extension:
            return None
        owners = [
            folder for folder in taxonomy.folders
            if any(PurePath(example).suffix.lower() == extension for example in folder.examples)
        ]
        return owners[0] if len(owners) == 1 else None

    def rank(self, state: FileState, taxonomy: TaxonomySpec) -> RankedFolder:
        declared = self._declared_binary_folder(state, taxonomy)
        if declared is not None:
            return RankedFolder(folder_id=declared.id, score=0.95, margin=1.0, abstained=False, source="user_extension_constraint")
        if getattr(self, "example_extension_fallback", False):
            inferred = self._example_extension_folder(state, taxonomy)
            if inferred is not None:
                return RankedFolder(inferred.id, 0.90, 1.0, False, "user_example_extension")
        ranked = self.score(state, taxonomy.folders)
        return self._rank_from_scores(ranked)

    def _rank_from_scores(self, ranked: Sequence[tuple[str, float]]) -> RankedFolder:
        if not ranked:
            return RankedFolder(None, 0.0, 0.0, True, "semantic_dual")
        null_score = next((score for folder_id, score in ranked if folder_id == VIRTUAL_ABSTAIN_FOLDER_ID), None)
        real = [(folder_id, score) for folder_id, score in ranked if folder_id != VIRTUAL_ABSTAIN_FOLDER_ID]
        if not real:
            return RankedFolder(None, float(null_score or 0.0), 0.0, True, "semantic_dual_virtual_abstain")
        folder_id, score = real[0]
        if null_score is not None:
            virtual_margin = null_score - score
            abstained = virtual_margin >= self.margin_threshold
            return RankedFolder(
                None if abstained else folder_id,
                null_score if abstained else score,
                virtual_margin if abstained else score - null_score,
                abstained,
                "semantic_dual_virtual_abstain",
            )
        runner_up = max((value for _, value in real[1:]), default=0.0)
        margin = score - runner_up
        abstained = score < self.score_threshold or margin < self.margin_threshold
        return RankedFolder(
            None if abstained else folder_id, score, margin, abstained,
            "semantic_dual_virtual_abstain" if getattr(self, "virtual_abstain", False) else "semantic_dual",
        )

    def rank_many(self, states: Sequence[FileState], taxonomy: TaxonomySpec) -> list[RankedFolder]:
        """Batch semantic encodings while retaining user constraints and abstention."""
        choices: list[RankedFolder | None] = [None] * len(states)
        undecided = []
        positions = []
        for index, state in enumerate(states):
            declared = self._declared_binary_folder(state, taxonomy)
            if declared is not None:
                choices[index] = RankedFolder(declared.id, 0.95, 1.0, False, "user_extension_constraint")
            elif getattr(self, "example_extension_fallback", False) and (inferred := self._example_extension_folder(state, taxonomy)) is not None:
                choices[index] = RankedFolder(inferred.id, 0.90, 1.0, False, "user_example_extension")
            else:
                undecided.append(state)
                positions.append(index)
        for index, ranked in zip(positions, self.score_many(undecided, taxonomy.folders)):
            choices[index] = self._rank_from_scores(ranked)
        return [choice for choice in choices if choice is not None]


def checkpoint_backend(checkpoint: Mapping) -> str:
    """Identify an explicit State-AI scorer checkpoint without guessing by name."""
    state = checkpoint.get("state_dict", {})
    if {"file_adapter.weight", "folder_adapter.weight"}.issubset(state):
        return "dual"
    if any(key.startswith("head.") for key in state):
        return "pair"
    raise ValueError("checkpoint is neither a dual-taxonomy nor a legacy pair-scorer checkpoint")


SEMANTIC_INPUT_FORMAT = "semantic_filename_content_v1"


def assert_semantic_dual_checkpoint(checkpoint: Mapping) -> None:
    """Reject legacy dual checkpoints trained with extension/path metadata.

    A checkpoint cannot become semantic merely because the runtime formatter was
    changed.  Requiring the recorded format prevents silent train/inference
    mismatch and makes extension-ablation claims auditable.
    """
    if checkpoint.get("input_format") != SEMANTIC_INPUT_FORMAT:
        raise ValueError(
            "dual checkpoint is not semantic-only; retrain with the current "
            "train.py before using it for open-taxonomy organization"
        )


def task_batches(tasks: Sequence, batch_size: int) -> Iterable[Sequence]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(tasks), batch_size):
        yield tasks[start:start + batch_size]


def hard_negative_ranking_loss(
    logits: torch.Tensor,
    candidate_folder_ids: Sequence[str],
    acceptable_folder_ids: Sequence[str],
    label_ranking: Sequence[str],
) -> torch.Tensor:
    """Emphasize teacher-ranked alternatives closest to the chosen folder.

    Standard within-taxonomy cross-entropy already compares the chosen folder
    with every candidate.  This auxiliary loss makes the informative close
    alternatives matter more: a teacher's second-ranked folder has weight 1,
    third-ranked 1/2, and so on.  It deliberately uses order rather than raw
    teacher probabilities, which are not calibrated across teacher models.
    """
    if logits.ndim != 1 or len(logits) != len(candidate_folder_ids):
        raise ValueError("logits must align with the candidate folder ids")
    if not label_ranking or not acceptable_folder_ids:
        return logits.sum() * 0.0
    if set(label_ranking) != set(candidate_folder_ids) or len(label_ranking) != len(candidate_folder_ids):
        raise ValueError("teacher ranking must contain every candidate exactly once")
    positions = {folder_id: index for index, folder_id in enumerate(label_ranking)}
    index = {folder_id: offset for offset, folder_id in enumerate(candidate_folder_ids)}
    positives = [folder_id for folder_id in acceptable_folder_ids if folder_id in index]
    if not positives:
        return logits.sum() * 0.0
    terms, weights = [], []
    for positive in positives:
        positive_rank = positions[positive]
        for negative in candidate_folder_ids:
            if negative in positives:
                continue
            # A ranking that places an alternative above the accepted folder is
            # inconsistent for a destination task; retain it with unit weight
            # rather than inventing a stronger teacher preference.
            distance = max(1, positions[negative] - positive_rank)
            terms.append(F.softplus(-(logits[index[positive]] - logits[index[negative]])))
            weights.append(1.0 / distance)
    if not terms:
        return logits.sum() * 0.0
    weight_tensor = torch.tensor(weights, dtype=logits.dtype, device=logits.device)
    return (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum()

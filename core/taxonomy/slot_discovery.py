"""Bridge collection slot assignments to confirmation-required taxonomy proposals."""

from __future__ import annotations

import re
from typing import Callable, Mapping

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from core.extractors.file_scanner import FileState
from core.discovery_text import format_discovery_file_record
from core.models.dual_taxonomy import mean_pool
from core.models.hf_load import from_pretrained_cached
from core.models.taxonomy_inducer import SEMANTIC_SLOT_INPUT_FORMATS, SlotTaxonomyInducer, load_slot_inducer_state
from core.taxonomy.spec import FolderSpec, TaxonomyEditType, TaxonomyProposal, TaxonomySpec


class SlotTaxonomyDiscoverer:
    """Convert learned collection clusters into bounded ADD proposals only."""

    def __init__(self, model: SlotTaxonomyInducer, embedder: Callable[[list[str]], torch.Tensor], *, min_cluster_size: int = 3, existence_threshold: float = 0.5, max_proposals: int = 5, max_cluster_fraction: float = 0.65, min_semantic_fraction: float = 0.60):
        self.model, self.embedder = model, embedder
        if min_cluster_size < 2 or max_proposals <= 0 or not 0 < max_cluster_fraction <= 1 or not 0 <= min_semantic_fraction <= 1:
            raise ValueError("invalid slot-discovery configuration")
        self.min_cluster_size, self.existence_threshold, self.max_proposals = min_cluster_size, existence_threshold, max_proposals
        self.max_cluster_fraction = max_cluster_fraction
        self.min_semantic_fraction = min_semantic_fraction

    def discover(self, unresolved_states: list[FileState], taxonomy: TaxonomySpec):
        texts = {
            state.file_id: format_discovery_file_record({
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
            })
            for state in unresolved_states
        }
        semantic_ids = {
            state.file_id for state in unresolved_states
            if len(state.content_sample.strip()) >= 80
            and not state.content_sample.startswith("[Binary File:")
            and not state.content_sample.startswith("[Empty File]")
        }
        return self.discover_texts(texts, taxonomy, semantic_ids=semantic_ids)

    def discover_texts(self, unresolved_texts: Mapping[str, str], taxonomy: TaxonomySpec, *, semantic_ids: set[str] | None = None):
        ids, texts = list(unresolved_texts), list(unresolved_texts.values())
        if len(ids) < self.min_cluster_size:
            return []
        with torch.no_grad():
            out = self.model(self.embedder(texts).unsqueeze(0))
        assignments = out["assignment_logits"][0].argmax(dim=-1).cpu().tolist()
        existence = torch.sigmoid(out["existence_logits"][0]).cpu().tolist()
        groups: dict[int, list[str]] = {}
        for file_id, slot in zip(ids, assignments): groups.setdefault(slot, []).append(file_id)
        proposals = []
        existing = {folder.id for folder in taxonomy.folders}
        for slot, members in groups.items():
            if (
                len(members) < self.min_cluster_size
                or len(members) / len(ids) > self.max_cluster_fraction
                or existence[slot] < self.existence_threshold
                or (semantic_ids is not None and sum(file_id in semantic_ids for file_id in members) / len(members) < self.min_semantic_fraction)
            ):
                continue
            keywords = self._keywords([unresolved_texts[file_id] for file_id in members])
            if not keywords:
                continue
            name = "Suggested: " + " / ".join(keywords[:2]).title()
            base_id = "suggested/" + "-".join(keywords[:2])
            folder_id, suffix = base_id, 2
            while folder_id in existing:
                folder_id = f"{base_id}-{suffix}"; suffix += 1
            existing.add(folder_id)
            proposals.append(TaxonomyProposal(
                operation=TaxonomyEditType.ADD,
                proposed_folders=(FolderSpec(folder_id, name, f"Learned collection cluster with {len(members)} files; review before adding.", examples=tuple(members[:5])),),
                affected_files=tuple(members), confidence=round(float(existence[slot]),4), utility=round(float(existence[slot])-0.05,4),
                rationale="Collection-conditioned latent-slot proposal; explicit approval required.",
            ))
        return sorted(proposals, key=lambda proposal: (proposal.utility, proposal.confidence), reverse=True)[:self.max_proposals]

    @staticmethod
    def _keywords(texts: list[str]) -> list[str]:
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=64)
            matrix = vectorizer.fit_transform(texts)
            scores = np.asarray(matrix.mean(axis=0)).ravel()
            terms = vectorizer.get_feature_names_out()
            generic = {"file", "filename", "path", "extension", "size", "bytes", "age", "mime", "binary", "content", "sample", "redacted", "inbox", "true", "false"}
            keywords = [re.sub(r"[^a-z0-9]+", "-", terms[index].lower()).strip("-") for index in scores.argsort()[::-1]]
            return [keyword for keyword in keywords if keyword and keyword not in generic][:3]
        except ValueError:
            return []


class MiniLMFileEmbedder:
    """Frozen file-state embedder matching the inducer training representation."""

    def __init__(self, *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "auto", batch_size: int = 32):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device, self.batch_size = torch.device(device), batch_size
        self.tokenizer = from_pretrained_cached(AutoTokenizer.from_pretrained, model_name)
        self.model = from_pretrained_cached(AutoModel.from_pretrained, model_name).to(self.device).eval()

    def __call__(self, texts: list[str]) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt")
                mask = encoded["attention_mask"].to(self.device)
                states = self.model(input_ids=encoded["input_ids"].to(self.device), attention_mask=mask).last_hidden_state
                chunks.append(mean_pool(states, mask))
        return torch.cat(chunks) if chunks else torch.empty((0, self.model.config.hidden_size), device=self.device)


def slot_model_config(checkpoint: Mapping) -> dict[str, int]:
    """Read architecture metadata, with a safe inference path for old checkpoints."""
    saved = checkpoint.get("model_config")
    if saved:
        return {key: int(saved[key]) for key in ("input_dim", "hidden_dim", "max_slots", "heads")}
    state = checkpoint.get("state_dict", {})
    try:
        projection, slots = state["file_projection.0.weight"], state["slot_queries"]
    except KeyError as error:
        raise ValueError("invalid slot-inducer checkpoint") from error
    return {"input_dim": int(projection.shape[1]), "hidden_dim": int(projection.shape[0]), "max_slots": int(slots.shape[0]), "heads": 4}


def load_slot_discoverer(checkpoint_path: str, *, device: str = "auto", min_cluster_size: int = 3, max_proposals: int = 5) -> SlotTaxonomyDiscoverer:
    """Load an explicit trained slot model; all proposed edits still require approval."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("input_format") not in SEMANTIC_SLOT_INPUT_FORMATS:
        raise ValueError("slot checkpoint predates semantic-only discovery inputs; retrain before use")
    model = SlotTaxonomyInducer(**slot_model_config(checkpoint))
    load_slot_inducer_state(model, checkpoint)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    target = torch.device(device)
    model.to(target).eval()
    return SlotTaxonomyDiscoverer(
        model, MiniLMFileEmbedder(device=str(target)), min_cluster_size=min_cluster_size, max_proposals=max_proposals,
        existence_threshold=float(checkpoint.get("existence_threshold", 0.5)),
    )

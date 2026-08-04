"""Optional visual fallback for opaque image files in an editable taxonomy."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from core.extractors.file_scanner import FileState
from core.models.taxonomy_scorer import RankedFolder
from core.taxonomy.spec import FolderSpec, TaxonomySpec


IMAGE_SUFFIXES = frozenset({".avif", ".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def is_visual_candidate(state: FileState) -> bool:
    """Use suffix only to select a decoder, never to select a destination."""
    return state.metadata.extension.lower() in IMAGE_SUFFIXES and Path(state.absolute_path).is_file()


def visual_folder_text(folder: FolderSpec) -> str:
    """Keep visual scoring tied to user-facing meaning, not IDs or constraints."""
    return f"{folder.name}. {folder.description}"


class CLIPVisualTaxonomyScorer:
    """Use frozen CLIP only when a semantic scorer safely abstains on an image.

    This is intentionally a fallback, not an uncalibrated fusion rule. It
    prevents image bytes from silently becoming a fixed extension taxonomy and
    leaves non-images and undecodable formats with the semantic scorer's safe
    abstention.
    """

    def __init__(
        self,
        semantic_scorer,
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        score_threshold: float = 0.0,
        margin_threshold: float = 0.10,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= margin_threshold < 1.0:
            raise ValueError("visual thresholds must be in [0, 1)")
        from transformers import AutoProcessor, CLIPModel

        from core.models.hf_load import from_pretrained_cached

        self.semantic_scorer = semantic_scorer
        self.device = torch.device(device)
        self.processor = from_pretrained_cached(AutoProcessor.from_pretrained, model_name)
        self.model = from_pretrained_cached(CLIPModel.from_pretrained, model_name).to(self.device).eval()
        self.score_threshold = score_threshold
        self.margin_threshold = margin_threshold
        self._folder_vector_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def _folder_vectors(self, folders: Sequence[FolderSpec]) -> torch.Tensor:
        texts = tuple(visual_folder_text(folder) for folder in folders)
        cached = self._folder_vector_cache.get(texts)
        if cached is None:
            encoded = self.processor(text=list(texts), padding=True, return_tensors="pt")
            with torch.no_grad():
                # Transformers 5 returns a model output from ``get_*_features``
                # for CLIP, while older releases return the projected tensor.
                # Calling public CLIP submodules is stable across both APIs.
                text_output = self.model.text_model(**{key: value.to(self.device) for key, value in encoded.items()})
                cached = self.model.text_projection(text_output.pooler_output)
                cached = torch.nn.functional.normalize(cached, p=2, dim=-1).detach()
            self._folder_vector_cache[texts] = cached
        return cached

    def _visual_scores(self, state: FileState, folders: Sequence[FolderSpec]) -> list[tuple[str, float]]:
        return self._visual_scores_many([state], folders).get(0, [])

    def _visual_scores_many(self, states: Sequence[FileState], folders: Sequence[FolderSpec]) -> dict[int, list[tuple[str, float]]]:
        try:
            from PIL import Image

            images, positions = [], []
            for index, state in enumerate(states):
                try:
                    with Image.open(state.absolute_path) as image:
                        images.append(image.convert("RGB"))
                    positions.append(index)
                except (OSError, ValueError):
                    continue
            if not images:
                return {}
            encoded = self.processor(images=images, return_tensors="pt")
        except (OSError, ValueError):
            return {}
        with torch.no_grad():
            image_output = self.model.vision_model(**{key: value.to(self.device) for key, value in encoded.items()})
            image_vector = self.model.visual_projection(image_output.pooler_output)
            image_vector = torch.nn.functional.normalize(image_vector, p=2, dim=-1)
            logits = self.model.logit_scale.exp().clamp(max=100.0) * (image_vector @ self._folder_vectors(folders).T)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
        return {
            index: sorted(zip((folder.id for folder in folders), values), key=lambda pair: pair[1], reverse=True)
            for index, values in zip(positions, probabilities)
        }

    def rank(self, state: FileState, taxonomy: TaxonomySpec) -> RankedFolder:
        semantic_choice = self.semantic_scorer.rank(state, taxonomy)
        if not semantic_choice.abstained or not is_visual_candidate(state):
            return semantic_choice
        ranked = self._visual_scores(state, taxonomy.folders)
        if not ranked:
            return semantic_choice
        folder_id, score = ranked[0]
        margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
        abstained = score < self.score_threshold or margin < self.margin_threshold
        return RankedFolder(None if abstained else folder_id, score, margin, abstained, source="clip_visual")

    def rank_many(self, states: Sequence[FileState], taxonomy: TaxonomySpec) -> list[RankedFolder]:
        semantic_rank_many = getattr(self.semantic_scorer, "rank_many", None)
        choices = semantic_rank_many(states, taxonomy) if callable(semantic_rank_many) else [self.semantic_scorer.rank(state, taxonomy) for state in states]
        candidates = [state for state, choice in zip(states, choices) if choice.abstained and is_visual_candidate(state)]
        candidate_positions = [index for index, (state, choice) in enumerate(zip(states, choices)) if choice.abstained and is_visual_candidate(state)]
        for relative_index, ranked in self._visual_scores_many(candidates, taxonomy.folders).items():
            folder_id, score = ranked[0]
            margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
            abstained = score < self.score_threshold or margin < self.margin_threshold
            choices[candidate_positions[relative_index]] = RankedFolder(None if abstained else folder_id, score, margin, abstained, "clip_visual")
        return choices

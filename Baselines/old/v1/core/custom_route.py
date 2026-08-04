from __future__ import annotations
import re
from functools import lru_cache
import numpy as np
from core.model import Student
from core.scan import FileState

TAX_BLURBS = {
    "documents/research": "academic research papers reports essays notes homework documents",
    "documents/financial": "finance invoices receipts budgets taxes banking payments bills",
    "code/projects": "source code programming scripts software development projects",
    "data/datasets": "datasets csv spreadsheets tables metrics data exports",
    "media/images": "photos images pictures screenshots graphics design",
    "media/audio_video": "video audio music recordings movies podcasts",
    "archives": "compressed zip rar 7z tar backup package installer dmg iso",
    "misc/uncategorized": "miscellaneous unsorted unknown temporary junk other",
}


@lru_cache(maxsize=1)
def _minilm():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _embed(texts: list[str]) -> np.ndarray:
    return np.asarray(_minilm().encode(texts, normalize_embeddings=True), dtype=np.float32)


def _folder_proto(folder: str, description: str | None = None) -> str:
    name = folder.split("/")[-1]
    if description and description.strip():
        return f"Folder '{name}': {description.strip()}"
    return f"Folder '{name}': files related to {name}"


def _file_text(state: FileState) -> str:
    return (
        f"File {state.metadata.filename} path {state.relative_path} "
        f"ext {state.metadata.extension} content {(state.content_sample or '')[:400]}"
    )


class CustomRouter:
    """Routes files into an arbitrary list of destination folders."""

    def __init__(
        self,
        student: Student,
        folders: list[str],
        *,
        descriptions: dict[str, str] | None = None,
        soft_weight: float = 0.75,
    ):
        folders = [f.strip().replace("\\", "/").strip("/") for f in folders if str(f).strip()]
        if not folders:
            raise ValueError("custom folders list is empty")
        self.student = student
        self.folders = folders
        self.soft_weight = float(soft_weight)
        self._stems = [f.split("/")[-1].lower() for f in folders]
        descs = descriptions or {}
        self._folder_emb = _embed([_folder_proto(f, descs.get(f)) for f in folders])

        tax_ids = list(student.taxonomy.classes)
        self._tax_ids = tax_ids
        tax_texts = [
            f"category {tid}: {TAX_BLURBS.get(tid, tid.replace('/', ' '))}"
            for tid in tax_ids
        ]
        tax_emb = _embed(tax_texts)
        sim = tax_emb @ self._folder_emb.T
        # temperature softmax over folders
        sim = sim / 0.05
        sim = sim - sim.max(axis=1, keepdims=True)
        exp = np.exp(sim)
        self._affinity = exp / exp.sum(axis=1, keepdims=True)

    def map_category(self, category: str) -> str:
        if category in self.folders:
            return category
        misc = self._misc_folder()
        if category in self._tax_ids:
            i = self._tax_ids.index(category)
            row = self._affinity[i]
            j = int(row.argmax())
            # weak projection (common when user folders don't cover this class) → Misc
            if float(row[j]) < 0.66 and misc is not None:
                return misc
            return self.folders[j]
        emb = _embed([_folder_proto(category)])[0]
        scores = emb @ self._folder_emb.T
        j = int(scores.argmax())
        if float(scores[j]) < 0.25 and misc is not None:
            return misc
        return self.folders[j]

    def _misc_folder(self) -> str | None:
        for f, s in zip(self.folders, self._stems):
            if s in {"misc", "other", "inbox", "unsorted", "unknown"}:
                return f
        return None

    def route(
        self,
        state: FileState,
        seed_category: str | None = None,
        *,
        probs: dict[str, float] | None = None,
        file_emb=None,  # unused — MiniLM path
        hard_seed: bool = False,
    ) -> tuple[str, float]:
        # exact folder name in filename/path (true open-vocab signal)
        blob = f"{state.metadata.filename or ''} {state.relative_path or ''}".lower()
        for i, stem in enumerate(self._stems):
            if len(stem) >= 4 and re.search(rf"(?<![a-z0-9]){re.escape(stem)}(?![a-z0-9])", blob):
                return self.folders[i], 0.92

        if hard_seed and seed_category:
            return self.map_category(seed_category), 0.90

        probs = probs if probs is not None else self.student.predict_probs(state)
        top = max(probs, key=probs.get)
        seed = seed_category if seed_category in self._tax_ids else top
        # primary decision = project KD class → folders (Misc if weak coverage)
        primary = self.map_category(seed)
        conf = float(probs.get(seed, probs.get(top, 0.5)))

        misc = self._misc_folder()
        if primary != misc:
            return primary, max(0.55, conf)

        # weak coverage: only leave Misc if direct MiniLM is decisive AND
        # the folder name/theme appears in the file text (avoids Travel/Cooking dumps)
        direct = _embed([_file_text(state)])[0] @ self._folder_emb.T
        j = int(direct.argmax())
        stem = self._stems[j]
        text = _file_text(state).lower()
        if float(direct[j]) >= 0.35 and len(stem) >= 4 and stem in text:
            return self.folders[j], float(direct[j])
        return (misc or primary), max(0.40, conf)


def get_router(
    student: Student,
    folders: list[str],
    descriptions: dict[str, str] | None = None,
) -> CustomRouter:
    return CustomRouter(student, folders, descriptions=descriptions)

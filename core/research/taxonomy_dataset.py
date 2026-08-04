"""Leakage-safe records for taxonomy-conditioned ranking experiments.

This module stores only the redacted state used by the model.  Raw workspaces
stay with their owners; manifest validation prevents a workspace source or file
content hash from crossing train/validation/test boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from core.taxonomy.spec import TaxonomySpec


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class TaxonomyRankingTask:
    """One file against a candidate taxonomy, with an explicit abstain label."""

    task_id: str
    workspace_id: str
    source_group_id: str
    taxonomy_id: str
    split: DatasetSplit
    file_id: str
    content_hash: str
    file_state: Mapping[str, Any]
    taxonomy: TaxonomySpec
    candidate_folder_ids: Tuple[str, ...]
    acceptable_folder_ids: Tuple[str, ...] = ()
    abstain: bool = False
    label_source: str = "human"
    label_ranking: Tuple[str, ...] = ()
    label_confidence: Optional[float] = None
    label_rationale: str = ""

    def __post_init__(self) -> None:
        if not self.task_id or not self.workspace_id or not self.source_group_id or not self.taxonomy_id:
            raise ValueError("task, workspace, source-group, and taxonomy ids are required")
        if not self.file_id or not self.content_hash:
            raise ValueError("file_id and content_hash are required")
        candidates = tuple(self.candidate_folder_ids)
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("candidate folders must be a non-empty unique sequence")
        known = {folder.id for folder in self.taxonomy.folders}
        if not set(candidates).issubset(known):
            raise ValueError("candidate folder is absent from taxonomy")
        if self.abstain and self.acceptable_folder_ids:
            raise ValueError("abstain task cannot have acceptable folders")
        if not self.abstain and not self.acceptable_folder_ids:
            raise ValueError("non-abstain task needs at least one acceptable folder")
        if not set(self.acceptable_folder_ids).issubset(set(candidates)):
            raise ValueError("acceptable folder must be a candidate")
        if self.label_ranking:
            if set(self.label_ranking) != set(candidates) or len(self.label_ranking) != len(candidates):
                raise ValueError("label ranking must contain every candidate folder exactly once")
        if self.label_confidence is not None and not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError("label confidence must be in [0, 1]")
        if not self.file_state.get("relative_path"):
            raise ValueError("file_state must contain a redacted relative_path")
        if self.file_state.get("absolute_path"):
            raise ValueError("manifest file_state must not contain absolute_path")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workspace_id": self.workspace_id,
            "source_group_id": self.source_group_id,
            "taxonomy_id": self.taxonomy_id,
            "split": self.split.value,
            "file_id": self.file_id,
            "content_hash": self.content_hash,
            "file_state": dict(self.file_state),
            "taxonomy": self.taxonomy.to_dict(),
            "candidate_folder_ids": list(self.candidate_folder_ids),
            "acceptable_folder_ids": list(self.acceptable_folder_ids),
            "abstain": self.abstain,
            "label_source": self.label_source,
            "label_ranking": list(self.label_ranking),
            "label_confidence": self.label_confidence,
            "label_rationale": self.label_rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaxonomyRankingTask":
        return cls(
            task_id=str(data["task_id"]),
            workspace_id=str(data["workspace_id"]),
            source_group_id=str(data["source_group_id"]),
            taxonomy_id=str(data["taxonomy_id"]),
            split=DatasetSplit(data["split"]),
            file_id=str(data["file_id"]),
            content_hash=str(data["content_hash"]),
            file_state=dict(data["file_state"]),
            taxonomy=TaxonomySpec.from_dict(dict(data["taxonomy"])),
            candidate_folder_ids=tuple(data["candidate_folder_ids"]),
            acceptable_folder_ids=tuple(data.get("acceptable_folder_ids") or ()),
            abstain=bool(data.get("abstain", False)),
            label_source=str(data.get("label_source", "human")),
            label_ranking=tuple(data.get("label_ranking") or ()),
            label_confidence=(
                float(data["label_confidence"]) if data.get("label_confidence") is not None else None
            ),
            label_rationale=str(data.get("label_rationale", "")),
        )


@dataclass
class BenchmarkManifest:
    """A benchmark split plus auditable provenance invariants."""

    name: str
    tasks: List[TaxonomyRankingTask] = field(default_factory=list)

    def validate(self) -> None:
        source_splits: Dict[str, DatasetSplit] = {}
        hash_splits: Dict[str, DatasetSplit] = {}
        task_ids = set()
        for task in self.tasks:
            if task.task_id in task_ids:
                raise ValueError(f"duplicate task id: {task.task_id}")
            task_ids.add(task.task_id)
            self._check_same_split(source_splits, task.source_group_id, task.split, "source group")
            self._check_same_split(hash_splits, task.content_hash, task.split, "content hash")

    @staticmethod
    def _check_same_split(
        values: Dict[str, DatasetSplit], key: str, split: DatasetSplit, label: str
    ) -> None:
        existing = values.get(key)
        if existing is not None and existing != split:
            raise ValueError(f"{label} crosses splits: {key} ({existing.value}, {split.value})")
        values[key] = split

    def counts(self) -> Dict[str, int]:
        out = {split.value: 0 for split in DatasetSplit}
        for task in self.tasks:
            out[task.split.value] += 1
        return out

    def validate_heldout_taxonomies(self) -> None:
        """Require every locked-test taxonomy id to be absent from train/validation."""
        seen = {
            task.taxonomy_id
            for task in self.tasks
            if task.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
        }
        heldout = {task.taxonomy_id for task in self.tasks if task.split == DatasetSplit.TEST}
        overlap = seen & heldout
        if overlap:
            raise ValueError(f"test taxonomy overlaps train/validation: {sorted(overlap)[:3]}")

    def write_jsonl(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(task.to_dict(), sort_keys=True) + "\n" for task in self.tasks),
            encoding="utf-8",
        )
        return output

    @classmethod
    def read_jsonl(cls, path: str | Path, name: str | None = None) -> "BenchmarkManifest":
        source = Path(path)
        tasks = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                tasks.append(TaxonomyRankingTask.from_dict(json.loads(line)))
            except Exception as error:  # retain line number for data curation
                raise ValueError(f"invalid task at {source}:{line_number}: {error}") from error
        manifest = cls(name or source.stem, tasks)
        manifest.validate()
        return manifest

"""Taxonomy-neutral data contracts used by ranking and discovery.

The legacy YAML taxonomy is a seed, not a model output vocabulary.  These
contracts let a model score arbitrary user folders while keeping every
taxonomy change explicit and reviewable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4


def _folder_id(value: str) -> str:
    value = (value or "").strip().replace("\\", "/").rstrip("/")
    parts = PurePosixPath(value).parts
    if not value or value.startswith("/") or ":" in value or any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"folder id must be a non-empty relative path: {value!r}")
    return value


@dataclass(frozen=True)
class FolderSpec:
    """One candidate destination in a user taxonomy."""

    id: str
    name: str
    description: str
    parent_id: Optional[str] = None
    examples: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _folder_id(self.id))
        if not self.name.strip() or not self.description.strip():
            raise ValueError("folder name and description are required")
        if self.parent_id is not None:
            object.__setattr__(self, "parent_id", _folder_id(self.parent_id))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderSpec":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            parent_id=data.get("parent_id"),
            examples=tuple(data.get("examples") or ()),
            constraints=tuple(data.get("constraints") or ()),
        )


@dataclass(frozen=True)
class TaxonomySpec:
    """Versioned, user-owned collection of candidate folders."""

    folders: Tuple[FolderSpec, ...]
    version: str = "user-v1"
    source: str = "user"

    def __post_init__(self) -> None:
        ids = [folder.id for folder in self.folders]
        if len(ids) != len(set(ids)):
            raise ValueError("taxonomy contains duplicate folder ids")
        known = set(ids)
        for folder in self.folders:
            if folder.parent_id and folder.parent_id not in known:
                raise ValueError(f"unknown parent folder: {folder.parent_id}")
            if folder.parent_id == folder.id:
                raise ValueError("folder cannot be its own parent")

    @classmethod
    def from_legacy_manager(cls, manager: Any, version: str = "legacy-seed-v1") -> "TaxonomySpec":
        """Adapt the fixed taxonomy only as an editable initial seed."""
        folders = []
        for category in sorted(manager.categories.values(), key=lambda item: item.id):
            parent = category.id.rsplit("/", 1)[0] if "/" in category.id else None
            # Legacy categories omit explicit parent nodes. Keep hierarchy as
            # display metadata rather than manufacturing fixed parent folders.
            folders.append(
                FolderSpec(
                    id=category.id,
                    name=category.name,
                    description=category.description,
                    parent_id=parent if parent in manager.categories else None,
                    constraints=tuple(f"extension:{ext}" for ext in category.extensions),
                )
            )
        return cls(folders=tuple(folders), version=version, source="legacy-seed")

    def folder(self, folder_id: str) -> FolderSpec:
        folder_id = _folder_id(folder_id)
        for folder in self.folders:
            if folder.id == folder_id:
                return folder
        raise KeyError(folder_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "folders": [folder.to_dict() for folder in self.folders],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaxonomySpec":
        return cls(
            folders=tuple(FolderSpec.from_dict(folder) for folder in data.get("folders") or ()),
            version=data.get("version", "user-v1"),
            source=data.get("source", "user"),
        )


class TaxonomyEditType(str, Enum):
    ADD = "ADD"
    RENAME = "RENAME"
    MERGE = "MERGE"
    SPLIT = "SPLIT"
    REMOVE = "REMOVE"
    KEEP = "KEEP"


@dataclass(frozen=True)
class TaxonomyProposal:
    """A discovered taxonomy edit. Approval is mandatory by default."""

    operation: TaxonomyEditType
    proposed_folders: Tuple[FolderSpec, ...]
    affected_files: Tuple[str, ...] = ()
    confidence: float = 0.0
    utility: float = 0.0
    rationale: str = ""
    proposal_id: str = field(default_factory=lambda: str(uuid4()))
    requires_confirmation: bool = True
    approved: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("proposal confidence must be in [0, 1]")
        if not self.rationale.strip():
            raise ValueError("proposal rationale is required")
        if self.approved and self.requires_confirmation is False:
            # Explicitly allowed only for future opt-in workflows; current
            # product paths construct proposals with confirmation enabled.
            return

    def approve(self) -> "TaxonomyProposal":
        return replace(self, approved=True)

    def reject(self) -> "TaxonomyProposal":
        return replace(self, approved=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["operation"] = self.operation.value
        return data


def has_unapproved_edits(proposals: Iterable[TaxonomyProposal]) -> bool:
    return any(proposal.requires_confirmation and not proposal.approved for proposal in proposals)

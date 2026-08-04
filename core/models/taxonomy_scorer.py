"""Shared open-taxonomy helpers. Pair scorer lives in canonical State-AI only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from core.taxonomy.spec import FolderSpec


@dataclass(frozen=True)
class RankedFolder:
    folder_id: Optional[str]
    score: float
    margin: float
    abstained: bool
    source: str = "semantic"


def format_folder_spec(folder: FolderSpec) -> str:
    examples = " | ".join(folder.examples[:5]) or "None"
    constraints = " | ".join(folder.constraints) or "None"
    parent = folder.parent_id or "None"
    return (
        f"Folder: {folder.name}\nID: {folder.id}\nParent: {parent}\n"
        f"Description: {folder.description}\nConstraints: {constraints}\nExamples: {examples}"
    )


def folder_prototype_texts(folder: FolderSpec) -> tuple[str, ...]:
    base = format_folder_spec(folder)
    isolated = tuple(
        f"Folder: {folder.name}\nDescription: {folder.description}\n"
        f"Representative Example: {example}"
        for example in folder.examples[:5]
    )
    return (base,) + isolated


def format_file_record(record: Mapping[str, Any]) -> str:
    meta = record.get("metadata") or record
    return (
        f"File: {meta.get('filename') or record.get('relative_path', '')}\n"
        f"Path: {record.get('relative_path', '')}\n"
        f"Extension: {meta.get('extension', '')} | Size: {meta.get('size_bytes', 0)} bytes | "
        f"Age: {meta.get('age_days', 0)} days\n"
        f"MIME: {meta.get('mime_type', '')} | Binary: {meta.get('is_binary', False)}\n"
        f"Content Sample:\n---\n{record.get('content_sample', '')}\n---"
    )

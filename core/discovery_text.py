"""Dependency-free semantic representation for open-taxonomy ML components.

Open-taxonomy routing and taxonomy discovery must not silently learn a fixed
extension, path, or age policy.  Both use this filename-stem-plus-content view.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Mapping


def format_semantic_file_record(record: Mapping[str, Any]) -> str:
    """Keep filename stem and semantic text; remove non-semantic type metadata."""
    meta = dict(record.get("metadata") or record)
    filename = str(meta.get("filename") or "")
    path = PurePath(filename)
    # Strip compound suffixes as well (.tar.gz), otherwise a hidden type
    # signal survives despite removing the final extension.
    while path.suffix:
        path = PurePath(path.stem)
    stem = path.name if filename else ""
    content = str(record.get("content_sample", ""))
    # The extractor's unreadable-binary marker is a type proxy, not semantics.
    if content.startswith("[Binary File:"):
        content = ""
    else:
        # Extraction modality is not a permitted feature for semantic routing:
        # preserve the words while dropping these local extraction markers.
        for prefix in ("[OCR Text]\n", "[Extracted Text]\n", "[Office Content]\n"):
            if content.startswith(prefix):
                content = content.removeprefix(prefix)
                break
    # Synthetic/public fixture builders sometimes stamp a source UID into a
    # document body.  It identifies a workspace/path rather than what the
    # document means, so do not let the semantic model treat it as content.
    content = "\n".join(line for line in content.splitlines() if not line.strip().lower().startswith("uid:"))
    return f"Filename: {stem}\nContent Sample:\n---\n{content}\n---"


def format_discovery_file_record(record: Mapping[str, Any]) -> str:
    """Compatibility name for the discovery caller."""
    return format_semantic_file_record(record)

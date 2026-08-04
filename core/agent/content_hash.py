"""
Content hashing helpers for workspace dedupe.

Path-based file_id (MD5 of absolute path) remains the stable identity key.
SHA-256 of file bytes is an additional signal used only for exact-content
deduplication and plan gold matching.

Empty (size==0) files all share the same SHA-256 and are excluded from
dedupe grouping so unique zero-byte placeholders are never quarantined.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from core.extractors.file_scanner import FileState

# Cap read size for huge binaries while remaining deterministic for sandbox suites.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024

# SHA-256 of empty bytes — all zero-byte files collide on this digest.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def sha256_file(path: str | Path, max_bytes: int = DEFAULT_MAX_BYTES) -> Optional[str]:
    """Return hex SHA-256 of file contents, or None if unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file_states(
    file_states: Iterable[FileState],
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Optional[str]]:
    """Map path-based file_id -> content SHA-256 (None if unreadable)."""
    out: Dict[str, Optional[str]] = {}
    for state in file_states:
        out[state.file_id] = sha256_file(state.absolute_path, max_bytes=max_bytes)
    return out


def _file_size_bytes(state: FileState) -> int:
    try:
        return int(state.metadata.size_bytes)
    except (AttributeError, TypeError, ValueError):
        try:
            return Path(state.absolute_path).stat().st_size
        except OSError:
            return -1


def is_dedupe_eligible(state: FileState, digest: Optional[str]) -> bool:
    """True when a file may participate in exact-content dedupe groups.

    Empty / zero-byte files all share EMPTY_SHA256 and must never be treated
    as duplicates of each other — they are unique placeholders by path/name.
    """
    if not digest or digest == EMPTY_SHA256:
        return False
    return _file_size_bytes(state) > 0


def group_by_content_hash(
    file_states: List[FileState],
    content_hashes: Dict[str, Optional[str]],
) -> Dict[str, List[FileState]]:
    """Group files that share an exact non-empty content hash (size > 0).

    Skips unreadable hashes, the empty-file digest, and size==0 files so that
    zero-byte placeholders are never quarantined as duplicates.
    """
    groups: Dict[str, List[FileState]] = {}
    for state in file_states:
        digest = content_hashes.get(state.file_id)
        if not is_dedupe_eligible(state, digest):
            continue
        groups.setdefault(digest, []).append(state)
    return {k: v for k, v in groups.items() if len(v) > 1}


def canonical_keep_path(states: List[FileState]) -> FileState:
    """Prefer shortest relative path, then lexicographically smallest name."""
    return sorted(
        states,
        key=lambda s: (len(Path(s.relative_path).parts), len(s.relative_path), s.relative_path.lower()),
    )[0]

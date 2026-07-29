from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path

SKIP = {".git", "__pycache__", ".venv", "node_modules", ".DS_Store", ".idea", ".vscode"}

SYNC_KEEP = {
    ".dropbox",
    ".dropbox_identity",
    ".sync_identity",
    ".synced_folder_icon",
    "._sync_token",
    "._sync_placeholder",
    ".localized",
}

TEXT_EXTS = {
    ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".html", ".css", ".js", ".ts", ".c", ".cpp", ".h", ".java",
    ".sh", ".tex", ".bib", ".rs", ".go", ".sql", ".xml", ".log",
}


@dataclass
class FileMeta:
    filename: str
    extension: str
    size_bytes: int
    age_days: float
    depth: int
    mime_type: str
    is_binary: bool


@dataclass
class FileState:
    absolute_path: str
    relative_path: str
    metadata: FileMeta
    content_sample: str
    target_class: str | None = None
    teacher_probs: list[float] | None = None


def _binary(path: Path, ext: str, mime: str) -> bool:
    if ext in TEXT_EXTS:
        return False
    if mime.startswith(("image/", "video/", "audio/", "application/zip", "application/pdf")):
        return True
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return True


def _sample(path: Path, meta: FileMeta, max_chars: int = 1000) -> str:
    if meta.is_binary:
        return f"[Binary File: {meta.mime_type}, Size: {meta.size_bytes} bytes, Extension: {meta.extension}]"
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        out = "\n".join(lines[:30])
        if len(out) > max_chars:
            out = out[:max_chars] + "..."
        return out or "[Empty File]"
    except OSError as e:
        return f"[Extraction Error: {e}]"


def _keep_file(name: str) -> bool:
    if name in SYNC_KEEP:
        return True
    if name.startswith("._"):
        return True  # AppleDouble / sync tokens
    if name.startswith("."):
        return False
    return True


def scan_folder(root: str) -> list[FileState]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"not a directory: {root}")

    out = []
    for p in root_path.rglob("*"):
        if not p.is_file():
            continue
        if not _keep_file(p.name):
            continue
        rel = p.relative_to(root_path)
        if any(part in SKIP for part in rel.parts):
            continue

        st = p.stat()
        ext = p.suffix.lower()
        mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        meta = FileMeta(
            filename=p.name,
            extension=ext,
            size_bytes=st.st_size,
            age_days=round((time.time() - st.st_mtime) / 86400, 2),
            depth=len(rel.parts) - 1,
            mime_type=mime,
            is_binary=_binary(p, ext, mime),
        )
        out.append(
            FileState(
                absolute_path=str(p),
                relative_path=str(rel).replace("\\", "/"),
                metadata=meta,
                content_sample=_sample(p, meta),
            )
        )
    return out

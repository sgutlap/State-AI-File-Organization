from __future__ import annotations

from pathlib import Path

# leading bytes each extension is expected to start with
MAGIC = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".webp": (b"RIFF",),
    ".pdf": (b"%PDF-",),
    ".zip": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    ".7z": (b"7z\xbc\xaf\x27\x1c",),
    ".rar": (b"Rar!\x1a\x07",),
    ".gz": (b"\x1f\x8b",),
    ".tgz": (b"\x1f\x8b",),
    ".mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
}


def is_corrupted(path: Path) -> bool:
    """True if the file is empty, unreadable, or its leading bytes
    don't match what its extension promises.

    Not wired in yet. To integrate:
      1. scan.py: add an is_corrupted field to FileMeta, set it in scan_folder
      2. taxonomy.yaml: add a quarantine/corrupted category
      3. moves.py: in build_plan, route flagged files there and skip the cascade
    """
    ext = path.suffix.lower()
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return True
    if size == 0:
        return True
    sigs = MAGIC.get(ext)
    if sigs and not any(head.startswith(s) for s in sigs):
        return True
    # mp4/mov magic sits at byte offset 4, not 0
    if ext in (".mp4", ".mov", ".m4a") and head[4:8] != b"ftyp":
        return True
    return False

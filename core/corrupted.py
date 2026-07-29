from __future__ import annotations

from pathlib import Path

import filetype

# extension aliases so jpg == jpeg, tgz == gz, etc.
ALIASES = {".jpeg": ".jpg", ".tgz": ".gz", ".mov": ".mp4", ".m4a": ".mp4"}


def is_corrupted(path: Path) -> bool:
    """True if the file is empty, unreadable, or its actual content
    (per magic bytes) doesn't match its extension.

    Not wired in yet. To integrate:
      1. scan.py: add an is_corrupted field to FileMeta, set it in scan_folder
      2. taxonomy.yaml: add a quarantine/corrupted category
      3. moves.py: in build_plan, route flagged files there and skip the cascade
    """
    ext = path.suffix.lower()
    try:
        if path.stat().st_size == 0:
            return True
        kind = filetype.guess(str(path))
    except OSError:
        return True
    if kind is None:
        # unknown to filetype (plain text etc.) -> nothing to verify
        return False
    detected = "." + kind.extension
    return ALIASES.get(ext, ext) != ALIASES.get(detected, detected)

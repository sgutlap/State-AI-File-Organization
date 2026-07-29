from __future__ import annotations

import json
from pathlib import Path

PREFS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "user_prefs.json"

DEFAULT_PREFS = {
    "user_bins": {"enabled": False, "folder_map": {}},
    "preferred_category_names": {},
    "rename_style": "normalize_noise",
    "dedupe_policy": "quarantine",
    "quarantine_dir": "_duplicates",
}


def load_prefs(path: Path | str | None = None) -> dict:
    p = Path(path) if path else PREFS_PATH
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(DEFAULT_PREFS, indent=2) + "\n")
        return dict(DEFAULT_PREFS)
    data = json.loads(p.read_text(encoding="utf-8"))
    out = dict(DEFAULT_PREFS)
    out.update(data)
    return out


def save_prefs(prefs: dict, path: Path | str | None = None) -> Path:
    p = Path(path) if path else PREFS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prefs, indent=2) + "\n")
    return p


def normalize_folder(path: str) -> str:
    s = (path or "").strip().replace("\\", "/").strip("/")
    if not s or ".." in s.split("/") or ":" in s:
        raise ValueError(f"bad folder name: {path!r}")
    return s


def folder_map(prefs: dict | None = None) -> dict[str, str]:
    prefs = prefs if prefs is not None else load_prefs()
    bins = prefs.get("user_bins") or {}
    if not bins.get("enabled"):
        return {}
    out: dict[str, str] = {}
    for src in (prefs.get("preferred_category_names") or {}, bins.get("folder_map") or {}):
        for cat_id, folder in src.items():
            if folder:
                out[str(cat_id)] = normalize_folder(str(folder))
    return out


def apply_bins(category: str, fmap: dict[str, str] | None = None) -> str:
    fmap = fmap if fmap is not None else folder_map()
    return fmap.get(category, category)


def set_bin(taxonomy_id: str, folder: str, *, enable: bool = True) -> dict:
    prefs = load_prefs()
    bins = dict(prefs.get("user_bins") or {"enabled": False, "folder_map": {}})
    fmap = dict(bins.get("folder_map") or {})
    fmap[taxonomy_id] = normalize_folder(folder)
    bins["folder_map"] = fmap
    bins["enabled"] = enable
    prefs["user_bins"] = bins
    legacy = dict(prefs.get("preferred_category_names") or {})
    legacy[taxonomy_id] = fmap[taxonomy_id]
    prefs["preferred_category_names"] = legacy
    save_prefs(prefs)
    return prefs


def clear_bins() -> dict:
    prefs = load_prefs()
    prefs["user_bins"] = {"enabled": False, "folder_map": {}}
    prefs["preferred_category_names"] = {}
    save_prefs(prefs)
    return prefs

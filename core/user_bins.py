from __future__ import annotations
import json
from pathlib import Path

PREFS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "user_prefs.json"

DEFAULT_PREFS = {
    "user_bins": {
        "enabled": False,
        "mode": "edit",
        "folders": [],
        "folder_map": {},
        "descriptions": {},
    },
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
    bins = dict(DEFAULT_PREFS["user_bins"])
    bins.update(out.get("user_bins") or {})
    out["user_bins"] = bins
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

def _clean_list(raw: list) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        try:
            f = normalize_folder(str(item))
        except ValueError:
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def default_folders() -> list[str]:
    """Stock KD taxonomy destinations."""
    from core.taxonomy import Taxonomy

    return list(Taxonomy().classes)


def active_folders(prefs: dict | None = None) -> list[str]:
    """Folders currently used for organizing (default, or edited default + extras)."""
    prefs = prefs if prefs is not None else load_prefs()
    bins = prefs.get("user_bins") or {}
    if bins.get("enabled"):
        cleaned = _clean_list(bins.get("folders") or [])
        if cleaned:
            return cleaned
    return default_folders()


def is_customized(prefs: dict | None = None) -> bool:
    prefs = prefs if prefs is not None else load_prefs()
    bins = prefs.get("user_bins") or {}
    if not bins.get("enabled"):
        return False
    active = _clean_list(bins.get("folders") or [])
    if not active:
        return False
    return set(active) != set(default_folders())


def custom_folders(prefs: dict | None = None) -> list[str]:
    """Non-empty ⇒ open-vocab router onto this list. Empty ⇒ native default path."""
    prefs = prefs if prefs is not None else load_prefs()
    if not is_customized(prefs):
        return []
    return active_folders(prefs)


def folder_descriptions(prefs: dict | None = None) -> dict[str, str]:
    prefs = prefs if prefs is not None else load_prefs()
    bins = prefs.get("user_bins") or {}
    raw = bins.get("descriptions") or {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        try:
            out[normalize_folder(str(k))] = str(v).strip()
        except ValueError:
            continue
    return out


def folder_map(prefs: dict | None = None) -> dict[str, str]:
    """Legacy rename map. Unused when an edited destination list is active."""
    prefs = prefs if prefs is not None else load_prefs()
    bins = prefs.get("user_bins") or {}
    if not bins.get("enabled"):
        return {}
    if custom_folders(prefs):
        return {}
    if (bins.get("mode") or "edit") == "rename":
        out: dict[str, str] = {}
        for src in (prefs.get("preferred_category_names") or {}, bins.get("folder_map") or {}):
            for cat_id, folder in src.items():
                if folder:
                    out[str(cat_id)] = normalize_folder(str(folder))
        return out
    return {}


def apply_bins(category: str, fmap: dict[str, str] | None = None) -> str:
    fmap = fmap if fmap is not None else folder_map()
    return fmap.get(category, category)


def set_active_folders(
    folders: list[str],
    *,
    descriptions: dict[str, str] | None = None,
) -> dict:
    """Save the full destination list (default ± edits + added folders)."""
    cleaned = _clean_list(folders)
    if not cleaned:
        return clear_bins()

    prefs = load_prefs()
    stock = default_folders()
    bins = dict(prefs.get("user_bins") or {})

    if set(cleaned) == set(stock):
        # identical to stock → native default path
        bins["enabled"] = False
        bins["mode"] = "edit"
        bins["folders"] = []
    else:
        bins["enabled"] = True
        bins["mode"] = "edit"
        bins["folders"] = cleaned

    if descriptions is not None:
        bins["descriptions"] = {
            normalize_folder(k): str(v).strip()
            for k, v in descriptions.items()
            if str(v).strip()
        }
    prefs["user_bins"] = bins
    save_prefs(prefs)
    return prefs


def set_custom_folders(
    folders: list[str],
    *,
    enable: bool = True,
    descriptions: dict[str, str] | None = None,
) -> dict:
    """Back-compat alias: save destination list (enable=False ⇒ revert)."""
    if not enable:
        return clear_bins()
    return set_active_folders(folders, descriptions=descriptions)


def set_bin(taxonomy_id: str, folder: str, *, enable: bool = True) -> dict:
    """Legacy: map one taxonomy id → folder name (rename mode)."""
    prefs = load_prefs()
    bins = dict(prefs.get("user_bins") or {"enabled": False, "folder_map": {}})
    fmap = dict(bins.get("folder_map") or {})
    fmap[taxonomy_id] = normalize_folder(folder)
    bins["folder_map"] = fmap
    bins["enabled"] = enable
    bins["mode"] = "rename"
    prefs["user_bins"] = bins
    legacy = dict(prefs.get("preferred_category_names") or {})
    legacy[taxonomy_id] = fmap[taxonomy_id]
    prefs["preferred_category_names"] = legacy
    save_prefs(prefs)
    return prefs


def clear_bins() -> dict:
    """Revert to stock default taxonomy."""
    prefs = load_prefs()
    prefs["user_bins"] = {
        "enabled": False,
        "mode": "edit",
        "folders": [],
        "folder_map": {},
        "descriptions": {},
    }
    prefs["preferred_category_names"] = {}
    save_prefs(prefs)
    return prefs
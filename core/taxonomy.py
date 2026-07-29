from __future__ import annotations
from pathlib import Path
import yaml
from core.user_bins import apply_bins, folder_map, load_prefs, set_bin

CUSTOM_PATH = Path(__file__).resolve().parent.parent / "custom_taxonomy.yaml"

class Taxonomy:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "taxonomy.yaml"
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.unknown_class = data.get("unknown_class", "misc/uncategorized")
        ids = [item["id"] for item in data.get("taxonomy", [])]
        if self.unknown_class not in ids:
            ids.append(self.unknown_class)
        self.classes = sorted(ids)

    @property
    def num_classes(self):
        return len(self.classes)

    def label_to_idx(self):
        return {c: i for i, c in enumerate(self.classes)}

    def idx_to_label(self):
        return {i: c for i, c in enumerate(self.classes)}


def load_custom(path=CUSTOM_PATH):
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("custom_categories", []))


def list_categories(taxonomy_yaml_path=None, custom_path=CUSTOM_PATH):
    """Folders the UI can sort into (taxonomy + user-bin names + custom)."""
    tax = Taxonomy(taxonomy_yaml_path)
    fmap = folder_map(load_prefs())
    # show remapped folder names when user bins are on
    built_in = [apply_bins(c, fmap) for c in tax.classes]
    # de-dupe while preserving order
    seen = set()
    out = []
    for c in built_in:
        if c not in seen:
            seen.add(c)
            out.append(c)
    for c in load_custom(custom_path):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def add_category(name, root, path=CUSTOM_PATH):
    """Create a custom destination folder and remember it for the UI."""
    name = name.strip().replace("\\", "/").strip("/")
    if not name or ".." in name.split("/") or ":" in name:
        raise ValueError(f"bad folder name: {name!r}")

    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)

    existing = load_custom(path)
    if name not in existing:
        existing.append(name)
        path.write_text(
            yaml.safe_dump({"custom_categories": existing}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return existing


def set_user_bin(taxonomy_id: str, folder_name: str):
    """Map a taxonomy class to a custom folder name (persisted)."""
    return set_bin(taxonomy_id, folder_name, enable=True)

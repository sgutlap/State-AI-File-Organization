from pathlib import Path

import yaml

from core.user_bins import active_folders, apply_bins, custom_folders, folder_map, load_prefs

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
    prefs = load_prefs()
    if custom_folders(prefs):
        return active_folders(prefs)
    tax = Taxonomy(taxonomy_yaml_path)
    fmap = folder_map(prefs)
    out, seen = [], set()
    for c in [apply_bins(x, fmap) for x in tax.classes] + load_custom(custom_path):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def add_category(name, root, path=CUSTOM_PATH):
    name = name.strip().replace("\\", "/").strip("/")
    if not name or ".." in name.split("/") or ":" in name:
        raise ValueError(f"bad folder name: {name!r}")
    (Path(root) / name).mkdir(parents=True, exist_ok=True)
    existing = load_custom(path)
    if name not in existing:
        existing.append(name)
        path.write_text(
            yaml.safe_dump({"custom_categories": existing}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return existing

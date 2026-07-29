from pathlib import Path
 
import yaml
 
 
class Taxonomy:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "taxonomy.yaml"
        data = yaml.safe_load(Path(config_path).read_text())
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
 
 

CUSTOM_PATH = Path(__file__).resolve().parent.parent / "custom_taxonomy.yaml"
 
 
def load_custom(path=CUSTOM_PATH):
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("custom_categories", []))
 
 
def list_categories(taxonomy_yaml_path=None, custom_path=CUSTOM_PATH):
    built_in = Taxonomy(taxonomy_yaml_path).classes
    custom = load_custom(custom_path)
    return built_in + [c for c in custom if c not in built_in]
 
 
def notify_taxonomy(folder_path: str):
    """
    Hook that fires whenever a new category folder gets created from
    the UI. Gets passed the full path to the new folder.
 
    Not implemented yet - this is where the actual taxonomy logic goes
    (whatever decides how files map to this new bucket). Left as a stub
    on purpose.
    """
    pass
 
 
def add_category(name, root, path=CUSTOM_PATH):
    name = name.strip()
    if not name or "/" in name or "\\" in name or ".." in name:
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
 
    notify_taxonomy(str(folder))
    return existing
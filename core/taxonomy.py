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

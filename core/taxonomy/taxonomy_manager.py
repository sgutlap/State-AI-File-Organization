"""Load taxonomy.yaml as an editable seed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_DEFAULT = Path(__file__).resolve().parents[2] / "taxonomy.yaml"


@dataclass
class TaxonomyCategory:
    id: str
    name: str
    description: str
    extensions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class TaxonomyManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else _DEFAULT
        if not self.config_path.is_file():
            raise FileNotFoundError(f"taxonomy seed not found: {self.config_path}")
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.unknown_class = data.get("unknown_class", "misc/uncategorized")
        self.categories: Dict[str, TaxonomyCategory] = {}
        for item in data.get("taxonomy") or []:
            self.categories[item["id"]] = TaxonomyCategory(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                extensions=[e.lower() for e in item.get("extensions") or []],
                keywords=[k.lower() for k in item.get("keywords") or []],
            )
        if self.unknown_class not in self.categories:
            self.categories[self.unknown_class] = TaxonomyCategory(
                id=self.unknown_class,
                name="Miscellaneous",
                description="Uncategorized or generic files",
            )

"""Create a copied flat-inbox fixture with labels derived from a separate rule file.

Use only on synthetic or licensed source trees.  The original tree is never
modified; folder labels stay in the manifest and are absent from inbox paths.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.content_hash import sha256_file
from core.config import ScanConfig
from core.extractors.file_scanner import FileScanner
from core.research.taxonomy_dataset import BenchmarkManifest, DatasetSplit, TaxonomyRankingTask
from core.taxonomy.spec import TaxonomySpec


def _destination(relative_path: str, rules: list[dict], default: str) -> str:
    for rule in rules:
        if fnmatch(relative_path, rule["glob"]):
            return rule["folder_id"]
    return default


def _excluded(relative_path: str, globs: list[str]) -> bool:
    return any(fnmatch(relative_path, pattern) for pattern in globs)


def materialize(
    source: Path,
    destination: Path,
    fixture: dict,
    source_group_id: str,
    split: DatasetSplit,
    *,
    include_content: bool = False,
) -> BenchmarkManifest:
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    taxonomy = TaxonomySpec.from_dict(fixture["taxonomy"])
    rules = list(fixture.get("rules") or [])
    exclude_globs = list(fixture.get("exclude_globs") or [])
    default = str(fixture.get("default_folder_id", "ABSTAIN"))
    valid_ids = {folder.id for folder in taxonomy.folders}
    if default != "ABSTAIN" and default not in valid_ids:
        raise ValueError("default_folder_id must be a taxonomy folder or ABSTAIN")
    for rule in rules:
        if rule.get("folder_id") not in valid_ids or not rule.get("glob"):
            raise ValueError("each rule needs a glob and a taxonomy folder_id")
    if not all(isinstance(pattern, str) and pattern for pattern in exclude_globs):
        raise ValueError("exclude_globs must contain non-empty strings")

    destination.mkdir(parents=True)
    inbox = destination / "inbox"
    inbox.mkdir()
    scanner = FileScanner(ScanConfig(include_content_samples=include_content))
    source_states = scanner.scan_directory(str(source))
    expected = {}
    for state in source_states:
        if _excluded(state.relative_path, exclude_globs):
            continue
        digest = hashlib.sha256(state.relative_path.encode("utf-8")).hexdigest()[:10]
        copied_name = f"{digest}_{Path(state.relative_path).name}"
        target = inbox / copied_name
        shutil.copy2(state.absolute_path, target)
        expected[f"inbox/{copied_name}"] = _destination(state.relative_path, rules, default)

    tasks = []
    for state in scanner.scan_directory(str(destination)):
        label = expected[state.relative_path]
        seed = f"{source_group_id}|{state.relative_path}|{taxonomy.version}"
        file_state = {"relative_path": state.relative_path, "metadata": state.metadata.to_dict()}
        if include_content:
            file_state["content_sample"] = state.content_sample
        tasks.append(TaxonomyRankingTask(
            task_id=hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
            workspace_id=f"fixture-{source_group_id}", source_group_id=source_group_id,
            taxonomy_id=taxonomy.version, split=split, file_id=state.file_id,
            content_hash=sha256_file(Path(state.absolute_path)),
            file_state=file_state,
            taxonomy=taxonomy, candidate_folder_ids=tuple(folder.id for folder in taxonomy.folders),
            acceptable_folder_ids=() if label == "ABSTAIN" else (label,),
            abstain=label == "ABSTAIN", label_source="fixture:path-rule",
        ))
    manifest = BenchmarkManifest(f"fixture-{source_group_id}", tasks)
    manifest.validate()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Synthetic/licensed organized source tree")
    parser.add_argument("--fixture", required=True, help="Taxonomy/rule JSON")
    parser.add_argument("--out-dir", required=True, help="New flat-inbox directory; must not exist")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-group-id", required=True)
    parser.add_argument("--split", choices=[item.value for item in DatasetSplit], default="test")
    parser.add_argument(
        "--include-content-local-only", action="store_true",
        help="Store local content snippets for semantic training; never use with a cloud labeler without separate consent.",
    )
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    manifest = materialize(
        Path(args.source), Path(args.out_dir), fixture, args.source_group_id, DatasetSplit(args.split),
        include_content=args.include_content_local_only,
    )
    manifest.write_jsonl(args.manifest)
    print(f"Materialized {len(manifest.tasks)} {args.split} tasks; fixture={args.fixture}")


if __name__ == "__main__":
    main()

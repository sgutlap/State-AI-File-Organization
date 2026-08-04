"""Validate a redacted taxonomy-conditioned benchmark manifest (.jsonl)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.research.taxonomy_dataset import BenchmarkManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="JSONL TaxonomyRankingTask manifest")
    parser.add_argument("--require-heldout-taxonomies", action="store_true")
    args = parser.parse_args()
    manifest = BenchmarkManifest.read_jsonl(args.manifest)
    if args.require_heldout_taxonomies:
        manifest.validate_heldout_taxonomies()
    print(f"valid manifest={manifest.name} counts={manifest.counts()}")


if __name__ == "__main__":
    main()

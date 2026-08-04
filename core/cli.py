"""
State AI CLI — open-taxonomy primary runtime.

Commands:
  scan, taxonomy seed, taxonomy discover, organize-open

Legacy fixed-taxonomy V1 lives under Baselines/old/v1 and is not imported here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


def handle_scan(args):
    from core.extractors.file_scanner import FileScanner

    target_dir = args.directory
    console.print(f"\n[bold cyan]Scanning directory:[/bold cyan] {target_dir}")
    scanner = FileScanner()
    file_states = scanner.scan_directory(target_dir)
    console.print(f"[bold green]Scanned {len(file_states)} files.[/bold green]\n")

    table = Table(title="Extracted File States (Sample)", show_header=True, header_style="bold magenta")
    table.add_column("File Name", style="cyan")
    table.add_column("Extension", style="yellow")
    table.add_column("Size", justify="right")
    table.add_column("Age (Days)", justify="right")
    table.add_column("Content Sample Snippet", style="dim", overflow="fold")
    for s in file_states[:15]:
        meta = s.metadata
        snippet = (s.content_sample or "").replace("\n", " ")[:60] + "..."
        table.add_row(meta.filename, meta.extension, f"{meta.size_bytes} B", f"{meta.age_days}", snippet)
    console.print(table)


def _load_taxonomy_spec(path: Optional[str]):
    from core.taxonomy.spec import TaxonomySpec

    if not path:
        from core.taxonomy.taxonomy_manager import TaxonomyManager

        return TaxonomySpec.from_legacy_manager(TaxonomyManager())
    with open(path, "r", encoding="utf-8") as handle:
        return TaxonomySpec.from_dict(json.load(handle))


def handle_taxonomy(args):
    from core.extractors.file_scanner import FileScanner
    from core.taxonomy.discovery import DiscoveryConfig, TaxonomyDiscoverer

    taxonomy = _load_taxonomy_spec(getattr(args, "taxonomy", None))
    if args.action == "seed":
        rendered = json.dumps(taxonomy.to_dict(), indent=2)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
            console.print(f"[green]Wrote editable taxonomy seed: {output}[/green]")
        else:
            console.print_json(rendered)
        return
    if args.action == "discover":
        scanner = FileScanner()
        states = scanner.scan_directory(args.directory)
        if args.slot_checkpoint:
            from core.taxonomy.slot_discovery import load_slot_discoverer

            discoverer = load_slot_discoverer(
                args.slot_checkpoint,
                device=args.device,
                min_cluster_size=args.min_cluster_size,
                max_proposals=args.max_proposals,
            )
        else:
            discoverer = TaxonomyDiscoverer(
                DiscoveryConfig(
                    min_cluster_size=args.min_cluster_size,
                    max_proposals=args.max_proposals,
                )
            )
        proposals = discoverer.discover(states, taxonomy)
        # Never mutates folders/taxonomy; proposals remain confirmation-required.
        console.print_json(
            json.dumps(
                {
                    "taxonomy_version": taxonomy.version,
                    "files_scanned": len(states),
                    "proposal_count": len(proposals),
                    "proposals": [proposal.to_dict() for proposal in proposals],
                    "requires_confirmation": True,
                },
                indent=2,
            )
        )
        return
    raise ValueError(f"unknown taxonomy action: {args.action}")


def handle_organize_open(args):
    """Semantic dual encoder path; dry-run by default."""
    import torch

    from core.agent.executor import PlanExecutor
    from core.agent.open_taxonomy_planner import OpenTaxonomyPlanner
    from core.config import ScanConfig
    from core.extractors.file_scanner import FileScanner
    from core.models.dual_taxonomy import (
        DualTaxonomyEncoder,
        DualTaxonomyScorer,
        assert_semantic_dual_checkpoint,
        checkpoint_backend,
    )
    from core.taxonomy.discovery import TaxonomyDiscoverer

    taxonomy = _load_taxonomy_spec(args.taxonomy)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"open-taxonomy checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    detected = checkpoint_backend(checkpoint)
    backend = detected if args.scorer_backend == "auto" else args.scorer_backend
    if backend != detected:
        raise ValueError(f"--scorer-backend={backend} does not match this {detected} checkpoint")
    if backend != "dual":
        raise ValueError(
            "this lean V2 runtime supports semantic dual checkpoints only; "
            f"got backend={backend}. Pair/hybrid/reranker paths remain in canonical State-AI."
        )
    assert_semantic_dual_checkpoint(checkpoint)
    virtual_abstain = bool(checkpoint.get("virtual_abstain", False))
    score_threshold, margin_threshold = 0.0, args.margin_threshold
    base_model = checkpoint.get("base_model", "sentence-transformers/all-MiniLM-L6-v2")
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        declared_checkpoint = calibration.get("calibration_checkpoint")
        if declared_checkpoint and Path(declared_checkpoint).resolve() != checkpoint_path.resolve():
            raise ValueError(
                "calibration artifact was fit for a different checkpoint path; "
                "point --checkpoint at the same path recorded in the calibration, "
                "or update calibration_checkpoint after copying"
            )
        declared_split = calibration.get("calibration_split")
        if declared_split and declared_split != "validation":
            raise ValueError("calibration artifact was not fit on validation predictions")
        score_threshold = float(calibration["abstain_threshold"])
        margin_threshold = float(calibration["margin_threshold"])
        if not 0.0 <= score_threshold <= 1.0 or margin_threshold < 0.0:
            raise ValueError("calibration thresholds are invalid")
        calibration_scorer = str(calibration.get("calibration_scorer", "dual"))
        expected_scorer = "virtual_dual" if virtual_abstain else "dual"
        if calibration_scorer != expected_scorer:
            raise ValueError("calibration artifact does not match this dual scorer")
    elif args.apply:
        raise ValueError(
            "refusing --apply with an uncalibrated dual scorer; provide a "
            "validation-only --calibration file or review the dry run"
        )
    model = DualTaxonomyEncoder(base_model)
    model.load_state_dict(checkpoint["state_dict"])
    scorer = DualTaxonomyScorer(
        model=model,
        base_model_name=base_model,
        device=args.device,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        folder_prototypes=bool(checkpoint.get("folder_prototypes", False)),
        example_extension_fallback=args.example_extension_fallback,
        virtual_abstain=virtual_abstain,
    )
    if args.visual_model:
        if args.apply:
            raise ValueError(
                "refusing --apply with an uncalibrated visual scorer; review the dry run "
                "until human-labelled media calibration is available"
            )
        from core.models.visual_taxonomy import CLIPVisualTaxonomyScorer

        scorer = CLIPVisualTaxonomyScorer(
            scorer,
            model_name=args.visual_model,
            device=args.device,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
        )
    planner = OpenTaxonomyPlanner(scorer, taxonomy)
    states = FileScanner(ScanConfig(image_ocr=args.image_ocr)).scan_directory(args.directory)
    if args.slot_checkpoint:
        from core.taxonomy.slot_discovery import load_slot_discoverer

        discoverer = load_slot_discoverer(args.slot_checkpoint, device=args.device)
    else:
        discoverer = TaxonomyDiscoverer() if args.discover else None
    plan = planner.plan(args.directory, states, discoverer=discoverer)
    PlanExecutor().execute(plan, dry_run=not args.apply, confirm=True, auto_skip_ask=False)


def main():
    parser = argparse.ArgumentParser(
        description="State AI: open-taxonomy file organization (V2 primary runtime)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_scan = subparsers.add_parser("scan", help="Scan directory and extract file state representation")
    p_scan.add_argument("directory", help="Target directory to scan")

    p_open = subparsers.add_parser("organize-open", help="Plan with a trained open-taxonomy dual encoder")
    p_open.add_argument("directory", help="Workspace copy to organize")
    p_open.add_argument("--taxonomy", required=True, help="User taxonomy JSON")
    p_open.add_argument("--checkpoint", required=True, help="Semantic dual checkpoint")
    p_open.add_argument("--device", default="auto")
    p_open.add_argument("--scorer-backend", choices=["auto", "dual"], default="auto")
    p_open.add_argument(
        "--example-extension-fallback",
        action="store_true",
        help="For opaque files only, apply an extension if examples uniquely identify a folder.",
    )
    p_open.add_argument(
        "--margin-threshold",
        type=float,
        default=0.10,
        help="Ambiguity margin below which the plan asks the user",
    )
    p_open.add_argument("--calibration", help="Validation-only abstention thresholds; required with --apply")
    p_open.add_argument("--visual-model", metavar="MODEL", help="Optional frozen CLIP image fallback (dry-run)")
    p_open.add_argument("--image-ocr", action="store_true", help="Local OCR text from images before scoring")
    p_open.add_argument("--discover", action="store_true", help="Add review-only ML taxonomy proposals")
    p_open.add_argument("--slot-checkpoint", help="Learned slot-discovery checkpoint; still confirmation-required")
    p_open.add_argument("--apply", action="store_true", help="Apply after confirmation; default is dry-run")

    p_taxonomy = subparsers.add_parser("taxonomy", help="Editable seed export or review-only discovery")
    p_taxonomy.add_argument("action", choices=["seed", "discover"])
    p_taxonomy.add_argument("directory", nargs="?", help="Workspace for discover")
    p_taxonomy.add_argument("--taxonomy", help="Editable TaxonomySpec JSON; default seed from configs/")
    p_taxonomy.add_argument("--out", help="Write seed JSON instead of printing")
    p_taxonomy.add_argument("--min-cluster-size", type=int, default=3)
    p_taxonomy.add_argument("--max-proposals", type=int, default=5)
    p_taxonomy.add_argument("--slot-checkpoint", help="Trained slot-discovery checkpoint")
    p_taxonomy.add_argument("--device", default="auto")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    if args.command == "scan":
        handle_scan(args)
    elif args.command == "organize-open":
        handle_organize_open(args)
    elif args.command == "taxonomy":
        if args.action == "discover" and not args.directory:
            parser.error("taxonomy discover requires a directory")
        handle_taxonomy(args)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_packet_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def parse_response(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("response file is empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSONL rows must be objects")
        return rows
    if isinstance(parsed, list) and all(isinstance(row, dict) for row in parsed):
        return parsed
    if isinstance(parsed, dict):
        for key in ("predictions", "results", "items"):
            values = parsed.get(key)
            if isinstance(values, list) and all(isinstance(row, dict) for row in values):
                return values
    raise ValueError("expected JSONL, object list, or object with predictions/results/items list")


def validate_rows(packet_ids: set[str], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    duplicates: list[str] = []
    malformed: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            malformed.append("missing_task_id")
            continue
        if task_id not in packet_ids:
            unknown.append(task_id)
            continue
        if task_id in by_id:
            duplicates.append(task_id)
            continue
        if not bool(row.get("abstained") or row.get("abstain")) and not (
            row.get("ranked_folder_ids")
            or row.get("raw_pred_folder_id")
            or row.get("predicted_folder_id")
            or row.get("destination")
            or row.get("folder")
        ):
            malformed.append(task_id)
            continue
        by_id[task_id] = row
    missing = sorted(packet_ids - set(by_id))
    audit = {
        "n_expected": len(packet_ids),
        "n_received": len(rows),
        "n_valid": len(by_id),
        "missing_task_ids": missing,
        "unknown_task_ids": sorted(set(unknown)),
        "duplicate_task_ids": sorted(set(duplicates)),
        "malformed_rows": malformed,
        "pass": not (missing or unknown or duplicates or malformed),
    }
    return [by_id[key] for key in sorted(by_id)], audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="packet directory or tasks.jsonl")
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="write valid subset despite audit failures")
    args = parser.parse_args()
    packet_path = args.packet / "tasks.jsonl" if args.packet.is_dir() else args.packet
    packet_ids = read_packet_ids(packet_path)
    rows, audit = validate_rows(packet_ids, parse_response(args.responses))
    audit_path = args.out.with_suffix(".collection_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit["pass"] and not args.allow_partial:
        raise SystemExit(f"response collection rejected; audit: {audit_path}")
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps({"predictions": str(args.out), "audit": str(audit_path), **audit}, indent=2))


if __name__ == "__main__":
    main()

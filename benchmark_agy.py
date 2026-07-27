from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.cascade import Cascade
from core.moves import load_student
from core.scan import scan_folder


def ask_agy(folder: Path, timeout=120) -> str:
    prompt = (
        "Look at the files in this folder. For each file, suggest ONE category path "
        "from: documents/research, documents/financial, code/projects, data/datasets, "
        "media/images, media/audio_video, archives, misc/uncategorized. "
        'Reply as JSON list of {"file": name, "category": id}. No other text.'
    )
    proc = subprocess.run(
        ["agy", "-p", prompt, "--add-dir", str(folder.resolve())],
        capture_output=True, text=True, timeout=timeout,
    )
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--ckpt", default="artifacts/student_model_ckpt")
    p.add_argument("--skip-agy", action="store_true")
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        sys.exit(1)

    student = load_student(args.ckpt)
    cascade = Cascade(student)
    states = scan_folder(str(folder))[: args.limit]

    print(f"\nour predictions ({len(states)} files):\n")
    ours = {}
    for s in states:
        d = cascade.decide(s)
        ours[s.metadata.filename] = d.category
        print(f"  {s.metadata.filename:40s}  {d.category:22s}  {d.confidence:.2f}  [{d.tier}]")

    if args.skip_agy:
        return
    if not shutil.which("agy"):
        print("\nagy not on PATH (use --skip-agy)")
        return

    with tempfile.TemporaryDirectory(prefix="agy_cmp_") as tmp:
        tmp_path = Path(tmp)
        for s in states:
            src = Path(s.absolute_path)
            dst = tmp_path / src.name
            n = 1
            while dst.exists():
                dst = tmp_path / f"{src.stem}_{n}{src.suffix}"
                n += 1
            shutil.copy2(src, dst)

        print(f"\nasking agy on {len(states)} files...")
        try:
            out = ask_agy(tmp_path, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("agy timed out")
            return

    print("\n--- agy output (truncated) ---")
    print(out[:2000])
    print("---")

    start, end = out.find("["), out.rfind("]")
    if start < 0 or end <= start:
        print("(no JSON list found)")
        return
    try:
        rows = json.loads(out[start : end + 1])
    except json.JSONDecodeError:
        print("(couldn't parse JSON)")
        return

    agree = n = 0
    for row in rows:
        name = row.get("file") or row.get("filename") or ""
        cat = row.get("category") or row.get("label") or ""
        if name not in ours:
            continue
        n += 1
        same = ours[name] == cat
        agree += int(same)
        print(f"  [{'Y' if same else 'N'}] {name}: us={ours[name]}  agy={cat}")
    if n:
        print(f"\nagree {agree}/{n} ({100 * agree / n:.0f}%)")


if __name__ == "__main__":
    main()

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SERVE_URL = "http://127.0.0.1:18765"
from core.moves import apply_plan, build_plan, duplicate_organized, load_student, print_plan


def _via_server(folder: Path, mode: str, threshold: float) -> bool:
    payload = json.dumps({"folder": str(folder), "mode": mode, "threshold": threshold}).encode()
    req = urllib.request.Request(
        f"{SERVE_URL}/organize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    if not data.get("ok"):
        print(data.get("error", "server error"))
        sys.exit(1)
    for line in data.get("log", []):
        print(line)
    return True


def main():
    p = argparse.ArgumentParser(description="organize a folder with state-ai student")
    p.add_argument("folder", help="folder path (Finder Quick Action passes this)")
    p.add_argument("--apply", action="store_true", help="move files in this folder")
    p.add_argument("--dupe", action="store_true", help="copy to '<name> Organized' in parent, then organize copy")
    p.add_argument("--ckpt", default="artifacts/student_model_ckpt")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-server", action="store_true", help="skip warm server, load model locally")
    args = p.parse_args()

    if args.apply and args.dupe:
        sys.exit("use --dupe OR --apply, not both")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")

    mode = "dupe" if args.dupe else ("apply" if args.apply else "dry")

    if not args.no_server and _via_server(folder, mode, args.threshold):
        return

    student = load_student(args.ckpt)

    if args.dupe:
        work = duplicate_organized(folder)
        plan = build_plan(str(work), student, conf_threshold=args.threshold)
        print_plan(plan)
        n = apply_plan(plan)
        print(f"done — {n} moves in {work}")
        return

    plan = build_plan(str(folder), student, conf_threshold=args.threshold)
    print_plan(plan)
    if args.apply:
        print(f"moved {apply_plan(plan)} files")
    else:
        print("dry-run (pass --apply or --dupe)")


if __name__ == "__main__":
    main()

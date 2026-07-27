import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.moves import apply_plan, build_plan, load_student, print_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--ckpt", default="artifacts/student_model_ckpt")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")

    student = load_student(args.ckpt)
    plan = build_plan(str(folder), student, conf_threshold=args.threshold)
    print_plan(plan)
    if args.apply:
        print(f"moved {apply_plan(plan)} files")
    else:
        print("dry-run (pass --apply to move)")


if __name__ == "__main__":
    main()

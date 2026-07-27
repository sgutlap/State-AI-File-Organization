import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.cascade import Cascade
from core.model import Student
from core.scan import scan_folder


@dataclass
class Move:
    src: str
    dst: str
    category: str
    confidence: float
    tier: str


@dataclass
class Plan:
    root: str
    moves: list[Move] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)


def load_student(ckpt="artifacts/student_model_ckpt"):
    s = Student()
    if Path(ckpt).exists():
        s.load(ckpt)
        print("loaded", ckpt)
    else:
        print("warning: missing ckpt at", ckpt)
    return s


def build_plan(root: str, student: Student, conf_threshold: float = 0.50) -> Plan:
    cascade = Cascade(student, threshold=max(conf_threshold, 0.65))
    root_path = Path(root).resolve()
    plan = Plan(root=str(root_path))
    needed = set()

    for state in scan_folder(root):
        d = cascade.decide(state)
        cat, conf = d.category, d.confidence
        if conf < conf_threshold:
            cat = student.taxonomy.unknown_class
            conf = max(conf, 0.40)

        dest_dir = root_path / cat
        # already there
        if Path(state.absolute_path).parent == dest_dir:
            continue

        dest = dest_dir / Path(state.absolute_path).name
        if dest.exists() and dest.resolve() != Path(state.absolute_path).resolve():
            n = 1
            while True:
                alt = dest_dir / f"{dest.stem}_{n}{dest.suffix}"
                if not alt.exists():
                    dest = alt
                    break
                n += 1

        needed.add(str(dest_dir))
        plan.moves.append(Move(state.absolute_path, str(dest), cat, round(conf, 4), d.tier))

    plan.dirs = sorted(needed)
    return plan


def print_plan(plan: Plan, limit: int = 40):
    print(f"\nplan for {plan.root}")
    print(f"  {len(plan.moves)} moves, {len(plan.dirs)} folders\n")
    for m in plan.moves[:limit]:
        name = Path(m.src).name
        try:
            rel = str(Path(m.dst).relative_to(plan.root))
        except ValueError:
            rel = m.dst
        print(f"  [{m.tier:9s}] {m.confidence*100:5.1f}%  {name}  ->  {rel}")
    if len(plan.moves) > limit:
        print(f"  ... +{len(plan.moves) - limit} more")
    print()


def apply_plan(plan: Plan) -> int:
    for d in plan.dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    n = 0
    for m in plan.moves:
        src, dst = Path(m.src), Path(m.dst)
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n += 1
    return n

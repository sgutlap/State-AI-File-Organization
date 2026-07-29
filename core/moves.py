from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.cascade import Cascade
from core.model import Student
from core.scan import FileState, scan_folder
from core.user_bins import apply_bins, folder_map, load_prefs

COPY_SKIP = shutil.ignore_patterns(".git", "__pycache__", ".venv", ".DS_Store", "node_modules")

SOURCE_TOPS = frozenset(
    {
        "ignored_files",
        "partial_sync",
        "conflicted_copies",
        "shared_with_me",
        "multi_device",
        "from_inbox",
        "from_spam",
        "from_sent",
        "attachments_by_date",
        "extracted_zips",
    }
)

SYNC_MARKER_RE = re.compile(
    r"^(?:desktop\.ini|thumbs\.db|\.ds_store|com\.apple\.timemachine\.supported|"
    r"\.localized|\.dropbox(?:_identity)?|\.sync(?:_identity)?|\.synced_folder_icon|"
    r"\._sync(?:_token|_placeholder)?|link_only_file\.txt|moved_files_here\.txt|"
    r"file_[123]\.txt)$",
    re.IGNORECASE,
)
SYNC_MARKER_SUBSTR = (
    ".dropbox",
    ".sync",
    "._sync",
    ".localized",
    "timemachine",
    "sync_token",
    "sync_identity",
    "sync_placeholder",
    "synced_folder",
)

NOISE_RE = re.compile(
    r"^(?P<stem>.+?)(?:\s*\((?P<n>\d+)\)|\s*[-_]?\s*copy(?:\s*\((?P<n2>\d+)\))?)$",
    re.IGNORECASE,
)

EXT_OVERRIDE = {
    ".png": "media/images",
    ".jpg": "media/images",
    ".jpeg": "media/images",
    ".gif": "media/images",
    ".webp": "media/images",
    ".svg": "media/images",
    ".fig": "media/images",
    ".psd": "media/images",
    ".mp4": "media/audio_video",
    ".mov": "media/audio_video",
    ".mp3": "media/audio_video",
    ".wav": "media/audio_video",
    ".zip": "archives",
    ".rar": "archives",
    ".7z": "archives",
    ".csv": "data/datasets",
    ".parquet": "data/datasets",
    ".tsv": "data/datasets",
}

PROJECT_MARKERS = ("package.json", "pyproject.toml", "Cargo.toml", "go.mod", "requirements.txt")


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


def duplicate_organized(src: Path) -> Path:
    src = src.resolve()
    parent = src.parent
    base = f"{src.name} Organized"
    dst = parent / base
    if dst.exists():
        n = 1
        while (parent / f"{src.name} Organized {n}").exists():
            n += 1
        dst = parent / f"{src.name} Organized {n}"
    shutil.copytree(src, dst, ignore=COPY_SKIP)
    print(f"copied to {dst}")
    return dst


def _is_sync_marker(name: str) -> bool:
    low = (name or "").lower()
    if SYNC_MARKER_RE.match(low):
        return True
    return any(s in low for s in SYNC_MARKER_SUBSTR)


def _normalize_name(filename: str) -> str | None:
    """Strip (1)/Copy noise. Returns new name or None."""
    p = Path(filename)
    stem, ext = p.stem, p.suffix
    m = NOISE_RE.match(stem)
    if not m:
        return None
    clean = m.group("stem").strip()
    if not clean or clean == stem:
        return None
    return f"{clean}{ext}"


def _file_hash(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _disambiguate(dest: Path, used: set[str]) -> Path:
    if str(dest) not in used and not dest.exists():
        return dest
    n = 1
    while True:
        alt = dest.parent / f"{dest.stem}_{n}{dest.suffix}"
        if str(alt) not in used and not alt.exists():
            return alt
        n += 1


def _project_roots(states: list[FileState], root: Path) -> dict[str, str]:
    """relative_path → project root relative path for keep-together."""
    members: dict[str, str] = {}
    by_parent: dict[str, list[FileState]] = {}
    for s in states:
        parent = str(Path(s.relative_path).parent).replace("\\", "/")
        by_parent.setdefault(parent, []).append(s)

    for parent, files in by_parent.items():
        names = {Path(f.metadata.filename).name.lower() for f in files}
        if not any(m.lower() in names for m in PROJECT_MARKERS):
            continue
        # Don't treat source dumps as projects
        top = parent.split("/")[0] if parent not in (".", "") else ""
        if top in SOURCE_TOPS:
            continue
        proj = parent if parent not in (".", "") else "."
        for f in files:
            members[f.relative_path] = proj
    return members


def build_plan(root: str, student: Student, conf_threshold: float = 0.50) -> Plan:
    prefs = load_prefs()
    fmap = folder_map(prefs)
    cascade = Cascade(student, threshold=max(conf_threshold, 0.65))
    root_path = Path(root).resolve()
    plan = Plan(root=str(root_path))
    needed: set[str] = set()
    used_dst: set[str] = set()
    claimed: set[str] = set()

    states = scan_folder(root)
    project_map = _project_roots(states, root_path)

    # Pass B1 — exact-hash dedupe (skip empty stubs — same hash, not real dupes)
    hashes: dict[str, list[FileState]] = {}
    for s in states:
        if s.metadata.size_bytes <= 0:
            continue
        h = _file_hash(Path(s.absolute_path))
        if h:
            hashes.setdefault(h, []).append(s)

    quarantine = prefs.get("quarantine_dir", "_duplicates")
    for group in hashes.values():
        if len(group) < 2:
            continue
        # keep first (shortest path), quarantine rest
        group_sorted = sorted(group, key=lambda x: (len(x.relative_path), x.relative_path))
        for dup in group_sorted[1:]:
            dest_dir = root_path / quarantine
            dest = _disambiguate(dest_dir / Path(dup.absolute_path).name, used_dst)
            needed.add(str(dest_dir))
            used_dst.add(str(dest))
            claimed.add(dup.absolute_path)
            plan.moves.append(
                Move(dup.absolute_path, str(dest), quarantine, 0.99, "dedupe")
            )

    # Pass A + B3 — classify and move
    for state in states:
        if state.absolute_path in claimed:
            continue

        d = cascade.decide(state)
        cat, conf = d.category, d.confidence
        ext = (state.metadata.extension or "").lower()

        # High-precision extension snap
        if ext in EXT_OVERRIDE and not any(
            x in state.metadata.filename.lower()
            for x in ("untitled", "temp", "download", "file", "document", "empty")
        ):
            cat = EXT_OVERRIDE[ext]

        # User bins: taxonomy id → folder name
        folder = apply_bins(cat, fmap)

        src_top = Path(state.relative_path).parts[0] if Path(state.relative_path).parts else ""
        force = src_top in SOURCE_TOPS or _is_sync_marker(state.metadata.filename)

        if conf < conf_threshold and not force:
            folder = apply_bins(student.taxonomy.unknown_class, fmap)
            conf = max(conf, 0.40)

        proj = project_map.get(state.relative_path)
        if proj and proj not in (".", "") and Path(proj).parts[0] not in SOURCE_TOPS:
            keep = apply_bins("code/projects", fmap)
            anchor = Path(proj).name
            try:
                within = Path(state.relative_path).relative_to(proj)
            except ValueError:
                within = Path(state.metadata.filename)
            dest_dir = root_path / keep / anchor / within.parent
            final_name = state.metadata.filename
            if prefs.get("rename_style") == "normalize_noise":
                final_name = _normalize_name(final_name) or final_name
            dest = _disambiguate(dest_dir / final_name, used_dst)
            if Path(state.absolute_path).resolve() == dest.resolve():
                continue
            needed.add(str(dest.parent))
            used_dst.add(str(dest))
            claimed.add(state.absolute_path)
            plan.moves.append(Move(state.absolute_path, str(dest), keep, round(conf, 4), d.tier))
            continue

        # Rename noise
        final_name = state.metadata.filename
        if prefs.get("rename_style") == "normalize_noise":
            final_name = _normalize_name(final_name) or final_name

        dest_dir = root_path / folder
        if Path(state.absolute_path).parent.resolve() == dest_dir.resolve() and final_name == state.metadata.filename:
            continue

        dest = _disambiguate(dest_dir / final_name, used_dst)
        if Path(state.absolute_path).resolve() == dest.resolve():
            continue

        needed.add(str(dest_dir))
        used_dst.add(str(dest))
        claimed.add(state.absolute_path)
        plan.moves.append(Move(state.absolute_path, str(dest), folder, round(max(conf, 0.85) if force else conf, 4), d.tier))

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
    # prune empty leftover dirs (best-effort)
    root = Path(plan.root)
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir()
            except OSError:
                pass
    return n

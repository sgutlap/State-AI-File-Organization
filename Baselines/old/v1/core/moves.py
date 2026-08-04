from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.cascade import Cascade
from core.custom_route import CustomRouter, get_router
from core.model import Student
from core.scan import FileState, scan_folder
from core.user_bins import apply_bins, custom_folders, folder_descriptions, folder_map, load_prefs

COPY_SKIP = shutil.ignore_patterns(".git", "__pycache__", ".venv", ".DS_Store", "node_modules")

SOURCE_TOPS = frozenset({
    "ignored_files", "partial_sync", "conflicted_copies", "shared_with_me",
    "multi_device", "from_inbox", "from_spam", "from_sent",
    "attachments_by_date", "extracted_zips",
})

DUMP_TOPS = frozenset({
    "downloads", "download", "desktop", "documents", "inbox", "tmp", "temp",
    "misc", "new folder", "new folder (2)",
})

SYNC_MARKER_RE = re.compile(
    r"^(?:desktop\.ini|thumbs\.db|\.ds_store|com\.apple\.timemachine\.supported|"
    r"\.localized|\.dropbox(?:_identity)?|\.sync(?:_identity)?|\.synced_folder_icon|"
    r"\._sync(?:_token|_placeholder)?|link_only_file\.txt|moved_files_here\.txt|"
    r"file_[123]\.txt)$",
    re.IGNORECASE,
)
SYNC_MARKER_SUBSTR = (
    ".dropbox", ".sync", "._sync", ".localized", "timemachine",
    "sync_token", "sync_identity", "sync_placeholder", "synced_folder",
)

NOISE_RE = re.compile(
    r"^(?P<stem>.+?)(?:\s*\((?P<n>\d+)\)|\s*[-_]?\s*copy(?:\s*\((?P<n2>\d+)\))?)$",
    re.IGNORECASE,
)

EXT_OVERRIDE = {
    ".png": "media/images", ".jpg": "media/images", ".jpeg": "media/images",
    ".gif": "media/images", ".webp": "media/images", ".svg": "media/images",
    ".fig": "media/images", ".psd": "media/images",
    ".mp4": "media/audio_video", ".mov": "media/audio_video",
    ".mp3": "media/audio_video", ".wav": "media/audio_video",
    ".zip": "archives", ".rar": "archives", ".7z": "archives",
    ".dmg": "archives", ".iso": "archives",
    ".csv": "data/datasets", ".parquet": "data/datasets", ".tsv": "data/datasets",
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


def load_student(ckpt="artifacts/model", quiet: bool = False):
    path = Path(ckpt)
    weight = path / "model.pt"
    if not weight.exists():
        raise FileNotFoundError(
            f"missing weights: {weight}\n"
            "Copy model.pt (~250MB) into artifacts/model/ "
            "(it is gitignored and not in the repo)."
        )
    s = Student(ckpt=path)
    if not quiet:
        print("loaded", path)
    return s


def duplicate_organized(src: Path) -> Path:
    src = src.resolve()
    parent = src.parent
    dst = parent / f"{src.name} Organized"
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
    return bool(SYNC_MARKER_RE.match(low)) or any(s in low for s in SYNC_MARKER_SUBSTR)


def _normalize_name(filename: str) -> str | None:
    p = Path(filename)
    m = NOISE_RE.match(p.stem)
    if not m:
        return None
    clean = m.group("stem").strip()
    if not clean or clean == p.stem:
        return None
    return f"{clean}{p.suffix}"


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


def _project_roots(states: list[FileState]) -> dict[str, str]:
    members: dict[str, str] = {}
    by_parent: dict[str, list[FileState]] = {}
    for s in states:
        parent = str(Path(s.relative_path).parent).replace("\\", "/")
        by_parent.setdefault(parent, []).append(s)

    for parent, files in by_parent.items():
        names = {f.metadata.filename.lower() for f in files}
        if not any(m.lower() in names for m in PROJECT_MARKERS):
            continue
        top = parent.split("/")[0] if parent not in (".", "") else ""
        if top.lower() in SOURCE_TOPS or top.lower() in DUMP_TOPS:
            continue
        if parent.lower() in DUMP_TOPS or Path(parent).name.lower() in DUMP_TOPS:
            continue
        proj = parent if parent not in (".", "") else "."
        for f in files:
            members[f.relative_path] = proj
    return members


def _resolve_folder(
    state: FileState,
    cat: str,
    conf: float,
    *,
    router: CustomRouter | None,
    fmap: dict[str, str],
    unknown: str,
    conf_threshold: float,
    force: bool,
    probs: dict[str, float] | None = None,
    hard_seed: bool = False,
) -> tuple[str, float]:
    """Pick destination folder — custom open-vocab taxonomy OR default/rename map."""
    if router is not None:
        if hard_seed or (force and cat):
            folder = router.map_category(cat)
            return folder, max(conf, 0.85)
        seed = unknown if (conf < conf_threshold and not force) else cat
        folder, rconf = router.route(
            state,
            seed_category=seed,
            probs=probs,
            hard_seed=False,
        )
        if conf < conf_threshold and not force:
            return folder, max(rconf, 0.40)
        return folder, max(conf, rconf)

    folder = apply_bins(cat, fmap)
    if conf < conf_threshold and not force:
        folder = apply_bins(unknown, fmap)
        conf = max(conf, 0.40)
    return folder, conf


def build_plan(root: str, student: Student, conf_threshold: float = 0.50) -> Plan:
    prefs = load_prefs()
    fmap = folder_map(prefs)
    folders = custom_folders(prefs)
    descs = folder_descriptions(prefs)
    router = get_router(student, folders, descriptions=descs) if folders else None
    cascade = Cascade(student, threshold=max(conf_threshold, 0.65))
    root_path = Path(root).resolve()
    plan = Plan(root=str(root_path))
    needed: set[str] = set()
    used_dst: set[str] = set()
    claimed: set[str] = set()

    states = scan_folder(root)
    project_map = _project_roots(states)

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
        keepers = sorted(group, key=lambda x: (len(x.relative_path), x.relative_path))
        for dup in keepers[1:]:
            dest_dir = root_path / quarantine
            dest = _disambiguate(dest_dir / Path(dup.absolute_path).name, used_dst)
            needed.add(str(dest_dir))
            used_dst.add(str(dest))
            claimed.add(dup.absolute_path)
            plan.moves.append(Move(dup.absolute_path, str(dest), quarantine, 0.99, "dedupe"))

    for state in states:
        if state.absolute_path in claimed:
            continue

        d = cascade.decide(state)
        cat, conf = d.category, d.confidence
        ext = (state.metadata.extension or "").lower()
        is_sync = _is_sync_marker(state.metadata.filename)
        probs = d.probs
        hard_seed = d.tier == "heuristic"

        if is_sync:
            cat = "misc/uncategorized"
            conf = max(conf, 0.90)
            hard_seed = True
        elif ext in EXT_OVERRIDE and not any(
            x in state.metadata.filename.lower()
            for x in ("untitled", "temp", "download", "file", "document", "empty")
        ):
            cat = EXT_OVERRIDE[ext]
            hard_seed = True

        src_top = Path(state.relative_path).parts[0] if Path(state.relative_path).parts else ""
        force = src_top in SOURCE_TOPS or is_sync

        folder, conf = _resolve_folder(
            state,
            cat,
            conf,
            router=router,
            fmap=fmap,
            unknown=student.taxonomy.unknown_class,
            conf_threshold=conf_threshold,
            force=force,
            probs=probs,
            hard_seed=hard_seed,
        )

        proj = project_map.get(state.relative_path)
        if proj and proj not in (".", "") and Path(proj).parts[0] not in SOURCE_TOPS:
            keep = (
                router.map_category("code/projects")
                if router is not None
                else apply_bins("code/projects", fmap)
            )
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
        plan.moves.append(
            Move(state.absolute_path, str(dest), folder, round(max(conf, 0.85) if force else conf, 4), d.tier)
        )

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
    root = Path(plan.root)
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir()
            except OSError:
                pass
    return n

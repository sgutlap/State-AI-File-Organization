from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import streamlit as st

from core.moves import apply_plan, build_plan, load_student

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "artifacts" / "student_model_ckpt"
RUNS = ROOT / "artifacts" / "ui_runs"
IGNORE = {".venv", ".git", "__pycache__", ".DS_Store"}


def tree_stats(folder: Path) -> dict:
    files, dirs, depth = 0, 0, 0
    top = []
    if not folder.is_dir():
        return {"files": 0, "dirs": 0, "depth": 0, "top": []}
    for p in folder.rglob("*"):
        if any(part in IGNORE for part in p.parts):
            continue
        rel = p.relative_to(folder)
        depth = max(depth, len(rel.parts))
        if p.is_file():
            files += 1
        elif p.is_dir():
            dirs += 1
    try:
        top = sorted(
            [c.name + ("/" if c.is_dir() else "") for c in folder.iterdir() if c.name not in IGNORE]
        )[:20]
    except OSError:
        top = []
    return {"files": files, "dirs": dirs, "depth": depth, "top": top}


def snapshot_names(folder: Path, limit: int = 40) -> list[str]:
    out = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        if any(part in IGNORE for part in p.parts):
            continue
        out.append(str(p.relative_to(folder)))
        if len(out) >= limit:
            break
    return out


@st.cache_resource(show_spinner="loading model...")
def get_student():
    return load_student(str(CKPT))


def copy_folder(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".DS_Store"),
    )


def run_stateai(folder: Path, student) -> dict:
    before = tree_stats(folder)
    before_files = snapshot_names(folder)
    t0 = time.perf_counter()
    plan = build_plan(str(folder), student)
    n = apply_plan(plan)
    ms = (time.perf_counter() - t0) * 1000
    after = tree_stats(folder)
    return {
        "ok": True,
        "name": "state-ai",
        "ms": ms,
        "moves": n,
        "planned": len(plan.moves),
        "before": before,
        "after": after,
        "before_files": before_files,
        "after_files": snapshot_names(folder),
        "plan_preview": [
            f"{Path(m.src).name} -> {Path(m.dst).relative_to(folder)}"
            for m in plan.moves[:25]
        ],
        "extra": "",
    }


def run_agy(folder: Path, timeout: int = 300) -> dict:
    before = tree_stats(folder)
    before_files = snapshot_names(folder)
    if not shutil.which("agy"):
        return {
            "ok": False,
            "name": "agy",
            "ms": 0,
            "moves": 0,
            "before": before,
            "after": before,
            "before_files": before_files,
            "after_files": before_files,
            "plan_preview": [],
            "extra": "agy not on PATH",
        }

    prompt = (
        f"Organize / clean this folder:\n{folder.resolve()}\n\n"
        "Goals:\n"
        "1. Put loose files into sensible folders (documents, media, code, data, archives, misc).\n"
        "2. Rename junk names if obvious (Untitled, Copy, (1)).\n"
        "3. Deduplicate exact copies under _duplicates/ if needed.\n"
        "4. Keep projects together.\n"
        "APPLY: you MAY create folders, rename, and move files under this path only.\n"
        "When done, print a short summary (folders created, files moved). "
        "If you know token usage, include it."
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["agy", "-p", prompt, "--add-dir", str(folder.resolve())],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(folder),
        )
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        err = None if proc.returncode == 0 else f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        out, err = "", f"timed out after {timeout}s"
    ms = (time.perf_counter() - t0) * 1000
    after = tree_stats(folder)
    after_files = snapshot_names(folder)
    # rough "steps" = lines that look like tool/file ops in output
    steps = sum(
        1
        for line in out.splitlines()
        if any(k in line.lower() for k in ("move", "mkdir", "renam", "creat", "wrote", "edit"))
    )
    return {
        "ok": err is None,
        "name": "agy",
        "ms": ms,
        "moves": max(0, before["files"] - after["files"]) + abs(len(after_files) - len(before_files)),
        "before": before,
        "after": after,
        "before_files": before_files,
        "after_files": after_files,
        "plan_preview": [],
        "extra": (err or "") + ("\n" if err and out else "") + out[:2500],
        "steps_guess": steps,
        "tokens": "n/a (agy rarely prints tokens)",
    }


def show_result(title: str, r: dict):
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric("time", f"{r['ms']:.0f} ms")
    c2.metric("files", r["after"]["files"])
    c3.metric("moves", r.get("moves", 0))

    left, right = st.columns(2)
    with left:
        st.markdown("**before**")
        st.code("\n".join(r["before"].get("top") or ["(empty)"]) or "(empty)")
        st.caption("paths")
        st.code("\n".join(r.get("before_files") or []) or "(empty)")
    with right:
        st.markdown("**after**")
        st.code("\n".join(r["after"].get("top") or ["(empty)"]) or "(empty)")
        st.caption("paths")
        st.code("\n".join(r.get("after_files") or []) or "(empty)")

    if r.get("plan_preview"):
        st.markdown("**moves**")
        st.code("\n".join(r["plan_preview"]))
    if r.get("extra"):
        with st.expander("agy log"):
            st.text(r["extra"])
    if "steps_guess" in r:
        st.caption(f"agy steps ~{r['steps_guess']} · tokens: {r.get('tokens')}")


def main():
    st.set_page_config(page_title="state-ai", layout="wide")
    st.title("state-ai")

    default = str((ROOT / "Messy-Folder").resolve())
    folder_in = st.text_input("folder", value=default)

    do_apply = st.checkbox("edit this folder", value=False)
    do_agy = st.checkbox("compare with agy", value=False)

    # warm cache early so first click isn't shocking
    if CKPT.exists():
        _ = get_student()
    else:
        st.error(f"missing checkpoint: {CKPT / 'student_model.pt'}")
        st.stop()

    if not st.button("run", type="primary"):
        return

    src = Path(folder_in).expanduser().resolve()
    if not src.is_dir():
        st.error(f"not a folder: {src}")
        return

    student = get_student()
    RUNS.mkdir(parents=True, exist_ok=True)

    if do_agy or not do_apply:
        work = RUNS / f"stateai_{int(time.time())}"
        copy_folder(src, work)
        st.write(f"folder is in: `{work.name}`")
        with st.spinner("state-ai organizing..."):
            r_sa = run_stateai(work, student)
        show_result("state-ai", r_sa)
    else:
        st.write(f"folder is in: `{src.name}`")
        with st.spinner("state-ai organizing..."):
            r_sa = run_stateai(src, student)
        show_result("state-ai", r_sa)

    if do_agy:
        agy_work = RUNS / f"agy_{int(time.time())}"
        copy_folder(src, agy_work)
        st.write(f"folder is in: `{agy_work.name}`")
        with st.spinner("agy organizing..."):
            r_agy = run_agy(agy_work)
        show_result("agy", r_agy)

        st.subheader("compare")
        a, b = st.columns(2)
        a.metric("state-ai time", f"{r_sa['ms']:.0f} ms")
        b.metric("agy time", f"{r_agy['ms']:.0f} ms")
        st.write(
            f"state-ai moves {r_sa.get('moves', 0)} · agy moves {r_agy.get('moves', 0)}"
        )



if __name__ == "__main__":
    main()

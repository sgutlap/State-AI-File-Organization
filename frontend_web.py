from __future__ import annotations

import shutil
import time
from pathlib import Path

import streamlit as st

from core.moves import apply_plan, build_plan, load_student
from core.taxonomy import list_categories
from core.user_bins import folder_map, load_prefs, save_prefs, set_bin

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "artifacts" / "student_model_ckpt"
RUNS = ROOT / "artifacts" / "ui_runs"


@st.cache_resource(show_spinner="loading model...")
def get_student():
    return load_student(str(CKPT), quiet=True)


def main():
    st.set_page_config(page_title="state-ai", layout="wide")
    st.title("state-ai")

    if not CKPT.exists():
        st.error(f"missing checkpoint: {CKPT}")
        st.stop()

    folder_in = st.text_input("folder", value=str((ROOT / "Messy-Folder").resolve()))
    edit_here = st.checkbox("edit this folder", value=False)

    with st.expander("custom folders (optional)"):
        prefs = load_prefs()
        bins_on = st.checkbox(
            "use my own folder names",
            value=bool((prefs.get("user_bins") or {}).get("enabled")),
        )
        student = get_student()
        fmap = folder_map(prefs)
        edits = {}
        for tid in student.taxonomy.classes:
            edits[tid] = st.text_input(tid, value=fmap.get(tid, tid), key=f"bin_{tid}")
        if st.button("save folder names"):
            if bins_on:
                for tid, folder in edits.items():
                    if folder.strip():
                        set_bin(tid, folder.strip(), enable=True)
            else:
                prefs = load_prefs()
                prefs.setdefault("user_bins", {})["enabled"] = False
                save_prefs(prefs)
            st.success("saved")
            st.rerun()

    if not st.button("run", type="primary"):
        return

    src = Path(folder_in).expanduser().resolve()
    if not src.is_dir():
        st.error(f"not a folder: {src}")
        return

    student = get_student()
    if edit_here:
        work = src
        st.write(f"editing `{work}`")
    else:
        RUNS.mkdir(parents=True, exist_ok=True)
        work = RUNS / f"run_{int(time.time())}"
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(
            src,
            work,
            ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".DS_Store"),
        )
        st.write(f"copy → `{work.name}`")

    with st.spinner("organizing..."):
        t0 = time.perf_counter()
        plan = build_plan(str(work), student)
        n = apply_plan(plan)
        ms = (time.perf_counter() - t0) * 1000

    c1, c2, c3 = st.columns(3)
    c1.metric("time", f"{ms:.0f} ms")
    c2.metric("moves", n)
    c3.metric("planned", len(plan.moves))

    left, right = st.columns(2)
    with left:
        st.markdown("**before**")
        try:
            st.code("\n".join(sorted(p.name for p in src.iterdir())[:25]) or "(empty)")
        except OSError:
            st.code("(can't list)")
    with right:
        st.markdown("**after**")
        try:
            st.code("\n".join(sorted(p.name for p in work.iterdir())[:25]) or "(empty)")
        except OSError:
            st.code("(can't list)")

    if plan.moves:
        st.markdown("**moves**")
        st.code(
            "\n".join(
                f"{Path(m.src).name}  →  {m.category}"
                for m in plan.moves[:40]
            )
        )


if __name__ == "__main__":
    main()

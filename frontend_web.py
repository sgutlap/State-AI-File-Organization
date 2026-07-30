from __future__ import annotations

import shutil
import time
from pathlib import Path

import streamlit as st

from core.moves import apply_plan, build_plan, load_student
from core.user_bins import (
    active_folders,
    clear_bins,
    default_folders,
    is_customized,
    load_prefs,
    set_active_folders,
)

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

    with st.expander("Taxonomy", expanded=True):
        prefs = load_prefs()
        stock = default_folders()
        current = active_folders(prefs)
        customized = is_customized(prefs)

        st.caption(
            "Starts as the default taxonomy. Add lines, delete lines, or rename — "
            "then save. This is the full set of folders files can go into."
        )
        folders_text = st.text_area(
            "one folder per line",
            value="\n".join(current),
            height=180,
            key="dest_folders_text",
            help="Default bins stay unless you delete them. New names are added on.",
        )
        c1, c2 = st.columns(2)
        if c1.button("save destinations", type="primary"):
            parsed = [ln.strip() for ln in folders_text.splitlines() if ln.strip()]
            if not parsed:
                st.error("need at least one destination folder")
            else:
                set_active_folders(parsed)
                if set(parsed) == set(stock):
                    st.success("stock default taxonomy")
                else:
                    added = [f for f in parsed if f not in stock]
                    removed = [f for f in stock if f not in parsed]
                    bits = []
                    if added:
                        bits.append("added " + ", ".join(added))
                    if removed:
                        bits.append("removed " + ", ".join(removed))
                    st.success(" · ".join(bits) if bits else "destinations saved")
                st.rerun()
        if c2.button("revert to default"):
            clear_bins()
            st.success("reverted to default taxonomy")
            st.rerun()

        if customized:
            st.caption("active → " + " · ".join(current))
        else:
            st.caption("active → default taxonomy")

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("time", f"{ms:.0f} ms")
    c2.metric("moves", n)
    c3.metric("planned", len(plan.moves))
    c4.metric("taxonomy", "customized" if is_customized() else "default")

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
            tops = sorted(p.name for p in work.iterdir())
            st.code("\n".join(tops[:25]) or "(empty)")
        except OSError:
            st.code("(can't list)")

    if plan.moves:
        st.markdown("**moves**")
        st.code(
            "\n".join(
                f"{Path(m.src).name}  →  {m.category}"
                for m in plan.moves[:50]
            )
        )


if __name__ == "__main__":
    main()

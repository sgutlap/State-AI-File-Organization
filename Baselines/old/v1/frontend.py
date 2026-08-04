import json
import os
import sys
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import messagebox, ttk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SERVE_URL = "http://127.0.0.1:18765"


def _plan_via_server(folder: str, threshold: float = 0.5):
    """Use warm serve.py if running to keep the Windows UI snappy."""
    from core.moves import Move, Plan

    payload = json.dumps({"folder": folder, "threshold": threshold}).encode()
    req = urllib.request.Request(
        f"{SERVE_URL}/plan",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    if not data.get("ok"):
        return None

    plan = Plan(root=str(data.get("root") or folder))
    plan.dirs = list(data.get("dirs") or [])
    for move in data.get("moves") or []:
        plan.moves.append(
            Move(
                src=str(move["src"]),
                dst=str(move["dst"]),
                category=str(move.get("category") or ""),
                confidence=float(move.get("confidence") or 0),
                tier=str(move.get("tier") or "student"),
            )
        )
    return plan


def main():
    if len(sys.argv) < 2:
        print("Usage: python frontend.py <folder>")
        sys.exit(1)

    folder_path = str(Path(sys.argv[1]).resolve())
    if not Path(folder_path).is_dir():
        messagebox.showerror("Organize", f"Not a folder:\n{folder_path}")
        sys.exit(1)

    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)
    sys.path.insert(0, str(root_dir))

    from core.moves import apply_plan, build_plan, load_student
    from core.taxonomy import list_categories
    from core.user_bins import normalize_folder

    categories = list_categories()
    original_categories = set(categories)
    plan = None

    root = tk.Tk()
    root.title("Organize")
    root.geometry("700x520")
    root.minsize(560, 400)

    content = tk.Frame(root, padx=14, pady=12)
    content.pack(fill="both", expand=True)

    def clear_content():
        root.unbind_all("<MouseWheel>")
        for child in content.winfo_children():
            child.destroy()

    def destination_for(move, category: str) -> Path:
        destination = Path(folder_path) / category / Path(move.dst).name
        used = {m.dst for m in plan.moves if m is not move}
        number = 1
        while str(destination) in used or (
            destination.exists() and destination.resolve() != Path(move.src).resolve()
        ):
            destination = destination.with_name(
                f"{Path(move.dst).stem}_{number}{Path(move.dst).suffix}"
            )
            number += 1
        return destination

    def fit_plan_to_taxonomy():
        if not categories:
            return
        fallback = "misc/uncategorized" if "misc/uncategorized" in categories else categories[0]
        for move in plan.moves:
            if move.category in original_categories and move.category not in categories:
                move.category = fallback
                move.dst = str(destination_for(move, fallback))
        plan.dirs = sorted({str(Path(move.dst).parent) for move in plan.moves})

    def render_done(moved: int):
        clear_content()
        panel = tk.Frame(content)
        panel.pack(expand=True)
        tk.Label(panel, text="Organization complete", font=("Segoe UI", 16, "bold")).pack(pady=6)
        tk.Label(panel, text=f"Moved {moved} files.", font=("Segoe UI", 10)).pack(pady=4)
        tk.Button(panel, text="Done", command=root.destroy, width=14).pack(pady=18)

    def apply_changes():
        try:
            plan.dirs = sorted({str(Path(move.dst).parent) for move in plan.moves})
            moved = apply_plan(plan)
        except OSError as error:
            messagebox.showerror("Organize", f"Could not apply changes:\n{error}", parent=root)
            return
        render_done(moved)

    def render_review():
        clear_content()

        tk.Label(content, text="Review proposed changes", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(
            content,
            text=f"{len(plan.moves)} files will be moved. Nothing changes until you click Apply.",
            font=("Segoe UI", 9),
            fg="#555",
        ).pack(anchor="w", pady=(2, 10))

        container = tk.Frame(content)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        rows = tk.Frame(canvas)
        rows.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        header = tk.Frame(rows)
        header.pack(fill="x")
        tk.Label(header, text="File", width=38, anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left")
        tk.Label(header, text="Move to", width=30, anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left")

        for move in plan.moves:
            row = tk.Frame(rows)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=Path(move.src).name, width=38, anchor="w", font=("Segoe UI", 9)).pack(
                side="left", padx=(0, 6)
            )
            combo = ttk.Combobox(row, values=categories, width=30, state="readonly", font=("Segoe UI", 9))
            combo.set(move.category)
            combo.pack(side="left")

            def change_destination(_event, selected=combo, selected_move=move):
                selected_move.category = selected.get()
                selected_move.dst = str(destination_for(selected_move, selected.get()))

            combo.bind("<<ComboboxSelected>>", change_destination)

        def on_wheel(event):
            if event.delta:
                canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        canvas.bind_all("<MouseWheel>", on_wheel)

        buttons = tk.Frame(content, pady=10)
        buttons.pack()
        tk.Button(buttons, text="Edit taxonomy", command=render_taxonomy, width=14).pack(side="left", padx=4)
        tk.Button(buttons, text="Apply", command=apply_changes, width=12, bg="#c8e6c9").pack(side="left", padx=4)
        tk.Button(buttons, text="Cancel", command=root.destroy, width=10).pack(side="left", padx=4)

    def organize_files(status_label):
        nonlocal plan
        if not categories:
            messagebox.showerror("Organize", "Add at least one taxonomy bin first.", parent=root)
            return

        status_label.config(text="Looking for files...")
        root.update_idletasks()
        warmed = _plan_via_server(folder_path)
        if warmed is not None:
            plan = warmed
        else:
            status_label.config(text="Loading model and looking for files...")
            root.update_idletasks()
            student = load_student(str(root_dir / "artifacts" / "model"), quiet=True)
            plan = build_plan(folder_path, student)
        fit_plan_to_taxonomy()
        render_review()

    def render_taxonomy():
        clear_content()

        tk.Label(content, text="Edit taxonomy", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(
            content,
            text="Add or remove destination bins before organizing your files.",
            font=("Segoe UI", 9),
            fg="#555",
        ).pack(anchor="w", pady=(2, 10))

        list_frame = tk.Frame(content)
        list_frame.pack(fill="both", expand=True)
        listbox = tk.Listbox(list_frame, font=("Segoe UI", 10), selectmode="extended")
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=list_scroll.set)
        listbox.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")

        def refresh_list():
            listbox.delete(0, "end")
            for category in categories:
                listbox.insert("end", category)

        refresh_list()

        add_row = tk.Frame(content, pady=8)
        add_row.pack(fill="x")
        new_category = tk.StringVar()
        entry = tk.Entry(add_row, textvariable=new_category, font=("Segoe UI", 10))
        entry.pack(side="left", fill="x", expand=True)

        def add_bin(_event=None):
            try:
                category = normalize_folder(new_category.get())
            except ValueError as error:
                messagebox.showerror("Organize", str(error), parent=root)
                return
            if category not in categories:
                categories.append(category)
                refresh_list()
            new_category.set("")
            entry.focus_set()

        def remove_bins():
            for index in reversed(listbox.curselection()):
                categories.pop(index)
            refresh_list()

        entry.bind("<Return>", add_bin)
        tk.Button(add_row, text="Add bin", command=add_bin, width=10).pack(side="left", padx=(6, 0))
        tk.Button(add_row, text="Remove selected", command=remove_bins, width=15).pack(side="left", padx=(6, 0))

        status = tk.Label(content, text="", font=("Segoe UI", 9), fg="#555")
        status.pack(anchor="w")

        buttons = tk.Frame(content, pady=10)
        buttons.pack()
        tk.Button(
            buttons,
            text="Organize files",
            command=lambda: organize_files(status),
            width=16,
            bg="#c8e6c9",
        ).pack(side="left", padx=4)
        tk.Button(buttons, text="Cancel", command=root.destroy, width=10).pack(side="left", padx=4)

    render_taxonomy()
    root.mainloop()


if __name__ == "__main__":
    main()

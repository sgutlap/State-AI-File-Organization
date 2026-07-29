from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ADD_NEW = "+ Add new folder..."


def main():
    if len(sys.argv) < 2:
        print("Usage: python frontend.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not Path(folder_path).is_dir():
        messagebox.showerror("Organize", f"Not a folder:\n{folder_path}")
        sys.exit(1)

    splash = tk.Tk()
    splash.title("Organize")
    splash.geometry("280x90")
    splash.resizable(False, False)
    tk.Label(splash, text="Analyzing files…", font=("Segoe UI", 11)).pack(expand=True)
    splash.update()

    ROOT = Path(__file__).resolve().parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from core.moves import apply_plan, build_plan, load_student
    from core.taxonomy import add_category, list_categories
    from core.user_bins import folder_map, load_prefs, set_bin

    student = load_student(str(ROOT / "artifacts" / "student_model_ckpt"))
    plan = build_plan(folder_path, student)
    plan_root = Path(plan.root)
    splash.destroy()

    if not plan.moves:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Organize Files", "Nothing to move — folder already looks organized.")
        return

    categories = list_categories()
    prefs = load_prefs()
    bins_on = bool((prefs.get("user_bins") or {}).get("enabled"))

    root = tk.Tk()
    root.title("Organize Files — Preview")
    root.geometry("720x520")
    root.minsize(560, 400)

    # Header
    header = tk.Frame(root, padx=12, pady=8)
    header.pack(fill="x")
    tk.Label(header, text="Organize Files", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(
        header,
        text=f"{len(plan.moves)} moves in  {folder_path}",
        font=("Segoe UI", 9),
        fg="#444",
    ).pack(anchor="w")
    tk.Label(
        header,
        text="Change a folder name below if you disagree. Optional: map whole categories via User bins.",
        font=("Segoe UI", 9),
        fg="#666",
    ).pack(anchor="w", pady=(2, 0))

    # User bins strip
    bins_frame = tk.LabelFrame(root, text="User bins (optional)", padx=8, pady=6)
    bins_frame.pack(fill="x", padx=12, pady=(0, 6))

    bins_var = tk.BooleanVar(value=bins_on)

    def refresh_categories():
        categories.clear()
        categories.extend(list_categories())
        for c in combos:
            cur = c.get()
            c["values"] = categories + [ADD_NEW]
            if cur in categories:
                c.set(cur)

    def toggle_bins():
        p = load_prefs()
        ub = dict(p.get("user_bins") or {"enabled": False, "folder_map": {}})
        ub["enabled"] = bool(bins_var.get())
        p["user_bins"] = ub
        from core.user_bins import save_prefs

        save_prefs(p)
        refresh_categories()
        status.set("User bins " + ("on" if ub["enabled"] else "off"))

    tk.Checkbutton(
        bins_frame,
        text="Use custom folder names for categories",
        variable=bins_var,
        command=toggle_bins,
        font=("Segoe UI", 9),
    ).pack(anchor="w")

    map_row = tk.Frame(bins_frame)
    map_row.pack(fill="x", pady=(4, 0))
    tax_ids = student.taxonomy.classes
    fmap = folder_map(prefs)
    tax_var = tk.StringVar(value=tax_ids[0] if tax_ids else "")
    folder_var = tk.StringVar(value=fmap.get(tax_var.get(), tax_var.get()))

    def on_tax_pick(_e=None):
        folder_var.set(fmap.get(tax_var.get(), tax_var.get()))

    ttk.Combobox(map_row, textvariable=tax_var, values=tax_ids, width=22, state="readonly").pack(
        side="left", padx=(0, 6)
    )
    tk.Label(map_row, text="→", font=("Segoe UI", 10)).pack(side="left")
    tk.Entry(map_row, textvariable=folder_var, width=22).pack(side="left", padx=6)

    def save_one_bin():
        try:
            set_bin(tax_var.get(), folder_var.get(), enable=True)
            bins_var.set(True)
            fmap.clear()
            fmap.update(folder_map())
            refresh_categories()
            # Remap matching moves to new folder
            for m in plan.moves:
                # only remap if move still uses old taxonomy-style path for this id
                if m.category in (tax_var.get(), fmap.get(tax_var.get())):
                    new_folder = folder_var.get().strip()
                    m.category = new_folder
                    m.dst = str(_dest_for(m, new_folder))
                    # update combo display
            for c, m in zip(combos, plan.moves):
                c.set(m.category)
            status.set(f"Saved bin: {tax_var.get()} → {folder_var.get()}")
        except ValueError as e:
            messagebox.showerror("User bins", str(e))

    tk.Button(map_row, text="Save mapping", command=save_one_bin).pack(side="left", padx=4)
    tax_var.trace_add("write", lambda *_: on_tax_pick())

    # Scrollable move list
    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=12, pady=4)

    canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    rows_frame = tk.Frame(canvas)
    rows_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=rows_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    hdr = tk.Frame(rows_frame)
    hdr.pack(fill="x", pady=(0, 4))
    tk.Label(hdr, text="File", width=34, anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left")
    tk.Label(hdr, text="Destination folder", width=28, anchor="w", font=("Segoe UI", 9, "bold")).pack(
        side="left"
    )

    combos: list[ttk.Combobox] = []

    def _dest_for(move, category: str) -> Path:
        dest_dir = plan_root / category
        dest = dest_dir / Path(move.src).name
        n = 1
        while dest.exists() and dest.resolve() != Path(move.src).resolve():
            dest = dest_dir / f"{Path(move.src).stem}_{n}{Path(move.src).suffix}"
            n += 1
        return dest

    def on_combo_change(_event, combo: ttk.Combobox, move):
        choice = combo.get()
        if choice == ADD_NEW:
            new_name = simpledialog.askstring(
                "New folder",
                f"Folder name for '{Path(move.src).name}':",
                parent=root,
            )
            if new_name and new_name.strip():
                try:
                    add_category(new_name, folder_path)
                except ValueError as e:
                    messagebox.showerror("Invalid name", str(e))
                    combo.set(move.category)
                    return
                if new_name not in categories:
                    categories.append(new_name)
                for c in combos:
                    c["values"] = categories + [ADD_NEW]
                combo.set(new_name)
                move.category = new_name
                move.dst = str(_dest_for(move, new_name))
            else:
                combo.set(move.category)
        else:
            move.category = choice
            move.dst = str(_dest_for(move, choice))

    for move in plan.moves:
        row = tk.Frame(rows_frame)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=Path(move.src).name, width=34, anchor="w", font=("Segoe UI", 9)).pack(
            side="left", padx=(0, 8)
        )
        combo = ttk.Combobox(
            row,
            values=categories + [ADD_NEW],
            width=28,
            state="readonly",
            font=("Segoe UI", 9),
        )
        combo.set(move.category)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e, c=combo, m=move: on_combo_change(e, c, m))
        combos.append(combo)

    # Footer
    status = tk.StringVar(value="")
    tk.Label(root, textvariable=status, font=("Segoe UI", 8), fg="#666").pack(anchor="w", padx=12)

    btn_frame = tk.Frame(root, pady=10)
    btn_frame.pack()

    def on_apply():
        # refresh dirs from possibly-edited destinations
        plan.dirs = sorted({str(Path(m.dst).parent) for m in plan.moves})
        count = apply_plan(plan)
        messagebox.showinfo("Done", f"Moved {count} files.")
        root.destroy()

    def on_add_category():
        name = simpledialog.askstring(
            "New folder",
            f"Folder name (created under {folder_path}):",
            parent=root,
        )
        if not name or not name.strip():
            return
        try:
            add_category(name, folder_path)
        except ValueError as e:
            messagebox.showerror("Invalid name", str(e))
            return
        refresh_categories()
        messagebox.showinfo("Added", f"Created: {Path(folder_path) / name.strip()}")

    tk.Button(btn_frame, text="+ Add folder", command=on_add_category, width=12).pack(
        side="left", padx=4
    )
    tk.Button(btn_frame, text="Apply moves", command=on_apply, width=14, bg="#c8e6c9").pack(
        side="left", padx=4
    )
    tk.Button(btn_frame, text="Cancel", command=root.destroy, width=10).pack(side="left", padx=4)

    root.mainloop()


if __name__ == "__main__":
    main()

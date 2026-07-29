import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ADD_NEW = "+ new folder..."


def main():
    if len(sys.argv) < 2:
        print("Usage: python frontend.py <folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not Path(folder_path).is_dir():
        messagebox.showerror("Organize", f"Not a folder:\n{folder_path}")
        sys.exit(1)

    splash = tk.Tk()
    splash.title("Organize")
    splash.geometry("260x80")
    splash.resizable(False, False)
    tk.Label(splash, text="Looking at files...", font=("Segoe UI", 10)).pack(expand=True)
    splash.update()

    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)
    sys.path.insert(0, str(root_dir))

    from core.moves import apply_plan, build_plan, load_student
    from core.taxonomy import add_category, list_categories
    from core.user_bins import folder_map, load_prefs, save_prefs, set_bin

    student = load_student(str(root_dir / "artifacts" / "student_model_ckpt"), quiet=True)
    plan = build_plan(folder_path, student)
    plan_root = Path(plan.root)
    splash.destroy()

    if not plan.moves:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Organize", "Nothing to move.")
        return

    categories = list_categories()
    prefs = load_prefs()
    bins_on = bool((prefs.get("user_bins") or {}).get("enabled"))

    root = tk.Tk()
    root.title("Organize")
    root.geometry("680x500")
    root.minsize(520, 360)

    top = tk.Frame(root, padx=10, pady=8)
    top.pack(fill="x")
    tk.Label(top, text=f"{len(plan.moves)} files to move", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    tk.Label(top, text=folder_path, font=("Segoe UI", 8), fg="#555").pack(anchor="w")

    bins_frame = tk.LabelFrame(root, text="custom folders (optional)", padx=8, pady=4)
    bins_frame.pack(fill="x", padx=10, pady=4)
    bins_var = tk.BooleanVar(value=bins_on)
    tax_ids = student.taxonomy.classes
    fmap = folder_map(prefs)
    tax_var = tk.StringVar(value=tax_ids[0] if tax_ids else "")
    folder_var = tk.StringVar(value=fmap.get(tax_var.get(), tax_var.get()))
    combos: list[ttk.Combobox] = []
    status = tk.StringVar(value="")

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
        save_prefs(p)
        refresh_categories()
        status.set("custom folders on" if ub["enabled"] else "custom folders off")

    tk.Checkbutton(
        bins_frame,
        text="use my own folder names",
        variable=bins_var,
        command=toggle_bins,
        font=("Segoe UI", 9),
    ).pack(anchor="w")

    map_row = tk.Frame(bins_frame)
    map_row.pack(fill="x", pady=2)
    ttk.Combobox(map_row, textvariable=tax_var, values=tax_ids, width=20, state="readonly").pack(
        side="left", padx=(0, 4)
    )
    tk.Label(map_row, text="→").pack(side="left")
    tk.Entry(map_row, textvariable=folder_var, width=20).pack(side="left", padx=4)

    def on_tax_pick(*_):
        folder_var.set(fmap.get(tax_var.get(), tax_var.get()))

    tax_var.trace_add("write", on_tax_pick)

    def save_one_bin():
        tax_id = tax_var.get()
        new_folder = folder_var.get().strip()
        old_folder = fmap.get(tax_id, tax_id)
        try:
            set_bin(tax_id, new_folder, enable=True)
            bins_var.set(True)
            fmap.clear()
            fmap.update(folder_map())
            refresh_categories()
            for m in plan.moves:
                if m.category in (tax_id, old_folder):
                    m.category = new_folder
                    name = Path(m.dst).name
                    dest_dir = plan_root / new_folder
                    dest = dest_dir / name
                    n = 1
                    while dest.exists() and dest.resolve() != Path(m.src).resolve():
                        dest = dest_dir / f"{Path(name).stem}_{n}{Path(name).suffix}"
                        n += 1
                    m.dst = str(dest)
            for c, m in zip(combos, plan.moves):
                c.set(m.category)
            status.set(f"{tax_id} → {new_folder}")
        except ValueError as e:
            messagebox.showerror("Organize", str(e))

    tk.Button(map_row, text="save", command=save_one_bin).pack(side="left", padx=2)

    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=10, pady=4)
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    rows = tk.Frame(canvas)
    rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=rows, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def on_wheel(event):
        delta = -1 * int(event.delta / 120) if event.delta else 0
        if delta:
            canvas.yview_scroll(delta, "units")

    canvas.bind_all("<MouseWheel>", on_wheel)

    hdr = tk.Frame(rows)
    hdr.pack(fill="x")
    tk.Label(hdr, text="file", width=34, anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left")
    tk.Label(hdr, text="put in", width=28, anchor="w", font=("Segoe UI", 9, "bold")).pack(side="left")

    def dest_for(move, category: str) -> Path:
        dest_dir = plan_root / category
        dest = dest_dir / Path(move.src).name
        n = 1
        while dest.exists() and dest.resolve() != Path(move.src).resolve():
            dest = dest_dir / f"{Path(move.src).stem}_{n}{Path(move.src).suffix}"
            n += 1
        return dest

    def on_combo(_e, combo, move):
        choice = combo.get()
        if choice == ADD_NEW:
            name = simpledialog.askstring("New folder", "folder name:", parent=root)
            if name and name.strip():
                try:
                    add_category(name, folder_path)
                except ValueError as err:
                    messagebox.showerror("Organize", str(err))
                    combo.set(move.category)
                    return
                if name not in categories:
                    categories.append(name)
                for c in combos:
                    c["values"] = categories + [ADD_NEW]
                combo.set(name)
                move.category = name
                move.dst = str(dest_for(move, name))
            else:
                combo.set(move.category)
        else:
            move.category = choice
            move.dst = str(dest_for(move, choice))

    for move in plan.moves:
        row = tk.Frame(rows)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=Path(move.src).name, width=34, anchor="w", font=("Segoe UI", 9)).pack(
            side="left", padx=(0, 6)
        )
        combo = ttk.Combobox(
            row, values=categories + [ADD_NEW], width=28, state="readonly", font=("Segoe UI", 9)
        )
        combo.set(move.category)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e, c=combo, m=move: on_combo(e, c, m))
        combos.append(combo)

    tk.Label(root, textvariable=status, font=("Segoe UI", 8), fg="#666").pack(anchor="w", padx=10)

    btns = tk.Frame(root, pady=8)
    btns.pack()

    def on_apply():
        plan.dirs = sorted({str(Path(m.dst).parent) for m in plan.moves})
        n = apply_plan(plan)
        messagebox.showinfo("Done", f"Moved {n} files.")
        root.destroy()

    def on_add():
        name = simpledialog.askstring("New folder", "folder name:", parent=root)
        if not name or not name.strip():
            return
        try:
            add_category(name, folder_path)
        except ValueError as err:
            messagebox.showerror("Organize", str(err))
            return
        refresh_categories()

    tk.Button(btns, text="+ folder", command=on_add, width=10).pack(side="left", padx=3)
    tk.Button(btns, text="Apply", command=on_apply, width=12, bg="#c8e6c9").pack(side="left", padx=3)
    tk.Button(btns, text="Cancel", command=root.destroy, width=10).pack(side="left", padx=3)

    root.mainloop()


if __name__ == "__main__":
    main()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
import sys
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from pathlib import Path
 
ADD_NEW_LABEL = "+ Add new category..."
 
 
def main():
    if len(sys.argv) < 2:
        print("Usage: python frontend.py <folder_path>")
        sys.exit(1)
 
    folder_path = sys.argv[1]
 
    splash = tk.Tk()
    splash.title("Organize")
    splash.geometry("250x80")
    lbl = tk.Label(splash, text="Analyzing files, please wait...", font=("Arial", 10))
    lbl.pack(expand=True)
    splash.update()
 
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.moves import load_student, build_plan, apply_plan
    from core.taxonomy import list_categories, add_category
 
    ckpt_path = "artifacts/student_model_ckpt"
    student = load_student(ckpt_path)
    plan = build_plan(folder_path, student)
    plan_root = Path(plan.root)
 
    splash.destroy()
 
    if not plan.moves:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Organize Files", "No files need to be moved!")
        return

    known_categories = list_categories()
 
    root = tk.Tk()
    root.title("Organize Files Preview")
    root.geometry("600x460")
 
    tk.Label(root, text=f"Files to move in: {folder_path}", font=("Arial", 11, "bold")).pack(pady=10)
    tk.Label(root, text="Don't agree with a category? Change it below.", font=("Arial", 9)).pack()
 
    def on_add_category():
        name = simpledialog.askstring(
            "New Category",
            f"Folder name (will be created in {folder_path}):",
            parent=root,
        )
        if not name or not name.strip():
            return
        try:
            updated = add_category(name, folder_path)  
        except ValueError as e:
            messagebox.showerror("Invalid name", str(e))
            return
        known_categories.clear()
        known_categories.extend(updated)
        for c in combos:
            c["values"] = known_categories + [ADD_NEW_LABEL]
        messagebox.showinfo("Category Added", f"Created folder: {Path(folder_path) / name.strip()}")
 
    tk.Button(root, text="+ Add Category", command=on_add_category).pack(pady=(4, 8))
 
    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=10, pady=5)
 
    canvas = tk.Canvas(container, borderwidth=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    rows_frame = tk.Frame(canvas)
 
    rows_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=rows_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
 
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
 
    combos: list[ttk.Combobox] = []
 
    def _dest_for(move, category: str) -> Path:
        """Recompute a destination path under the new category, avoiding collisions."""
        dest_dir = plan_root / category
        dest = dest_dir / Path(move.src).name
        n = 1
        while dest.exists() and dest.resolve() != Path(move.src).resolve():
            dest = dest_dir / f"{Path(move.src).stem}_{n}{Path(move.src).suffix}"
            n += 1
        return dest
 
    def on_combo_change(event, combo: ttk.Combobox, move):
        choice = combo.get()
        if choice == ADD_NEW_LABEL:
            new_name = simpledialog.askstring(
                "New Category",
                f"Folder name for '{Path(move.src).name}':",
                parent=root,
            )
            if new_name and new_name.strip():
                add_category(new_name, folder_path) 
                if new_name not in known_categories:
                    known_categories.append(new_name)
                for c in combos:
                    c["values"] = known_categories + [ADD_NEW_LABEL]
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
        row.pack(fill="x", pady=2)
 
        file_name = Path(move.src).name
        tk.Label(row, text=file_name, width=32, anchor="w").pack(side="left", padx=(0, 8))
 
        combo = ttk.Combobox(
            row,
            values=known_categories + [ADD_NEW_LABEL],
            width=28,
            state="readonly",
        )
        combo.set(move.category)  
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e, c=combo, m=move: on_combo_change(e, c, m))
        combos.append(combo)
 
    def on_apply():
        count = apply_plan(plan)
        messagebox.showinfo("Done", f"Successfully moved {count} files!")
        root.destroy()
 
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
 
    tk.Button(btn_frame, text="Apply Moves", command=on_apply, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(btn_frame, text="Cancel", command=root.destroy).pack(side="left", padx=5)
 
    root.mainloop()
 
 
if __name__ == "__main__":
    main()
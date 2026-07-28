import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import tkinter as tk
from pathlib import Path

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


    ckpt_path = "artifacts/student_model_ckpt"
    student = load_student(ckpt_path)
    plan = build_plan(folder_path, student)

    splash.destroy()

    if not plan.moves:
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Organize Files", "No files need to be moved!")
        return

    root = tk.Tk()
    root.title("Organize Files Preview")
    root.geometry("400x300")

    tk.Label(root, text=f"Files to move in: {folder_path}", font=("Arial", 11, "bold")).pack(pady=10)

    listbox = tk.Listbox(root, width=50, height=10)
    listbox.pack(pady=5)

    for move in plan.moves:
        file_name = Path(move.src).name
        listbox.insert(tk.END, f"{file_name} -> {move.category}")

    def on_apply():
        from tkinter import messagebox
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
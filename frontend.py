# Need to do this


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path




sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.moves import load_student, build_plan, apply_plan




def main():


    if len(sys.argv) < 2:
        print("Usage: python frontend.py <folder_path>")
        sys.exit(1)


    folder_path = sys.argv[1]
    ckpt_path = "artifacts/student_model_ckpt"




    student = load_student(ckpt_path)
    plan = build_plan(folder_path, student)


    if not plan.moves:
        messagebox.showinfo("Organize Files", "No files need to be moved!")
        return


    root = tk.Tk()
    root.title("Organize Files Preview")
    root.geometry("400x300")




    lbl = tk.Label(root, text=f"Files to move in: {folder_path}", font=("Arial", 11, "bold"))
    lbl.pack(pady=10)


    listbox = tk.Listbox(root, width=50, height=10)
    listbox.pack(pady=5)


    for move in plan.moves:
        file_name = Path(move.src).name
        listbox.insert(tk.END, f"{file_name} -> {move.category}")


    def on_apply():
        count = apply_plan(plan)
        messagebox.showinfo("Done", f"Successfully moved {count} files!")
        root.destroy()


 
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)


    btn_apply = tk.Button(btn_frame, text="Apply Moves", command=on_apply, bg="lightgreen")
    btn_apply.pack(side="left", padx=5)


    btn_cancel = tk.Button(btn_frame, text="Cancel", command=root.destroy)
    btn_cancel.pack(side="left", padx=5)


    root.mainloop()




if __name__ == "__main__":
    main()

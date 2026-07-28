#once you have this u need to open ur terminal, cs inside the state-ai...folder (the name of this one)
#and then you need to run: python install_context_menu.py


import shutil
import sys
import winreg #literly the only way to add stuff by right clicking
from pathlib import Path

def install():

    python_exe = Path(sys.executable).parent / "pythonw.exe"
    if not python_exe.exists():
        python_exe = sys.executable

    script_path = Path(__file__).resolve().parent / "frontend.py"

    command = f'"{python_exe}" "{script_path}" "%V"' #the v thing is a plceholder tbat fill in the file we clicked on

    registry_paths = [
        r"Directory\Background\shell\OrganizeWithAI",
        r"Directory\shell\OrganizeWithAI"
    ]


    try:
        for reg_path in registry_paths:
     
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\\" + reg_path)
            winreg.SetValue(key, "", winreg.REG_SZ, "Organize Folder")
        
            sub_key = winreg.CreateKey(key, "command")
            winreg.SetValue(sub_key, "", winreg.REG_SZ, command)
           
            winreg.CloseKey(sub_key)
            winreg.CloseKey(key)

        print("yay! we did it! All hail team 404 file found!")
    except Exception as e:
        print("trouble")


if __name__ == "__main__":
    install()

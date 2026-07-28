import sys
import winreg
from pathlib import Path

def install():
    python_exe = Path(sys.executable).parent / "pythonw.exe"
    if not python_exe.exists():
        python_exe = sys.executable

    script_path = Path(__file__).resolve().parent / "frontend.py"
    

    command = f'"{python_exe}" "{script_path}" "%V"'

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

        print("Successfully re-registered 'Organize Folder' in Windows Context Menu!")
    except Exception as e:
        print(f"Error registering context menu: {e}")

if __name__ == "__main__":
    install()
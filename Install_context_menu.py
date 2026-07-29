from __future__ import annotations

import sys
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None  


def install() -> None:
    if winreg is None:
        print("winreg only works on Windows.")
        sys.exit(1)

    python_exe = Path(sys.executable).parent / "pythonw.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    script_path = Path(__file__).resolve().parent / "frontend.py"
    command = f'"{python_exe}" "{script_path}" "%V"'

    registry_paths = [
        r"Directory\Background\shell\OrganizeWithAI",
        r"Directory\shell\OrganizeWithAI",
    ]

    for reg_path in registry_paths:
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\\" + reg_path)
        winreg.SetValue(key, "", winreg.REG_SZ, "Organize Folder")
        sub_key = winreg.CreateKey(key, "command")
        winreg.SetValue(sub_key, "", winreg.REG_SZ, command)
        winreg.CloseKey(sub_key)
        winreg.CloseKey(key)

    print("Installed: right-click a folder → Organize Folder")
    print(f"  script: {script_path}")


if __name__ == "__main__":
    install()

import sys
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None


def install():
    if winreg is None:
        print("Windows only.")
        sys.exit(1)

    python_exe = Path(sys.executable).parent / "pythonw.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    script = Path(__file__).resolve().parent / "frontend.py"
    command = f'"{python_exe}" "{script}" "%V"'

    for reg_path in (
        r"Directory\Background\shell\OrganizeWithAI",
        r"Directory\shell\OrganizeWithAI",
    ):
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\\" + reg_path)
        winreg.SetValue(key, "", winreg.REG_SZ, "Organize Folder")
        sub = winreg.CreateKey(key, "command")
        winreg.SetValue(sub, "", winreg.REG_SZ, command)
        winreg.CloseKey(sub)
        winreg.CloseKey(key)

    print("done — right-click a folder → Organize Folder")


if __name__ == "__main__":
    install()

# PyInstaller build: `pyinstaller platescanner.spec` (or run build.ps1)
# Produces dist/PlateScanner/PlateScanner.exe (one-folder build).
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

models = Path("models")
datas = collect_data_files("easyocr")
if models.is_dir():
    datas.append((str(models), "models"))
else:
    print("WARNING: ./models missing, run tools/fetch_models.py or OCR will not work offline")

a = Analysis(
    ["run.py"],
    datas=datas,
    hiddenimports=collect_submodules("easyocr"),
    excludes=["tkinter", "matplotlib", "IPython", "PyQt5", "PyQt6"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PlateScanner",
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="PlateScanner", upx=False)

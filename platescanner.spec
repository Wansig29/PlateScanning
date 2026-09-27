# PyInstaller build: `pyinstaller platescanner.spec` (or run build.ps1)
# Produces dist/PlateScanner/PlateScanner.exe (one-folder build).
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

models = Path("models") / "alpr"
datas = collect_data_files("fast_plate_ocr") + collect_data_files("open_image_models")
if models.is_dir():
    datas.append((str(models), "models/alpr"))
else:
    print("WARNING: ./models/alpr missing, run tools/fetch_models.py or plate reading will not work offline")

a = Analysis(
    ["run.py"],
    datas=datas,
    hiddenimports=collect_submodules("fast_plate_ocr.inference") + collect_submodules("open_image_models.detection"),
    excludes=["tkinter", "matplotlib", "IPython", "PyQt5", "PyQt6", "torch", "torchvision", "easyocr"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="PlateScanner",
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="PlateScanner", upx=False)

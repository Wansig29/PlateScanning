"""Download the EasyOCR English models into ./models (run once, while online).

PyInstaller bundles this folder so the gate laptop never needs internet
access to load the OCR engine.
"""
from pathlib import Path

import easyocr

target = Path(__file__).resolve().parent.parent / "models"
target.mkdir(exist_ok=True)
easyocr.Reader(["en"], gpu=False, model_storage_directory=str(target), download_enabled=True)
print("Models saved to", target, [p.name for p in target.iterdir()])

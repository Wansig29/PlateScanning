"""Download the plate detector and plate OCR models into ./models/alpr (run once, while online).

PyInstaller bundles this folder so the gate laptop never needs internet
access to load the models. Downloads the models named in the settings
(ocr.detector_model and every model in ocr.ocr_model).
"""
import sys
from pathlib import Path

from fast_plate_ocr.inference import hub as ocr_hub
from open_image_models.detection.core import hub as det_hub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner.config import OcrConfig  # noqa: E402

target = Path(__file__).resolve().parent.parent / "models" / "alpr"
target.mkdir(parents=True, exist_ok=True)
cfg = OcrConfig()
det_hub.download_model(cfg.detector_model, save_directory=target)
for name in [n.strip() for n in cfg.ocr_model.split(",") if n.strip()]:
    ocr_hub.download_model(name, save_directory=target / name)
print("Models saved to", target)
for p in sorted(target.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(target)}  ({p.stat().st_size / 1e6:.1f} MB)")

"""Download the plate detector and plate OCR models into ./models/alpr (run once, while online).

PyInstaller bundles this folder so the gate laptop never needs internet
access to load the models. Downloads the models named in the settings
(ocr.detector_model and every model in ocr.ocr_model).
"""
import sys
from pathlib import Path
from typing import Callable

from fast_plate_ocr.inference import hub as ocr_hub
from open_image_models.detection.core import hub as det_hub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platescanner.config import OcrConfig  # noqa: E402


def _download(download: Callable[..., Path | tuple[Path, ...]], *args, **kwargs) -> None:
    """Call a hub download_model function, retrying once with force_download if it left
    an empty file behind (these libraries only check that a cached file exists, not that
    it downloaded successfully, so a network hiccup can cache a 0-byte file forever)."""
    def paths_of(result) -> tuple[Path, ...]:
        return result if isinstance(result, tuple) else (result,)

    paths = paths_of(download(*args, **kwargs))
    empty = [p for p in paths if p.stat().st_size == 0]
    if empty:
        paths = paths_of(download(*args, force_download=True, **kwargs))
        empty = [p for p in paths if p.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Download left an empty file, even after retrying: {empty}")


target = Path(__file__).resolve().parent.parent / "models" / "alpr"
target.mkdir(parents=True, exist_ok=True)
cfg = OcrConfig()
_download(det_hub.download_model, cfg.detector_model, save_directory=target)
for name in [n.strip() for n in cfg.ocr_model.split(",") if n.strip()]:
    _download(ocr_hub.download_model, name, save_directory=target / name)
print("Models saved to", target)
for p in sorted(target.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(target)}  ({p.stat().st_size / 1e6:.1f} MB)")

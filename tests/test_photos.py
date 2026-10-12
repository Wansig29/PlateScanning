"""Photos over the size limit are compressed down to it; smaller ones are kept as they are."""
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QImage  # noqa: E402

from platescanner import api, photos  # noqa: E402
from platescanner.config import Config  # noqa: E402

MB = 1 << 20


def _noise(h, w, ext=".png", params=()):
    img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    return cv2.imencode(ext, img, list(params))[1].tobytes()


def _decoded(data: bytes) -> QImage:
    img = QImage()
    assert img.loadFromData(data)
    return img


def test_small_photo_is_kept_byte_for_byte():
    data = _noise(200, 300)
    assert photos.shrink_to(data, 5 * MB) is data


def test_big_png_is_compressed_under_the_limit():
    data = _noise(3000, 4000)                         # ~34 MB of camera-like noise
    assert len(data) > 25 * MB
    out = photos.shrink_to(data, 5 * MB)
    assert out is not None and len(out) <= 5 * MB
    img = _decoded(out)
    assert max(img.width(), img.height()) <= photos.MAX_SIDE
    assert abs(img.width() / img.height() - 4000 / 3000) < 0.01    # same shape


def test_big_jpeg_shrinks_to_a_tight_limit():
    data = _noise(2000, 2000, ".jpg", (cv2.IMWRITE_JPEG_QUALITY, 100))
    out = photos.shrink_to(data, 300 * 1024)
    assert out is not None and len(out) <= 300 * 1024


def test_transparent_png_becomes_white_not_black():
    img = np.zeros((1500, 1500, 4), np.uint8)               # fully transparent...
    img[..., :3] = (np.random.default_rng(1).random((1500, 1500, 3)) * 255).astype(np.uint8)
    img[:100, :100] = (0, 0, 0, 0)                           # ...corner stays see-through
    img[100:, :, 3] = 255
    data = cv2.imencode(".png", img)[1].tobytes()
    out = _decoded(photos.shrink_to(data, len(data) // 2))
    assert out.pixelColor(10, 10).lightness() > 240


def test_rotated_phone_photo_keeps_its_orientation():
    # A portrait phone photo stored landscape with EXIF "rotate 90": compressed upright.
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QImageIOHandler, QImageWriter
    src = QImage(1200, 800, QImage.Format.Format_RGB32)
    src.fill(0xFF336699)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    w = QImageWriter(buf, b"jpeg")
    w.setTransformation(QImageIOHandler.Transformation.TransformationRotate90)
    w.setQuality(100)
    assert w.write(src)
    data = bytes(buf.data())
    out = _decoded(photos.shrink_to(data, len(data) - 1))
    assert out.height() > out.width()


@pytest.mark.parametrize("data", [b"not a picture" * 1000, b"\x89PNG\r\n\x1a\n" + b"\0" * 5000])
def test_not_a_picture_gives_none(data):
    assert photos.shrink_to(data, 100) is None


def test_download_compresses_a_photo_over_5_mb(monkeypatch, tmp_path):
    from test_core import PhotoResp
    big = _noise(2000, 2000)                          # ~11 MB: over 5 MB, under the 25 MB download cap
    assert api.MAX_PHOTO_BYTES < len(big) < api.MAX_DOWNLOAD_BYTES
    client = api.ApiClient(Config().api)
    chunks = [big[i:i + MB] for i in range(0, len(big), MB)]
    monkeypatch.setattr(client.session, "get", lambda url, **kw: PhotoResp(chunks, {"Content-Type": "image/png"}))
    dest = tmp_path / "p.png"
    assert client.download("https://x/p.png", dest)
    assert dest.stat().st_size <= api.MAX_PHOTO_BYTES and not list(tmp_path.glob("*.part"))
    from PySide6.QtGui import QPixmap
    assert not QPixmap(str(dest)).isNull()           # opens even though it's now a JPEG in a .png name


def test_download_over_the_25_mb_cap_is_not_fetched(monkeypatch, tmp_path):
    from test_core import PhotoResp
    client = api.ApiClient(Config().api)
    resp = PhotoResp([b"x" * MB] * 26, {"Content-Type": "image/png"})
    monkeypatch.setattr(client.session, "get", lambda url, **kw: resp)
    assert not client.download("https://x/p.png", tmp_path / "p.png")
    assert not list(tmp_path.iterdir())

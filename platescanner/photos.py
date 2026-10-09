"""Shrinking downloaded photos (owner photos, violation evidence) to a size limit.

A photo over the limit is re-saved as a JPEG: first at lower quality, then, if that is
not enough, at a smaller size, until it fits. Photos at or under the limit are kept
exactly as they came. Qt does the decoding: it reads the picture's size before decoding
it and refuses pictures that would need more than its allocation limit (256 MB) of
memory, so a small file claiming enormous dimensions can't exhaust the laptop's memory.
"""
from __future__ import annotations

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter

# Longest side kept: far more than the dashboard shows, enough to zoom into evidence.
MAX_SIDE = 2560
# Tried in turn at each size until the photo fits.
QUALITIES = (90, 80, 70, 60)
# Each further try shrinks the picture by this factor, down to MIN_SIDE.
STEP = 0.75
MIN_SIDE = 480


def _jpeg(img: QImage, quality: int) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPEG", quality)
    return bytes(buf.data())


def _flatten(img: QImage) -> QImage:
    """JPEG has no transparency: put a transparent picture on white instead of black."""
    if not img.hasAlphaChannel():
        return img.convertToFormat(QImage.Format.Format_RGB32)
    out = QImage(img.size(), QImage.Format.Format_RGB32)
    out.fill(QColor("white"))
    p = QPainter(out)
    p.drawImage(0, 0, img)
    p.end()
    return out


def shrink_to(data: bytes, limit: int) -> bytes | None:
    """`data` if it is at most `limit` bytes, else the same picture as a JPEG of at most
    `limit` bytes. None if it isn't a readable picture or can't be made small enough."""
    if len(data) <= limit:
        return data
    buf = QBuffer()
    buf.setData(QByteArray(data))
    buf.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buf)
    reader.setAutoTransform(True)  # phone photos: apply the EXIF rotation before it is dropped
    size = reader.size()
    if not size.isValid() or size.isEmpty():
        return None
    longest = max(size.width(), size.height())
    if longest > MAX_SIDE:  # let the decoder scale while reading: less memory than full size
        reader.setScaledSize(size.scaled(MAX_SIDE, MAX_SIDE, Qt.AspectRatioMode.KeepAspectRatio))
    img = reader.read()
    if img.isNull():
        return None
    img = _flatten(img)
    while True:
        for quality in QUALITIES:
            out = _jpeg(img, quality)
            if len(out) <= limit:
                return out
        side = int(max(img.width(), img.height()) * STEP)
        if side < MIN_SIDE:
            return None
        img = img.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)

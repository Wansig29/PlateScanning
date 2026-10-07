"""Build the app icon and badge from the PSAU logo (platescanner/assets/psau_logo.png).

    python tools/make_icon.py

Writes app.ico / app.png (window + taskbar icon) and logo_badge.png (top bar and sign-in screen): the
transparent logo on a white rounded tile so its green strokes stay readable on the dark theme.
"""
from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "platescanner" / "assets"
CREAM = (255, 255, 255, 255)


def badge(size: int = 256) -> Image.Image:
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle((0, 0, size - 1, size - 1), radius=size * 0.22, fill=CREAM)
    logo = Image.open(ASSETS / "psau_logo.png").convert("RGBA")
    inner = int(size * 0.84)
    logo = logo.resize((inner, inner), Image.LANCZOS)
    tile.alpha_composite(logo, ((size - inner) // 2, (size - inner) // 2))
    return tile


if __name__ == "__main__":
    b = badge(256)
    b.save(ASSETS / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    b.save(ASSETS / "app.png")
    b.save(ASSETS / "logo_badge.png")
    print("Wrote icon and badge to", ASSETS)

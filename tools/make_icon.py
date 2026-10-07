"""Build the app icon and badge from the PSAU logo (platescanner/assets/psau_logo.png).

    python tools/make_icon.py

Writes app.ico / app.png (window + taskbar icon) and logo_badge.png (top bar and sign-in screen): the
transparent logo on a white circle so its green strokes stay readable on the dark theme.
"""
from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parent.parent / "platescanner" / "assets"
CREAM = (255, 255, 255, 255)


def badge(size: int = 256) -> Image.Image:
    big = size * 4  # draw large, then downsample for a smooth circle edge
    tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(tile).ellipse((0, 0, big - 1, big - 1), fill=CREAM)
    inner = int(big * 0.84)
    logo = Image.open(ASSETS / "psau_logo.png").convert("RGBA").resize((inner, inner), Image.LANCZOS)
    tile.alpha_composite(logo, ((big - inner) // 2, (big - inner) // 2))
    return tile.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    b = badge(256)
    b.save(ASSETS / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    b.save(ASSETS / "app.png")
    b.save(ASSETS / "logo_badge.png")
    print("Wrote icon and badge to", ASSETS)

"""Draw the app icon (platescanner/assets/app.ico): the blue "P" brand mark over a plate.

    python tools/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "platescanner" / "assets" / "app.ico"
BLUE, WHITE, INK = (59, 130, 246, 255), (255, 255, 255, 255), (11, 16, 22, 255)


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(size: int = 256) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 256
    d.rounded_rectangle((8 * s, 8 * s, 248 * s, 248 * s), radius=52 * s, fill=BLUE)
    f = font(int(150 * s))
    d.text((128 * s, 104 * s), "P", font=f, fill=WHITE, anchor="mm")
    # A little licence plate under the P.
    d.rounded_rectangle((52 * s, 176 * s, 204 * s, 226 * s), radius=9 * s, fill=WHITE, outline=INK,
                        width=max(1, int(5 * s)))
    for i in range(6):  # characters on the plate
        x = 70 * s + i * 21 * s
        d.rounded_rectangle((x, 189 * s, x + 13 * s, 213 * s), radius=3 * s, fill=INK)
    return img


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    draw().save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    draw().save(OUT.with_suffix(".png"))
    print("Wrote", OUT)

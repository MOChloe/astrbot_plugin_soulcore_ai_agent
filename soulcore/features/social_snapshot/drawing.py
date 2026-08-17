from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .errors import SocialSnapshotError, SocialSnapshotErrorCode

DEFAULT_FONT_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "fonts" / "SoulCoreSansSC-Regular.ttf"
)


@lru_cache(maxsize=32)
def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


@dataclass(frozen=True, slots=True)
class FontBook:
    path: Path = DEFAULT_FONT_PATH

    def get(self, size: int) -> ImageFont.FreeTypeFont:
        if not self.path.is_file():
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.FONT_UNAVAILABLE, "bundled font is unavailable"
            )
        try:
            return _load_font(self.path, size)
        except OSError as exc:
            raise SocialSnapshotError(
                SocialSnapshotErrorCode.FONT_UNAVAILABLE, "bundled font is unavailable"
            ) from exc


def rgb(value: str) -> tuple[int, int, int]:
    color = ImageColor.getrgb(value)
    return int(color[0]), int(color[1]), int(color[2])


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
) -> tuple[str, ...]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for character in paragraph:
            candidate = current + character
            box = draw.textbbox((0, 0), candidate, font=text_font)
            if current and box[2] - box[0] > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        lines.append(current or " ")
    return tuple(lines)


def line_height(text_font: ImageFont.FreeTypeFont, spacing: int = 8) -> int:
    box = text_font.getbbox("国Ag")
    return int(box[3] - box[1] + spacing)


def draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: tuple[str, ...],
    position: tuple[int, int],
    text_font: ImageFont.FreeTypeFont,
    *,
    fill: str,
    spacing: int = 8,
) -> int:
    x, y = position
    height = line_height(text_font, spacing)
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        y += height
    return y


def cover_image(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    converted = source.convert("RGB")
    scale = max(width / converted.width, height / converted.height)
    resized = converted.resize(
        (max(width, round(converted.width * scale)), max(height, round(converted.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def rounded_avatar(
    source: Image.Image | None,
    size: int,
    *,
    color: str,
    label: str,
    fonts: FontBook,
) -> Image.Image:
    if source is None:
        avatar = Image.new("RGB", (size, size), rgb(color))
        avatar_draw = ImageDraw.Draw(avatar)
        label_font = fonts.get(max(18, size // 3))
        glyph = label[:1] or "?"
        box = avatar_draw.textbbox((0, 0), glyph, font=label_font)
        avatar_draw.text(
            ((size - (box[2] - box[0])) / 2, (size - (box[3] - box[1])) / 2 - box[1]),
            glyph,
            fill="white",
            font=label_font,
        )
    else:
        avatar = cover_image(source, (size, size))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    output.paste(avatar, (0, 0), mask)
    return output


def draw_android_status_bar(
    image: Image.Image,
    *,
    clock: str,
    battery_percent: int,
    charging: bool,
    fonts: FontBook,
    foreground: str = "#161616",
    background: str = "#ffffff",
) -> None:
    draw = ImageDraw.Draw(image)
    width = image.width
    scale = width / 873
    draw.rectangle((0, 0, width, round(84 * scale)), fill=background)
    status_font = fonts.get(max(18, round(28 * scale)))
    draw.text((round(38 * scale), round(22 * scale)), clock, font=status_font, fill=foreground)
    right = width - round(36 * scale)
    battery_width = round(52 * scale)
    battery_height = round(25 * scale)
    top = round(26 * scale)
    left = right - battery_width
    draw.rounded_rectangle(
        (left, top, right, top + battery_height),
        radius=max(2, round(5 * scale)),
        outline=foreground,
        width=max(1, round(2 * scale)),
    )
    draw.rectangle(
        (
            right + round(3 * scale),
            top + round(8 * scale),
            right + round(7 * scale),
            top + round(17 * scale),
        ),
        fill=foreground,
    )
    inner_width = max(0, round((battery_width - 8 * scale) * battery_percent / 100))
    fill = "#19b45b" if charging else foreground
    if inner_width:
        draw.rounded_rectangle(
            (
                left + round(4 * scale),
                top + round(4 * scale),
                left + round(4 * scale) + inner_width,
                top + battery_height - round(4 * scale),
            ),
            radius=max(1, round(3 * scale)),
            fill=fill,
        )
    signal_x = left - round(88 * scale)
    for index, height in enumerate((8, 13, 18, 23)):
        x = signal_x + round(index * 8 * scale)
        draw.rectangle(
            (
                x,
                top + battery_height - round(height * scale),
                x + round(4 * scale),
                top + battery_height,
            ),
            fill=foreground,
        )


def draw_disclosure(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    y: int,
    fonts: FontBook,
    text: str = "AI演绎",
) -> int:
    font = fonts.get(max(16, round(width * 0.022)))
    box = draw.textbbox((0, 0), text, font=font)
    label_width = box[2] - box[0] + 30
    height = box[3] - box[1] + 18
    left = width - label_width - 26
    draw.rounded_rectangle((left, y, width - 26, y + height), radius=height // 2, fill="#e7e7e7")
    draw.text((left + 15, y + 7 - box[1]), text, font=font, fill="#676767")
    return int(y + height)


def draw_bottom_gesture(image: Image.Image, *, background: str = "#ffffff") -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    scale = width / 873
    bar_width = round(270 * scale)
    y = height - round(22 * scale)
    draw.rectangle((0, height - round(45 * scale), width, height), fill=background)
    draw.rounded_rectangle(
        ((width - bar_width) // 2, y, (width + bar_width) // 2, y + round(8 * scale)),
        radius=round(4 * scale),
        fill="#1f1f1f",
    )

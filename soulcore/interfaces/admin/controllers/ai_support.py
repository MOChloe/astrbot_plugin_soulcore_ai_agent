"""AI administrator adapter constants and probe helpers."""

from __future__ import annotations

import io
import uuid
import wave
from pathlib import Path

AI_CAPABILITIES = (
    "chat.completion",
    "conversation.turn_buffer",
    "conversation.group_interjection",
    "conversation.group_reply_relocation",
    "conversation.timer_lifecycle_review",
    "conversation.response_polish",
    "conversation.summary",
    "memory.reasoning",
    "text.completion",
    "sticker.collect",
    "sticker.check",
    "vision.describe",
    "image.generate",
    "audio.transcribe",
    "audio.speech",
    "web.search",
    "web.read",
    "file.generate",
)


def build_vision_probe_challenge() -> tuple[str, bytes]:
    """Create a private-free image whose random code must be read from pixels."""

    from PIL import Image, ImageDraw, ImageFont

    challenge = uuid.uuid4().hex[:6].upper()
    image = Image.new("RGB", (480, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 12, 468, 168), radius=18, outline="black", width=5)
    font_path = (
        Path(__file__).resolve().parents[3] / "assets" / "fonts" / "SoulCoreSansSC-Regular.ttf"
    )
    font = ImageFont.truetype(str(font_path), 76)
    label = f"CODE {challenge}"
    bounds = draw.textbbox((0, 0), label, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((480 - width) / 2, (180 - height) / 2 - bounds[1]),
        label,
        fill="black",
        font=font,
    )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return challenge, output.getvalue()


def build_audio_probe_sample() -> bytes:
    """Build a short credential-free PCM WAV for an STT transport probe."""

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 6_400)
    return output.getvalue()

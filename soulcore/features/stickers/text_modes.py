"""Text presentation modes for generated stickers."""

TEXT_MODE_NONE = "NONE"
TEXT_MODE_INTEGRATED_TEXT = "INTEGRATED_TEXT"

GENERATED_STICKER_TEXT_MODES = frozenset(
    {
        TEXT_MODE_NONE,
        TEXT_MODE_INTEGRATED_TEXT,
    }
)

__all__ = [
    "GENERATED_STICKER_TEXT_MODES",
    "TEXT_MODE_INTEGRATED_TEXT",
    "TEXT_MODE_NONE",
]

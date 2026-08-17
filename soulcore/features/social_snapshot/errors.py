from __future__ import annotations

from enum import StrEnum


class SocialSnapshotErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_THEME = "UNSUPPORTED_THEME"
    UNSUPPORTED_MODE = "UNSUPPORTED_MODE"
    UNSUPPORTED_ENTRY = "UNSUPPORTED_ENTRY"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    ASSET_MISSING = "ASSET_MISSING"
    ASSET_INVALID = "ASSET_INVALID"
    ASSET_TOO_LARGE = "ASSET_TOO_LARGE"
    FONT_UNAVAILABLE = "FONT_UNAVAILABLE"
    RENDER_FAILED = "RENDER_FAILED"


class SocialSnapshotError(ValueError):
    """A redaction-safe domain or rendering failure."""

    def __init__(self, code: SocialSnapshotErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


def invalid(message: str) -> SocialSnapshotError:
    return SocialSnapshotError(SocialSnapshotErrorCode.INVALID_REQUEST, message)

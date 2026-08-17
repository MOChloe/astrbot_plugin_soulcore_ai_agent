"""Neutral escaping for untrusted text that resembles model identity syntax."""

from __future__ import annotations

import re

_MODEL_IDENTITY_MARK = re.compile(r"\{\[([^{}\[\]\r\n]{1,160})\]\}")


def escape_untrusted_identity_syntax(value: str) -> str:
    """Prevent user-authored lookalikes from impersonating trusted identity marks."""

    escaped = _MODEL_IDENTITY_MARK.sub(
        lambda match: match.group(0).replace("{[", "｛［").replace("]}", "］｝"),
        str(value or ""),
    )
    return escaped


__all__ = ["escape_untrusted_identity_syntax"]

"""Stable text identity used across feature boundaries."""

from __future__ import annotations

import hashlib
from typing import Any


def content_fingerprint(value: Any) -> str:
    canonical = " ".join(str(value or "").split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() if canonical else ""

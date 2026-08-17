"""Provider-neutral conservative token accounting shared by context producers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class TokenCountMode(StrEnum):
    EXACT = "EXACT"
    ESTIMATED = "ESTIMATED"


@dataclass(frozen=True, slots=True)
class TokenMeasurement:
    tokens: int
    mode: TokenCountMode
    model_id: str = ""


class TokenItem(Protocol):
    @property
    def speaker(self) -> str: ...

    @property
    def body(self) -> Any: ...


class TokenMeter(Protocol):
    def count_text(self, value: str) -> int: ...

    def measure_item(self, item: TokenItem) -> TokenMeasurement: ...

    def measure(self, items: Sequence[TokenItem]) -> TokenMeasurement: ...


_CJK_RE = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


class ConservativeTokenMeter:
    """Deterministic fallback used when the provider exposes no tokenizer."""

    MESSAGE_OVERHEAD = 6
    MULTIMODAL_TOKENS = {
        "image": 2_048,
        "audio": 4_096,
        "voice": 4_096,
        "file": 2_048,
        "video": 8_192,
    }

    def __init__(self, model_id: str = "") -> None:
        self.model_id = str(model_id or "")

    def count_text(self, value: str) -> int:
        text = str(value or "")
        if not text:
            return 0
        cjk = len(_CJK_RE.findall(text))
        return cjk + math.ceil((len(text) - cjk) / 3)

    def count_text_prefixes(
        self,
        value: str,
        ends: Sequence[int],
    ) -> dict[int, int]:
        """Count many character prefixes in one scan with ``count_text`` parity."""

        text = str(value or "")
        ordered = sorted({int(end) for end in ends})
        if any(end < 0 or end > len(text) for end in ordered):
            raise ValueError("text prefix boundary is outside the source text")
        result: dict[int, int] = {}
        cursor = 0
        cjk = 0
        non_cjk = 0
        for end in ordered:
            chunk = text[cursor:end]
            chunk_cjk = len(_CJK_RE.findall(chunk))
            cjk += chunk_cjk
            non_cjk += len(chunk) - chunk_cjk
            result[end] = cjk + math.ceil(non_cjk / 3)
            cursor = end
        return result

    def count_value(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return self.count_text(value)
        if isinstance(value, bytes):
            return max(1, math.ceil(len(value) / 3))
        if isinstance(value, Mapping):
            kind = str(value.get("kind") or value.get("type") or "").lower()
            base = self.MULTIMODAL_TOKENS.get(kind, 0)
            serializable = {
                str(key): val
                for key, val in value.items()
                if str(key).lower() not in {"data", "bytes", "base64"}
            }
            return base + self.count_text(
                json.dumps(serializable, ensure_ascii=False, sort_keys=True, default=str)
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return sum(self.count_value(item) for item in value)
        return self.count_text(str(value))

    def measure_item(self, item: TokenItem) -> TokenMeasurement:
        tokens = self.MESSAGE_OVERHEAD + self.count_text(item.speaker)
        tokens += self.count_value(item.body)
        return TokenMeasurement(tokens, TokenCountMode.ESTIMATED, self.model_id)

    def measure(self, items: Sequence[TokenItem]) -> TokenMeasurement:
        item_tokens = sum(self.measure_item(item).tokens for item in items)
        return TokenMeasurement(item_tokens, TokenCountMode.ESTIMATED, self.model_id)

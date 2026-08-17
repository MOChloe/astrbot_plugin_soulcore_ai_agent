"""Chinese-aware deterministic lexical projection for FTS and exact matching."""

from __future__ import annotations

import re
import threading
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import jieba

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_DATE_TOKEN = re.compile(r"\b(?:19|20)\d{2}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2})?日?\b")
_GENERIC_QUERY_ANCHORS = frozenset(
    {
        "有没有",
        "为什么",
        "怎么样",
        "什么情况",
        "什么地方",
        "其中一个",
        "这个计划",
        "相关情况",
        "之前",
        "之后",
        "后来",
        "现在",
        "最近",
        "当时",
        "对方",
        "事情",
        "情况",
        "改动",
        "修改",
        "修正",
        "改变",
        "比如",
        "改成",
        "说过",
        "回来",
    }
)
_TOKENIZER_LOCK = threading.Lock()
_TOKENIZER: jieba.Tokenizer | None = None
_TOKENIZER_CACHE_DIR: Path | None = None


def configure_tokenizer_cache(cache_dir: Path) -> None:
    """Keep Jieba's generated dictionary cache inside SoulCore-managed storage."""

    global _TOKENIZER_CACHE_DIR
    resolved = Path(cache_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    with _TOKENIZER_LOCK:
        _TOKENIZER_CACHE_DIR = resolved
        if _TOKENIZER is not None and not _TOKENIZER.initialized:
            _TOKENIZER.tmp_dir = str(resolved)
            _TOKENIZER.cache_file = "soulcore-jieba.cache"


def _tokenizer() -> jieba.Tokenizer:
    global _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            tokenizer = jieba.Tokenizer()
            if _TOKENIZER_CACHE_DIR is not None:
                tokenizer.tmp_dir = str(_TOKENIZER_CACHE_DIR)
                tokenizer.cache_file = "soulcore-jieba.cache"
            _TOKENIZER = tokenizer
        return _TOKENIZER


def normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def lexical_tokens(value: object, *, aliases: Iterable[object] = ()) -> tuple[str, ...]:
    text = normalize_text(value)
    alias_values = tuple(normalize_text(item) for item in aliases if normalize_text(item))
    combined = " ".join((text, *alias_values)).strip()
    if not combined:
        return ()
    tokens: list[str] = []
    tokens.extend(_LATIN_OR_NUMBER.findall(combined))
    tokens.extend(_DATE_TOKEN.findall(combined))
    for word in _tokenizer().cut(combined, cut_all=False):
        normalized = normalize_text(word)
        if len(normalized) >= 2 or _LATIN_OR_NUMBER.fullmatch(normalized):
            tokens.append(normalized)
    for run in _CJK_RUN.findall(combined):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) <= 8:
            tokens.append(run)
    tokens.extend(alias_values)
    return tuple(dict.fromkeys(token for token in tokens if token and len(token) <= 80))


def fts_document(value: object, *, aliases: Iterable[object] = ()) -> str:
    return " ".join(lexical_tokens(value, aliases=aliases))


def fts_query(value: object, *, maximum_terms: int = 32) -> str:
    tokens = lexical_tokens(value)[: max(1, int(maximum_terms))]
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


def token_overlap(query: Iterable[str], candidate: Iterable[str]) -> float:
    left, right = set(query), set(candidate)
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / max(1.0, (len(left) * len(right)) ** 0.5)


def token_containment(query: Iterable[str], candidate: Iterable[str]) -> float:
    """Measure how fully the shorter side is contained in the other.

    Cosine-like overlap is intentionally conservative for similarly sized
    documents, but it under-rates concise corrections when the query contains
    a long temporal description.  FTS rank plus shorter-side containment keeps
    that evidence without weakening the prefetch confidence gate.
    """

    left, right = set(query), set(candidate)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def has_distinctive_token_overlap(query: Iterable[str], candidate: Iterable[str]) -> bool:
    """Return whether both sides share a non-generic complete Chinese term.

    ``lexical_tokens`` emits Chinese bigrams for recall breadth, plus Jieba
    words and short complete runs.  Requiring at least three CJK characters
    therefore distinguishes an actual lexical anchor such as ``旧书店`` from
    a coincidental bigram.  Generic question/temporal words are excluded so
    this signal cannot turn ``有没有`` or ``后来`` into evidence by itself.
    """

    shared = set(query) & set(candidate)
    return any(
        len(token) >= 3
        and token not in _GENERIC_QUERY_ANCHORS
        and _CJK_RUN.fullmatch(token) is not None
        for token in shared
    )


__all__ = [
    "configure_tokenizer_cache",
    "fts_document",
    "fts_query",
    "has_distinctive_token_overlap",
    "lexical_tokens",
    "normalize_text",
    "token_containment",
    "token_overlap",
]

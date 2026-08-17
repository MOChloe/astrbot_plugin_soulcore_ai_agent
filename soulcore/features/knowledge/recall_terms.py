"""Deterministic text normalization and lexical term extraction."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]+", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)
_MEANINGFUL_TERM_RE = re.compile(r"[a-z0-9\u3400-\u9fff]", re.IGNORECASE)
STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "也",
    "就",
    "都",
    "而",
    "但",
    "啊",
    "呀",
    "吗",
    "呢",
    "吧",
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "什么",
    "怎么",
    "一个",
    "一些",
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "to",
    "of",
    "in",
    "on",
}
GENERIC_QUERY_TERMS = {
    "一下",
    "之前",
    "事情",
    "关于",
    "刚才",
    "可以",
    "可否",
    "后来",
    "告诉",
    "如何",
    "帮我",
    "情况",
    "想问",
    "想知道",
    "提到",
    "有没有",
    "现在",
    "目前",
    "能不能",
    "能",
    "请",
    "请问",
    "说过",
    "说说",
    "这个",
    "这些",
    "这座",
    "那个",
    "那些",
    "那座",
    "问题",
    "怎么样",
    "怎样",
    "怎么回事",
    "知道",
    "最近",
    "仍然",
    "是否",
    "发生",
    "被",
    "把",
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "tell",
    "show",
    "about",
    "please",
    "current",
    "currently",
    "recent",
    "recently",
    "status",
    "happen",
    "happened",
    "know",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(_PUNCT_RE.sub(" ", text).split())


def _fallback_terms(text: str) -> list[str]:
    result: list[str] = []
    for match in _WORD_RE.findall(text):
        token = match.strip()
        if not token or token in STOPWORDS:
            continue
        result.append(token)
        if re.fullmatch(r"[\u3400-\u9fff]+", token) and len(token) > 2:
            for size in (2, 3, 4):
                if len(token) < size:
                    continue
                result.extend(token[index : index + size] for index in range(len(token) - size + 1))
    return result


def _segmented_terms(text: str) -> list[str]:
    try:  # The packaged runtime installs jieba; source-only use keeps a fallback.
        import jieba  # type: ignore

        return [normalize_text(item) for item in jieba.cut_for_search(text)]
    except Exception:
        return []


def search_terms(value: Any) -> list[str]:
    """NFKC/case/punctuation normalization plus Chinese search segmentation."""

    text = normalize_text(value)
    if not text:
        return []
    values = _segmented_terms(text)
    values.extend(_fallback_terms(text))
    return list(dict.fromkeys(item for item in values if item and item not in STOPWORDS))


def bounded_search_terms(value: Any, *, limit: int = 16) -> list[str]:
    """Extract bounded natural terms without discarding short names such as ``猫`` or ``AI``."""

    normalized = normalize_text(value)
    if not normalized:
        return []
    segmented = _segmented_terms(normalized)
    candidates = [normalized, *(segmented or _fallback_terms(normalized))]
    terms: list[str] = []
    for term in candidates:
        term = normalize_text(term)
        if (
            not term
            or not _MEANINGFUL_TERM_RE.search(term)
            or term in STOPWORDS
            or term in GENERIC_QUERY_TERMS
        ):
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) >= max(1, min(int(limit), 32)):
            break
    return terms


__all__ = [
    "GENERIC_QUERY_TERMS",
    "STOPWORDS",
    "bounded_search_terms",
    "normalize_text",
    "search_terms",
]

"""Pure evidence and impression matching helpers for Main Core player profiles."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _resolve_current_evidence(
    collector: Any,
    target: Any,
    evidence_ref: str,
    *,
    impression: str,
) -> tuple[int, str]:
    candidates = target_evidence_messages(target)
    requested = str(evidence_ref or "").strip()
    if requested:
        return _resolve_requested_evidence(
            collector,
            candidates,
            requested,
            impression=impression,
        )
    supported = [
        (message_id, text)
        for message_id, text in candidates
        if is_usable_profile_evidence(text, impression)
    ]
    if len(supported) == 1:
        return supported[0]
    if not supported and len(candidates) == 1 and not _looks_hypothetical(candidates[0][1]):
        return candidates[0]
    if not supported:
        raise ValueError("当前输入没有唯一一条能直接支持这件认识的本人文字")
    raise ValueError("当前输入有多条可能依据，请用一条当前可见消息短引用明确选择")


def _resolve_requested_evidence(
    collector: Any,
    candidates: list[tuple[int, str]],
    requested: str,
    *,
    impression: str,
) -> tuple[int, str]:
    allowlist = dict(collector.message_ref_allowlist or {})
    allowed = allowlist.get(requested)
    if allowed is None and requested in collector.model_reference_map:
        requested = str(collector.model_reference_map[requested])
        allowed = allowlist.get(requested)
    if allowed is None:
        raise ValueError("依据消息不属于本次行动的当前可见对话")
    message_id = int(allowed.get("ledger_message_id") or 0)
    selected = [(item_id, text) for item_id, text in candidates if item_id == message_id]
    if len(selected) != 1:
        raise ValueError("依据消息不是所选现实聊天对象本人说出的可见文字")
    if not is_usable_profile_evidence(selected[0][1], impression):
        raise ValueError("所选消息不能直接支持这件稳定认识")
    return selected[0]


def target_evidence_messages(target: Any) -> list[tuple[int, str]]:
    raw = tuple(target.evidence_messages or ())
    if not raw:
        raw = ((int(target.evidence_message_id or 0), str(target.evidence_text or "")),)
    result: list[tuple[int, str]] = []
    seen: set[int] = set()
    for message_id, text in raw:
        normalized_id = int(message_id or 0)
        normalized_text = normalize_evidence_text(text)
        if normalized_id < 1 or len(normalized_text) < 2 or normalized_id in seen:
            continue
        seen.add(normalized_id)
        result.append((normalized_id, normalized_text))
    return result


def is_usable_profile_evidence(message: str, impression: str) -> bool:
    if _looks_hypothetical(message):
        return False
    message_key = normalized_impression(message)
    impression_key = normalized_impression(impression)
    if not message_key or not impression_key:
        return False
    if message_key in impression_key or impression_key in message_key:
        return True
    message_pairs = character_pairs(message_key)
    impression_pairs = character_pairs(impression_key)
    if not message_pairs or not impression_pairs:
        return False
    overlap = len(message_pairs & impression_pairs)
    return overlap / max(1, min(len(message_pairs), len(impression_pairs))) >= 0.6


def _looks_hypothetical(value: str) -> bool:
    return bool(
        re.search(
            r"(?:假如|假设|如果有人|比如|举个例子|例如|虚构人物|故事人物|别人说|某个人)",
            str(value or ""),
        )
    )


def normalized_impression(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"(?:原来的印象|这条印象|对方|这个人|他|她|本人)", "", text)
    return "".join(char for char in text if char.isalnum())


def character_pairs(value: str) -> set[str]:
    return {value[index : index + 2] for index in range(max(0, len(value) - 1))}


def impression_match_score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    if query in candidate or candidate in query:
        return min(len(query), len(candidate)) / max(len(query), len(candidate))
    query_pairs = character_pairs(query)
    candidate_pairs = character_pairs(candidate)
    if not query_pairs or not candidate_pairs:
        return 0.0
    return len(query_pairs & candidate_pairs) / len(query_pairs | candidate_pairs)


def normalize_evidence_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def evidence_message_id(target: Any, quote: str) -> int:
    candidates = tuple(getattr(target, "evidence_messages", ()) or ())
    if not candidates:
        candidates = (
            (
                int(getattr(target, "evidence_message_id", 0) or 0),
                str(getattr(target, "evidence_text", "") or ""),
            ),
        )
    for message_id, text in candidates:
        if int(message_id) > 0 and quote in normalize_evidence_text(text):
            return int(message_id)
    raise ValueError("证据原文不属于所选对象本人的本轮可见消息")


__all__ = [
    "_resolve_current_evidence",
    "evidence_message_id",
    "impression_match_score",
    "normalize_evidence_text",
    "normalized_impression",
]

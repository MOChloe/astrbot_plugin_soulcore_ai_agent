"""Frozen retrieval policy, intent recognizers, and internal candidate records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

CHANGE_INTENT = re.compile(
    r"(变化|改变|以前.*现在|原来.*现在|过去.*如今|不再|改成|修订|修正|纠正|更正|前后)"
)
CURRENT_INTENT = re.compile(r"(现在|当前|如今|目前|最新|此刻)")
HISTORY_INTENT = re.compile(r"(过去|以前|曾经|当时|之前|后来|之后|先后|哪次|发生)")
RECENT_INTENT = re.compile(r"(最近|刚才|刚刚|最新|上次|昨天|今天|前天)")
COMPLEX_INTENT = re.compile(
    r"(怎么认识|之间.*关系|通过谁|为什么.*后来|先.*再|多次|分别|哪些人|"
    r"(?:参加|参与).*(?:的人|谁)|谁.*(?:现在)?住)"
)
PARTICIPANT_INTENT = re.compile(r"(?:参加|参与).*(?:的人|谁)")
DATE_HINT = re.compile(r"(?:19|20)\d{2}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2})?日?")
CORRECTION_WORDS = re.compile(r"(纠正|修正|错误|不再|已经不是|取代|冲突|更正)")
GENERIC_ENTITY_NAMES = frozenset({"角色", "对方", "用户", "交谈者", "聊天原话"})


@dataclass(frozen=True, slots=True)
class RecallPolicyV1:
    version: str = "RecallPolicyV1"
    prefetch_timeout_seconds: float = 30.0
    rrf_k: int = 60
    bm25_limit: int = 80
    dense_limit: int = 80
    exact_limit: int = 40
    graph_limit: int = 30
    rerank_limit: int = 30
    final_limit: int = 5
    prefetch_limit: int = 6
    explicit_confidence: float = 0.38
    prefetch_confidence: float = 0.82
    mmr_lambda: float = 0.76
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    exact_weight: float = 1.35
    graph_weight: float = 0.65
    scene_weight: float = 0.55
    rerank_weight: float = 1.45


@dataclass(slots=True)
class RecallCandidate:
    document: dict[str, Any]
    route_ranks: dict[str, int]
    signals: dict[str, float]
    rrf: float = 0.0
    confidence: float = 0.0


@dataclass(slots=True)
class MatrixCacheEntry:
    document_keys: tuple[str, ...]
    matrix: np.ndarray


__all__ = [
    "CHANGE_INTENT",
    "COMPLEX_INTENT",
    "CORRECTION_WORDS",
    "CURRENT_INTENT",
    "DATE_HINT",
    "GENERIC_ENTITY_NAMES",
    "HISTORY_INTENT",
    "MatrixCacheEntry",
    "PARTICIPANT_INTENT",
    "RECENT_INTENT",
    "RecallCandidate",
    "RecallPolicyV1",
]

"""Stable request/result contracts at the Recall boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class RecallMode(StrEnum):
    PREFETCH = "PREFETCH"
    EXPLICIT = "EXPLICIT"
    ADMIN_PROBE = "ADMIN_PROBE"


def _bounded_recent(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized:
            result.append(normalized[:2000])
    return tuple(result)


def _nonempty_strings(values: Iterable[Any]) -> frozenset[str]:
    return frozenset(normalized for value in values if (normalized := str(value)))


def _upper_strings(values: Iterable[Any]) -> frozenset[str]:
    result: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().upper()
        if normalized:
            result.add(normalized)
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class RecallRequest:
    profile_id: str
    instance_id: str
    need: str
    mode: RecallMode
    current_time: datetime
    recent_visible_context: tuple[str, ...] = ()
    visible_source_fingerprints: frozenset[str] = frozenset()
    excluded_document_keys: frozenset[str] = frozenset()
    allowed_source_types: frozenset[str] = frozenset()
    allowed_authority_statuses: frozenset[str] = frozenset()
    token_budget: int = 1200

    def normalized(self) -> RecallRequest:
        return RecallRequest(
            profile_id=str(self.profile_id or "").strip(),
            instance_id=str(self.instance_id or "").strip(),
            need=str(self.need or "").strip()[:4000],
            mode=RecallMode(self.mode),
            current_time=self.current_time,
            recent_visible_context=_bounded_recent(self.recent_visible_context[-6:]),
            visible_source_fingerprints=_nonempty_strings(self.visible_source_fingerprints),
            excluded_document_keys=_nonempty_strings(self.excluded_document_keys),
            allowed_source_types=_upper_strings(self.allowed_source_types),
            allowed_authority_statuses=_upper_strings(self.allowed_authority_statuses),
            token_budget=max(64, min(int(self.token_budget or 1200), 4000)),
        )


@dataclass(frozen=True, slots=True)
class RecallEvidence:
    """One server-validated evidence document.

    Internal keys and scores are retained for exclusion and diagnostics, but the
    model-facing renderer deliberately never exposes them.
    """

    document_key: str
    source_type: str
    authority_status: str
    statement: str
    time_label: str
    evidence_note: str
    source_fingerprint: str
    source_message_ids: tuple[int, ...] = ()
    confidence: float = 0.0
    retrieval_routes: tuple[str, ...] = ()
    occurred_at: str = ""
    valid_from: str = ""
    valid_until: str = ""

    def public_view(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "time": self.time_label,
            "evidence": self.evidence_note,
            "source_type": self.source_type,
            "status": self.authority_status,
        }


@dataclass(frozen=True, slots=True)
class RecallChange:
    before: RecallEvidence
    after: RecallEvidence
    explanation: str

    def public_view(self) -> dict[str, Any]:
        return {
            "before": self.before.public_view(),
            "after": self.after.public_view(),
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class RecallReadiness:
    code: str
    label: str
    embedding_provider_id: str = ""
    rerank_provider_id: str = ""
    generation_id: int | None = None
    indexed_documents: int = 0
    dense_documents: int = 0
    embedded_documents: int = 0
    pending_tasks: int = 0
    coverage: float = 0.0
    detail: str = ""

    def public_view(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "embedding_provider_id": self.embedding_provider_id,
            "rerank_provider_id": self.rerank_provider_id,
            "generation_id": self.generation_id,
            "indexed_documents": self.indexed_documents,
            "dense_documents": self.dense_documents,
            "embedded_documents": self.embedded_documents,
            "pending_tasks": self.pending_tasks,
            "coverage": round(self.coverage, 4),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RecallBundle:
    current_facts: tuple[RecallEvidence, ...] = ()
    historical_events: tuple[RecallEvidence, ...] = ()
    changes: tuple[RecallChange, ...] = ()
    conflicts: tuple[RecallEvidence, ...] = ()
    refusal: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    readiness: RecallReadiness | None = None

    @property
    def evidence(self) -> tuple[RecallEvidence, ...]:
        values: list[RecallEvidence] = [*self.current_facts, *self.historical_events]
        values.extend(self.conflicts)
        for change in self.changes:
            values.extend((change.before, change.after))
        return tuple(dict.fromkeys(values))

    @property
    def document_keys(self) -> frozenset[str]:
        return frozenset(item.document_key for item in self.evidence)

    @property
    def reliable(self) -> bool:
        return bool(self.evidence) and not self.refusal

    def public_view(self, *, include_diagnostics: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "current": [item.public_view() for item in self.current_facts],
            "history": [item.public_view() for item in self.historical_events],
            "changes": [item.public_view() for item in self.changes],
            "uncertain": [item.public_view() for item in self.conflicts],
            "refusal": self.refusal,
        }
        if self.readiness is not None:
            result["readiness"] = self.readiness.public_view()
        if include_diagnostics:
            result["diagnostics"] = dict(self.diagnostics)
        return result


__all__ = [
    "RecallBundle",
    "RecallChange",
    "RecallEvidence",
    "RecallMode",
    "RecallReadiness",
    "RecallRequest",
]

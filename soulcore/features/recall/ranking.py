"""Deterministic scoring, temporal selection, graph math, and evidence rendering."""

from __future__ import annotations

import difflib
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from ...contracts.text_fingerprint import content_fingerprint
from .domain import RecallEvidence, RecallMode, RecallRequest
from .policy import (
    CHANGE_INTENT as _CHANGE_INTENT,
)
from .policy import (
    COMPLEX_INTENT as _COMPLEX_INTENT,
)
from .policy import (
    CURRENT_INTENT as _CURRENT_INTENT,
)
from .policy import (
    DATE_HINT as _DATE_HINT,
)
from .policy import (
    GENERIC_ENTITY_NAMES as _GENERIC_ENTITY_NAMES,
)
from .policy import (
    HISTORY_INTENT as _HISTORY_INTENT,
)
from .policy import (
    PARTICIPANT_INTENT as _PARTICIPANT_INTENT,
)
from .policy import RECENT_INTENT as _RECENT_INTENT
from .policy import (
    RecallCandidate as _Candidate,
)
from .policy import RecallPolicyV1
from .tokenization import lexical_tokens, normalize_text, token_overlap


class RecallSelectionMixin:
    if TYPE_CHECKING:
        policy: RecallPolicyV1
        _ensure_change_pairs: Callable[[list[_Candidate], Sequence[_Candidate], int], None]
        _prefetch_safe: Callable[[RecallRequest, _Candidate], bool]
        _time_sort_key: Callable[[Mapping[str, object]], tuple[int, float, str]]

    def _select_candidates(
        self, request: RecallRequest, candidates: Sequence[_Candidate]
    ) -> list[_Candidate]:
        filtered = self._eligible_candidates(request, candidates)
        limit = (
            self.policy.prefetch_limit
            if request.mode is RecallMode.PREFETCH
            else self.policy.final_limit
        )
        selected = self._mmr_candidates(request, filtered, limit)
        if _CHANGE_INTENT.search(request.need):
            self._ensure_change_pairs(selected, candidates, limit)
        return selected[:limit]

    def _eligible_candidates(
        self, request: RecallRequest, candidates: Sequence[_Candidate]
    ) -> list[_Candidate]:
        threshold = (
            self.policy.prefetch_confidence
            if request.mode is RecallMode.PREFETCH
            else self.policy.explicit_confidence
        )
        visible = set(request.visible_source_fingerprints)
        excluded = set(request.excluded_document_keys)
        allowed_source_types = set(request.allowed_source_types)
        allowed_authority_statuses = set(request.allowed_authority_statuses)
        filtered = [
            item
            for item in candidates
            if self._candidate_is_eligible(
                request,
                item,
                threshold=threshold,
                visible=visible,
                excluded=excluded,
                allowed_source_types=allowed_source_types,
                allowed_authority_statuses=allowed_authority_statuses,
            )
        ]
        if _RECENT_INTENT.search(request.need):
            filtered.sort(
                key=lambda item: (
                    self._time_sort_key(item.document),
                    item.confidence,
                    item.rrf,
                ),
                reverse=True,
            )
        else:
            filtered.sort(
                key=lambda item: (
                    -self._intent_adjusted_confidence(request, item),
                    -item.rrf,
                    item.document["document_key"],
                )
            )
        return filtered

    def _candidate_is_eligible(
        self,
        request: RecallRequest,
        candidate: _Candidate,
        *,
        threshold: float,
        visible: set[str],
        excluded: set[str],
        allowed_source_types: set[str],
        allowed_authority_statuses: set[str],
    ) -> bool:
        document = candidate.document
        if self._intent_adjusted_confidence(request, candidate) < threshold:
            return False
        if (
            allowed_source_types
            and str(document.get("source_type") or "").upper() not in allowed_source_types
        ):
            return False
        if (
            allowed_authority_statuses
            and str(document.get("authority_status") or "").upper()
            not in allowed_authority_statuses
        ):
            return False
        if str(document["document_key"]) in excluded:
            return False
        if str(document.get("source_fingerprint") or "") in visible:
            return False
        if content_fingerprint(document.get("content", "")) in visible:
            return False
        return self._prefetch_safe(request, candidate)

    def _mmr_candidates(
        self, request: RecallRequest, filtered: list[_Candidate], limit: int
    ) -> list[_Candidate]:
        selected: list[_Candidate] = []
        change_intent = bool(_CHANGE_INTENT.search(request.need))
        per_source: dict[tuple[str, str], int] = defaultdict(int)
        while filtered and len(selected) < limit:
            best_index = self._best_mmr_index(
                request, filtered, selected, per_source, change_intent
            )
            candidate = filtered.pop(best_index)
            source = self._candidate_source(candidate)
            if per_source[source] >= (2 if change_intent else 1):
                continue
            selected.append(candidate)
            per_source[source] += 1
        return selected

    def _best_mmr_index(
        self,
        request: RecallRequest,
        candidates: Sequence[_Candidate],
        selected: Sequence[_Candidate],
        per_source: Mapping[tuple[str, str], int],
        change_intent: bool,
    ) -> int:
        best_index, best_score = 0, -1.0
        for index, candidate in enumerate(candidates):
            source = self._candidate_source(candidate)
            if per_source.get(source, 0) >= (2 if change_intent else 1):
                continue
            diversity = self._candidate_diversity(candidate, selected)
            score = (
                self.policy.mmr_lambda * self._intent_adjusted_confidence(request, candidate)
                - (1.0 - self.policy.mmr_lambda) * diversity
            )
            if score > best_score:
                best_index, best_score = index, score
        return best_index

    @staticmethod
    def _candidate_source(candidate: _Candidate) -> tuple[str, str]:
        return (
            str(candidate.document["source_type"]),
            str(candidate.document["source_key"]),
        )

    @staticmethod
    def _candidate_diversity(candidate: _Candidate, selected: Sequence[_Candidate]) -> float:
        diversity = max(
            (
                token_overlap(
                    lexical_tokens(candidate.document.get("content", "")),
                    lexical_tokens(item.document.get("content", "")),
                )
                for item in selected
            ),
            default=0.0,
        )
        return diversity * 0.2 if "graph" in candidate.route_ranks else diversity

    @staticmethod
    def _intent_adjusted_confidence(request: RecallRequest, candidate: _Candidate) -> float:
        score = candidate.confidence
        if _CHANGE_INTENT.search(request.need):
            return score
        status = str(candidate.document.get("authority_status") or "")
        source_type = str(candidate.document.get("source_type") or "")
        if _CURRENT_INTENT.search(request.need):
            current_fact = status == "CURRENT" and source_type in {
                "WORLD_INFO",
                "ROLE_CURRENT",
            }
            score += 0.1 if current_fact else -0.08
        elif _HISTORY_INTENT.search(request.need):
            historical_event = status == "HISTORICAL" or source_type in {
                "MEMORY",
                "ROLE_EVENT",
                "DIALOGUE_SUMMARY",
                "MESSAGE",
            }
            score += 0.05 if historical_event else -0.02
        return max(0.0, min(1.0, score))


_CURRENT_OVERVIEW_INTENT = re.compile(r"(怎么样|如何|什么情况|近况|状态)")
_ASSERTED_CORRECTION = re.compile(
    r"(不再|不打算|不准备|取消(?:了)?|改成|改为|纠正为|更正为|修正为|取代)"
)
_INTERROGATIVE = re.compile(r"(什么|有没有|是否|为何|为什么|怎么|哪(?:里|个|些)|[?？])")
_CORRECTION_PREFETCH_MAX_LEXICAL_RANK = 5


class RecallRankingMixin:
    if TYPE_CHECKING:
        _entity_index_cache: dict[tuple[str, str], dict[str, tuple[str, ...]]]
        _remember_scope: Callable[[Any, Any, Any], None]

    def _confidence(self, request: RecallRequest, candidate: _Candidate) -> float:
        base, lexical, entity, predicate = self._direct_signal_confidence(candidate)
        base = self._entity_confidence_floor(
            request,
            candidate,
            base=base,
            entity=entity,
            lexical=lexical,
            predicate=predicate,
        )
        base = max(base, self._graph_confidence_floor(request, candidate))
        if "scene" in candidate.route_ranks:
            base = max(base, 0.44)
        base += self._support_confidence_adjustment(request, candidate)
        if self._anchored_message_correction(request, candidate):
            # This is a final calibrated guarantee, not a route floor. Raw
            # messages normally receive a prefetch precision penalty, but a
            # top-BM25 declarative correction that shares a distinctive term
            # with an explicit change question has already passed the stricter
            # rule below and must remain above the prefetch threshold.
            base = max(base, 0.86)
        base += self._message_question_adjustment(request, candidate)
        return max(0.0, min(0.99, base))

    @staticmethod
    def _message_question_adjustment(request: RecallRequest, candidate: _Candidate) -> float:
        """Keep prior questions from masquerading as change evidence.

        A lexical rewrite of the current question can legitimately outrank the
        answer in FTS. For a change-intent query, a raw interrogative message is
        still useful as context but is not itself evidence that the change
        happened. A bounded penalty drops a lone lexical echo below the
        explicit threshold while allowing independently corroborated messages
        to remain candidates.
        """

        if str(candidate.document.get("source_type") or "") != "MESSAGE":
            return 0.0
        if _CHANGE_INTENT.search(request.need) is None:
            return 0.0
        content = str(candidate.document.get("content") or "")
        return -0.22 if _INTERROGATIVE.search(content) is not None else 0.0

    @staticmethod
    def _direct_signal_confidence(candidate: _Candidate) -> tuple[float, float, float, float]:
        signals = candidate.signals
        exact = min(1.0, max(0.0, signals.get("exact", 0.0)))
        lexical = min(1.0, max(0.0, signals.get("lexical", 0.0)))
        dense = max(0.0, min(1.0, (signals.get("dense", -1.0) + 1.0) / 2.0))
        rerank = max(0.0, min(1.0, signals.get("rerank", 0.0)))
        lexical_anchor = max(0.0, min(1.0, signals.get("lexical_anchor", 0.0)))
        base = max(exact, lexical * 0.84, dense * 0.82, rerank * 0.94)
        lexical_rank = candidate.route_ranks.get("lexical")
        if (
            lexical_rank is not None
            and lexical_rank <= 3
            and (lexical >= 0.28 or lexical_anchor >= 0.20)
        ):
            # BM25 already established that this is one of the best lexical
            # documents in the scope.  Concise evidence may be diluted by the
            # query length and projection labels, so a verified complete-term
            # anchor can calibrate it without lowering the global threshold.
            base = max(base, 0.58)
        elif lexical_rank is not None and lexical >= 0.28 and lexical_rank <= 5:
            base = max(base, 0.48)
        entity = max(0.0, min(1.0, signals.get("entity", 0.0)))
        predicate = max(0.0, min(1.0, signals.get("predicate", 0.0)))
        return base, lexical, entity, predicate

    @staticmethod
    def _graph_confidence_floor(request: RecallRequest, candidate: _Candidate) -> float:
        graph_rank = candidate.route_ranks.get("graph")
        if graph_rank is None:
            return 0.0
        if _CHANGE_INTENT.search(request.need) and graph_rank <= 4:
            return 0.46
        if _COMPLEX_INTENT.search(request.need) and graph_rank <= 3:
            return 0.44
        if _HISTORY_INTENT.search(request.need) and graph_rank <= 2:
            return 0.40
        return 0.0

    @staticmethod
    def _support_confidence_adjustment(request: RecallRequest, candidate: _Candidate) -> float:
        signals = candidate.signals
        source_type = str(candidate.document.get("source_type") or "")
        lexical = signals.get("lexical", 0.0)
        message_predicate_supported = source_type != "MESSAGE" or lexical >= 0.25
        strong_routes = sum(
            (
                lexical >= 0.25,
                signals.get("dense", -1.0) >= 0.35,
                signals.get("exact", 0.0) >= 0.55,
                signals.get("entity", 0.0) >= 0.7 and message_predicate_supported,
                signals.get("predicate", 0.0) >= 0.1 and message_predicate_supported,
                signals.get("rerank", 0.0) >= 0.5,
            )
        )
        adjustment = max(0, strong_routes - 1) * 0.045
        evidence_count = len(candidate.document.get("evidence") or ())
        if evidence_count >= 2:
            adjustment += 0.04
        if request.mode is RecallMode.PREFETCH and candidate.document["source_type"] == "MESSAGE":
            adjustment -= 0.08
        return adjustment

    @classmethod
    def _entity_confidence_floor(
        cls,
        request: RecallRequest,
        candidate: _Candidate,
        *,
        base: float,
        entity: float,
        lexical: float,
        predicate: float,
    ) -> float:
        if not entity:
            return base
        base = max(base, 0.34)
        if str(candidate.document.get("source_type") or "") == "MESSAGE":
            # A participant name legitimately matches every message they sent.
            # Generic interrogative overlap is not independent corroboration;
            # require a strong lexical match before promoting one such message
            # to factual evidence.
            if lexical >= 0.25:
                return max(base, 0.82 if entity >= 0.95 else 0.76)
            # Multi-branch questions such as “两个人分别喜欢什么” must retain
            # each exact participant's relevant statement even when short text
            # and projection labels dilute cosine-style overlap. Predicate
            # support plus a bounded lexical floor keeps unrelated messages
            # from the same frequent participant below the threshold.
            if (
                request.mode is RecallMode.EXPLICIT
                and _COMPLEX_INTENT.search(request.need) is not None
                and predicate >= 0.1
                and lexical >= 0.18
            ):
                return max(base, 0.46)
            return base
        supported = any(
            (
                lexical >= 0.25,
                predicate >= 0.1,
                cls._current_overview_prefetch(request, candidate),
                cls._scoped_current_world_info(request, candidate, entity),
            )
        )
        return max(base, 0.82 if entity >= 0.95 else 0.76) if supported else base

    @staticmethod
    def _current_overview_prefetch(request: RecallRequest, candidate: _Candidate) -> bool:
        # Generic current-state questions may have little predicate overlap, but
        # exact entity + FTS is strong for this narrow overview shape.
        return (
            request.mode is RecallMode.PREFETCH
            and "lexical" in candidate.route_ranks
            and _CURRENT_INTENT.search(request.need) is not None
            and _CURRENT_OVERVIEW_INTENT.search(request.need) is not None
        )

    @staticmethod
    def _anchored_message_correction(request: RecallRequest, candidate: _Candidate) -> bool:
        if request.mode is not RecallMode.PREFETCH or _CHANGE_INTENT.search(request.need) is None:
            return False
        if str(candidate.document.get("source_type") or "") != "MESSAGE":
            return False
        if candidate.route_ranks.get("lexical", 10**9) > _CORRECTION_PREFETCH_MAX_LEXICAL_RANK:
            return False
        if candidate.signals.get("lexical_anchor", 0.0) < 0.20:
            return False
        content = str(candidate.document.get("content") or "")
        return (
            _ASSERTED_CORRECTION.search(content) is not None
            and _INTERROGATIVE.search(content) is None
        )

    @staticmethod
    def _scoped_current_world_info(
        request: RecallRequest, candidate: _Candidate, entity: float
    ) -> bool:
        # A constrained terminal consumer asks for the named entity's active
        # canon; this rule deliberately does not apply to general recall.
        return (
            request.mode is RecallMode.PREFETCH
            and request.allowed_source_types == frozenset({"WORLD_INFO"})
            and request.allowed_authority_statuses == frozenset({"CURRENT"})
            and candidate.document.get("source_type") == "WORLD_INFO"
            and candidate.document.get("authority_status") == "CURRENT"
            and entity >= 0.95
            and "lexical" in candidate.route_ranks
        )

    @staticmethod
    def _exact_score(
        query: str,
        document: Mapping[str, Any],
        *,
        normalized_query: str = "",
        query_tokens: Sequence[str] = (),
    ) -> float:
        normalized = normalized_query or normalize_text(query)
        if not normalized:
            return 0.0
        names = list(document.get("_recall_normalized_names") or ())
        content = str(
            document.get("_recall_normalized_content")
            or normalize_text(document.get("content", ""))
        )
        if len(normalized) >= 3 and normalized in content:
            return 0.94
        hints = _DATE_HINT.findall(normalized)
        if hints and any(
            hint
            in " ".join(
                str(document.get(key) or "") for key in ("occurred_at", "valid_from", "valid_until")
            )
            for hint in hints
        ):
            return 0.91
        content_tokens = document.get("_recall_content_tokens") or lexical_tokens(
            content, aliases=names
        )
        overlap = token_overlap(query_tokens or lexical_tokens(normalized), content_tokens)
        return overlap if overlap >= 0.28 else 0.0

    def _entity_route(
        self,
        profile_id: str,
        instance_id: str,
        normalized_query: str,
        documents: Sequence[Mapping[str, Any]],
    ) -> tuple[list[tuple[str, float]], frozenset[str]]:
        scope = (profile_id, instance_id)
        index = self._entity_index_cache.get(scope)
        if index is None:
            mutable: dict[str, list[str]] = defaultdict(list)
            for document in documents:
                for name in document.get("_recall_normalized_names", ()):
                    if name:
                        mutable[str(name)].append(str(document["document_key"]))
            index = {name: tuple(dict.fromkeys(keys)) for name, keys in mutable.items()}
            self._remember_scope(self._entity_index_cache, scope, index)
        name_scores = {
            name: 1.0
            for name in index
            if len(name) >= 2 and name not in _GENERIC_ENTITY_NAMES and name in normalized_query
        }
        if not name_scores:
            name_scores = {
                name: score
                for name in index
                if (score := self._fuzzy_entity_score(normalized_query, (name,))) > 0
            }
        document_scores: dict[str, float] = {}
        for name, score in name_scores.items():
            for document_key in index[name]:
                document_scores[document_key] = max(document_scores.get(document_key, 0.0), score)
        route = sorted(document_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return route, frozenset(document_scores)

    @staticmethod
    def _predicate_score(normalized_query: str, document: Mapping[str, Any]) -> float:
        query = normalized_query
        content = str(document.get("_recall_normalized_content") or "")
        names = sorted(
            (str(item) for item in document.get("_recall_normalized_names", ()) if str(item)),
            key=len,
            reverse=True,
        )
        for name in names:
            if name in query:
                query = query.replace(name, " ")
            content = content.replace(name, " ")
        query_tokens = lexical_tokens(query)
        if not query_tokens:
            return 0.0
        return token_overlap(query_tokens, lexical_tokens(content))

    @staticmethod
    def _fuzzy_entity_score(query: str, names: Sequence[str]) -> float:
        best = 0.0
        query_numbers = tuple(re.findall(r"\d+", query))
        for name in names:
            if name in _GENERIC_ENTITY_NAMES or not (2 <= len(name) <= 20):
                continue
            name_numbers = tuple(re.findall(r"\d+", name))
            if query_numbers and name_numbers and query_numbers != name_numbers:
                continue
            for width in range(max(2, len(name) - 1), min(len(query), len(name) + 1) + 1):
                for start in range(0, len(query) - width + 1):
                    ratio = difflib.SequenceMatcher(
                        None,
                        name,
                        query[start : start + width],
                        autojunk=False,
                    ).ratio()
                    best = max(best, ratio)
        if best >= 0.72:
            return 0.88
        if best >= 0.5 and any(len(name) <= 3 for name in names):
            return 0.72
        return 0.0

    @staticmethod
    def _expansion_seed_quality(candidate: _Candidate) -> float:
        exact = max(0.0, min(1.0, candidate.signals.get("exact", 0.0)))
        lexical = max(0.0, min(1.0, candidate.signals.get("lexical", 0.0)))
        dense_raw = candidate.signals.get("dense")
        dense = max(0.0, min(1.0, (float(dense_raw) + 1.0) / 2.0)) if dense_raw is not None else 0.0
        entity = max(0.0, min(1.0, candidate.signals.get("entity", 0.0)))
        if str(candidate.document.get("source_type") or "") == "MESSAGE":
            # High-cardinality participant entities retrieve candidates, but
            # cannot seed graph diffusion alone.  Otherwise every message from
            # that participant becomes a high-quality starting point.
            entity = 0.0
        predicate = max(0.0, min(1.0, candidate.signals.get("predicate", 0.0)))
        return max(exact, lexical, dense, entity, predicate)

    @staticmethod
    def _allowed_edges(need: str) -> set[str]:
        if _PARTICIPANT_INTENT.search(need):
            # Event -> participant -> current fact.  Chronological and scene
            # edges would leak into adjacent events that merely happened near
            # the named event and overwhelm the actual participant branch.
            return {
                "EVIDENCE_FOR",
                "PARTICIPATED_IN",
                "MENTIONS_ENTITY",
                "REVISED_BY",
                "SUPERSEDED_BY",
                "DERIVED_CURRENT_STATE",
            }
        if _CHANGE_INTENT.search(need):
            return {
                "REVISED_BY",
                "SUPERSEDED_BY",
                "CONFLICTS_WITH",
                "BEFORE",
                "AFTER",
                "EVIDENCE_FOR",
            }
        if _COMPLEX_INTENT.search(need):
            return {
                "EVIDENCE_FOR",
                "PARTICIPATED_IN",
                "MENTIONS_ENTITY",
                "REVISED_BY",
                "SUPERSEDED_BY",
                "CONFLICTS_WITH",
                "BEFORE",
                "AFTER",
                "DERIVED_CURRENT_STATE",
                "BELONGS_TO_SCENE",
                "BELONGS_TO_TOPIC",
            }
        if _HISTORY_INTENT.search(need):
            return {
                "EVIDENCE_FOR",
                "REVISED_BY",
                "SUPERSEDED_BY",
                "CONFLICTS_WITH",
                "BEFORE",
                "AFTER",
                "DERIVED_CURRENT_STATE",
                "BELONGS_TO_SCENE",
                "BELONGS_TO_TOPIC",
            }
        return {
            "EVIDENCE_FOR",
            "MENTIONS_ENTITY",
            "REVISED_BY",
            "SUPERSEDED_BY",
            "DERIVED_CURRENT_STATE",
            "BELONGS_TO_SCENE",
        }

    @staticmethod
    def _benchmark_query(document: Mapping[str, Any]) -> str:
        names = [
            str(item).strip()
            for item in document.get("entity_names", ())
            if str(item).strip() and str(item).strip() not in _GENERIC_ENTITY_NAMES
        ]
        title = str(document.get("title") or "").strip()
        generic_titles = {"记忆", "角色经历", "角色当前状态", "会话摘要", "聊天原话"}
        if names:
            return f"关于{names[0]}的相关情况"
        if title and title not in generic_titles:
            return title[:120]
        content = str(document.get("content") or "").strip()
        return re.split(r"[。！？\n]", content, maxsplit=1)[0][:120]

    @staticmethod
    def _stratified_sample(documents: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in documents:
            buckets[
                (str(item.get("source_type") or ""), str(item.get("authority_status") or ""))
            ].append(item)
        selected: list[dict[str, Any]] = []
        ordered = [buckets[key] for key in sorted(buckets)]
        while len(selected) < limit and any(ordered):
            for bucket in ordered:
                if bucket and len(selected) < limit:
                    selected.append(bucket.pop(0))
        return selected

    @staticmethod
    def _percentile(values: Sequence[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(item) for item in values)
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
        return ordered[index]

    @staticmethod
    def _personalized_page_rank(
        seeds: Mapping[str, float], adjacency: Mapping[str, Sequence[tuple[str, float]]]
    ) -> dict[str, float]:
        total = sum(max(0.0, value) for value in seeds.values()) or 1.0
        personalization = {key: max(0.0, value) / total for key, value in seeds.items()}
        mass = dict(personalization)
        for _ in range(3):
            next_mass: dict[str, float] = defaultdict(float)
            for key, value in personalization.items():
                next_mass[key] += 0.35 * value
            for source, value in mass.items():
                targets = adjacency.get(source, ())
                weight_total = sum(weight for _, weight in targets)
                if weight_total <= 0:
                    continue
                for target, weight in targets:
                    next_mass[target] += 0.65 * value * weight / weight_total
            mass = dict(next_mass)
        return mass

    @classmethod
    def _prefetch_safe(cls, request: RecallRequest, candidate: _Candidate) -> bool:
        if request.mode is not RecallMode.PREFETCH:
            return True
        routes = set(candidate.route_ranks)
        evidence_count = len(candidate.document.get("evidence") or ())
        return (
            "exact" in routes
            or {"entity", "lexical"} <= routes
            or {"entity", "predicate"} <= routes
            or len(routes & {"lexical", "dense", "rerank"}) >= 2
            or evidence_count >= 2
            or cls._anchored_message_correction(request, candidate)
        )

    @staticmethod
    def _ensure_change_pairs(
        selected: list[_Candidate], candidates: Sequence[_Candidate], limit: int
    ) -> None:
        by_source: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
        for item in candidates:
            if item.document["source_type"] in {"MEMORY", "WORLD_INFO"}:
                by_source[
                    (str(item.document["source_type"]), str(item.document["source_key"]))
                ].append(item)
        selected_keys = {str(item.document["document_key"]) for item in selected}
        for item in tuple(selected):
            source = (str(item.document["source_type"]), str(item.document["source_key"]))
            siblings = sorted(
                by_source.get(source, ()), key=lambda row: int(row.document["source_revision"])
            )
            if len(siblings) < 2:
                continue
            index = siblings.index(item)
            sibling = siblings[index - 1] if index > 0 else siblings[1]
            key = str(sibling.document["document_key"])
            if key in selected_keys:
                continue
            if len(selected) >= limit:
                selected[-1] = sibling
            else:
                selected.append(sibling)
            selected_keys.add(key)
            break

    @staticmethod
    def _validated_vector(value: Any, dimension: int) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1 or vector.size != dimension or not np.isfinite(vector).all():
            raise ValueError("Embedding Provider 返回了无效向量")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Embedding Provider 返回了零向量")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    @staticmethod
    def _time_sort_key(document: Mapping[str, Any]) -> tuple[int, float, str]:
        value = str(
            document.get("occurred_at")
            or document.get("valid_from")
            or document.get("recorded_from")
            or ""
        )
        if not value:
            return (0, 0.0, "")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.utcoffset() is None:
                parsed = parsed.replace(tzinfo=UTC)
            else:
                parsed = parsed.astimezone(UTC)
            return (1, parsed.timestamp(), value)
        except (OverflowError, ValueError):
            return (0, 0.0, value)

    @staticmethod
    def _time_label(document: Mapping[str, Any]) -> str:
        valid_from = str(document.get("valid_from") or "")
        valid_until = str(document.get("valid_until") or "")
        occurred = str(document.get("occurred_at") or "")
        if valid_from and valid_until:
            return f"有效于 {valid_from} 至 {valid_until}"
        if valid_from:
            return f"自 {valid_from} 起有效"
        if occurred:
            return f"发生于 {occurred}"
        return "时间未确定"

    @staticmethod
    def _evidence_note(document: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
        source_labels = {
            "MEMORY": "正式记忆修订",
            "WORLD_INFO": "正式 WorldInfo 修订",
            "ROLE_EVENT": "角色经历",
            "ROLE_CURRENT": "角色当前状态",
            "DIALOGUE_SUMMARY": "会话摘要及其成员范围",
            "MESSAGE": "可见聊天原话",
        }
        message_count = len(
            {int(value) for row in rows for value in row.get("message_ids", ()) if int(value) > 0}
        )
        base = source_labels.get(str(document.get("source_type") or ""), "已保存资料")
        if message_count:
            return f"来自{base}，并关联 {message_count} 条原话证据"
        return f"来自{base}"

    @classmethod
    def _render_section(
        cls, lines: list[str], title: str, values: Sequence[RecallEvidence]
    ) -> None:
        if not values:
            return
        lines.append(title + "：")
        for item in values:
            lines.append(
                "- "
                + cls._safe_text(item.statement)
                + cls._time_suffix(item)
                + "。依据："
                + cls._safe_text(item.evidence_note)
            )

    @staticmethod
    def _time_suffix(item: RecallEvidence) -> str:
        return f"（{item.time_label}）" if item.time_label else ""

    @staticmethod
    def _safe_text(value: str) -> str:
        return (
            str(value or "")
            .replace("[[", "［［")
            .replace("]]", "］］")
            .replace("<", "＜")
            .replace(">", "＞")
            .strip()
        )

    @staticmethod
    def _truncate_tokens(value: str, token_budget: int) -> str:
        maximum_chars = max(256, int(token_budget) * 3)
        if len(value) <= maximum_chars:
            return value
        return value[: maximum_chars - 18].rstrip() + "\n（其余证据已省略）"

    @staticmethod
    def _finish_diagnostics(diagnostics: dict[str, Any], started: float) -> dict[str, Any]:
        diagnostics["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return diagnostics


__all__ = ["RecallRankingMixin"]

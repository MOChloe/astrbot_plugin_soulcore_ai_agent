"""Chinese-first, media-agnostic sticker retrieval and diversity ranking.

The retriever deliberately consumes only the text projection produced by the
sticker Check boundary.  Static images and animated GIFs therefore follow the
same ranking path and their pixels never leak into Main Core.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .domain import StickerItem, StickerItemStatus

_ENGLISH_WORD = re.compile(r"[a-z0-9][a-z0-9_+'-]*", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

# These are query hints, not a closed ontology.  Direct matches against the
# item's structured fields are also retained, so administrator-defined labels
# continue to work without code changes.
_EMOTION_HINTS: dict[str, frozenset[str]] = {
    "开心": frozenset({"开心", "高兴", "快乐", "笑死", "哈哈", "乐", "爽"}),
    "无语": frozenset({"无语", "沉默", "离谱", "难评", "汗", "尴尬"}),
    "生气": frozenset({"生气", "愤怒", "火大", "气死", "恼火"}),
    "难过": frozenset({"难过", "伤心", "委屈", "想哭", "哭了", "悲伤"}),
    "惊讶": frozenset({"惊讶", "震惊", "居然", "什么", "卧槽", "啊"}),
    "害怕": frozenset({"害怕", "恐惧", "怕", "瑟瑟发抖"}),
    "得意": frozenset({"得意", "骄傲", "赢了", "拿下", "厉害"}),
    "喜欢": frozenset({"喜欢", "爱了", "可爱", "心动", "亲亲"}),
}

_SPEECH_ACT_HINTS: dict[str, frozenset[str]] = {
    "AGREE": frozenset({"同意", "赞同", "确实", "对对对", "就是"}),
    "REFUSE": frozenset({"拒绝", "不要", "不行", "达咩", "禁止"}),
    "QUESTION": frozenset({"为什么", "怎么", "什么", "真的吗", "吗", "呢"}),
    "COMPLAIN": frozenset({"吐槽", "离谱", "受不了", "难评", "无语"}),
    "COMFORT": frozenset({"安慰", "没事", "抱抱", "别难过", "会好的"}),
    "THANK": frozenset({"谢谢", "感谢", "多谢", "爱你"}),
    "APOLOGIZE": frozenset({"对不起", "抱歉", "错了", "道歉"}),
    "TEASE": frozenset({"嘲笑", "笑你", "笨蛋", "就这", "菜"}),
    "ANNOUNCE": frozenset({"宣布", "来了", "登场", "注意", "开饭"}),
}


@dataclass(frozen=True, slots=True)
class StickerUsageWindow:
    recent_item_ids: frozenset[str] = frozenset()
    recent_cluster_ids: frozenset[str] = frozenset()
    item_last_run: Mapping[str, int | str] | None = None
    cluster_last_run: Mapping[str, int | str] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> StickerUsageWindow:
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            recent_item_ids=frozenset(
                str(value) for value in payload.get("recent_item_ids", ()) if str(value)
            ),
            recent_cluster_ids=frozenset(
                str(value) for value in payload.get("recent_cluster_ids", ()) if str(value)
            ),
            item_last_run={
                str(key): value for key, value in dict(payload.get("item_last_run") or {}).items()
            },
            cluster_last_run={
                str(key): value
                for key, value in dict(payload.get("cluster_last_run") or {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class RankedSticker:
    row: StickerItem
    item_id: str
    semantic_group: str
    visual_group: str
    relevance: float
    affect_match: float
    persona_preference: float
    reinforcement: float
    novelty: float
    stable_tie: float
    recently_used: bool
    recent_semantic_group: bool

    @property
    def suppression_level(self) -> int:
        if self.recently_used:
            return 2
        if self.recent_semantic_group:
            return 1
        return 0

    @property
    def sort_key(self) -> tuple[float, ...]:
        # The order is a product requirement.  Reinforcement can only decide
        # among otherwise similarly relevant/persona-compatible items and
        # therefore cannot monopolise the workset.
        return (
            self.relevance,
            self.affect_match,
            self.persona_preference,
            self.reinforcement,
            self.novelty,
            self.stable_tie,
        )

    @property
    def display_score(self) -> float:
        return (
            self.relevance * 100.0
            + self.affect_match * 10.0
            + self.persona_preference * 5.0
            + self.reinforcement
            + self.novelty
        )


def tokenize_sticker_text(value: Any) -> frozenset[str]:
    """Return jieba terms plus CJK bigrams and English words.

    CJK bigrams are intentionally added even when jieba recognises a longer
    phrase.  This makes short colloquial queries such as ``无语`` or ``笑死``
    recall descriptions containing the same phrase without falling back to a
    whole-sentence SQL ``LIKE``.
    """

    text = _normal_text(value)
    if not text:
        return frozenset()
    terms: set[str] = set(_ENGLISH_WORD.findall(text))
    try:  # requirements installs jieba; source-only tests keep a deterministic fallback.
        import jieba  # type: ignore

        segmented = jieba.cut_for_search(text, HMM=True)
    except ImportError:
        segmented = (*_ENGLISH_WORD.findall(text), *_CJK_RUN.findall(text))
    for token in segmented:
        normalized = _normal_token(token)
        if normalized and (_contains_cjk(normalized) or len(normalized) >= 2):
            terms.add(normalized)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            terms.add(run)
        else:
            terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return frozenset(term for term in terms if term)


def structured_search_tokens(query: Any) -> tuple[str, ...]:
    """Stable token list suitable for a repository-side token index query."""

    normalized = _normal_text(query)
    tokens = set(tokenize_sticker_text(normalized))
    for label, hints in (*_EMOTION_HINTS.items(), *_SPEECH_ACT_HINTS.items()):
        if any(term in normalized or term in tokens for term in hints):
            tokens.add(_normal_text(label))
    return tuple(sorted(tokens, key=lambda item: (len(item), item)))


class StickerRetrievalRanker:
    def rank(
        self,
        rows: Sequence[StickerItem],
        *,
        current_text: str,
        recent_texts: Sequence[str] = (),
        requirements: str = "",
        run_id: int | str = 0,
        usage: StickerUsageWindow | None = None,
    ) -> list[RankedSticker]:
        active = [row for row in rows if row.status is StickerItemStatus.ACTIVE]
        if not active:
            return []
        usage = usage or StickerUsageWindow()
        current_tokens = tokenize_sticker_text(current_text)
        recent_tokens: set[str] = set()
        # Recent dialogue is deliberately bounded and weaker than the current
        # player message.  It resolves pronouns without turning the full Run
        # Prompt or old history into a retrieval query.
        for text in tuple(recent_texts)[-8:]:
            recent_tokens.update(tokenize_sticker_text(text))
        requirement_tokens = _positive_requirement_tokens(requirements)
        current_normal = _normal_text(current_text)
        query_emotions = self._infer_labels(
            current_normal, current_tokens | recent_tokens, _EMOTION_HINTS
        )
        query_acts = self._infer_labels(
            current_normal, current_tokens | recent_tokens, _SPEECH_ACT_HINTS
        )

        document_tokens = [self._row_tokens(row) for row in active]
        frequencies: Counter[str] = Counter()
        for tokens in document_tokens:
            frequencies.update(tokens)
        total = len(active)

        ranked: list[RankedSticker] = []
        for row, tokens in zip(active, document_tokens, strict=False):
            candidate = self._rank_one(
                row,
                tokens,
                current_tokens=current_tokens,
                recent_tokens=recent_tokens,
                requirement_tokens=requirement_tokens,
                current_normal=current_normal,
                query_emotions=query_emotions,
                query_acts=query_acts,
                frequencies=frequencies,
                total=total,
                usage=usage,
                run_id=run_id,
            )
            if candidate is not None:
                ranked.append(candidate)

        # Recently used resources are held back as a fallback pool.  Inside
        # each pool the product ranking order above remains authoritative.
        return sorted(
            ranked,
            key=lambda item: (-item.suppression_level, *item.sort_key, item.item_id),
            reverse=True,
        )

    @classmethod
    def _rank_one(
        cls,
        row: StickerItem,
        tokens: frozenset[str],
        *,
        current_tokens: frozenset[str],
        recent_tokens: set[str],
        requirement_tokens: frozenset[str],
        current_normal: str,
        query_emotions: frozenset[str],
        query_acts: frozenset[str],
        frequencies: Counter[str],
        total: int,
        usage: StickerUsageWindow,
        run_id: int | str,
    ) -> RankedSticker | None:
        item_id = row.item_id
        if not item_id:
            return None
        semantic = _semantic_group(row, item_id)
        visual = row.visual_group or row.phash or item_id
        relevance = cls._relevance(
            row,
            tokens,
            current_tokens=current_tokens,
            recent_tokens=recent_tokens,
            current_normal=current_normal,
            frequencies=frequencies,
            total=total,
        )
        affect = cls._affect(row, current_normal, query_emotions, query_acts)
        persona = _bounded_float(row.persona_score, 0.0, 1.0, 0.5)
        preference = persona + min(1.0, len(requirement_tokens & tokens) * 0.25)
        reinforcement = _bounded_float(row.reinforcement_score, -100.0, 100.0, 0.0) / 100.0
        item_recent = item_id in usage.recent_item_ids
        cluster_recent = semantic in usage.recent_cluster_ids
        novelty = 0.0 if item_recent else (0.5 if cluster_recent else 1.0)
        return RankedSticker(
            row=row,
            item_id=item_id,
            semantic_group=semantic,
            visual_group=visual,
            relevance=relevance,
            affect_match=affect,
            persona_preference=preference,
            reinforcement=reinforcement,
            novelty=novelty,
            stable_tie=_stable_fraction(f"sticker:{run_id}:{item_id}"),
            recently_used=item_recent,
            recent_semantic_group=cluster_recent,
        )

    @classmethod
    def _relevance(
        cls,
        row: StickerItem,
        tokens: frozenset[str],
        *,
        current_tokens: frozenset[str],
        recent_tokens: set[str],
        current_normal: str,
        frequencies: Counter[str],
        total: int,
    ) -> float:
        current_overlap = current_tokens & tokens
        recent_overlap = (recent_tokens - current_tokens) & tokens
        score = sum(cls._idf(total, frequencies[token]) * 3.0 for token in current_overlap)
        score += sum(cls._idf(total, frequencies[token]) for token in recent_overlap)
        semantic_text = _normal_text(row.semantic_key)
        return score + (8.0 if semantic_text and semantic_text in current_normal else 0.0)

    @staticmethod
    def _affect(
        row: StickerItem,
        current_normal: str,
        query_emotions: frozenset[str],
        query_acts: frozenset[str],
    ) -> float:
        emotion = _normal_text(row.emotion)
        speech_act = _normal_text(row.speech_act).upper()
        emotion_match = emotion and (emotion in query_emotions or emotion in current_normal)
        act_match = speech_act and (
            speech_act in query_acts or _normal_text(speech_act) in current_normal
        )
        return (2.0 if emotion_match else 0.0) + (2.0 if act_match else 0.0)

    @staticmethod
    def diverse_select(
        ranked: Sequence[RankedSticker],
        *,
        limit: int,
        semantic_variant_limit: int = 2,
    ) -> list[RankedSticker]:
        """Round-robin semantic groups while enforcing one visual variant.

        Fresh candidates are exhausted before recent semantic groups, and an
        exact item used in the last ten successful foreground turns is only
        considered when both earlier pools cannot fill the requested limit.
        """

        maximum = max(0, int(limit))
        if maximum <= 0:
            return []
        selected: list[RankedSticker] = []
        selected_ids: set[str] = set()
        visual_groups: set[str] = set()
        semantic_counts: Counter[str] = Counter()
        for suppression in (0, 1, 2):
            StickerRetrievalRanker._select_suppression_pool(
                ranked,
                suppression=suppression,
                maximum=maximum,
                semantic_variant_limit=semantic_variant_limit,
                selected=selected,
                selected_ids=selected_ids,
                visual_groups=visual_groups,
                semantic_counts=semantic_counts,
            )
            if len(selected) >= maximum:
                break
        return selected

    @staticmethod
    def _select_suppression_pool(
        ranked: Sequence[RankedSticker],
        *,
        suppression: int,
        maximum: int,
        semantic_variant_limit: int,
        selected: list[RankedSticker],
        selected_ids: set[str],
        visual_groups: set[str],
        semantic_counts: Counter[str],
    ) -> None:
        groups: dict[str, list[RankedSticker]] = defaultdict(list)
        for item in ranked:
            if item.suppression_level == suppression and item.item_id not in selected_ids:
                groups[item.semantic_group].append(item)
        queues = sorted(
            groups.values(),
            key=lambda values: (values[0].sort_key, values[0].item_id),
            reverse=True,
        )
        while queues and len(selected) < maximum:
            next_queues: list[list[RankedSticker]] = []
            progress = False
            for queue in queues:
                item = StickerRetrievalRanker._next_variant(
                    queue, visual_groups, semantic_counts, semantic_variant_limit
                )
                if item is not None:
                    selected.append(item)
                    selected_ids.add(item.item_id)
                    visual_groups.add(item.visual_group)
                    semantic_counts[item.semantic_group] += 1
                    progress = True
                if queue and semantic_counts[queue[0].semantic_group] < semantic_variant_limit:
                    next_queues.append(queue)
                if len(selected) >= maximum:
                    break
            if not progress:
                break
            queues = next_queues

    @staticmethod
    def _next_variant(
        queue: list[RankedSticker],
        visual_groups: set[str],
        semantic_counts: Counter[str],
        semantic_variant_limit: int,
    ) -> RankedSticker | None:
        while queue:
            item = queue.pop(0)
            if item.visual_group in visual_groups:
                continue
            if semantic_counts[item.semantic_group] >= semantic_variant_limit:
                continue
            return item
        return None

    @staticmethod
    def _idf(total: int, frequency: int) -> float:
        return math.log((total + 1.0) / (max(0, frequency) + 1.0)) + 1.0

    @staticmethod
    def _infer_labels(
        normalized_text: str,
        tokens: Iterable[str],
        hints: Mapping[str, frozenset[str]],
    ) -> frozenset[str]:
        token_set = set(tokens)
        return frozenset(
            label
            for label, terms in hints.items()
            if any(term in token_set or term in normalized_text for term in terms)
        )

    @staticmethod
    def _row_tokens(row: StickerItem) -> frozenset[str]:
        values: list[Any] = [
            row.compact_name,
            row.compact_description,
            row.visible_text,
            row.semantic_key,
            row.emotion,
            row.speech_act,
            row.search_index,
        ]
        values.extend(row.search_keywords)
        if row.metadata:
            extra = row.metadata.get("search_keywords") or ()
            values.extend((extra,) if isinstance(extra, str) else extra)
        result: set[str] = set()
        for value in values:
            result.update(tokenize_sticker_text(value))
        return frozenset(result)


def _semantic_group(row: StickerItem, item_id: str) -> str:
    return row.cluster_id or _normal_text(row.semantic_key) or item_id


def _positive_requirement_tokens(value: Any) -> frozenset[str]:
    """Use soft/positive preferences without rewarding explicitly banned themes."""

    output: set[str] = set()
    for clause in re.split(r"[。！？!?；;\n]+", _normal_text(value)):
        if not clause or any(
            marker in clause for marker in ("禁止", "不得", "不能", "不要", "不许")
        ):
            continue
        cleaned = re.sub(r"(?:必须|只允许|应当|应该|尽量|优先)", " ", clause)
        output.update(tokenize_sticker_text(cleaned))
    return frozenset(output)


def _normal_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).lower().split())


def _normal_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9_+'\-\u3400-\u4dbf\u4e00-\u9fff]+", "", _normal_text(value))


def _contains_cjk(value: str) -> bool:
    return bool(_CJK_RUN.search(value))


def _bounded_float(value: Any, lower: float, upper: float, fallback: float) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
        return fallback


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


__all__ = [
    "RankedSticker",
    "StickerRetrievalRanker",
    "StickerUsageWindow",
    "structured_search_tokens",
    "tokenize_sticker_text",
]

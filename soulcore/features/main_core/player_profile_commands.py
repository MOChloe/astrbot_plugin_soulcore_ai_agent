"""Run-scoped player-profile reads and deferred mutations for Main Core."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from ...shared.token_meter import ConservativeTokenMeter
from ..identity import encode_identity_template_for_model
from ..player_profiles import (
    AddProfileEntry,
    PlayerProfileEntry,
    PlayerProfileError,
    PlayerProfileProjectionRequest,
    ProfileCategory,
    ProfileEntryDraft,
    ProfileEntryStatus,
    ProfileLayer,
    ProfileMessageEvidence,
    ProfileSensitivity,
    ProfileSourceType,
    ReviseProfileEntry,
    WithdrawProfileEntry,
    apply_profile_command,
    build_compact_projection,
)
from .command_context import _active
from .player_profile_evidence import (
    _resolve_current_evidence,
)
from .player_profile_evidence import (
    evidence_message_id as _evidence_message_id,
)
from .player_profile_evidence import (
    impression_match_score as _impression_match_score,
)
from .player_profile_evidence import (
    normalize_evidence_text as _normalize_evidence_text,
)
from .player_profile_evidence import (
    normalized_impression as _normalized_impression,
)

MAX_PROFILE_MUTATIONS_PER_RUN = 12
PRIVATE_TARGET_REF = "private-counterpart"
_MODEL_LAYERS = {
    "DECLARED_FACT": "PLAYER_FACT",
    "INFERRED_OBSERVATION": "AI_OBSERVATION",
}
_MODEL_SOURCES = {
    "DIRECT_STATEMENT": "PLAYER_STATEMENT",
    "STRONG_TEXT_EVIDENCE": "STRONG_MESSAGE_EVIDENCE",
    "DIRECT_CORRECTION": "PLAYER_CORRECTION",
    "INFERRED_OBSERVATION": "AI_OBSERVATION",
}


async def remember_player_profile(
    _event: Any,
    content: str,
    member_ref: str = "",
    evidence_ref: str = "",
) -> dict[str, Any] | str:
    """Stage one stable, evidence-bound impression through the natural facade."""

    collector = _active()
    value = str(content or "").strip()
    if len(value) < 2:
        return _not_executed("想记住的内容不能为空", "INVALID_IMPRESSION")
    try:
        target = _resolve_target(collector, member_ref)
        _message_id, quote = _resolve_current_evidence(
            collector,
            target,
            evidence_ref,
            impression=value,
        )
    except ValueError as exc:
        return _not_executed(str(exc), "INSUFFICIENT_OR_AMBIGUOUS_EVIDENCE")
    return await update_player_profile(
        _event,
        "ADD",
        member_ref=member_ref,
        layer=ProfileLayer.PLAYER_FACT.value,
        category=_natural_profile_category(value).value,
        text=value,
        source_type=ProfileSourceType.PLAYER_STATEMENT.value,
        evidence_quote=quote,
    )


async def revise_player_profile(
    _event: Any,
    original_impression: str,
    new_impression: str,
    evidence_ref: str = "",
) -> dict[str, Any] | str:
    """Revise exactly one existing impression; never guess among plausible entries."""

    collector = _active()
    replacement = str(new_impression or "").strip()
    if len(replacement) < 2:
        return _not_executed("改成的认识不能为空", "INVALID_IMPRESSION")
    resolved = _resolve_impression(collector, original_impression)
    if isinstance(resolved, dict):
        return resolved
    target, entry = resolved
    if _normalized_impression(entry.text) == _normalized_impression(replacement):
        return _not_executed("新的认识与原来的印象没有实际变化", "UNCHANGED_IMPRESSION")
    if str(evidence_ref or "").strip():
        try:
            message_id, quote = _resolve_current_evidence(
                collector,
                target,
                evidence_ref,
                impression=replacement,
            )
        except ValueError as exc:
            return _not_executed(str(exc), "INSUFFICIENT_OR_AMBIGUOUS_EVIDENCE")
        evidence = (
            ProfileMessageEvidence(
                scope=target.scope,
                message_ref=f"ledger-message:{message_id}",
                note="Main Core run-visible text evidence",
            ),
        )
        return _stage_resolved_impression_change(
            collector,
            target,
            entry,
            operation="REVISE",
            replacement=replacement,
            evidence=evidence,
            evidence_message_id=message_id,
            evidence_quote=quote,
            reuse_existing_evidence=False,
        )
    return _stage_resolved_impression_change(
        collector,
        target,
        entry,
        operation="REVISE",
        replacement=replacement,
        evidence=entry.evidence,
        evidence_message_id=0,
        evidence_quote="",
        reuse_existing_evidence=True,
    )


async def forget_player_profile(
    _event: Any,
    impression: str,
) -> dict[str, Any] | str:
    """Withdraw exactly one impression without deleting its source conversation."""

    collector = _active()
    resolved = _resolve_impression(collector, impression)
    if isinstance(resolved, dict):
        return resolved
    target, entry = resolved
    return _stage_resolved_impression_change(
        collector,
        target,
        entry,
        operation="WITHDRAW",
        replacement="",
        evidence=entry.evidence,
        evidence_message_id=0,
        evidence_quote="",
        reuse_existing_evidence=True,
    )


async def recall_player_profile(
    _event: Any,
    member_ref: str = "",
) -> dict[str, Any] | str:
    """Return natural current impressions plus precise run-local edit references."""

    collector = _active()
    try:
        target = _resolve_target(collector, member_ref)
    except (KeyError, TypeError, ValueError):
        return _not_executed(
            "只能回顾当前私聊对象或本轮可见的现实群成员",
            "INVALID_REAL_CHAT_PERSON",
        )
    snapshot = getattr(target, "virtual_snapshot", None) or target.snapshot
    entries = sorted(
        snapshot.effective_entries,
        key=lambda item: (item.confirmed_at, item.updated_at, item.entry_id),
        reverse=True,
    )
    display_name = str(target.display_name or "").strip()
    heading = f"对{display_name}的印象" if display_name else "对这个人的印象"
    if not entries:
        return {"content": f"{heading}：目前还没有形成可回顾的稳定认识。"}
    lines = [f"{heading}："]
    for index, entry in enumerate(entries, start=1):
        nature = "对方明确说过" if entry.layer is ProfileLayer.PLAYER_FACT else "自己的观察"
        lines.append(f"{index}. {nature}：{entry.text}")
    lines.append("需要精确改动时，以上条目按顺序对应随后给出的短引用。")
    return {
        "content": "\n".join(lines),
        "profile_entry_refs": [entry.entry_id for entry in entries],
    }


async def view_player_profile(_event: Any, member_ref: str = "") -> dict[str, Any] | str:
    """Return one real group member's run-frozen profile within the current token share."""

    collector = _active()
    try:
        target = _resolve_target(collector, member_ref)
        if not str(getattr(target, "member_ref", "") or ""):
            raise ValueError("私聊人物肖像已经自动提供，不存在查看动作")
        snapshot = getattr(target, "virtual_snapshot", None) or target.snapshot
        token_limit = max(0, int(collector.player_profile_query_token_limit or 0))
        if token_limit < 1:
            return _not_executed("当前上下文已没有可安全载入人物认识的空间", "NO_CONTEXT_SPACE")
        projection = _query_projection(snapshot, token_limit)
        display_name = str(getattr(target, "display_name", "") or "").strip()
        heading = f"{display_name}的人物认识" if display_name else "所选群成员的人物认识"
        if projection.rendered:
            content = f"{heading}：\n{_model_identity_text(collector, projection.rendered)}"
            if projection.truncated:
                content += (
                    "\n本轮上下文空间有限，其余认识未载入；如有必要可围绕更具体的问题再次查看。"
                )
        else:
            content = f"{heading}：目前没有已保存的有效认识。"
        return {
            "content": content,
            "profile_entry_refs": [entry.entry_id for entry in projection.entries],
        }
    except (KeyError, TypeError, ValueError):
        return _not_executed(
            "只能查看本轮当前输入或近期对话中可见的现实群成员",
            "INVALID_REAL_GROUP_MEMBER",
        )


async def update_player_profile(
    _event: Any,
    operation: str,
    member_ref: str = "",
    entry_ref: str = "",
    layer: str = "",
    category: str = "",
    text: str = "",
    source_type: str = "",
    confidence: float | None = None,
    sensitivity: str = "",
    evidence_quote: str = "",
) -> dict[str, Any] | str:
    collector = _active()
    if len(collector.player_profile_mutations) >= MAX_PROFILE_MUTATIONS_PER_RUN:
        return _not_executed("本次行动的人物认识变更已经达到安全上限", "MUTATION_LIMIT")
    try:
        target = _resolve_target(collector, member_ref)
        normalized = _normalized_payload(
            operation=operation,
            member_ref=member_ref,
            entry_ref=entry_ref,
            layer=layer,
            category=category,
            text=text,
            source_type=source_type,
            confidence=confidence,
            sensitivity=sensitivity,
            evidence_quote=evidence_quote,
        )
        fingerprint = hashlib.sha256(
            json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        existing = collector.player_profile_mutation_fingerprints.get(fingerprint)
        if existing is not None:
            mutation = collector.player_profile_mutations[existing]
            return _result(mutation, idempotent=True)
        mutation = _build_mutation(collector, target, normalized)
        collector.player_profile_mutation_fingerprints[fingerprint] = len(
            collector.player_profile_mutations
        )
        collector.player_profile_mutations.append(mutation)
        _advance_virtual_target(target, mutation, collector.player_profile_confirmed_at)
        return _result(mutation, idempotent=False)
    except PlayerProfileError as exc:
        return _not_executed(exc.public_message, exc.code.value)
    except (KeyError, TypeError, ValueError):
        return _not_executed(
            "目标、原认识、证据原文或变更字段无效；人物肖像没有写入",
            "INVALID_PROFILE_CHANGE",
        )


def _model_identity_text(collector: Any, value: str) -> str:
    catalog = getattr(collector, "identity_catalog", None)
    if catalog is None:
        return value
    return encode_identity_template_for_model(value, catalog)


def _natural_profile_category(value: str) -> ProfileCategory:
    text = str(value or "")
    if re.search(r"(?:不喜欢|讨厌|厌恶)", text):
        return ProfileCategory.DISLIKE
    if "喜欢" in text:
        return ProfileCategory.LIKE
    if re.search(r"(?:兴趣|爱好|热衷)", text):
        return ProfileCategory.INTEREST
    if re.search(r"(?:习惯|总会|经常|通常)", text):
        return ProfileCategory.HABIT
    if re.search(r"(?:别提|不要问|不想聊|避开)", text):
        return ProfileCategory.AVOID_TOPIC
    if re.search(r"(?:边界|不接受|不能叫|别叫)", text):
        return ProfileCategory.BOUNDARY
    if re.search(r"(?:称呼|叫我|昵称|别名)", text):
        return ProfileCategory.RELATIONSHIP_NAME
    return ProfileCategory.OTHER


def _resolve_impression(
    collector: Any,
    descriptor: str,
) -> tuple[Any, PlayerProfileEntry] | dict[str, Any]:
    requested = str(descriptor or "").strip()
    if not requested:
        return _not_executed("需要说明想改动哪条印象", "IMPRESSION_NOT_FOUND")
    internal = collector.model_reference_map.get(requested, requested)
    candidates = _active_impressions(collector)
    direct = [
        (target, entry)
        for target, entry in candidates
        if entry.entry_id == str(internal or "").strip()
    ]
    if len(direct) == 1:
        return direct[0]
    query = _normalized_impression(requested)
    scored = [
        (_impression_match_score(query, _normalized_impression(entry.text)), target, entry)
        for target, entry in candidates
    ]
    matched = [(target, entry) for score, target, entry in scored if score >= 0.62]
    if len(matched) == 1:
        return matched[0]
    if not matched:
        return _not_executed(
            "没有找到能由这段描述唯一对应的已有印象；请先想想对这个人的印象",
            "IMPRESSION_NOT_FOUND",
        )
    limited = matched[:3]
    entry_ids = [entry.entry_id for _target, entry in limited]
    descriptions = "；".join(
        f"{index}. {entry.text}" for index, (_target, entry) in enumerate(limited, start=1)
    )
    return {
        "ok": False,
        "error": "AMBIGUOUS_IMPRESSION",
        "message": (
            "未执行：这段描述可能对应多条印象，请从少量候选中明确选择；"
            f"候选按顺序对应随后给出的短引用：{descriptions}"
        ),
        "profile_entry_refs": entry_ids,
    }


def _active_impressions(collector: Any) -> list[tuple[Any, PlayerProfileEntry]]:
    result: list[tuple[Any, PlayerProfileEntry]] = []
    for target in dict(collector.player_profile_targets or {}).values():
        snapshot = getattr(target, "virtual_snapshot", None) or target.snapshot
        for entry in snapshot.effective_entries:
            result.append((target, entry))
    result.sort(
        key=lambda item: (
            item[1].confirmed_at,
            item[1].updated_at,
            item[1].entry_id,
        ),
        reverse=True,
    )
    return result


def _stage_resolved_impression_change(
    collector: Any,
    target: Any,
    entry: PlayerProfileEntry,
    *,
    operation: str,
    replacement: str,
    evidence: tuple[Any, ...],
    evidence_message_id: int,
    evidence_quote: str,
    reuse_existing_evidence: bool,
) -> dict[str, Any]:
    if len(collector.player_profile_mutations) >= MAX_PROFILE_MUTATIONS_PER_RUN:
        return _not_executed("本次行动的人物认识变更已经达到安全上限", "MUTATION_LIMIT")
    snapshot = getattr(target, "virtual_snapshot", None) or target.snapshot
    current = snapshot.find_entry(entry.entry_id)
    if current is None or current.status is not ProfileEntryStatus.ACTIVE:
        return _not_executed("所选印象已经不是可修改的当前认识", "STALE_IMPRESSION")
    if not evidence:
        return _not_executed("所选印象没有可复用的受控依据", "NO_CONTROLLED_EVIDENCE")
    fingerprint_payload = {
        "operation": operation,
        "target_ref": str(target.target_ref),
        "entry_id": current.entry_id,
        "entry_version": current.version,
        "replacement": replacement,
        "evidence": [repr(item) for item in evidence],
        "reuse_existing_evidence": reuse_existing_evidence,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    existing_index = collector.player_profile_mutation_fingerprints.get(fingerprint)
    if existing_index is not None:
        return _result(collector.player_profile_mutations[existing_index], idempotent=True)
    ordinal = len(collector.player_profile_mutations) + 1
    idempotency_key = f"main-core-profile:{collector.core_run_id}:{ordinal}"
    confirmed_at = collector.player_profile_confirmed_at
    if not isinstance(confirmed_at, datetime):
        confirmed_at = datetime.now(UTC)
    if operation == "REVISE":
        payload = {
            "layer": "",
            "category": "",
            "text": replacement,
            "sensitivity": "",
            "confidence": None,
        }
        command: Any = ReviseProfileEntry(
            idempotency_key=idempotency_key,
            scope=target.scope,
            expected_profile_version=snapshot.version,
            entry_id=current.entry_id,
            expected_entry_version=current.version,
            draft=_revised_draft(current, payload, evidence, confirmed_at),
        )
    elif operation == "WITHDRAW":
        command = WithdrawProfileEntry(
            idempotency_key=idempotency_key,
            scope=target.scope,
            expected_profile_version=snapshot.version,
            entry_id=current.entry_id,
            expected_entry_version=current.version,
            evidence=evidence,
        )
    else:
        raise ValueError("unsupported natural impression operation")
    mutation = {
        "command": command,
        "target_ref": target.target_ref,
        "member_ref": target.member_ref,
        "sender_id": target.sender_id,
        "evidence_message_id": int(evidence_message_id),
        "evidence_quote": str(evidence_quote or ""),
        "reuse_existing_evidence": bool(reuse_existing_evidence),
        "operation": operation,
        "entry_ref": current.entry_id,
        "pending_profile_version": snapshot.version + 1,
    }
    collector.player_profile_mutation_fingerprints[fingerprint] = len(
        collector.player_profile_mutations
    )
    collector.player_profile_mutations.append(mutation)
    _advance_virtual_target(target, mutation, confirmed_at)
    return _result(mutation, idempotent=False)


def _query_projection(snapshot: Any, token_limit: int) -> Any:
    meter = ConservativeTokenMeter()
    content_limit = max(1, int(token_limit) - 48)
    low, high = 1, max(1, content_limit * 4)
    best = build_compact_projection(
        snapshot,
        PlayerProfileProjectionRequest(snapshot.scope, 1),
    )
    while low <= high:
        middle = (low + high) // 2
        candidate = build_compact_projection(
            snapshot,
            PlayerProfileProjectionRequest(snapshot.scope, middle),
        )
        if meter.count_text(candidate.rendered) <= content_limit:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _not_executed(message: str, code: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(code or "PROFILE_CHANGE_NOT_EXECUTED"),
        "message": f"未执行：{str(message or '人物肖像操作不符合当前边界').strip()}",
    }


def _resolve_target(collector: Any, member_ref: str) -> Any:
    targets = dict(collector.player_profile_targets or {})
    requested = str(member_ref or "").strip()
    is_group = any(bool(target.member_ref) for target in targets.values())
    if is_group:
        if not requested or requested not in targets:
            raise ValueError("人物短引用不属于本轮")
        target = targets[requested]
        allowed = collector.member_ref_allowlist.get(requested)
        if not allowed or str(allowed.get("sender_id") or "") != str(target.sender_id):
            raise ValueError("人物短引用不属于本轮")
        return target
    if requested:
        raise ValueError("私聊人物目标无需填写")
    target = targets.get(PRIVATE_TARGET_REF)
    if target is None:
        raise ValueError("本轮没有可用的人物肖像目标")
    return target


def _normalized_payload(**values: Any) -> dict[str, Any]:
    operation = str(values["operation"] or "").strip().upper()
    if operation not in {"ADD", "REVISE", "WITHDRAW"}:
        raise ValueError("操作只能是新增、修正或撤回")
    payload = {
        "operation": operation,
        "member_ref": str(values["member_ref"] or "").strip(),
        "entry_ref": str(values["entry_ref"] or "").strip(),
        "layer": _MODEL_LAYERS.get(
            str(values["layer"] or "").strip().upper(),
            str(values["layer"] or "").strip().upper(),
        ),
        "category": str(values["category"] or "").strip().upper(),
        "text": str(values["text"] or "").strip(),
        "source_type": _MODEL_SOURCES.get(
            str(values["source_type"] or "").strip().upper(),
            str(values["source_type"] or "").strip().upper(),
        ),
        "confidence": None if values["confidence"] is None else float(values["confidence"]),
        "sensitivity": str(values["sensitivity"] or "").strip().upper(),
        "evidence_quote": _normalize_evidence_text(values["evidence_quote"]),
    }
    _validate_operation_payload(payload)
    return payload


def _validate_operation_payload(payload: dict[str, Any]) -> None:
    operation = payload["operation"]
    if len(payload["evidence_quote"]) < 2:
        raise ValueError("证据原文必须逐字复制所选对象本人的可见文字")
    if operation == "ADD":
        if payload["entry_ref"]:
            raise ValueError("新增人物认识时不能填写原认识编号")
        if not all(payload[field] for field in ("layer", "category", "text", "source_type")):
            raise ValueError("新增条目缺少必填字段")
        payload["sensitivity"] = payload["sensitivity"] or ProfileSensitivity.NORMAL.value
        return
    if not payload["entry_ref"]:
        raise ValueError("修正或撤回需要已有认识的临时编号")
    if operation == "WITHDRAW":
        if (
            any(
                payload[field]
                for field in ("layer", "category", "text", "source_type", "sensitivity")
            )
            or payload["confidence"] is not None
        ):
            raise ValueError("撤回只接受人物、原认识和证据原文")
        return
    if payload["source_type"]:
        raise ValueError("修正时证据类型由当前条目自动确定")
    if not any(
        (
            payload["layer"],
            payload["category"],
            payload["text"],
            payload["sensitivity"],
            payload["confidence"] is not None,
        )
    ):
        raise ValueError("修正至少需要一个变化字段")


def _build_mutation(collector: Any, target: Any, payload: dict[str, Any]) -> dict[str, Any]:
    evidence_id = _evidence_message_id(target, payload["evidence_quote"])
    if evidence_id < 1:
        raise ValueError("需要一条本轮可见的对方消息作为证据")
    scope = target.scope
    evidence = (
        ProfileMessageEvidence(
            scope=scope,
            message_ref=f"ledger-message:{evidence_id}",
            note="Main Core run-visible text evidence",
        ),
    )
    snapshot = getattr(target, "virtual_snapshot", None) or target.snapshot
    ordinal = len(collector.player_profile_mutations) + 1
    idempotency_key = f"main-core-profile:{collector.core_run_id}:{ordinal}"
    operation = payload["operation"]
    confirmed_at = collector.player_profile_confirmed_at
    if not isinstance(confirmed_at, datetime):
        confirmed_at = datetime.now(UTC)
    if operation == "ADD":
        entry_id = f"profile-entry:{collector.core_run_id}:{ordinal}"
        command = AddProfileEntry(
            idempotency_key=idempotency_key,
            scope=scope,
            expected_profile_version=snapshot.version,
            entry_id=entry_id,
            draft=_add_draft(payload, evidence, confirmed_at),
        )
    else:
        entry_id = _resolve_entry_id(target, snapshot, payload["entry_ref"])
        existing = snapshot.find_entry(entry_id) if entry_id else None
        if existing is None or existing.status is not ProfileEntryStatus.ACTIVE:
            raise ValueError("原认识编号不在本轮可修改范围内")
        if operation == "REVISE":
            command = ReviseProfileEntry(
                idempotency_key=idempotency_key,
                scope=scope,
                expected_profile_version=snapshot.version,
                entry_id=entry_id,
                expected_entry_version=existing.version,
                draft=_revised_draft(existing, payload, evidence, confirmed_at),
            )
        else:
            command = WithdrawProfileEntry(
                idempotency_key=idempotency_key,
                scope=scope,
                expected_profile_version=snapshot.version,
                entry_id=entry_id,
                expected_entry_version=existing.version,
                evidence=evidence,
            )
    return {
        "command": command,
        "target_ref": target.target_ref,
        "member_ref": target.member_ref,
        "sender_id": target.sender_id,
        "evidence_message_id": evidence_id,
        "evidence_quote": payload["evidence_quote"],
        "operation": operation,
        "entry_ref": entry_id,
        "pending_profile_version": snapshot.version + 1,
    }


def _add_draft(
    payload: dict[str, Any],
    evidence: tuple[ProfileMessageEvidence, ...],
    confirmed_at: datetime,
) -> ProfileEntryDraft:
    return ProfileEntryDraft(
        layer=ProfileLayer(payload["layer"]),
        category=ProfileCategory(payload["category"]),
        text=payload["text"],
        source_type=ProfileSourceType(payload["source_type"]),
        evidence=evidence,
        confirmed_at=confirmed_at,
        confidence=payload["confidence"],
        sensitivity=ProfileSensitivity(payload["sensitivity"]),
    )


def _revised_draft(
    existing: PlayerProfileEntry,
    payload: dict[str, Any],
    evidence: tuple[ProfileMessageEvidence, ...],
    confirmed_at: datetime,
) -> ProfileEntryDraft:
    layer = ProfileLayer(payload["layer"]) if payload["layer"] else existing.layer
    confidence = payload["confidence"]
    if layer is ProfileLayer.PLAYER_FACT:
        source_type = ProfileSourceType.PLAYER_CORRECTION
        if confidence is not None:
            raise ValueError("明确事实不接受可信度")
        confidence = None
    else:
        source_type = ProfileSourceType.AI_OBSERVATION
        if confidence is None:
            confidence = existing.confidence
        if confidence is None:
            raise ValueError("切换为观察项时必须提供可信度")
    return ProfileEntryDraft(
        layer=layer,
        category=(
            ProfileCategory(payload["category"]) if payload["category"] else existing.category
        ),
        text=payload["text"] or existing.text,
        source_type=source_type,
        evidence=evidence,
        confirmed_at=confirmed_at,
        confidence=confidence,
        sensitivity=(
            ProfileSensitivity(payload["sensitivity"])
            if payload["sensitivity"]
            else existing.sensitivity
        ),
    )


def _resolve_entry_id(target: Any, snapshot: Any, reference: str) -> str:
    requested = str(reference or "").strip()
    direct = snapshot.find_entry(requested) if requested else None
    if direct is not None:
        return requested
    return _ensure_entry_map(target, snapshot).get(requested, "")


def _ensure_entry_map(target: Any, snapshot: Any) -> dict[str, str]:
    existing = dict(getattr(target, "entry_ref_map", None) or {})
    known_ids = set(existing.values())
    active = sorted(
        snapshot.effective_entries,
        key=lambda entry: (entry.confirmed_at, entry.updated_at, entry.entry_id),
        reverse=True,
    )
    next_index = 1
    for entry in active:
        if entry.entry_id in known_ids:
            continue
        while f"E{next_index}" in existing:
            next_index += 1
        existing[f"E{next_index}"] = entry.entry_id
        known_ids.add(entry.entry_id)
        next_index += 1
    target.entry_ref_map = existing
    return existing


def _advance_virtual_target(target: Any, mutation: dict[str, Any], now: Any) -> None:
    snapshot = getattr(target, "virtual_snapshot", None) or target.snapshot
    applied_at = now if isinstance(now, datetime) else datetime.now(UTC)
    result = apply_profile_command(snapshot, mutation["command"], now=applied_at)
    target.virtual_snapshot = result.snapshot
    target.virtual_profile_version = result.snapshot.version
    target.entry_ref_map = dict(getattr(target, "entry_ref_map", None) or {})
    _ensure_entry_map(target, result.snapshot)
    target.virtual_entry_versions = {
        entry.entry_id: (entry.version, entry.status) for entry in result.snapshot.entries
    }


def _result(mutation: dict[str, Any], *, idempotent: bool) -> dict[str, Any]:
    labels = {
        "ADD": "新增人物认识",
        "REVISE": "人物认识修正",
        "WITHDRAW": "人物认识撤回",
    }
    operation = str(mutation["operation"])
    state = "此前已暂存" if idempotent else "已暂存"
    return {
        "content": (
            f"{labels.get(operation, operation)}{state}，将在本次行动最终成功提交时一并生效。"
        ),
        "profile_entry_ref": mutation["entry_ref"],
    }


__all__ = [
    "MAX_PROFILE_MUTATIONS_PER_RUN",
    "forget_player_profile",
    "recall_player_profile",
    "remember_player_profile",
    "revise_player_profile",
    "update_player_profile",
    "view_player_profile",
]

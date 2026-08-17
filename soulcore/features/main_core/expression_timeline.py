"""Strict validation for Main Core expression timelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_EXPRESSION_ITEM_DELAY_SECONDS = 120
MAX_EXPRESSION_BATCH_DELAY_SECONDS = 300
MAX_RETRACTION_STEPS = 6
MAX_MENTION_MEMBERS = 10
VISIBLE_EXPRESSION_KINDS = {"TEXT", "IMAGE", "STICKER", "FILE"}
VOICE_PRESENTATION = "VOICE"
_VOICE_METADATA_FIELDS = ("presentation", "as_voice", "voice_parse_audit")
_SCENE_NARRATION_FIELDS = ("scene_narration_before", "scene_narration_after")


def normalize_expression_steps(
    expression_steps: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], str]:
    """Validate only the current expression timeline contract."""

    if not isinstance(expression_steps, list):
        return [], "error: 发送时间线格式无效"
    return _normalize_step_sequence(
        expression_steps,
        strict=True,
    )


def _normalize_step_sequence(
    steps: list[dict[str, Any]],
    *,
    strict: bool,
) -> tuple[list[dict[str, Any]], str]:
    normalized: list[dict[str, Any]] = []
    visible_steps: list[dict[str, Any]] = []
    retract_count = 0
    retraction_targets: set[tuple[str, str | int]] = set()
    total_delay = 0
    saw_visible = False
    for step_index, raw in enumerate(steps):
        item, error = _normalize_expression_step(raw, step_index, strict=strict)
        if error:
            return [], error
        total_delay += item["delay_after_previous_seconds"]
        if total_delay > MAX_EXPRESSION_BATCH_DELAY_SECONDS:
            return [], "error: 整组发送的延迟总时长不能超过 300 秒"
        if item["kind"] == "RETRACT":
            retract_count += 1
            if retract_count > MAX_RETRACTION_STEPS:
                return [], "error: 本轮最多撤回六条消息"
            if "target_message_ref" in item and saw_visible:
                return [], ("error: 撤回已有消息必须放在所有新发送内容之前")
            target_ordinal = item.get("target_output_ordinal")
            target_identity: tuple[str, str | int]
            if target_ordinal is not None:
                target_identity = ("output", int(target_ordinal))
            else:
                target_identity = ("message", str(item["target_message_ref"]))
            if target_identity in retraction_targets:
                return [], f"error: 第 {step_index + 1} 个撤回指令重复指向同一条消息"
            retraction_targets.add(target_identity)
            if target_ordinal is not None and target_ordinal > len(visible_steps):
                return [], (f"error: 第 {step_index + 1} 个撤回指令必须指向本轮更早的发送内容")
        else:
            visible_steps.append(item)
            saw_visible = True
        normalized.append(item)
    return normalized, ""


def _normalize_expression_step(
    raw: Any, step_index: int, *, strict: bool
) -> tuple[dict[str, Any], str]:
    if not isinstance(raw, dict):
        return {}, f"error: 第 {step_index + 1} 条发送内容格式无效"
    kind = str(raw.get("kind") or "").strip().upper()
    if kind == "RETRACT":
        return _normalize_retract_step(raw, step_index)
    if kind not in VISIBLE_EXPRESSION_KINDS:
        return {}, (f"error: 第 {step_index + 1} 条发送内容类型无效")
    label = "expression step"
    delay, error = _expression_delay(raw, step_index, label=label)
    if error:
        return {}, error
    can_be_interrupted, error = _expression_interruptibility(raw, kind, step_index, strict=strict)
    if error:
        return {}, error
    item, error = _expression_content(raw, step_index, kind, delay, can_be_interrupted, label=label)
    if error:
        return {}, error
    item.update(_normalize_voice_metadata(raw, kind))
    narration, error = _normalize_scene_narration_metadata(raw, step_index)
    if error:
        return {}, error
    item.update(narration)
    addressing, error = _normalize_visible_addressing(raw, step_index)
    if error:
        return {}, error
    item.update(addressing)
    memo = str(raw.get("memo") or "").strip()
    if memo:
        item["memo"] = memo
    return item, ""


def _normalize_retract_step(raw: dict[str, Any], step_index: int) -> tuple[dict[str, Any], str]:
    if str(raw.get("memo") or "").strip():
        return {}, f"error: 第 {step_index + 1} 个撤回指令不能携带留话"
    delay, error = _expression_delay(raw, step_index, label="RETRACT step")
    if error:
        return {}, error
    message_ref = str(raw.get("target_message_ref") or "").strip()
    target_ordinal = raw.get("target_output_ordinal")
    has_ordinal = target_ordinal is not None
    if bool(message_ref) == bool(has_ordinal):
        return {}, (f"error: 第 {step_index + 1} 个撤回指令必须且只能指定一个目标")
    item: dict[str, Any] = {
        "kind": "RETRACT",
        "delay_after_previous_seconds": delay,
    }
    if message_ref:
        item["target_message_ref"] = message_ref
        return item, ""
    if isinstance(target_ordinal, bool) or not isinstance(target_ordinal, int):
        return {}, (f"error: 第 {step_index + 1} 个撤回指令的本轮消息序号必须是正整数")
    if target_ordinal < 1:
        return {}, (f"error: 第 {step_index + 1} 个撤回指令的本轮消息序号必须从 1 开始")
    item["target_output_ordinal"] = target_ordinal
    return item, ""


def _normalize_visible_addressing(
    raw: dict[str, Any], step_index: int
) -> tuple[dict[str, Any], str]:
    result: dict[str, Any] = {}
    if "reply_to_message_ref" in raw:
        reply_ref = str(raw.get("reply_to_message_ref") or "").strip()
        if not reply_ref:
            return {}, (f"error: 第 {step_index + 1} 条消息的回复目标不能为空")
        result["reply_to_message_ref"] = reply_ref
    if "mention_member_refs" not in raw:
        return result, ""
    raw_mentions = raw.get("mention_member_refs")
    if not isinstance(raw_mentions, list):
        return {}, f"error: 第 {step_index + 1} 条消息的提及对象格式无效"
    mentions: list[str] = []
    for raw_ref in raw_mentions:
        if not isinstance(raw_ref, str) or not raw_ref.strip():
            return {}, (f"error: 第 {step_index + 1} 条消息包含空的提及对象")
        ref = raw_ref.strip()
        if ref not in mentions:
            mentions.append(ref)
    if len(mentions) > MAX_MENTION_MEMBERS:
        return {}, (f"error: 第 {step_index + 1} 条消息最多提及十个人")
    if mentions:
        result["mention_member_refs"] = mentions
    return result, ""


def _expression_delay(raw: dict[str, Any], ordinal: int, *, label: str) -> tuple[int, str]:
    delay = raw.get("delay_after_previous_seconds", 0)
    if isinstance(delay, bool) or not isinstance(delay, int):
        return 0, f"error: 第 {ordinal + 1} 条内容的延迟必须是整数"
    if delay < 0 or delay > MAX_EXPRESSION_ITEM_DELAY_SECONDS:
        return 0, f"error: 第 {ordinal + 1} 条内容的延迟必须在 0 到 120 秒之间"
    return delay, ""


def _expression_interruptibility(
    raw: dict[str, Any], kind: str, ordinal: int, *, strict: bool
) -> tuple[bool, str]:
    if strict and "can_be_interrupted" not in raw:
        return False, (f"error: 第 {ordinal + 1} 条内容必须明确填写是否允许被打断")
    can_be_interrupted = _interruptibility(raw, kind)
    if can_be_interrupted is None:
        return False, f"error: 第 {ordinal + 1} 条内容的是否允许被打断格式无效"
    return can_be_interrupted, ""


def _expression_content(
    raw: dict[str, Any],
    ordinal: int,
    kind: str,
    delay: int,
    can_be_interrupted: bool,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    text = str(raw.get("text") or "").strip()
    asset_ref = str(raw.get("asset_ref_id") or "").strip()
    if kind == "TEXT":
        if not text:
            return {}, f"error: 第 {ordinal + 1} 条文字内容不能为空"
        if asset_ref:
            return {}, f"error: 第 {ordinal + 1} 条文字不能同时携带资源"
        return {
            "kind": kind,
            "text": text,
            "delay_after_previous_seconds": delay,
            "can_be_interrupted": can_be_interrupted,
        }, ""
    if not asset_ref:
        return {}, f"error: 第 {ordinal + 1} 条媒体或文件内容必须填写短引用"
    if text:
        return {}, f"error: 第 {ordinal + 1} 条媒体或文件内容不能同时填写文字"
    return {
        "kind": kind,
        "asset_ref_id": asset_ref,
        "delay_after_previous_seconds": delay,
        "can_be_interrupted": can_be_interrupted,
    }, ""


def _interruptibility(raw: dict[str, Any], kind: str) -> bool | None:
    if "can_be_interrupted" not in raw:
        return kind != "FILE"
    value = raw["can_be_interrupted"]
    return value if isinstance(value, bool) else None


def _normalize_voice_metadata(raw: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind != "TEXT":
        return {}
    presentation = str(raw.get("presentation") or "").strip().upper()
    as_voice = False
    generated_audit: dict[str, str] | None = None
    if "as_voice" in raw:
        as_voice, generated_audit = _relaxed_voice_boolean(raw.get("as_voice"))
    if presentation and presentation not in {"TEXT", VOICE_PRESENTATION}:
        generated_audit = generated_audit or {
            "status": "FALLBACK_TO_TEXT",
            "reason": "UNRECOGNIZED_PRESENTATION",
            "raw_value": presentation[:80],
        }
    elif presentation == "TEXT" and as_voice:
        generated_audit = generated_audit or {
            "status": "FALLBACK_TO_TEXT",
            "reason": "PRESENTATION_BOOLEAN_MISMATCH",
            "raw_value": "TEXT/as_voice=true",
        }
    result: dict[str, Any] = {}
    if presentation == VOICE_PRESENTATION or (not presentation and as_voice):
        result["presentation"] = VOICE_PRESENTATION
        result["as_voice"] = True
    audit = _bounded_voice_parse_audit(raw.get("voice_parse_audit"))
    if audit is None:
        audit = generated_audit
    if audit is not None:
        result["voice_parse_audit"] = audit
    return result


def _relaxed_voice_boolean(value: Any) -> tuple[bool, dict[str, str] | None]:
    if isinstance(value, bool):
        return value, None
    text = str(value or "").strip()
    normalized = text.casefold()
    if normalized in {"1", "true", "yes", "y", "是", "允许", "可"}:
        return True, None
    if not normalized or normalized in {"0", "false", "no", "n", "否", "不允许", "不可"}:
        return False, None
    return False, {
        "status": "FALLBACK_TO_TEXT",
        "reason": "UNRECOGNIZED_BOOLEAN",
        "raw_value": text[:80],
    }


def _bounded_voice_parse_audit(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    result = {
        "status": str(value.get("status") or "FALLBACK_TO_TEXT")[:40],
        "reason": str(value.get("reason") or "UNRECOGNIZED_BOOLEAN")[:80],
        "raw_value": str(value.get("raw_value") or "")[:80],
    }
    return result


def _normalize_scene_narration_metadata(
    raw: Mapping[str, Any], step_index: int
) -> tuple[dict[str, list[str]], str]:
    result: dict[str, list[str]] = {}
    for field in _SCENE_NARRATION_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        values = _scene_narration_values(value)
        if values is None:
            return {}, f"error: 第 {step_index + 1} 条消息的旁白格式无效"
        if values:
            result[field] = values
    return result, ""


def _scene_narration_values(value: Any) -> list[str] | None:
    if isinstance(value, str):
        raw_values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = value
    else:
        return None
    return [str(item or "").strip() for item in raw_values if str(item or "").strip()]


def has_scene_narration_metadata(steps: Sequence[Mapping[str, Any]]) -> bool:
    for step in steps:
        for field in _SCENE_NARRATION_FIELDS:
            values = _scene_narration_values(step.get(field))
            if values:
                return True
    return False


def has_voice_expression_metadata(steps: Sequence[Mapping[str, Any]]) -> bool:
    """Whether expression ordinals carry a real voice presentation marker."""

    return any(
        str(step.get("presentation") or "").strip().upper() == VOICE_PRESENTATION
        or (not str(step.get("presentation") or "").strip() and step.get("as_voice") is True)
        for step in steps
    )


def restore_unbound_voice_parse_audit(
    original_steps: Sequence[Mapping[str, Any]],
    replacement_steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep a text fallback warning without binding ordinary text bubble topology."""

    restored = [dict(step) for step in replacement_steps]
    audits = [
        audit
        for step in original_steps
        if (audit := _bounded_voice_parse_audit(step.get("voice_parse_audit"))) is not None
    ]
    if not audits:
        return restored
    combined = audits[0]
    if len(audits) > 1:
        raw_values = [audit["raw_value"] for audit in audits if audit["raw_value"]]
        combined = {
            "status": "FALLBACK_TO_TEXT",
            "reason": "MULTIPLE_UNRECOGNIZED_BOOLEAN_VALUES",
            "raw_value": " | ".join(raw_values)[:80],
        }
    for step in restored:
        if str(step.get("kind") or "").strip().upper() == "TEXT":
            step["voice_parse_audit"] = combined
            break
    return restored


def restore_expression_voice_metadata(
    original_step: Mapping[str, Any], replacement_step: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore model-external voice metadata from the original stable ordinal."""

    replacement = dict(replacement_step)
    for field in _VOICE_METADATA_FIELDS:
        replacement.pop(field, None)
    presentation = str(original_step.get("presentation") or "").strip().upper()
    if presentation == VOICE_PRESENTATION or (
        not presentation and original_step.get("as_voice") is True
    ):
        replacement["presentation"] = VOICE_PRESENTATION
        replacement["as_voice"] = True
    audit = _bounded_voice_parse_audit(original_step.get("voice_parse_audit"))
    if audit is not None:
        replacement["voice_parse_audit"] = audit
    return replacement


def restore_expression_scene_narration(
    original_step: Mapping[str, Any], replacement_step: Mapping[str, Any]
) -> dict[str, Any]:
    replacement = dict(replacement_step)
    for field in _SCENE_NARRATION_FIELDS:
        replacement.pop(field, None)
        values = _scene_narration_values(original_step.get(field)) or []
        if values:
            replacement[field] = values
    return replacement


__all__ = [
    "VOICE_PRESENTATION",
    "has_scene_narration_metadata",
    "has_voice_expression_metadata",
    "normalize_expression_steps",
    "restore_expression_scene_narration",
    "restore_expression_voice_metadata",
    "restore_unbound_voice_parse_audit",
]

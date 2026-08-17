"""Bounded, model-safe transient context for ResponsePolish."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import unescape
from typing import Any

from ...shared.prompt_document import xml_text
from ...shared.token_meter import ConservativeTokenMeter
from ..identity import encode_identity_template_for_model, project_identity_text_for_model

RESPONSE_POLISH_LOCAL_CONTEXT_TOKENS = 2_048
RESPONSE_POLISH_INTERNAL_CONTEXT_TOKENS = 2_048


def fit_text_edges(
    value: str,
    token_limit: int,
    meter: ConservativeTokenMeter,
    *,
    xml_escaped: bool = False,
) -> str:
    text = str(value or "").strip()
    limit = max(0, int(token_limit))
    count = lambda candidate: meter.count_text(  # noqa: E731
        xml_text(candidate) if xml_escaped else candidate
    )
    if not text or limit <= 0:
        return ""
    if count(text) <= limit:
        return text
    marker = "\n…\n"
    if count(marker) > limit:
        return ""
    low, high = 0, len(text)
    best = marker.strip()
    while low <= high:
        kept = (low + high) // 2
        prefix_chars = (kept + 1) // 2
        suffix_chars = kept // 2
        prefix = text[:prefix_chars].rstrip()
        suffix = text[len(text) - suffix_chars :].lstrip() if suffix_chars else ""
        candidate = marker.join(part for part in (prefix, suffix) if part)
        if count(candidate) <= limit:
            best = candidate
            low = kept + 1
        else:
            high = kept - 1
    return best


def _project_internal_source(
    value: str,
    identity_context: Any | None,
    identity_catalog: Any | None,
) -> str:
    text = str(value or "").strip()
    if not text or identity_context is None or identity_catalog is None:
        return text
    return project_identity_text_for_model(
        text,
        identity_catalog,
        scope=str(identity_context.scope),
    )


def _render_internal_memos(
    expressions: Sequence[Mapping[str, Any]],
    *,
    identity_catalog: Any | None,
) -> str:
    rendered: list[str] = []
    for ordinal, item in enumerate(expressions, start=1):
        memo = str(item.get("memo") or "").strip()
        if not memo:
            continue
        if identity_catalog is not None:
            memo = encode_identity_template_for_model(memo, identity_catalog)
        rendered.append(f"第{ordinal}条消息发送后的留话：\n{memo}")
    return "\n\n".join(rendered)


def _fit_internal_materials(
    working_text: str,
    memos: str,
    *,
    meter: ConservativeTokenMeter,
) -> tuple[str, str, bool]:
    work = str(working_text or "").strip()
    memo = str(memos or "").strip()
    if not work and not memo:
        return "", "", False

    total = RESPONSE_POLISH_INTERNAL_CONTEXT_TOKENS
    if work and memo:
        work_limit = total // 2
        memo_limit = total - work_limit
        work_cost = meter.count_text(xml_text(work))
        memo_cost = meter.count_text(xml_text(memo))
        work_spare = max(0, work_limit - work_cost)
        memo_spare = max(0, memo_limit - memo_cost)
        work_limit += memo_spare
        memo_limit += work_spare
    elif work:
        work_limit, memo_limit = total, 0
    else:
        work_limit, memo_limit = 0, total

    fitted_work = fit_text_edges(work, work_limit, meter, xml_escaped=True)
    fitted_memo = fit_text_edges(memo, memo_limit, meter, xml_escaped=True)
    return fitted_work, fitted_memo, fitted_work != work or fitted_memo != memo


def project_internal_materials(
    working_text: str,
    expressions: Sequence[Mapping[str, Any]],
    identity_context: Any | None,
    identity_catalog: Any | None,
    *,
    meter: ConservativeTokenMeter,
) -> tuple[str, str, bool]:
    projected_working_text = _project_internal_source(
        working_text,
        identity_context,
        identity_catalog,
    )
    projected_memos = _render_internal_memos(
        expressions,
        identity_catalog=identity_catalog,
    )
    return _fit_internal_materials(
        projected_working_text,
        projected_memos,
        meter=meter,
    )


def internal_copy_reference(document: str) -> str:
    """Recover only the projected internal material that actually reached the model."""

    values: list[str] = []
    source = str(document or "")
    for label in ("原稿形成时的相关考虑", "发送后连续性留话"):
        opening = f"<{label}>\n"
        closing = f"\n</{label}>"
        _, found, remainder = source.partition(opening)
        if not found:
            continue
        content, found, _ = remainder.partition(closing)
        if found and content.strip():
            values.append(unescape(content.strip()))
    return "\n".join(values)


def _fit_recent_dialogue(
    values: Sequence[str],
    *,
    token_limit: int,
    meter: ConservativeTokenMeter,
) -> tuple[list[str], int]:
    lines = [str(value).strip() for value in values if str(value).strip()]
    selected: list[str] = []
    used = 0
    newline_tokens = meter.count_text("\n")
    for line in reversed(lines):
        separator = newline_tokens if selected else 0
        remaining = max(0, int(token_limit)) - used - separator
        if remaining <= 0:
            break
        cost = meter.count_text(line)
        if cost <= remaining:
            selected.insert(0, line)
            used += separator + cost
            continue
        if not selected:
            fitted = fit_text_edges(line, remaining, meter)
            if fitted:
                selected.insert(0, fitted)
        break
    return selected, len(lines) - len(selected)


def project_polish_conversation(
    conversation: Any,
    request: Any,
    identity_context: Any | None,
    identity_catalog: Any | None,
    *,
    meter: ConservativeTokenMeter,
) -> tuple[list[str], str, int, bool]:
    current_lines = list(conversation.current_lines)
    dialogue = list(conversation.recent_lines)
    if identity_context is not None and identity_catalog is not None:
        scope = str(identity_context.scope)

        def project(value: str) -> str:
            return project_identity_text_for_model(value, identity_catalog, scope=scope)

        current_lines = [project(value) for value in current_lines]
        dialogue = [project(value) for value in dialogue]
    current = "\n".join(current_lines)
    state_gate = str(request.metadata.get("state_gate_expression_context") or "").strip()
    if state_gate and identity_context is not None and identity_catalog is not None:
        state_gate = project_identity_text_for_model(
            state_gate,
            identity_catalog,
            scope=str(identity_context.scope),
        )
    raw_current = "\n".join(part for part in (current, xml_text(state_gate[:1000])) if part)
    fitted_current = fit_text_edges(
        raw_current,
        RESPONSE_POLISH_LOCAL_CONTEXT_TOKENS,
        meter,
    )
    remaining = max(
        0,
        RESPONSE_POLISH_LOCAL_CONTEXT_TOKENS - meter.count_text(fitted_current),
    )
    fitted_dialogue, dropped = _fit_recent_dialogue(
        dialogue,
        token_limit=remaining,
        meter=meter,
    )
    return fitted_dialogue, fitted_current, dropped, fitted_current != raw_current


__all__ = [
    "RESPONSE_POLISH_INTERNAL_CONTEXT_TOKENS",
    "RESPONSE_POLISH_LOCAL_CONTEXT_TOKENS",
    "fit_text_edges",
    "internal_copy_reference",
    "project_internal_materials",
    "project_polish_conversation",
]

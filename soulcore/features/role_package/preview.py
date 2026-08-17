"""Player-facing, grouped import preview without database identities or raw JSON."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..character_model import model_to_payload
from .domain import ImportState, ParsedRolePackage, RoleDatabaseSnapshot
from .model_patch import PORTRAIT_SCOPES

_FIELD_LABELS = {
    "identity.name": "名字",
    "identity.aliases": "也会回应的称呼",
    "identity.overview": "角色简介",
    "identity.facts": "重要事实",
    "personality.traits_and_values": "性格与看重的事",
    "personality.thinking_and_behavior": "通常怎样想、怎样做",
    "personality.habits_and_emotions": "习惯与情绪",
    "social.interaction_style": "与人相处的方式",
    "social.boundaries": "相处时坚持的分寸",
    "preferences.likes_and_interests": "喜欢与兴趣",
    "preferences.dislikes": "不喜欢的事",
    "language.speaking_style": "说话的感觉",
    "language.messaging_habits": "发消息的习惯",
    "language.address_habits": "怎样称呼别人",
    "dialogue_reference": "对话参考",
    "visual.appearance": "外貌",
    "visual.clothing": "常见穿着",
    "visual.visual_boundaries": "形象上不会改变的事",
    "capabilities.abilities": "能力",
    "capabilities.knowledge_scope": "知道和熟悉的事",
    "capabilities.limitations": "做不到的事",
    "main_core_modes.self_initiated": "主动联系方式",
    "main_core_styles.relationship_context": "关系与通信背景",
    "main_core_styles.speaking_style": "消息表达方式",
    "main_core_styles.sticker_style": "表情表达方式",
    "main_core_styles.thinking_style": "思考与相处方式",
    "main_core_styles.content_style": "聊天内容倾向",
    "main_core_styles.conversation_content": "自定义聊天内容规则",
    "response_polish.writing_correction": "表达修正规则",
    "story_styles.involvement": "故事参与方式",
    "story_styles.stance": "故事立场",
    "background_creation.world_change": "世界变化方式",
    "background_creation.story_boundary": "故事扩展边界",
    "background_creation.imagination": "想象方式",
    "background_creation.temperature": "创作气质",
    "world_brief": "世界整体印象",
    "world_rules": "始终成立的规则",
    "life_direction": "人生方向",
    "world_texture": "生活气质",
    "expansion_policy": "未写明内容的扩展方式",
}
_REQUIRED_PROMPTS = {
    "main_core_styles.relationship_context",
    "background_creation.story_boundary",
}


def build_import_preview(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> dict[str, Any]:
    groups = [
        _character_group(package, snapshot, state),
        _prompt_group(package, snapshot, state),
        _trigger_group(package, snapshot, state),
        _world_group(package, snapshot, state),
        _portrait_group(package, snapshot, state),
    ]
    changed_items = sum(
        1 for group in groups for item in group["items"] if item["action"] not in {"keep"}
    )
    warnings = [
        item["summary"]
        for group in groups
        for item in group["items"]
        if item["action"] == "restore_default"
    ]
    return {
        "package_title": package.title,
        "target_role_name": snapshot.title,
        "has_changes": state.changed,
        "changed_item_count": changed_items,
        "summary": (
            f"将修改 {changed_items} 项内容。未出现在角色包中的内容会保留。"
            if state.changed
            else "角色包内容与当前角色一致，不需要写入。"
        ),
        "groups": groups,
        "warnings": warnings,
        "preserve_rule": "角色包未包含的字段、世界资料和立绘都会保留当前值。",
    }


def _character_group(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> dict[str, Any]:
    patch = _character_patch(package)
    before = model_to_payload(snapshot.character)
    after = model_to_payload(state.character)
    items: list[dict[str, Any]] = []
    for section, raw in patch.items():
        if section in {"custom_prompts", "trigger_rules"}:
            continue
        if section == "dialogue_reference":
            items.append(_value_item("dialogue_reference", raw, before[section], after[section]))
            continue
        if not isinstance(raw, Mapping):
            continue
        for field, incoming in raw.items():
            path = f"{section}.{field}"
            items.append(_value_item(path, incoming, before[section][field], after[section][field]))
    if not items:
        items.append(_keep_item("角色资料", "包内未包含角色资料，保留当前内容。"))
    return _group("character", "角色资料", items)


def _prompt_group(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> dict[str, Any]:
    patch = _character_patch(package).get("custom_prompts")
    before = model_to_payload(snapshot.character)["custom_prompts"]
    after = model_to_payload(state.character)["custom_prompts"]
    items: list[dict[str, Any]] = []
    if isinstance(patch, Mapping):
        for group, raw_fields in patch.items():
            if not isinstance(raw_fields, Mapping):
                continue
            for field, incoming in raw_fields.items():
                path = f"{group}.{field}"
                if incoming == "" and path in _REQUIRED_PROMPTS:
                    items.append(
                        {
                            "label": _FIELD_LABELS.get(path, path),
                            "action": "restore_default",
                            "action_label": "恢复默认",
                            "summary": (
                                f"“{_FIELD_LABELS.get(path, path)}”是必需规则，"
                                "空值会恢复当前 SoulCore 系统默认。"
                            ),
                            "before": _summary(before[group][field]),
                            "after": _summary(after[group][field]),
                            "detail": str(after[group][field]),
                            "expandable": True,
                        }
                    )
                else:
                    items.append(
                        _value_item(
                            path,
                            incoming,
                            before[group][field],
                            after[group][field],
                            expandable=True,
                        )
                    )
    if not items:
        items.append(_keep_item("自定义行为", "包内未包含自定义行为，保留当前规则。"))
    return _group("prompts", "自定义行为", items)


def _trigger_group(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> dict[str, Any]:
    patch = _character_patch(package)
    if "trigger_rules" not in patch:
        return _group(
            "triggers",
            "触发规则",
            [_keep_item("触发规则", "包内未包含触发规则，保留当前规则。")],
        )
    incoming = patch["trigger_rules"]
    before = model_to_payload(snapshot.character)["trigger_rules"]
    after = model_to_payload(state.character)["trigger_rules"]
    changed = before != after
    empty = not incoming
    return _group(
        "triggers",
        "触发规则",
        [
            {
                "label": "全部触发规则",
                "action": "clear" if changed and empty else ("replace" if changed else "keep"),
                "action_label": "清除"
                if changed and empty
                else ("整体替换" if changed else "保留"),
                "summary": (
                    "将清除当前全部触发规则。"
                    if changed and empty
                    else (
                        f"将用角色包中的 {len(after)} 组规则整体替换当前规则。"
                        if changed
                        else "角色包中的触发规则与当前内容一致。"
                    )
                ),
                "before": f"{len(before)} 组",
                "after": f"{len(after)} 组",
                "detail": _trigger_text(after),
                "expandable": bool(after),
            }
        ],
    )


def _world_group(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> dict[str, Any]:
    root = package.role.get("world")
    patch = dict(root) if isinstance(root, Mapping) else {}
    items = _world_definition_items(patch, snapshot, state)
    for key, label in (("lore", "世界资料"), ("boundaries", "创作边界")):
        if key in patch:
            items.append(_world_collection_item(key, label, snapshot, state))
    if not items:
        items.append(_keep_item("世界资料", "包内未包含世界内容，保留当前世界。"))
    return _group("world", "世界资料", items)


def _world_definition_items(
    patch: Mapping[str, Any],
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> list[dict[str, Any]]:
    definition = patch.get("definition")
    if not isinstance(definition, Mapping):
        return []
    return [
        _value_item(
            str(field),
            incoming,
            snapshot.world_definition[str(field)],
            state.world_definition[str(field)],
        )
        for field, incoming in definition.items()
    ]


def _world_collection_item(
    key: str,
    label: str,
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> dict[str, Any]:
    before = snapshot.lore if key == "lore" else snapshot.boundaries
    after = state.lore if key == "lore" else state.boundaries
    changed = _records(before) != _records(after)
    cleared = changed and not after
    return {
        "label": label,
        "action": "clear" if cleared else ("replace" if changed else "keep"),
        "action_label": "清除" if cleared else ("整体替换" if changed else "保留"),
        "summary": (
            f"将清除当前全部{label}。"
            if cleared
            else (
                f"将用角色包中的 {len(after)} 条内容整体替换当前{label}。"
                if changed
                else f"角色包中的{label}与当前内容一致。"
            )
        ),
        "before": f"{len(before)} 条",
        "after": f"{len(after)} 条",
        "detail": _world_records_text(after, boundary=key == "boundaries"),
        "expandable": bool(after),
    }


def _portrait_group(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
    state: ImportState,
) -> dict[str, Any]:
    del package
    items: list[dict[str, Any]] = []
    for scope in PORTRAIT_SCOPES:
        label = "私聊立绘" if scope == "private" else "群聊立绘"
        action = state.portrait_actions[scope]["action"]
        changed = state.portrait_changed[scope]
        current = snapshot.portraits.get(scope)
        if action == "keep":
            items.append(_keep_item(label, f"包内未包含{label}，保留当前图片。"))
        elif action == "clear":
            items.append(
                {
                    "label": label,
                    "action": "clear" if changed else "keep",
                    "action_label": "删除" if changed else "保留",
                    "summary": f"将删除当前{label}。"
                    if changed
                    else f"当前没有{label}，无需删除。",
                    "before": "已有图片" if current else "没有图片",
                    "after": "没有图片",
                    "detail": "",
                    "expandable": False,
                }
            )
        else:
            items.append(
                {
                    "label": label,
                    "action": "replace" if changed else "keep",
                    "action_label": "替换" if changed else "保留",
                    "summary": f"将使用角色包中的图片替换{label}。"
                    if changed
                    else f"{label}图片内容与当前一致。",
                    "before": "已有图片" if current else "没有图片",
                    "after": "角色包图片",
                    "detail": "",
                    "expandable": False,
                }
            )
    return _group("portraits", "立绘", items)


def _value_item(
    path: str,
    incoming: Any,
    before: Any,
    after: Any,
    *,
    expandable: bool = False,
) -> dict[str, Any]:
    changed = before != after
    cleared = incoming == "" or incoming == []
    action = "clear" if changed and cleared else ("modify" if changed else "keep")
    action_label = "清除" if action == "clear" else ("修改" if action == "modify" else "保留")
    label = _FIELD_LABELS.get(path, path)
    return {
        "label": label,
        "action": action,
        "action_label": action_label,
        "summary": (
            f"将清除{label}。"
            if action == "clear"
            else (f"将修改{label}。" if action == "modify" else f"{label}与当前内容一致。")
        ),
        "before": _summary(before),
        "after": _summary(after),
        "detail": _detail(after),
        "expandable": expandable or len(_detail(after)) > 120,
    }


def _keep_item(label: str, summary: str) -> dict[str, Any]:
    return {
        "label": label,
        "action": "keep",
        "action_label": "保留",
        "summary": summary,
        "before": "当前内容",
        "after": "保持不变",
        "detail": "",
        "expandable": False,
    }


def _group(key: str, title: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"key": key, "title": title, "items": items}


def _character_patch(package: ParsedRolePackage) -> dict[str, Any]:
    value = package.role.get("character")
    return dict(value) if isinstance(value, Mapping) else {}


def _summary(value: Any) -> str:
    if value == "" or value == [] or value == ():
        return "空"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return f"{len(value)} 项"
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "…"


def _detail(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _trigger_text(values: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(values, start=1):
        keys = "、".join(str(key) for key in item.get("keys") or ())
        lines.append(f"{index}. Key：{keys}（回看 {item.get('lookback_turns')} 轮）")
        lines.append(str(item.get("content") or ""))
    return "\n\n".join(lines)


def _world_records_text(values: Sequence[Mapping[str, Any]], *, boundary: bool) -> str:
    lines: list[str] = []
    for item in values:
        if boundary:
            lines.append(f"{item.get('category')}：{item.get('rule_text')}")
        else:
            lines.append(f"{item.get('title')}\n{item.get('content')}")
    return "\n\n".join(lines)


def _records(values: Sequence[Mapping[str, Any]]) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        sorted(
            tuple(sorted((str(key), repr(value)) for key, value in item.items())) for item in values
        )
    )


__all__ = ["build_import_preview"]

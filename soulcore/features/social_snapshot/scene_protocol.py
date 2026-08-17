"""Model-facing, run-scoped protocol for deterministic social snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class SocialSnapshotSceneProtocolError(ValueError):
    """A precise, model-correctable scene document error."""


@dataclass(frozen=True, slots=True)
class SocialSnapshotPreset:
    label: str
    theme: str
    mode: str
    entry_sections: tuple[str, ...]
    supports_draft: bool


SOCIAL_SNAPSHOT_PRESETS: tuple[SocialSnapshotPreset, ...] = (
    SocialSnapshotPreset("QQ私聊", "mobile_chat", "private_chat", ("时间", "消息", "图片"), True),
    SocialSnapshotPreset("QQ群聊", "mobile_chat", "group_chat", ("时间", "消息", "图片"), True),
    SocialSnapshotPreset("微信私聊", "wechat", "private_chat", ("时间", "消息", "图片"), True),
    SocialSnapshotPreset("微信群聊", "wechat", "group_chat", ("时间", "消息", "图片"), True),
    SocialSnapshotPreset(
        "钉钉私聊", "dingtalk", "private_chat", ("时间", "消息", "图片", "文件"), True
    ),
    SocialSnapshotPreset(
        "钉钉群聊", "dingtalk", "group_chat", ("时间", "消息", "图片", "文件"), True
    ),
    SocialSnapshotPreset("微博动态", "weibo_feed", "feed", ("动态", "评论", "转发"), False),
    SocialSnapshotPreset("X动态", "x", "feed", ("动态", "评论", "转发"), False),
    SocialSnapshotPreset("小红书笔记", "xiaohongshu", "note", ("笔记", "评论"), True),
)
SOCIAL_SNAPSHOT_PRESET_LABELS = tuple(item.label for item in SOCIAL_SNAPSHOT_PRESETS)
_PRESET_BY_LABEL = {item.label: item for item in SOCIAL_SNAPSHOT_PRESETS}


@dataclass(frozen=True, slots=True)
class _FieldSpec:
    label: str
    required: bool = False
    media_reference: bool = False


@dataclass(frozen=True, slots=True)
class _SectionSpec:
    name: str
    fields: tuple[_FieldSpec, ...]
    entry_kind: str = ""

    @property
    def by_label(self) -> dict[str, _FieldSpec]:
        return {item.label: item for item in self.fields}


_INTERFACE_FIELDS = (
    _FieldSpec("标题", required=True),
    _FieldSpec("副标题"),
    _FieldSpec("界面时间"),
)
_PERSON = _SectionSpec(
    "人物",
    (
        _FieldSpec("标识", required=True),
        _FieldSpec("名称", required=True),
        _FieldSpec("方向"),
        _FieldSpec("头像", media_reference=True),
        _FieldSpec("徽章"),
        _FieldSpec("颜色"),
    ),
)
_QUOTE_FIELDS = (
    _FieldSpec("回复发送者"),
    _FieldSpec("回复正文"),
    _FieldSpec("回复媒体说明"),
    _FieldSpec("回复时间"),
)
_FIELD_HINTS = {
    "标题": "界面顶部文字",
    "副标题": "标题下的补充文字",
    "界面时间": "状态栏文字，例如 22:48",
    "未发送草稿": "输入框中尚未发送的文字",
    "标识": "字母或数字开头，其余可用字母、数字、_、.、:、-，最多 64 字符",
    "名称": "人物显示名",
    "方向": "左或右；省略为左",
    "头像": "一个当前可见的 I 短引用",
    "徽章": "显示名旁的短标记",
    "颜色": "六位十六进制颜色，例如 #7f8c9a",
    "内容": "时间分隔条文字",
    "发送者": "已声明的人物标识",
    "正文": "可见文字，可含冒号和竖线",
    "时间": "该内容项显示的时间文字",
    "媒体": "一个当前可见的 I 短引用",
    "图片": "一个当前可见的 I 短引用",
    "文件": "一个当前可见的 I 短引用",
    "回复发送者": "被回复内容显示的发送者名称",
    "回复正文": "被回复的文字",
    "回复媒体说明": "被回复媒体的文字说明，不是图片引用",
    "回复时间": "被回复内容显示的时间文字",
}
_ENTRY_SPECS: dict[str, _SectionSpec] = {
    "时间": _SectionSpec(
        "时间",
        (_FieldSpec("内容", required=True),),
        "timestamp",
    ),
    "消息": _SectionSpec(
        "消息",
        (
            _FieldSpec("发送者", required=True),
            _FieldSpec("正文"),
            _FieldSpec("时间"),
            _FieldSpec("媒体", media_reference=True),
            *_QUOTE_FIELDS,
        ),
        "message",
    ),
    "图片": _SectionSpec(
        "图片",
        (
            _FieldSpec("发送者", required=True),
            _FieldSpec("图片", required=True, media_reference=True),
            _FieldSpec("正文"),
            _FieldSpec("时间"),
            *_QUOTE_FIELDS,
        ),
        "image",
    ),
    "文件": _SectionSpec(
        "文件",
        (
            _FieldSpec("发送者", required=True),
            _FieldSpec("正文"),
            _FieldSpec("文件", media_reference=True),
            _FieldSpec("时间"),
            *_QUOTE_FIELDS,
        ),
        "file",
    ),
    "动态": _SectionSpec(
        "动态",
        (
            _FieldSpec("发送者", required=True),
            _FieldSpec("正文"),
            _FieldSpec("时间"),
            _FieldSpec("媒体", media_reference=True),
        ),
        "post",
    ),
    "笔记": _SectionSpec(
        "笔记",
        (
            _FieldSpec("发送者", required=True),
            _FieldSpec("正文"),
            _FieldSpec("时间"),
            _FieldSpec("媒体", media_reference=True),
        ),
        "post",
    ),
    "评论": _SectionSpec(
        "评论",
        (
            _FieldSpec("发送者", required=True),
            _FieldSpec("正文"),
            _FieldSpec("时间"),
            _FieldSpec("媒体", media_reference=True),
        ),
        "comment",
    ),
    "转发": _SectionSpec(
        "转发",
        (
            _FieldSpec("发送者", required=True),
            _FieldSpec("正文"),
            _FieldSpec("时间"),
            _FieldSpec("媒体", media_reference=True),
        ),
        "repost",
    ),
}


@dataclass(slots=True)
class _FieldValue:
    text: str
    line: int


@dataclass(slots=True)
class _Record:
    section: str
    line: int
    fields: dict[str, _FieldValue] = field(default_factory=dict)
    current_field: str = ""


def social_snapshot_preset(label: str) -> SocialSnapshotPreset:
    preset = _PRESET_BY_LABEL.get(str(label or "").strip())
    if preset is None:
        raise SocialSnapshotSceneProtocolError(
            "社交截图类型只能是：" + "、".join(SOCIAL_SNAPSHOT_PRESET_LABELS)
        )
    return preset


def render_social_snapshot_format(preset: SocialSnapshotPreset) -> str:
    """Return the complete strict scene schema for one selected interface."""

    sections = _section_specs(preset)
    field_lines = _render_schema_description(preset, sections)
    example = _format_example(preset)
    draft_limit = "；未发送草稿最多 500 字" if preset.supports_draft else ""
    ordering_rule = {
        "feed": (
            "内容区块按写下的顺序显示。评论或转发应紧跟它所对应的动态；协议没有父级字段，"
            "不要用文字伪造嵌套关系。"
        ),
        "note": (
            "内容区块按写下的顺序显示。评论应紧跟它所对应的笔记；协议没有父级字段，"
            "不要用文字伪造嵌套关系。"
        ),
    }.get(preset.mode, "内容区块按写下的顺序显示。")
    return "\n".join(
        (
            f"已选界面：{preset.label}",
            "区块标记独占一行，字段写成“字段：值”；多行值的续行缩进。可选字段没有内容时"
            "省略整行。不得自造区块、字段或 I 短引用。",
            ordering_rule,
            (
                "限制：标题最多 120 字，每项正文最多 500 字，总文本最多 8000 字"
                f"{draft_limit}；最多 12 个人物、60 个内容项、5 个内容媒体引用；"
                "头像与内容媒体合计最多 12 个不同引用。"
            ),
            "允许的区块与字段：",
            *field_lines,
            "以下示例只演示结构，不代表本轮事实，也不要求照抄内容：",
            example,
        )
    )


def parse_social_snapshot_scene(
    preset: SocialSnapshotPreset,
    text: str,
    *,
    reference_map: Mapping[str, Any],
) -> dict[str, Any]:
    records = _parse_records(preset, text)
    interface = [item for item in records if item.section == "界面"]
    people_records = [item for item in records if item.section == "人物"]
    entry_records = [item for item in records if item.section not in {"界面", "人物"}]
    if len(interface) != 1:
        raise SocialSnapshotSceneProtocolError("场景必须且只能包含一个【界面】区块")
    if not people_records:
        raise SocialSnapshotSceneProtocolError("场景至少需要一个【人物】区块")
    if not entry_records:
        raise SocialSnapshotSceneProtocolError("场景至少需要一个内容区块")

    ui_record = interface[0]
    title = _required_value(ui_record, "标题")
    draft = _optional_value(ui_record, "未发送草稿")
    people, people_ids = _compile_people(people_records, reference_map)
    items = [
        _compile_entry(record, people_ids=people_ids, reference_map=reference_map)
        for record in entry_records
    ]
    ui = {
        key: value
        for key, value in (
            ("subtitle", _optional_value(ui_record, "副标题")),
            ("clock", _optional_value(ui_record, "界面时间")),
        )
        if value
    }
    return {
        "theme": preset.theme,
        "mode": preset.mode,
        "title": title,
        "people": people,
        "items": items,
        "draft": draft,
        "ui": ui,
    }


def _section_specs(preset: SocialSnapshotPreset) -> tuple[_SectionSpec, ...]:
    interface_fields = (
        *_INTERFACE_FIELDS,
        *((_FieldSpec("未发送草稿"),) if preset.supports_draft else ()),
    )
    return (
        _SectionSpec("界面", interface_fields),
        _PERSON,
        *(_ENTRY_SPECS[name] for name in preset.entry_sections),
    )


def _parse_records(preset: SocialSnapshotPreset, text: str) -> list[_Record]:
    raw_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    allowed = {item.name: item for item in _section_specs(preset)}
    all_sections = {"界面", "人物", *_ENTRY_SPECS}
    records: list[_Record] = []
    current: _Record | None = None
    for line_number, raw in enumerate(raw_text.splitlines(), start=1):
        current = _consume_record_line(
            preset,
            raw=raw,
            line_number=line_number,
            allowed=allowed,
            all_sections=all_sections,
            records=records,
            current=current,
        )
    if not records:
        raise SocialSnapshotSceneProtocolError("场景内容为空")
    _validate_required_fields(records, allowed)
    return records


def _consume_record_line(
    preset: SocialSnapshotPreset,
    *,
    raw: str,
    line_number: int,
    allowed: Mapping[str, _SectionSpec],
    all_sections: set[str],
    records: list[_Record],
    current: _Record | None,
) -> _Record | None:
    if not raw.strip():
        return current
    stripped = raw.strip()
    section = _section_header(
        preset,
        raw=raw,
        stripped=stripped,
        line_number=line_number,
        allowed=allowed,
        all_sections=all_sections,
    )
    if section is not None:
        record = _Record(section, line_number)
        records.append(record)
        return record
    if current is None:
        raise SocialSnapshotSceneProtocolError(f"场景第{line_number}行不属于任何区块")
    if raw != raw.lstrip():
        if not current.current_field:
            raise SocialSnapshotSceneProtocolError(f"场景第{line_number}行是没有所属字段的续行")
        previous = current.fields[current.current_field]
        previous.text = previous.text + "\n" + stripped
        return current
    label, value = _field_line(raw, line_number)
    if label not in allowed[current.section].by_label:
        raise SocialSnapshotSceneProtocolError(
            f"场景第{line_number}行的【{current.section}】不支持字段“{label}”"
        )
    if label in current.fields:
        raise SocialSnapshotSceneProtocolError(f"场景第{line_number}行重复字段“{label}”")
    if not value:
        raise SocialSnapshotSceneProtocolError(
            f"场景第{line_number}行字段“{label}”不能为空；不用时请删除整行"
        )
    current.fields[label] = _FieldValue(value, line_number)
    current.current_field = label
    return current


def _validate_required_fields(
    records: Sequence[_Record], allowed: Mapping[str, _SectionSpec]
) -> None:
    for record in records:
        spec = allowed[record.section]
        missing = [
            item.label for item in spec.fields if item.required and item.label not in record.fields
        ]
        if missing:
            raise SocialSnapshotSceneProtocolError(
                f"【{record.section}】第{record.line}行开始，缺少必填字段：{'、'.join(missing)}"
            )


def _section_header(
    preset: SocialSnapshotPreset,
    *,
    raw: str,
    stripped: str,
    line_number: int,
    allowed: Mapping[str, _SectionSpec],
    all_sections: set[str],
) -> str | None:
    if not (stripped.startswith("【") and stripped.endswith("】")):
        return None
    if raw != raw.lstrip():
        raise SocialSnapshotSceneProtocolError(f"场景第{line_number}行的区块标记不能缩进")
    section = stripped[1:-1].strip()
    if section not in all_sections:
        raise SocialSnapshotSceneProtocolError(f"场景第{line_number}行包含未知区块【{section}】")
    if section not in allowed:
        raise SocialSnapshotSceneProtocolError(
            f"{preset.label}不支持【{section}】区块（第{line_number}行）"
        )
    return section


def _field_line(raw: str, line_number: int) -> tuple[str, str]:
    positions = [position for marker in ("：", ":") if (position := raw.find(marker)) >= 0]
    if not positions:
        raise SocialSnapshotSceneProtocolError(f"场景第{line_number}行必须是“字段：值”或缩进续行")
    position = min(positions)
    label = raw[:position].strip()
    if not label:
        raise SocialSnapshotSceneProtocolError(f"场景第{line_number}行缺少字段名")
    return label, raw[position + 1 :].strip()


def _compile_people(
    records: Sequence[_Record], reference_map: Mapping[str, Any]
) -> tuple[list[dict[str, str]], set[str]]:
    people: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for record in records:
        identifier = _required_value(record, "标识")
        if identifier in identifiers:
            raise SocialSnapshotSceneProtocolError(
                f"【人物】第{record.line}行使用了重复标识“{identifier}”"
            )
        identifiers.add(identifier)
        direction = _optional_value(record, "方向") or "左"
        if direction not in {"左", "右"}:
            raise SocialSnapshotSceneProtocolError(
                f"【人物】第{record.line}行的方向只能是“左”或“右”"
            )
        avatar = _optional_value(record, "头像")
        people.append(
            {
                key: value
                for key, value in (
                    ("id", identifier),
                    ("name", _required_value(record, "名称")),
                    ("side", {"左": "left", "右": "right"}[direction]),
                    (
                        "avatar",
                        _resolve_media_reference(avatar, record, "头像", reference_map)
                        if avatar
                        else "",
                    ),
                    ("badge", _optional_value(record, "徽章")),
                    ("color", _optional_value(record, "颜色")),
                )
                if value
            }
        )
    return people, identifiers


def _compile_entry(
    record: _Record,
    *,
    people_ids: set[str],
    reference_map: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _ENTRY_SPECS[record.section]
    sender = _optional_value(record, "发送者")
    if sender and sender not in people_ids:
        raise SocialSnapshotSceneProtocolError(
            f"【{record.section}】第{record.line}行引用了未知发送者“{sender}”"
        )
    media_label = (
        "图片" if record.section == "图片" else "文件" if record.section == "文件" else "媒体"
    )
    media = _optional_value(record, media_label)
    text = _optional_value(record, "内容") or _optional_value(record, "正文")
    if record.section != "时间" and not text and not media:
        raise SocialSnapshotSceneProtocolError(
            f"【{record.section}】第{record.line}行必须填写正文或媒体引用"
        )
    quote = _compile_quote(record)
    return {
        key: value
        for key, value in (
            ("k", spec.entry_kind),
            ("by", sender),
            ("text", text),
            ("time", _optional_value(record, "时间")),
            (
                "media",
                _resolve_media_reference(media, record, media_label, reference_map)
                if media
                else "",
            ),
            ("quote", quote),
        )
        if value
    }


def _compile_quote(record: _Record) -> dict[str, str] | None:
    values = {
        "sender": _optional_value(record, "回复发送者"),
        "text": _optional_value(record, "回复正文"),
        "media_label": _optional_value(record, "回复媒体说明"),
        "time": _optional_value(record, "回复时间"),
    }
    if not any(values.values()):
        return None
    if not values["sender"]:
        raise SocialSnapshotSceneProtocolError(
            f"【{record.section}】第{record.line}行填写回复内容时必须填写回复发送者"
        )
    if not values["text"] and not values["media_label"]:
        raise SocialSnapshotSceneProtocolError(
            f"【{record.section}】第{record.line}行的回复必须包含回复正文或回复媒体说明"
        )
    return {key: value for key, value in values.items() if value}


def _resolve_media_reference(
    value: str,
    record: _Record,
    field_label: str,
    reference_map: Mapping[str, Any],
) -> str:
    if not value.startswith("I") or not value[1:].isdigit():
        raise SocialSnapshotSceneProtocolError(
            f"【{record.section}】第{record.line}行字段“{field_label}”只能填写可见的 I 短引用"
        )
    internal = reference_map.get(value)
    if internal is None:
        raise SocialSnapshotSceneProtocolError(
            f"【{record.section}】第{record.line}行使用了未知图片短引用“{value}”"
        )
    return str(internal)


def _required_value(record: _Record, label: str) -> str:
    item = record.fields.get(label)
    if item is None or not item.text.strip():
        raise SocialSnapshotSceneProtocolError(
            f"【{record.section}】第{record.line}行缺少必填字段“{label}”"
        )
    return item.text.strip()


def _optional_value(record: _Record, label: str) -> str:
    item = record.fields.get(label)
    return item.text.strip() if item is not None else ""


def _render_schema_description(
    preset: SocialSnapshotPreset,
    sections: tuple[_SectionSpec, ...],
) -> list[str]:
    fixed_names = {"界面", "人物", "时间"}
    fixed = [spec for spec in sections if spec.name in fixed_names]
    content = [spec for spec in sections if spec.name not in fixed_names]
    lines: list[str] = []
    for spec in fixed:
        lines.extend(_render_section_description(spec))
    if preset.mode in {"private_chat", "group_chat"}:
        lines.extend(_render_chat_content_schema(content))
    else:
        lines.extend(_render_grouped_content_schema(content))
    return lines


def _render_chat_content_schema(specs: Sequence[_SectionSpec]) -> list[str]:
    if not specs:
        return []
    common_labels, common_fields = _common_chat_fields(specs)
    section_names = "、".join(f"【{spec.name}】" for spec in specs)
    lines = [f"- {section_names}的共用字段："]
    lines.extend(_render_field_line(field) for field in common_fields)
    if any(field.label == "回复发送者" for field in common_fields):
        lines.append(
            "  - 使用回复字段时，必须填写`回复发送者`，并至少填写`回复正文`或`回复媒体说明`。"
        )
    for spec in specs:
        unique_fields = tuple(field for field in spec.fields if field.label not in common_labels)
        lines.append(f"- 【{spec.name}】（可重复）：使用上述共用字段")
        lines.extend(_render_field_line(field) for field in unique_fields)
        note = _section_note(spec.name)
        if note:
            lines.append(note)
    return lines


def _common_chat_fields(
    specs: Sequence[_SectionSpec],
) -> tuple[set[str], tuple[_FieldSpec, ...]]:
    common_labels = {field.label for field in specs[0].fields}
    for spec in specs[1:]:
        common_labels.intersection_update(field.label for field in spec.fields)
    common_fields = tuple(field for field in specs[0].fields if field.label in common_labels)
    return common_labels, common_fields


def _render_grouped_content_schema(specs: Sequence[_SectionSpec]) -> list[str]:
    groups: dict[tuple[_FieldSpec, ...], list[_SectionSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.fields, []).append(spec)
    lines: list[str] = []
    for grouped_specs in groups.values():
        names = "、".join(f"【{spec.name}】" for spec in grouped_specs)
        repeat = "各自可重复" if len(grouped_specs) > 1 else "可重复"
        lines.append(f"- {names}（{repeat}）：")
        lines.extend(_render_field_line(field) for field in grouped_specs[0].fields)
        note = _section_note(grouped_specs[0].name)
        if note:
            lines.append(note)
    return lines


def _render_section_description(spec: _SectionSpec) -> list[str]:
    repeat = "一次" if spec.name == "界面" else "可重复"
    lines = [f"- 【{spec.name}】（{repeat}）"]
    lines.extend(_render_field_line(field_spec) for field_spec in spec.fields)
    note = _section_note(spec.name)
    if note:
        lines.append(note)
    return lines


def _render_field_line(field_spec: _FieldSpec) -> str:
    requirement = "必填" if field_spec.required else "可选"
    hint = _FIELD_HINTS.get(field_spec.label, "")
    description = f"{requirement}；{hint}" if hint else requirement
    return f"  - `{field_spec.label}`：{description}"


def _section_note(name: str) -> str:
    return {
        "消息": "  - 正文和媒体至少填写一项；媒体是随消息附带的图片。",
        "图片": "",
        "文件": "  - 正文（文件名或说明）和文件引用至少填写一项。",
        "动态": "  - 正文和媒体至少填写一项。",
        "笔记": "  - 正文和媒体至少填写一项。",
        "评论": "  - 正文和媒体至少填写一项。",
        "转发": "  - 正文和媒体至少填写一项。",
    }.get(name, "")


def _format_example(preset: SocialSnapshotPreset) -> str:
    if preset.mode in {"private_chat", "group_chat"}:
        title = "周末摸鱼群" if preset.mode == "group_chat" else "和阿青的聊天"
        lines = [
            "【界面】",
            f"标题：{title}",
            "界面时间：22:48",
            "",
            "【人物】",
            "标识：me",
            "名称：小可",
            "方向：右",
            "",
            "【人物】",
            "标识：qing",
            "名称：阿青",
            "方向：左",
            "",
            "【消息】",
            "发送者：qing",
            "正文：今晚还打不打",
            "时间：22:46",
        ]
        if preset.mode == "group_chat":
            lines.extend(
                (
                    "",
                    "【消息】",
                    "发送者：me",
                    "正文：我晚点上线",
                    "时间：22:47",
                )
            )
        if preset.theme == "dingtalk":
            lines.extend(
                (
                    "",
                    "【文件】",
                    "发送者：qing",
                    "正文：季度预算.xlsx",
                    "时间：22:47",
                )
            )
        return "\n".join(lines)
    section = "笔记" if preset.mode == "note" else "动态"
    lines = [
        "【界面】",
        f"标题：{preset.label}示例",
        "",
        "【人物】",
        "标识：me",
        "名称：小可",
        "",
        "【人物】",
        "标识：qing",
        "名称：阿青",
        "",
        f"【{section}】",
        "发送者：me",
        "正文：今天遇到一件挺有意思的事。",
        "时间：22:46",
        "",
        "【评论】",
        "发送者：qing",
        "正文：展开说说？",
        "时间：22:48",
    ]
    if preset.mode == "feed":
        lines.extend(
            (
                "",
                "【转发】",
                "发送者：qing",
                "正文：分享给也会感兴趣的人。",
                "时间：22:49",
            )
        )
    return "\n".join(lines)


__all__ = [
    "SOCIAL_SNAPSHOT_PRESETS",
    "SOCIAL_SNAPSHOT_PRESET_LABELS",
    "SocialSnapshotPreset",
    "SocialSnapshotSceneProtocolError",
    "parse_social_snapshot_scene",
    "render_social_snapshot_format",
    "social_snapshot_preset",
]

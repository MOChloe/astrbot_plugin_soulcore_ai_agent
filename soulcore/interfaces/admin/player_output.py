"""Readable player projection for accepted model output."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ...features.ai.service import parse_model_turn

_REFERENCE_PRESENTATIONS = {
    "图片": "已选择一张图片",
    "表情": "已选择一个表情",
    "文件": "已选择一个文件",
    "回复": "已选择回复一条消息",
    "提及": "已选择群成员",
    "消息": "已选择一条消息",
    "本轮消息": "已选择本轮的一条消息",
    "人物": "已选择一位群成员",
    "依据": "已选择一条依据消息",
    "原来的印象": "已选择一条已有印象",
    "印象": "已选择一条已有印象",
    "参考图片": "已选择参考图片",
    "用到的材料": "已选择相关材料",
    "哪件事": "已选择一项安排",
}
_COMMAND_TITLES = {
    "制定Plan": ("plan", "制定 Plan"),
    "发文字": ("text", "发送文字"),
    "发图片": ("image", "发送图片"),
    "发表情": ("sticker", "发送表情"),
    "发文件": ("file", "发送文件"),
    "撤回": ("retract", "撤回消息"),
}
_EXPRESSION_TITLES = {
    "TEXT": ("text", "发送文字"),
    "IMAGE": ("image", "发送图片"),
    "STICKER": ("sticker", "发送表情"),
    "FILE": ("file", "发送文件"),
    "RETRACT": ("retract", "撤回消息"),
}
_MEDIA_FIELD_VALUES = {
    "IMAGE": ("图片", "已选择一张图片"),
    "STICKER": ("表情", "已选择一个表情"),
    "FILE": ("文件", "已选择一个文件"),
}


class PlayerOutputProjector:
    """Turn accepted protocol evaluations into player-readable cards."""

    @staticmethod
    def accepted(evaluation: Mapping[str, Any]) -> bool:
        return (
            bool(evaluation.get("accepted"))
            or str(evaluation.get("validation_status") or "").upper() == "ACCEPTED"
        )

    @classmethod
    def main_core_cards(cls, evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
        source = cls._model_payload_text(evaluation)
        parsed = parse_model_turn(source) if source else None
        cards: list[dict[str, Any]] = []
        if parsed is not None and not parsed.errors and parsed.working_text:
            cards.append(
                {
                    "kind": "thought",
                    "title": "当时的想法",
                    "content": parsed.working_text,
                    "fields": [],
                }
            )
        for command in cls._commands(evaluation, parsed):
            card = cls._command_card(command)
            if card is not None:
                cards.append(card)
        return cards

    @classmethod
    def _commands(cls, evaluation: Mapping[str, Any], parsed: Any) -> list[Mapping[str, Any]]:
        commands = cls._mapping_sequence(evaluation.get("parsed_commands"))
        if commands:
            return commands
        if parsed is None or parsed.errors:
            return []
        return [
            {
                "ordinal": item.ordinal,
                "name": item.name,
                "parameters": {
                    **dict(item.parameters),
                    **({"内容": item.unlabeled_content} if item.unlabeled_content else {}),
                },
            }
            for item in parsed.commands
        ]

    @classmethod
    def _model_payload_text(cls, evaluation: Mapping[str, Any]) -> str:
        payload = str(evaluation.get("payload_text") or "")
        if payload.strip():
            return payload
        texts = [
            str(item.get("text") or "")
            for item in cls._mapping_sequence(evaluation.get("model_visible_items"))
            if str(item.get("text") or "").strip()
        ]
        return "\n".join(texts)

    @classmethod
    def _command_card(cls, command: Mapping[str, Any]) -> dict[str, Any] | None:
        name = str(command.get("name") or "").strip()
        if not name:
            return None
        kind, title = _COMMAND_TITLES.get(name, ("action", name))
        content = ""
        fields: list[dict[str, str]] = []
        for raw_label, raw_value in cls._mapping(command.get("parameters")).items():
            projected = cls._command_parameter(raw_label, raw_value)
            if projected is None:
                continue
            label, value = projected
            if label in {"内容", "正文"} and not content:
                content = value
            else:
                fields.append({"label": cls._parameter_label(label), "value": value})
        if not content and not fields:
            return None
        return {"kind": kind, "title": title, "content": content, "fields": fields}

    @classmethod
    def _command_parameter(cls, raw_label: Any, raw_value: Any) -> tuple[str, str] | None:
        label = str(raw_label or "").strip()
        if not label or not re.search(r"[\u3400-\u9fff]", label):
            return None
        value = cls._parameter_value(label, raw_value)
        return (label, value) if value else None

    @classmethod
    def expression_cards(cls, value: Any) -> list[dict[str, Any]]:
        cards = []
        for item in cls._mapping_sequence(value):
            card = cls._expression_card(item)
            if card is not None:
                cards.append(card)
        return cards

    @classmethod
    def _expression_card(cls, item: Mapping[str, Any]) -> dict[str, Any] | None:
        kind = str(item.get("kind") or "").strip().upper()
        if kind not in _EXPRESSION_TITLES:
            return None
        card_kind, title = _EXPRESSION_TITLES[kind]
        if kind == "TEXT" and cls._is_voice(item):
            title = "发送语音"
        return {
            "kind": card_kind,
            "title": title,
            "content": str(item.get("text") or "").strip() if kind == "TEXT" else "",
            "fields": cls._expression_fields(item, kind),
        }

    @classmethod
    def _expression_fields(cls, item: Mapping[str, Any], kind: str) -> list[dict[str, str]]:
        fields = [
            {
                "label": "延迟",
                "value": f"{max(0, int(item.get('delay_after_previous_seconds') or 0))} 秒",
            }
        ]
        if kind != "RETRACT":
            fields.append(
                {
                    "label": "可被打断",
                    "value": "是" if bool(item.get("can_be_interrupted", True)) else "否",
                }
            )
        prefix = cls._expression_prefix_fields(item, kind)
        return [*prefix, *fields]

    @classmethod
    def _expression_prefix_fields(cls, item: Mapping[str, Any], kind: str) -> list[dict[str, str]]:
        if kind == "TEXT":
            return cls._text_prefix_fields(item)
        if kind in _MEDIA_FIELD_VALUES:
            label, value = _MEDIA_FIELD_VALUES[kind]
            return [{"label": label, "value": value}]
        if kind == "RETRACT" and cls._has_retract_target(item):
            return [{"label": "目标", "value": "已选择撤回一条消息"}]
        return []

    @staticmethod
    def _text_prefix_fields(item: Mapping[str, Any]) -> list[dict[str, str]]:
        fields = []
        if str(item.get("reply_to_message_ref") or "").strip():
            fields.append({"label": "回复", "value": "已选择回复一条消息"})
        mentions = [ref for ref in item.get("mention_member_refs") or () if str(ref or "").strip()]
        if mentions:
            fields.insert(0, {"label": "提及", "value": f"已选择 {len(mentions)} 位群成员"})
        return fields

    @staticmethod
    def _is_voice(item: Mapping[str, Any]) -> bool:
        return (
            bool(item.get("as_voice"))
            or str(item.get("presentation") or "").strip().upper() == "VOICE"
        )

    @staticmethod
    def _has_retract_target(item: Mapping[str, Any]) -> bool:
        return (
            bool(str(item.get("target_message_ref") or "").strip())
            or item.get("target_output_ordinal") is not None
        )

    @classmethod
    def _parameter_value(cls, label: str, value: Any) -> str:
        if label in _REFERENCE_PRESENTATIONS:
            return _REFERENCE_PRESENTATIONS[label]
        if label == "链接":
            return cls._link_value(value)
        if label == "延迟":
            return cls._delay_value(value)
        if label == "可被打断":
            return "是" if cls._truthy(value) else "否"
        if label == "语音":
            return "语音" if cls._truthy(value) else "文字"
        return cls._plain_value(value)

    @staticmethod
    def _link_value(value: Any) -> str:
        text = str(value or "").strip()
        return text if re.match(r"https?://", text, flags=re.IGNORECASE) else "已选择一条网页资料"

    @staticmethod
    def _delay_value(value: Any) -> str:
        text = str(value or "").strip()
        return f"{text} 秒" if re.fullmatch(r"\d+", text) else text

    @staticmethod
    def _plain_value(value: Any) -> str:
        if isinstance(value, Mapping):
            return ""
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return str(value or "").strip()
        scalars = []
        for item in value:
            if isinstance(item, Mapping):
                continue
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                continue
            text = str(item).strip()
            if text:
                scalars.append(text)
        return "、".join(scalars)

    @staticmethod
    def _parameter_label(label: str) -> str:
        return "留给之后的自己" if label == "留话" else label

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().casefold() in {"1", "true", "yes", "是"}

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        return [item for item in value if isinstance(item, Mapping)]


__all__ = ["PlayerOutputProjector"]

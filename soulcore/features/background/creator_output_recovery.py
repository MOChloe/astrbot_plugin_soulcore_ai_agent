"""Deterministic syntax recovery for background creator output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .domain import BackgroundAuthorKind
from .output_contract import BackgroundOutputError, TextBlock

CURRENT_FIELDS = (
    "时间",
    "地点",
    "正在做",
    "身体状态",
    "心情",
    "打算",
    "当前牵挂",
)
OPTIONAL_FIELD_MAP = {
    "正在做": "doing",
    "身体状态": "body_state",
    "心情": "mood",
    "打算": "intention",
    "当前牵挂": "current_concern",
}
_RETIRED_TAGS = frozenset({"故事源", "普通帧", "关键帧", "持久变化", "六小时前"})
_ALL_CURRENT_TAGS = frozenset(
    {
        "世界变化",
        "人生方向",
        "故事模组",
        "经历",
        "介入模组",
        "模组已了结",
        "留下变化",
        "留下变化已解决",
        "角色现在",
        "关键帧交接点",
    }
)
RESERVED_TAGS = frozenset((*_ALL_CURRENT_TAGS, *_RETIRED_TAGS))
_LEGACY_PARAMETER = re.compile(
    r"\[\[\s*(?:"
    r"正文|无新增|无变化|模组正文|可能入口|经历正文|留下变化|持久变化|"
    r"时间|地点|正在做|身体状态|心情|打算|当前牵挂"
    r")\s*\]\]"
)
_STRUCTURAL_NAMES = "|".join(
    re.escape(name) for name in sorted(RESERVED_TAGS, key=len, reverse=True)
)
_LOOSE_STRUCTURAL_TAG = re.compile(
    rf"[<＜]\s*(?P<close>[/／]?)\s*(?P<name>{_STRUCTURAL_NAMES})\s*[>＞]"
)
_CANONICAL_STRUCTURAL_TAG = re.compile(rf"<(?P<close>/)?(?P<name>{_STRUCTURAL_NAMES})>")
_FULL_UNKNOWN_WRAPPER = re.compile(
    r"^\s*<(?P<name>[^/<>&\s]+)>\s*(?P<body>.*)\s*</(?P=name)>\s*$",
    re.S,
)
_FIELD_LINE = re.compile(
    r"^\s*(?:>\s*)?(?:[-*+]\s*)?"
    r"(?P<label>(?:\*\*|__|`)?[^：:=|]+?(?:\*\*|__|`)?)"
    r"\s*[：:=]\s*(?P<value>.*)\s*$"
)
_TABLE_FIELD_LINE = re.compile(r"^\s*\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]*?)\s*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*$")
_PRESENTATION_TEXT = re.compile(
    r"^(?:好的[，,!！]?\s*)?(?:以下(?:是|为).*(?:输出|结果)|输出(?:如下|结果)|结果如下|"
    r"here(?:'s| is)\s+(?:the\s+)?(?:output|result)|sure[,.!]?)\s*[：:]?$",
    re.I,
)


@dataclass(frozen=True, slots=True)
class SnapshotNormalization:
    current_view: dict[str, Any] | None
    canonical_body: str
    normalizations: tuple[str, ...]
    error: str = ""


@dataclass(slots=True)
class _BlockScanner:
    allowed: frozenset[str]
    main_tag: str
    current_tag: str
    frame: bool
    snapshot_only: bool
    blocks: list[TextBlock] = field(default_factory=list)
    normalizations: list[str] = field(default_factory=list)
    active_name: str = ""
    active_parts: list[str] = field(default_factory=list)
    outside_parts: list[str] = field(default_factory=list)

    def feed(self, between: str, *, name: str, closing: bool) -> None:
        (self.active_parts if self.active_name else self.outside_parts).append(between)
        if name not in self.allowed:
            raise BackgroundOutputError(
                f"background creator returned incompatible outer tag: {name}"
            )
        if closing:
            self._close(name)
        else:
            self._open(name)

    def complete(self, tail: str) -> tuple[tuple[TextBlock, ...], tuple[str, ...]]:
        if self.active_name:
            self.active_parts.append(tail)
            self._finish_active(missing_close=True)
        elif tail.strip():
            self.outside_parts.append(tail)
        trailing = "".join(self.outside_parts).strip()
        if trailing:
            _consume_outside(
                trailing,
                blocks=self.blocks,
                main_tag=self.main_tag,
                normalizations=self.normalizations,
                allow_untagged_main=False,
            )
        if not self.blocks:
            raise BackgroundOutputError("background creator returned no usable blocks")
        return tuple(self.blocks), unique(self.normalizations)

    def _open(self, name: str) -> None:
        if self.active_name:
            self._finish_active(missing_close=True)
        leading = "".join(self.outside_parts).strip()
        self.outside_parts = []
        if leading:
            _consume_outside(
                leading,
                blocks=self.blocks,
                main_tag=self.main_tag,
                normalizations=self.normalizations,
                allow_untagged_main=(
                    self.frame and not self.snapshot_only and name != self.main_tag
                ),
            )
        self.active_name = name
        self.active_parts = []

    def _close(self, name: str) -> None:
        if self.active_name:
            if self.active_name != name:
                raise BackgroundOutputError(
                    f"background creator has mismatched tags: {self.active_name} -> {name}"
                )
            self._finish_active()
            return
        candidate = "".join(self.outside_parts).strip()
        self.outside_parts = []
        if not candidate or not _can_infer_missing_open(
            name,
            main_tag=self.main_tag,
            current_tag=self.current_tag,
            blocks=self.blocks,
        ):
            raise BackgroundOutputError(
                f"background creator closing tag {name} has no matching opener"
            )
        self.blocks.append(TextBlock(name, candidate))
        self.normalizations.append("missing_opening_tag_synthesized")

    def _finish_active(self, *, missing_close: bool = False) -> None:
        body = "".join(self.active_parts).strip()
        if not body:
            raise BackgroundOutputError(f"{self.active_name}正文不能为空")
        self.blocks.append(TextBlock(self.active_name, body))
        if missing_close:
            self.normalizations.append("missing_closing_tag_synthesized")
        self.active_name = ""
        self.active_parts = []


def prepare_output(raw: str) -> tuple[str, tuple[str, ...]]:
    if not raw.strip():
        raise BackgroundOutputError("background creator returned empty output")
    _reject_retired_markup(raw)
    normalizations: list[str] = []
    value = raw.strip()
    lines = value.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        value = "\n".join(lines[1:-1]).strip()
        normalizations.append("markdown_fence_removed")
    normalized_tags = _LOOSE_STRUCTURAL_TAG.sub(
        lambda match: f"<{'/' if match.group('close') else ''}{match.group('name')}>",
        value,
    )
    if normalized_tags != value:
        value = normalized_tags
        normalizations.append("structural_tag_punctuation_normalized")
    _reject_retired_markup(value)
    wrapper = _FULL_UNKNOWN_WRAPPER.fullmatch(value)
    if wrapper is not None and wrapper.group("name") not in RESERVED_TAGS:
        body = wrapper.group("body").strip()
        if _CANONICAL_STRUCTURAL_TAG.search(body):
            value = body
            normalizations.append("presentation_wrapper_removed")
    value, headings_changed = _normalize_headings(value)
    if headings_changed:
        normalizations.append("markdown_headings_normalized")
    return value.strip(), unique(normalizations)


def _normalize_headings(raw: str) -> tuple[str, bool]:
    names = "|".join(re.escape(name) for name in sorted(_ALL_CURRENT_TAGS, key=len, reverse=True))
    heading = re.compile(
        rf"^\s*(?:>\s*)?(?:#{{1,6}}\s*)?(?:[-*+]\s*)?"
        rf"(?:(?:\*\*|__|`)(?P<wrapped>{names})(?:\*\*|__|`)|(?P<plain>{names}))"
        rf"\s*[：:]?\s*$"
    )
    changed = False
    lines: list[str] = []
    for line in raw.splitlines():
        match = heading.fullmatch(line)
        if match is None:
            lines.append(line)
            continue
        lines.append(f"<{match.group('wrapped') or match.group('plain')}>")
        changed = True
    return "\n".join(lines), changed


def recover_blocks(
    raw: str,
    *,
    allowed: frozenset[str],
    main_tag: str,
    current_tag: str,
    frame: bool,
    snapshot_only: bool = False,
) -> tuple[tuple[TextBlock, ...], tuple[str, ...]]:
    markers = list(_CANONICAL_STRUCTURAL_TAG.finditer(raw))
    if not markers:
        if _looks_like_unknown_wrapper(raw):
            raise BackgroundOutputError("background creator returned an unknown outer wrapper")
        if snapshot_only:
            return (TextBlock(current_tag, raw.strip()),), ("plain_snapshot_fields_wrapped",)
        if frame:
            return _plain_frame_blocks(raw, main_tag=main_tag, current_tag=current_tag)
        return (TextBlock(main_tag, raw.strip()),), ("plain_body_wrapped",)
    scanner = _BlockScanner(
        allowed=allowed,
        main_tag=main_tag,
        current_tag=current_tag,
        frame=frame,
        snapshot_only=snapshot_only,
    )
    cursor = 0
    for marker in markers:
        scanner.feed(
            raw[cursor : marker.start()],
            name=marker.group("name"),
            closing=bool(marker.group("close")),
        )
        cursor = marker.end()
    return scanner.complete(raw[cursor:])


def _consume_outside(
    text: str,
    *,
    blocks: list[TextBlock],
    main_tag: str,
    normalizations: list[str],
    allow_untagged_main: bool,
) -> None:
    if _PRESENTATION_TEXT.fullmatch(text.strip()):
        normalizations.append("presentation_text_removed")
        return
    if allow_untagged_main and not any(block.name == main_tag for block in blocks):
        blocks.append(TextBlock(main_tag, text.strip()))
        normalizations.append("untagged_main_body_wrapped")
        return
    raise BackgroundOutputError("background creator contains ambiguous text outside its blocks")


def _plain_frame_blocks(
    raw: str,
    *,
    main_tag: str,
    current_tag: str,
) -> tuple[tuple[TextBlock, ...], tuple[str, ...]]:
    experience, snapshot = _split_plain_snapshot(raw)
    blocks: list[TextBlock] = []
    normalizations = ["plain_experience_wrapped"]
    if experience.strip():
        blocks.append(TextBlock(main_tag, experience.strip()))
    if snapshot.strip():
        blocks.append(TextBlock(current_tag, snapshot.strip()))
        normalizations.append("trailing_snapshot_fields_wrapped")
    if not blocks:
        raise BackgroundOutputError("生活帧没有可保留的经历正文")
    return tuple(blocks), tuple(normalizations)


def _split_plain_snapshot(raw: str) -> tuple[str, str]:
    lines = raw.strip().splitlines()
    index = len(lines)
    saw_field = False
    while index > 0:
        candidate = lines[index - 1]
        if not candidate.strip() or _TABLE_SEPARATOR.fullmatch(candidate):
            index -= 1
            continue
        if _parse_field_candidate(candidate) is None:
            break
        saw_field = True
        index -= 1
    if not saw_field:
        return raw.strip(), ""
    snapshot = "\n".join(lines[index:]).strip()
    if "地点" not in _snapshot_fields(snapshot):
        return raw.strip(), ""
    return "\n".join(lines[:index]).strip(), snapshot


def normalize_upper_blocks(
    blocks: tuple[TextBlock, ...],
    author_kind: BackgroundAuthorKind,
) -> tuple[tuple[TextBlock, ...], tuple[str, ...]]:
    tag = main_tag(author_kind)
    if any(block.name != tag for block in blocks):
        raise BackgroundOutputError(f"creator can only return {tag} blocks")
    bodies = [block.body.strip() for block in blocks if block.body.strip()]
    if not bodies:
        raise BackgroundOutputError(f"{tag}正文不能为空")
    if author_kind is BackgroundAuthorKind.LIFE_DIRECTION:
        if len(bodies) == 1:
            return (TextBlock(tag, bodies[0]),), ()
        return (TextBlock(tag, "\n\n".join(bodies)),), ("duplicate_main_blocks_merged",)
    if len(bodies) <= 2:
        return tuple(TextBlock(tag, body) for body in bodies), ()
    return (
        TextBlock(tag, bodies[0]),
        TextBlock(tag, "\n\n".join(bodies[1:])),
    ), ("excess_main_blocks_merged",)


def normalize_frame_blocks(
    blocks: tuple[TextBlock, ...],
    *,
    current_tag: str,
    authoritative_time: str,
) -> tuple[tuple[TextBlock, ...], tuple[str, ...]]:
    grouped, grouped_normalizations = _group_frame_blocks(blocks, current_tag=current_tag)
    normalized, content_normalizations = _normalize_frame_content_blocks(grouped)
    snapshot, snapshot_normalizations = _select_current_snapshot(
        grouped.get(current_tag) or (),
        current_tag=current_tag,
        authoritative_time=authoritative_time,
    )
    if snapshot is not None:
        normalized.append(snapshot)
    return tuple(normalized), unique(
        (*grouped_normalizations, *content_normalizations, *snapshot_normalizations)
    )


def _group_frame_blocks(
    blocks: tuple[TextBlock, ...],
    *,
    current_tag: str,
) -> tuple[dict[str, list[TextBlock]], tuple[str, ...]]:
    grouped: dict[str, list[TextBlock]] = {}
    normalizations: list[str] = []
    for block in blocks:
        name = block.name
        if name in {"角色现在", "关键帧交接点"}:
            if name != current_tag:
                normalizations.append("snapshot_tag_corrected")
            name = current_tag
        grouped.setdefault(name, []).append(TextBlock(name, block.body))
    return grouped, unique(normalizations)


def _normalize_frame_content_blocks(
    grouped: dict[str, list[TextBlock]],
) -> tuple[list[TextBlock], tuple[str, ...]]:
    experiences = grouped.get("经历") or []
    if not experiences:
        raise BackgroundOutputError("生活帧缺少可保留的经历正文")
    normalizations: list[str] = []
    normalized = [TextBlock("经历", "\n\n".join(block.body for block in experiences))]
    if len(experiences) > 1:
        normalizations.append("duplicate_experience_blocks_merged")
    _append_merged_blocks(
        normalized,
        grouped,
        tags=("介入模组", "模组已了结"),
        separator=", ",
        normalization="duplicate_reference_blocks_merged",
        normalizations=normalizations,
    )
    _append_merged_blocks(
        normalized,
        grouped,
        tags=("留下变化",),
        separator="\n\n",
        normalization="duplicate_leftover_blocks_merged",
        normalizations=normalizations,
    )
    _append_merged_blocks(
        normalized,
        grouped,
        tags=("留下变化已解决",),
        separator=", ",
        normalization="duplicate_retire_blocks_merged",
        normalizations=normalizations,
    )
    return normalized, unique(normalizations)


def _append_merged_blocks(
    target: list[TextBlock],
    grouped: dict[str, list[TextBlock]],
    *,
    tags: tuple[str, ...],
    separator: str,
    normalization: str,
    normalizations: list[str],
) -> None:
    for tag in tags:
        values = grouped.get(tag) or []
        if not values:
            continue
        target.append(TextBlock(tag, separator.join(block.body for block in values)))
        if len(values) > 1:
            normalizations.append(normalization)


def _select_current_snapshot(
    current_blocks: list[TextBlock] | tuple[TextBlock, ...],
    *,
    current_tag: str,
    authoritative_time: str,
) -> tuple[TextBlock | None, tuple[str, ...]]:
    if not current_blocks:
        return None, ()
    snapshots = [
        normalize_snapshot(block.body, authoritative_time=authoritative_time)
        for block in current_blocks
    ]
    normalizations = [item for snapshot in snapshots for item in snapshot.normalizations]
    complete = [snapshot for snapshot in snapshots if snapshot.current_view is not None]
    locations = {
        str(snapshot.current_view.get("location") or "")
        for snapshot in complete
        if snapshot.current_view is not None
    }
    if len(locations) > 1:
        normalizations.append("conflicting_snapshot_locations_require_repair")
        return None, unique(normalizations)
    if not complete:
        return None, unique(normalizations)
    if len(current_blocks) > 1:
        normalizations.append("duplicate_snapshot_last_complete_kept")
    return TextBlock(current_tag, complete[-1].canonical_body), unique(normalizations)


def normalize_snapshot_repair_blocks(
    blocks: tuple[TextBlock, ...],
    *,
    current_tag: str,
    authoritative_time: str,
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    candidates, changes = _repair_snapshot_candidates(blocks, current_tag=current_tag)
    if not candidates:
        raise BackgroundOutputError(f"快照修复轮没有返回{current_tag}字段")
    snapshots = [
        normalize_snapshot(block.body, authoritative_time=authoritative_time)
        for block in candidates
    ]
    complete = [snapshot for snapshot in snapshots if snapshot.current_view is not None]
    _reject_conflicting_repair_locations(complete)
    if not complete:
        error = snapshots[-1].error if snapshots else f"{current_tag}缺少地点"
        raise BackgroundOutputError(error or f"{current_tag}缺少地点")
    normalizations = [*changes]
    normalizations.extend(item for snapshot in snapshots for item in snapshot.normalizations)
    if len(snapshots) > 1:
        normalizations.append("duplicate_snapshot_last_complete_kept")
    chosen = complete[-1]
    return (
        dict(chosen.current_view or {}),
        canonical_document((TextBlock(current_tag, chosen.canonical_body),)),
        unique(normalizations),
    )


def _repair_snapshot_candidates(
    blocks: tuple[TextBlock, ...],
    *,
    current_tag: str,
) -> tuple[list[TextBlock], tuple[str, ...]]:
    candidates: list[TextBlock] = []
    changes: list[str] = []
    for block in blocks:
        if block.name not in {"角色现在", "关键帧交接点"}:
            continue
        if block.name != current_tag:
            changes.append("snapshot_tag_corrected")
        candidates.append(TextBlock(current_tag, block.body))
    return candidates, unique(changes)


def _reject_conflicting_repair_locations(snapshots: list[SnapshotNormalization]) -> None:
    locations = {
        str(snapshot.current_view.get("location") or "")
        for snapshot in snapshots
        if snapshot.current_view is not None
    }
    if len(locations) > 1:
        raise BackgroundOutputError("快照修复轮返回了互相冲突的地点")


def normalize_snapshot(body: str, *, authoritative_time: str) -> SnapshotNormalization:
    values, collected = _collect_snapshot_values(body)
    normalizations = list(collected)
    location_values = list(dict.fromkeys(values.get("地点") or ()))
    if len(location_values) > 1:
        return _snapshot_failure(
            "角色快照包含互相冲突的地点",
            (*normalizations, "conflicting_location_fields"),
        )
    if not location_values:
        return _snapshot_failure("角色快照缺少可用的地点", normalizations)
    chosen_time = _choose_snapshot_time(
        values.get("时间") or (),
        authoritative_time=authoritative_time,
        normalizations=normalizations,
    )
    if not chosen_time:
        return _snapshot_failure("角色快照缺少可用的时间", normalizations)
    params: dict[str, str] = {"时间": chosen_time, "地点": location_values[0]}
    _add_optional_snapshot_fields(params, values=values, normalizations=normalizations)
    canonical_body = "\n".join(
        f"{name}：{params[name]}" for name in CURRENT_FIELDS if name in params
    )
    return SnapshotNormalization(
        current_view=current_view_from_params(params),
        canonical_body=canonical_body,
        normalizations=unique(normalizations),
    )


def _collect_snapshot_values(
    body: str,
) -> tuple[dict[str, list[str]], tuple[str, ...]]:
    values: dict[str, list[str]] = {}
    normalizations: list[str] = []
    for line in str(body or "").splitlines():
        if not line.strip() or _TABLE_SEPARATOR.fullmatch(line):
            if _TABLE_SEPARATOR.fullmatch(line):
                normalizations.append("markdown_table_separator_removed")
            continue
        parsed = _parse_field_candidate(line)
        if parsed is None:
            normalizations.append("snapshot_unstructured_line_dropped")
            continue
        name, value, syntax = parsed
        if name not in CURRENT_FIELDS:
            normalizations.append("unknown_snapshot_field_dropped")
            continue
        if syntax != "canonical":
            normalizations.append("snapshot_field_syntax_normalized")
        value = _unwrap_markdown(value.strip())
        if not value or _is_placeholder(value):
            code = (
                "invalid_required_snapshot_field_removed"
                if name in {"时间", "地点"}
                else "empty_optional_snapshot_field_dropped"
            )
            normalizations.append(code)
            continue
        values.setdefault(name, []).append(value)
    return values, unique(normalizations)


def _choose_snapshot_time(
    raw_values: list[str] | tuple[str, ...],
    *,
    authoritative_time: str,
    normalizations: list[str],
) -> str:
    time_values = list(dict.fromkeys(raw_values))
    if not time_values and str(authoritative_time or "").strip():
        time_values = [str(authoritative_time).strip()]
        normalizations.append("authoritative_snapshot_time_injected")
    if not time_values:
        return ""
    if len(time_values) > 1:
        chosen = str(authoritative_time or "").strip() or time_values[-1]
        normalizations.append("duplicate_time_fields_normalized")
        return chosen
    return time_values[0]


def _add_optional_snapshot_fields(
    params: dict[str, str],
    *,
    values: dict[str, list[str]],
    normalizations: list[str],
) -> None:
    for name in CURRENT_FIELDS[2:]:
        candidates = list(dict.fromkeys(values.get(name) or ()))
        if not candidates:
            continue
        if len(candidates) > 1:
            normalizations.append("duplicate_optional_snapshot_field_last_kept")
        params[name] = candidates[-1]


def _snapshot_failure(error: str, normalizations: Any) -> SnapshotNormalization:
    return SnapshotNormalization(
        current_view=None,
        canonical_body="",
        normalizations=unique(normalizations),
        error=error,
    )


def _snapshot_fields(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in str(body or "").splitlines():
        parsed = _parse_field_candidate(line)
        if parsed is not None and parsed[0] in CURRENT_FIELDS:
            result[parsed[0]] = parsed[1]
    return result


def _parse_field_candidate(line: str) -> tuple[str, str, str] | None:
    table = _TABLE_FIELD_LINE.fullmatch(line)
    if table is not None:
        label = _unwrap_markdown(table.group("label").strip())
        return label, table.group("value").strip(), "markdown_table"
    match = _FIELD_LINE.fullmatch(line)
    if match is None:
        return None
    label = _unwrap_markdown(match.group("label").strip())
    syntax = "canonical" if f"{label}：" in line and line.strip().startswith(label) else "variant"
    return label, match.group("value").strip(), syntax


def _unwrap_markdown(value: str) -> str:
    result = str(value or "").strip()
    for marker in ("**", "__", "`"):
        if result.startswith(marker) and result.endswith(marker) and len(result) >= len(marker) * 2:
            result = result[len(marker) : -len(marker)].strip()
    return result


def _is_placeholder(value: str) -> bool:
    raw = str(value or "").strip()
    normalized = raw.strip("【】[]（）()<>《》").strip()
    if normalized in {
        "可选",
        "必填",
        "无",
        "没有",
        "暂无",
        "未知",
        "待补充",
        "清空",
        "-",
        "—",
        "/",
    }:
        return True
    wrapped = len(raw) >= 2 and raw[0] in "【[（(<《" and raw[-1] in "】]）)>》"
    return wrapped and normalized.startswith(("可选", "必填"))


def current_view_from_params(params: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "narrative_time": params["时间"],
        "location": params["地点"],
    }
    for prompt_name, field_name in OPTIONAL_FIELD_MAP.items():
        if prompt_name in params:
            result[field_name] = params[prompt_name]
    return result


def last_snapshot_body(blocks: tuple[TextBlock, ...]) -> str:
    candidates = [block.body for block in blocks if block.name in {"角色现在", "关键帧交接点"}]
    return candidates[-1] if candidates else ""


def main_tag(kind: BackgroundAuthorKind) -> str:
    return {
        BackgroundAuthorKind.WORLD: "世界变化",
        BackgroundAuthorKind.LIFE_DIRECTION: "人生方向",
        BackgroundAuthorKind.STORY_SOURCE: "故事模组",
        BackgroundAuthorKind.KEYFRAME: "经历",
        BackgroundAuthorKind.ORDINARY: "经历",
    }[kind]


def allowed_tags(
    kind: BackgroundAuthorKind,
    *,
    opening_keyframe: bool,
) -> frozenset[str]:
    if kind in {
        BackgroundAuthorKind.WORLD,
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
    }:
        return frozenset({main_tag(kind)})
    current_tag = "关键帧交接点" if opening_keyframe else "角色现在"
    return frozenset(
        {
            "经历",
            "介入模组",
            "模组已了结",
            "留下变化",
            "留下变化已解决",
            current_tag,
        }
    )


def recovery_allowed_tags(kind: BackgroundAuthorKind) -> frozenset[str]:
    if kind in {
        BackgroundAuthorKind.WORLD,
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
    }:
        return frozenset({main_tag(kind)})
    return frozenset(
        {
            "经历",
            "介入模组",
            "模组已了结",
            "留下变化",
            "留下变化已解决",
            "角色现在",
            "关键帧交接点",
        }
    )


def _can_infer_missing_open(
    name: str,
    *,
    main_tag: str,
    current_tag: str,
    blocks: list[TextBlock],
) -> bool:
    if name == main_tag and not any(block.name == main_tag for block in blocks):
        return True
    return name in {current_tag, "角色现在", "关键帧交接点"}


def canonical_document(blocks: tuple[TextBlock, ...]) -> str:
    return "\n\n".join(f"<{block.name}>\n{block.body.strip()}\n</{block.name}>" for block in blocks)


def _looks_like_unknown_wrapper(raw: str) -> bool:
    match = _FULL_UNKNOWN_WRAPPER.fullmatch(str(raw or ""))
    return match is not None and match.group("name") not in RESERVED_TAGS


def _reject_retired_markup(raw: str) -> None:
    if _LEGACY_PARAMETER.search(raw):
        raise BackgroundOutputError(
            "background creator output cannot contain legacy [[字段]] parameters"
        )
    retired = "|".join(re.escape(name) for name in sorted(_RETIRED_TAGS, key=len, reverse=True))
    if re.search(rf"[<＜]\s*[/／]?\s*(?:{retired})\s*[>＞]", raw):
        raise BackgroundOutputError("background creator output contains a retired outer tag")


def unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))

"""Render bounded MainCore state into stable context and append-only turn blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ...contracts.ai_models import AIPromptCacheSemanticKind
from ...shared.prompt_document import (
    CompiledPrompt,
    PromptBlock,
    PromptCacheBoundary,
    compile_prompt_document,
    xml_text,
)
from ..conversation import ContextSource
from . import turn_prompt
from .roleplay_prompt_contracts import BoundedPromptState
from .roleplay_prompt_stable import (
    role_identity_prompt_blocks,
    role_protocol_prompt_blocks,
)

MAIN_CORE_PROMPT_PROTOCOL_VERSION = "main-core-roleplay-v29"
_RECENT_DIALOGUE_MESSAGE_LIMIT = 10
_OLDER_HISTORY_TOOL_HINT = (
    "这里只显示了压缩后的历史；需要了解更早内容的具体事实、事件或变化时，使用“回想”查询。"
)

_AGENT_ACTION_AND_OUTPUT_PROTOCOL = (
    "这是多轮行动。每轮只做当前一步：soulcore_continue 返回完整行动结果并继续；"
    "soulcore_final 提交最终表达并结束。每轮只选一个，text 只写完整 XML。没有这两个工具时改用：\n"
    "<继续行动>\n……完整 XML……\n</继续行动>\n"
    "或\n"
    "<最终表达>\n……完整 XML……\n</最终表达>\n\n"
    "草稿可写在工具或外层之外。继续行动只放非终止指令；最终表达只放终止指令。"
    "一批中任一项错误，整批都不执行。互不依赖的行动可同批；依赖前一项结果的行动留到下一轮。\n\n"
    "例一（Plan → 结果 → 最终表达）：\n"
    "soulcore_continue：\n"
    "<制定Plan>\n[[内容]]：先确定最终表达的目标、取舍和完成标准。\n</制定Plan>\n"
    "返回：制定Plan：成功\nPlan 已保存；下一轮继续。\n"
    "soulcore_final：\n"
    "<发文字>\n[[内容]]：现在给出的最终表达。\n</发文字>\n\n"
    "例二（查询 → 结果 → 最终表达）：\n"
    "soulcore_continue：\n<查资料>\n[[想知道什么]]：明天本地是否下雨？\n</查资料>\n"
    "收到查询结果后，再用 soulcore_final 写最终表达。\n\n"
    "例三（错误重试）：\n"
    "若把<查资料>放进 soulcore_final，整批不会执行。改用 soulcore_continue 提交查询，收到结果后再"
    "用 soulcore_final。"
)

_FINAL_ONLY_OUTPUT_PROTOCOL = (
    "本次只提交最终表达：使用 soulcore_final，text 只写完整 XML。没有该工具时使用：\n"
    "<最终表达>\n……完整 XML……\n</最终表达>\n"
    "草稿可写在工具或外层之外；一批中任一项错误，整批都不提交。"
)


def _action_and_output_protocol(state: BoundedPromptState) -> str:
    has_continue = any(spec.prompt_visible and not spec.terminal for spec in state.registry.specs)
    return _AGENT_ACTION_AND_OUTPUT_PROTOCOL if has_continue else _FINAL_ONLY_OUTPUT_PROTOCOL


class RolePlayPromptRenderingMixin:
    """Rendering half of ``RolePlayPromptCompiler`` without input preparation."""

    def _render_bounded_prompt(self, state: BoundedPromptState) -> CompiledPrompt:
        context_blocks = self._bounded_context_blocks(state)
        turn_blocks = self._bounded_turn_blocks(state)
        compiled = compile_prompt_document(
            context_blocks,
            turn_blocks,
            model_id=state.model_id,
            reference_map=state.reference_map,
            trim_reasons=state.trim_reasons,
            image_urls=state.image_urls,
            prompt_protocol_version=MAIN_CORE_PROMPT_PROTOCOL_VERSION,
            cache_rebase_reasons=state.cache_rebase_reasons,
        )
        source_ids = tuple(
            sorted(
                set(state.current_message_ids)
                | {
                    int(message_id)
                    for source in (
                        ContextSource.CHARACTER_INTENT,
                        ContextSource.HISTORY_SUMMARY,
                        ContextSource.PLAYER_PROFILE,
                        ContextSource.STICKER,
                        ContextSource.CURRENT_WEB_RESOURCE,
                        ContextSource.CURRENT_DIALOGUE,
                    )
                    for message_id in state.working_message_ids[source]
                    if int(message_id) > 0
                }
                | {
                    int(ledger_id)
                    for ledger_id, public in state.message_reference_by_ledger_id.items()
                    if int(ledger_id) > 0 and f"[{public}]" in compiled.document
                }
            )
        )
        summary_ids = tuple(
            sorted(
                {
                    int(summary_id)
                    for values in state.working_summary_ids.values()
                    for summary_id in values
                    if int(summary_id) > 0
                }
            )
        )
        summary_id_set = set(summary_ids)
        summary_coverage = tuple(
            row for row in state.summary_coverage if int(row[0]) in summary_id_set
        )
        return replace(
            compiled,
            source_message_ids=source_ids,
            source_summary_ids=summary_ids,
            source_summary_coverage=summary_coverage,
            background_item_refs=tuple(
                ref
                for source in (
                    ContextSource.BACKGROUND_STORY,
                    ContextSource.BACKGROUND_WORLD,
                    ContextSource.BACKGROUND_EXPERIENCE,
                    ContextSource.BACKGROUND_KEYFRAME,
                    ContextSource.ROLE_LATEST_EXPERIENCE,
                    ContextSource.BACKGROUND_LEFTOVER,
                    ContextSource.ROLE_LIFE_DIRECTION,
                )
                for ref in state.working_item_refs[source]
                if ref
            ),
        )

    def _bounded_context_blocks(self, state: BoundedPromptState) -> list[PromptBlock]:
        blocks = [
            *role_protocol_prompt_blocks(state),
            PromptBlock("指令列表", state.registry.list_text()),
            # Registry text is trusted protocol syntax; callable templates stay unescaped.
            PromptBlock("指令详情", state.registry.details_text()),
            PromptBlock(
                "行动与输出协议",
                _action_and_output_protocol(state),
                cache_boundaries=(
                    PromptCacheBoundary(
                        "main-core-protocol",
                        AIPromptCacheSemanticKind.PROTOCOL,
                        1,
                        selection_reason="固定协议末端",
                    ),
                ),
            ),
            *role_identity_prompt_blocks(state),
            PromptBlock(
                "人生方向",
                self._background_block(state, ContextSource.ROLE_LIFE_DIRECTION),
            ),
            PromptBlock(
                "世界动向",
                self._background_block(state, ContextSource.BACKGROUND_WORLD),
            ),
            PromptBlock(
                "角色现在",
                self._background_block(state, ContextSource.ROLE_STATE),
            ),
            PromptBlock("近期经历", self._recent_experience_block(state)),
            PromptBlock(
                "留下变化",
                self._background_block(state, ContextSource.BACKGROUND_LEFTOVER),
            ),
            PromptBlock(
                "别处",
                self._background_block(state, ContextSource.BACKGROUND_STORY),
            ),
            PromptBlock("交互对象肖像", self._joined(state.working[ContextSource.PLAYER_PROFILE])),
            PromptBlock("历史对话", self._history_block(state)),
        ]
        return self._attach_cache_boundaries_to_last_nonempty(
            blocks,
            (
                PromptCacheBoundary(
                    "main-core-context",
                    AIPromptCacheSemanticKind.CONTEXT,
                    2,
                    selection_reason="角色背景、交互对象肖像与历史摘要末端",
                ),
            ),
        )

    def _bounded_turn_blocks(self, state: BoundedPromptState) -> list[PromptBlock]:
        dialogue, recent_dialogue, rebase_reasons = self._dialogue_blocks(state)
        state.cache_rebase_reasons = tuple(dict.fromkeys(rebase_reasons))
        blocks = [
            dialogue,
            PromptBlock(
                "当前消息中的网页",
                self._joined(state.working[ContextSource.CURRENT_WEB_RESOURCE]),
            ),
            PromptBlock("表情包", self._joined(state.working[ContextSource.STICKER])),
            PromptBlock(
                "仍在考虑的打算",
                self._joined(state.working[ContextSource.CHARACTER_INTENT]),
            ),
            PromptBlock("当前处境", state.situation_note),
            PromptBlock("当前模式方式", state.mode_guidance),
            PromptBlock("身份引用", xml_text(state.identity_catalog_text)),
            PromptBlock("角色触发提醒", turn_prompt.trigger_reminder_xml(state.trigger_reminders)),
            PromptBlock("本轮要求", state.thinking_requirement),
            PromptBlock("本轮补充信息", state.runtime_note),
            recent_dialogue,
            PromptBlock("当前时间", state.current_time),
        ]
        return self._attach_cache_boundaries_to_last_nonempty(
            blocks,
            (
                PromptCacheBoundary(
                    "main-core-run-base",
                    AIPromptCacheSemanticKind.CURRENT_RUN,
                    4,
                    selection_reason="稳定场景与当前输入末端；Agent 历史由 Provider 继续追加",
                ),
            ),
        )

    def _dialogue_blocks(
        self, state: BoundedPromptState
    ) -> tuple[PromptBlock, PromptBlock, list[str]]:
        entries = self._rendered_dialogue_entries(state)
        split_at = max(0, len(entries) - _RECENT_DIALOGUE_MESSAGE_LIMIT)
        earlier_entries = entries[:split_at]
        recent_entries = entries[split_at:]
        earlier_lines = [line for line, _message_id in earlier_entries]
        earlier_ids = [message_id for _line, message_id in earlier_entries]
        earlier_content = self._joined(earlier_lines)
        recent_content = self._joined([line for line, _message_id in recent_entries])
        offsets = self._joined_item_end_offsets(earlier_lines)
        boundaries = self._dialogue_cache_boundaries(offsets)
        previous_boundary, rebase_reason = self._previous_dialogue_boundary(
            entries,
            earlier_ids,
            offsets,
            state.previous_context_message_ids,
        )
        if previous_boundary is not None:
            boundaries.append(previous_boundary)
        rebase_reasons = [rebase_reason] if rebase_reason else []
        return (
            PromptBlock(
                "对话时间线",
                earlier_content,
                cache_boundaries=tuple(boundaries),
            ),
            PromptBlock("最近对话", recent_content),
            rebase_reasons,
        )

    @staticmethod
    def _rendered_dialogue_entries(
        state: BoundedPromptState,
    ) -> list[tuple[str, int]]:
        lines = [*state.working[ContextSource.CURRENT_DIALOGUE], *state.current_lines]
        message_ids = [
            *state.working_message_ids[ContextSource.CURRENT_DIALOGUE],
            *state.current_line_message_ids,
        ]
        return [
            (str(line).strip(), int(message_id))
            for line, message_id in zip(lines, message_ids, strict=True)
            if str(line).strip()
        ]

    @staticmethod
    def _dialogue_cache_boundaries(
        offsets: Sequence[int],
    ) -> list[PromptCacheBoundary]:
        boundaries: list[PromptCacheBoundary] = []
        for index, offset in enumerate(offsets):
            is_earlier_end = index == len(offsets) - 1
            boundaries.append(
                PromptCacheBoundary(
                    (
                        "main-core-dialogue-older-current"
                        if is_earlier_end
                        else f"main-core-dialogue-older-line-{index + 1}"
                    ),
                    AIPromptCacheSemanticKind.CURRENT_DIALOGUE,
                    3,
                    content_end=offset,
                    selection_reason=(
                        "较早对话末端；最近 10 条保留在动态尾部"
                        if is_earlier_end
                        else "较早对话消息末端内部候选边界"
                    ),
                )
            )
        return boundaries

    @staticmethod
    def _previous_dialogue_boundary(
        entries: Sequence[tuple[str, int]],
        earlier_ids: Sequence[int],
        offsets: Sequence[int],
        previous_context_message_ids: Sequence[int],
    ) -> tuple[PromptCacheBoundary | None, str]:
        previous_ids = set(previous_context_message_ids)
        if not previous_ids:
            return None, ""
        visible_ids = {message_id for _line, message_id in entries if message_id > 0}
        if not previous_ids.issubset(visible_ids):
            return None, "上一 Run 的消息已被 FillBudget 淘汰，已退回背景上下文"
        previous_visible_ids = tuple(
            dict.fromkeys(
                message_id
                for _line, message_id in entries
                if message_id > 0 and message_id in previous_ids
            )
        )
        previous_older_count = max(
            0,
            len(previous_visible_ids) - _RECENT_DIALOGUE_MESSAGE_LIMIT,
        )
        if not previous_older_count:
            return None, ""
        previous_end_id = previous_visible_ids[previous_older_count - 1]
        previous_index = earlier_ids.index(previous_end_id)
        return (
            PromptCacheBoundary(
                "main-core-dialogue-previous-run",
                AIPromptCacheSemanticKind.PREVIOUS_DIALOGUE,
                2,
                content_end=offsets[previous_index],
                selection_reason="上一 MainCore Run 的较早对话末端",
            ),
            "",
        )

    @staticmethod
    def _attach_cache_boundaries_to_last_nonempty(
        blocks: Sequence[PromptBlock],
        boundaries: tuple[PromptCacheBoundary, ...],
    ) -> list[PromptBlock]:
        result = list(blocks)
        for index in range(len(result) - 1, -1, -1):
            if not result[index].render():
                continue
            result[index] = replace(
                result[index],
                cache_boundaries=(*result[index].cache_boundaries, *boundaries),
            )
            break
        return result

    @staticmethod
    def _joined_item_end_offsets(values: Sequence[str]) -> tuple[int, ...]:
        offsets: list[int] = []
        cursor = 0
        rendered_count = 0
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            if rendered_count:
                cursor += 1
            cursor += len(text)
            offsets.append(cursor)
            rendered_count += 1
        return tuple(offsets)

    @staticmethod
    def _joined(values: Sequence[str]) -> str:
        return "\n".join(str(value).strip() for value in values if str(value).strip())

    def _history_block(self, state: BoundedPromptState) -> str:
        content = self._joined(
            (
                *state.working[ContextSource.HISTORY_SUMMARY],
                *state.working[ContextSource.HISTORY_FRAGMENT],
            )
        )
        recall_available = state.registry.get("回想") is not None
        if not state.has_searchable_earlier_history or not recall_available:
            return content
        return f"{content}\n\n{_OLDER_HISTORY_TOOL_HINT}".strip()

    def _background_block(
        self,
        state: BoundedPromptState,
        sources: ContextSource | Sequence[ContextSource],
    ) -> str:
        selected = (sources,) if isinstance(sources, ContextSource) else tuple(sources)
        items = tuple(item for source in selected for item in state.working[source])
        if selected == (ContextSource.BACKGROUND_STORY,):
            return "\n\n——\n\n".join(str(item).strip() for item in items if str(item).strip())
        return self._joined(items)

    def _recent_experience_block(self, state: BoundedPromptState) -> str:
        sources = (
            ContextSource.BACKGROUND_EXPERIENCE,
            ContextSource.BACKGROUND_KEYFRAME,
            ContextSource.ROLE_LATEST_EXPERIENCE,
        )
        entries: list[tuple[tuple[int, int, int, int], str]] = []
        for source_rank, source in enumerate(sources):
            values = state.working[source]
            sequences = state.working_sequences[source]
            for index, content in enumerate(values):
                # The background projection assigns rank 0 to the time-latest
                # protected experience and increasing ranks toward the past.
                # Descending rank therefore renders one old-to-new chronology
                # across ordinary and keyframe sources.
                order = (-sequences[index], source_rank, index, 0)
                entries.append((order, content))
        content = self._joined(
            tuple(content for _order, content in sorted(entries, key=lambda item: item[0]))
        )
        return content


__all__ = [
    "MAIN_CORE_PROMPT_PROTOCOL_VERSION",
    "RolePlayPromptRenderingMixin",
]

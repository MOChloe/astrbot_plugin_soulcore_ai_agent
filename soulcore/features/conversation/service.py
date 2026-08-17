"""Instance-owned conversation assembly and summary scheduling."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from html import unescape
from typing import Any, Protocol

from ...contracts.message_reference import safe_model_identity
from ...contracts.models import ConversationMessage, MessageDirection
from ...contracts.runtime_limits import DURABLE_AI_MAX_ATTEMPTS
from ...contracts.thinking import thinking_policy_from_value
from ..identity import (
    build_identity_catalog,
    escape_untrusted_identity_syntax,
    project_identity_text_for_model,
)
from ..knowledge.ports import KnowledgeRepositoryPort
from ..media.ports import MediaRepositoryPort
from ..profiles.ports import ProfilesRepositoryPort
from ..profiles.service import ProfileRuntimeGate
from ..stickers.service import StickerService
from .context import (
    BACKGROUND_FILL_WEIGHT,
    DEFAULT_SOURCE_POLICIES,
    BudgetClass,
    ConservativeTokenMeter,
    ContextBudgetConfig,
    ContextItem,
    ContextSource,
    DefaultContextProjector,
    RequestBudgetGuard,
)
from .history import ConversationHistoryMixin, ConversationHistoryPage
from .player_profile_context import PlayerProfileContextMixin
from .ports import AITaskSchedulerPort, ConversationRepositoryPort
from .preparation_inputs import ContextPreparationInputs, ConversationInputLoaderMixin
from .preparation_items import ContextItemAssemblerMixin
from .preparation_result import ContextCompilationMixin, PreparedMainCoreContext


def model_identity_projection(
    value: str,
    identity_context: Any,
    identity_catalog: Any | None,
) -> str:
    catalog = identity_catalog or build_identity_catalog(identity_context)
    return project_identity_text_for_model(
        str(value or ""),
        catalog,
        scope=str(identity_context.scope),
    )


def project_identity_context_items(
    items: list[Any],
    identity_context: Any,
    identity_catalog: Any,
) -> list[Any]:
    """Project trusted blocks; current messages were already escaped/projected at ingestion."""

    result = []
    for item in items:
        if item.source in {
            ContextSource.CURRENT_PLAYER_MESSAGE,
            ContextSource.CURRENT_DIALOGUE,
        }:
            result.append(item)
            continue
        result.append(
            replace(
                item,
                body=model_identity_projection(
                    str(item.body or ""), identity_context, identity_catalog
                ),
            )
        )
    return result


DIALOGUE_SUMMARY_RECENT_TURNS = 20
INTERRUPTED_UNSENT_STATUS = "INTERRUPTED_UNSENT"
_INBOUND_IMAGE_PLACEHOLDER = re.compile(r"\[对方发送了一张图片(?:：[^\]\r\n]*)?\]")


class _ContextThinkingPolicy(Protocol):
    fill_ratio: float


class ConversationContextService(
    ConversationInputLoaderMixin,
    ContextItemAssemblerMixin,
    PlayerProfileContextMixin,
    ContextCompilationMixin,
    ConversationHistoryMixin,
):
    """Compile one Main Core request from SoulCore's own SQLite ledger."""

    def __init__(
        self,
        repository: ConversationRepositoryPort,
        *,
        media_repository: MediaRepositoryPort,
        profiles: ProfilesRepositoryPort,
        ai_tasks: AITaskSchedulerPort,
        runtime_gate: ProfileRuntimeGate,
        knowledge: KnowledgeRepositoryPort,
        stickers: StickerService,
        platform_messages: Any,
        player_profiles: Any,
        identity: Any,
    ) -> None:
        self.repository = repository
        self.media_repository = media_repository
        self.profiles = profiles
        self.ai_tasks = ai_tasks
        self.runtime_gate = runtime_gate
        self.projector = DefaultContextProjector()
        self.knowledge = knowledge
        self.stickers = stickers
        self.platform_messages = platform_messages
        self.player_profiles = player_profiles
        self.identity = identity

    async def thinking_policy(self, profile_id: str) -> _ContextThinkingPolicy:
        profile = await self.profiles.get_profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        return thinking_policy_from_value(profile.thinking_complexity)

    async def budget_config_for_profile(self, profile_id: str, role: Any) -> ContextBudgetConfig:
        return self.budget_config(role, await self.thinking_policy(profile_id))

    async def recent_inbound_interaction_turns(
        self,
        profile_id: str,
        instance_id: str,
        *,
        limit: int = 50,
        through_message_id: int | None = None,
    ) -> tuple[str, ...]:
        """Return newest-first inbound text grouped by full interaction boundaries."""

        maximum = max(1, min(int(limit), 50))
        turns: list[str] = []
        current: list[str] = []
        inside_inbound = False
        cursor = int(through_message_id) if through_message_id is not None else None
        inspected = 0
        while len(turns) < maximum and inspected < 10_000:
            rows = await self.repository.list_instance_messages(
                profile_id,
                instance_id,
                through_message_id=cursor,
                limit=min(200, 10_000 - inspected),
                ascending=False,
                context_eligible_only=True,
            )
            if not rows:
                break
            inspected += len(rows)
            current, inside_inbound = self._collect_inbound_turns(
                rows, turns, current, inside_inbound, maximum
            )
            cursor = min(int(message.message_id) for message in rows) - 1
            if len(rows) < 200 or cursor < 1:
                break
        if inside_inbound and len(turns) < maximum:
            turns.append("\n".join(reversed(current)))
        return tuple(turns[:maximum])

    @staticmethod
    def _collect_inbound_turns(
        rows: Sequence[ConversationMessage],
        turns: list[str],
        current: list[str],
        inside_inbound: bool,
        maximum: int,
    ) -> tuple[list[str], bool]:
        for message in rows:
            if message.direction == MessageDirection.INBOUND:
                inside_inbound = True
                text = str(message.plain_text or "").strip()
                if text:
                    current.append(text)
                continue
            if inside_inbound:
                turns.append("\n".join(reversed(current)))
                current = []
                inside_inbound = False
                if len(turns) >= maximum:
                    break
        return current, inside_inbound

    def budget_config(
        self,
        role: Any,
        thinking_policy: _ContextThinkingPolicy | None = None,
    ) -> ContextBudgetConfig:
        if thinking_policy is None:
            return ContextBudgetConfig(
                int(role.max_context_tokens),
                int(role.target_context_tokens),
            ).normalized()
        return ContextBudgetConfig(
            int(role.max_context_tokens),
            int(role.target_context_tokens),
            thinking_policy.fill_ratio,
        ).normalized()

    async def prepare(
        self,
        *,
        profile_id: str,
        instance_id: str,
        role: Any,
        run_prompt: str,
        model_id: str = "",
        provider_context_limit: int | None = None,
        current_message_id: int | None = None,
        current_message_ids: Sequence[int] = (),
        core_run_id: int = 0,
        active_intents: Sequence[Any] = (),
        current_web_resources: Sequence[Any] = (),
        background_view: Any | None = None,
        thinking_policy: _ContextThinkingPolicy | None = None,
        defer_provider_selection: bool = False,
        custom_prompt_text: str = "",
    ) -> PreparedMainCoreContext:
        await self.runtime_gate.require_enabled(profile_id, instance_id)
        thinking_policy = thinking_policy or await self.thinking_policy(profile_id)
        inputs = await self._load_preparation_inputs(
            profile_id=profile_id,
            instance_id=instance_id,
            role=role,
            model_id=model_id,
            run_prompt=run_prompt,
            current_message_id=current_message_id,
            current_message_ids=current_message_ids,
            core_run_id=core_run_id,
            current_web_resources=current_web_resources,
            thinking_policy=thinking_policy,
        )
        items, stickers = await self._assemble_context_items(
            profile_id=profile_id,
            instance_id=instance_id,
            run_prompt=run_prompt,
            current_message_id=current_message_id,
            core_run_id=core_run_id,
            active_intents=active_intents,
            inputs=inputs,
            background_view=background_view,
        )
        player_profile_targets = await self._append_player_profile_context(
            items,
            inputs,
            profile_id=profile_id,
            instance_id=instance_id,
            provider_context_limit=provider_context_limit,
            defer_provider_selection=defer_provider_selection,
            custom_prompt_text=custom_prompt_text,
        )
        # Player profiles are appended after the other semantic sources, so
        # trusted identity templates are projected only after this final append.
        items = project_identity_context_items(
            items,
            inputs.identity_context,
            inputs.identity_catalog,
        )
        result = await self._compile_context_result(
            profile_id=profile_id,
            instance_id=instance_id,
            model_id=model_id,
            core_run_id=core_run_id,
            provider_context_limit=provider_context_limit,
            defer_provider_selection=defer_provider_selection,
            items=items,
            stickers=stickers,
            inputs=inputs,
            custom_prompt_text=custom_prompt_text,
        )
        return PreparedMainCoreContext(
            compiled=result.compiled,
            guard=RequestBudgetGuard(inputs.meter),
            effective_max_tokens=result.compiled.report.effective_max_tokens,
            model_id=model_id,
            background_enabled=bool(
                background_view is not None and getattr(background_view, "enabled", False)
            ),
            stickers=result.stickers,
            message_ref_allowlist=result.message_ref_allowlist,
            member_ref_allowlist=result.member_ref_allowlist,
            player_profile_targets=player_profile_targets,
            current_turn=self._current_turn_projection(
                inputs,
                visible_interrupted={
                    item.item_id: item.body
                    for item in result.compiled.items
                    if item.metadata.get("current_turn_shadow")
                },
            ),
            identity_context=result.identity_context,
            identity_catalog=result.identity_catalog,
            history_before_message_id=result.history_before_message_id,
            visible_history_summary_ids=result.visible_history_summary_ids,
            visible_message_ids=result.visible_message_ids,
            visible_history_fingerprints=result.visible_history_fingerprints,
            visible_recall_document_keys=frozenset(
                str(key)
                for item in result.compiled.items
                if item.source is ContextSource.HISTORY_FRAGMENT
                for key in item.metadata.get("document_keys", ())
                if str(key)
            ),
            visible_summary_coverage=result.visible_summary_coverage,
            has_searchable_earlier_history=(
                inputs.has_searchable_earlier_history
                or any(
                    item_id.startswith(("history-fragment:", "message:", "interrupted:"))
                    for item_id in result.compiled.report.dropped_item_ids
                )
            ),
        )

    def _current_turn_projection(
        self,
        inputs: ContextPreparationInputs,
        *,
        visible_interrupted: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        result = []
        for message in inputs.current_messages:
            for index, interrupted in enumerate(
                self.interrupted_expression_messages(message), start=1
            ):
                projection_id = f"interrupted:{message.message_id}:{index}"
                if projection_id not in visible_interrupted:
                    continue
                result.append(
                    {
                        "projection_id": projection_id,
                        "ledger_message_id": 0,
                        "speaker": "assistant",
                        "text": visible_interrupted[projection_id],
                        "occurred_at": interrupted.occurred_at,
                        "sender_name": str(inputs.identity_context.character_name or "")[:80],
                        "sender_id": "soulcore",
                        "member_ref": "",
                        "interrupted_unsent": True,
                        "interrupting_message_id": int(message.message_id),
                    }
                )
            sender_id = str(message.sender_id or "")
            result.append(
                {
                    "ledger_message_id": int(message.message_id),
                    "speaker": str(message.role or "user"),
                    "text": self.project_message(
                        message,
                        include_sender=False,
                        identity_context=inputs.identity_context,
                        identity_catalog=inputs.identity_catalog,
                    ),
                    "occurred_at": message.occurred_at,
                    "sender_name": self._current_sender_name(
                        message,
                        inputs.identity_context,
                    )[:80],
                    "sender_id": str(message.sender_id or ""),
                    "member_ref": inputs.member_ref_by_sender_id.get(sender_id, ""),
                }
            )
        return tuple(result)

    @staticmethod
    def _current_sender_name(message: Any, identity_context: Any) -> str:
        participant = identity_context.participant_by_id.get(str(message.sender_id or ""))
        if participant is not None and participant.display_name:
            return str(participant.display_name)
        return str(message.sender_name or "")

    @staticmethod
    def _compact_structured_row(row: Mapping[str, Any], *, fields: Sequence[str]) -> dict[str, Any]:
        """Project bounded structured resources without carrying source chat text."""

        result: dict[str, Any] = {}
        for name in fields:
            value = row.get(name)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, str):
                result[name] = value[:800]
            elif isinstance(value, (bool, int, float)):
                result[name] = value
            elif isinstance(value, (list, tuple)):
                result[name] = [str(item)[:200] for item in value[:8]]
            else:
                result[name] = str(value)[:200]
        return result

    async def maybe_enqueue_summary(
        self,
        profile_id: str,
        instance_id: str,
        role: Any,
    ) -> Any | None:
        if not await self.runtime_gate.is_enabled(profile_id, instance_id):
            return None
        summary = await self.repository.get_latest_dialogue_summary(profile_id, instance_id)
        covered_through = summary.covered_through_message_id if summary else None
        window = await self.dialogue_summary_window(
            profile_id,
            instance_id,
            covered_through_message_id=covered_through,
        )
        target = int(window.get("target_message_id") or 0)
        if target < 1:
            return None
        trigger = (await self.budget_config_for_profile(profile_id, role)).summary_trigger_tokens
        meter = ConservativeTokenMeter()
        messages, _has_older_dialogue = await self._load_dialogue_candidates_for_fill(
            profile_id,
            instance_id,
            after_message_id=covered_through,
            excluded_message_ids=(),
            fill_token_budget=trigger,
            meter=meter,
        )
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        include_sender = bool(instance and str(instance.scope) == "group")
        items: list[ContextItem] = []
        sequence = 0
        for message in messages:
            entries = (*self.interrupted_expression_messages(message), message)
            for index, entry in enumerate(entries):
                sequence += 1
                items.append(
                    ContextItem(
                        item_id=(
                            f"message:{message.message_id}"
                            if entry is message
                            else f"interrupted:{message.message_id}:{index + 1}"
                        ),
                        budget_class=BudgetClass.DATA,
                        source=ContextSource.CURRENT_DIALOGUE,
                        speaker=entry.role,
                        body=self.project_message(entry, include_sender=include_sender),
                        sequence=sequence,
                    )
                )
        if meter.measure(items).tokens < trigger:
            return None
        return await self.ai_tasks.create_ai_task(
            profile_id,
            "DIALOGUE_SUMMARY",
            instance_id=instance_id,
            task_class="BACKGROUND",
            capability="text.completion",
            priority=-20,
            mutex_key=f"dialogue-summary:{instance_id}",
            idempotency_key=f"dialogue-summary:{target}:{summary.summary_id if summary else 0}",
            input_data={
                "target_message_id": target,
                "base_summary_id": summary.summary_id if summary else None,
            },
            recovery_policy="RESUME_CHECKPOINT",
            max_attempts=DURABLE_AI_MAX_ATTEMPTS,
        )

    async def dialogue_summary_window(
        self,
        profile_id: str,
        instance_id: str,
        *,
        covered_through_message_id: int | None,
    ) -> dict[str, int | None]:
        return await self.repository.get_dialogue_summary_window(
            profile_id,
            instance_id,
            after_message_id=covered_through_message_id,
            keep_recent_turns=DIALOGUE_SUMMARY_RECENT_TURNS,
        )

    async def diagnostics(
        self,
        profile_id: str,
        instance_id: str,
        role: Any,
    ) -> dict[str, Any]:
        policy = await self.thinking_policy(profile_id)
        config = self.budget_config(role, policy)
        summary = await self.repository.get_latest_dialogue_summary(profile_id, instance_id)
        tasks = await self.ai_tasks.list_ai_tasks(
            profile_id=profile_id,
            instance_id=instance_id,
            task_type="DIALOGUE_SUMMARY",
            limit=1,
        )
        latest_build = await self.repository.get_context_build_report(profile_id, instance_id)
        return {
            "token_mode": latest_build.token_count_mode if latest_build else "ESTIMATED",
            "provider_context_window": None,
            "effective_max_tokens": (
                latest_build.hard_token_limit if latest_build else config.max_context_tokens
            ),
            "message_count": await self.repository.count_instance_messages(profile_id, instance_id),
            "summary": summary,
            "summary_task": tasks[0] if tasks else None,
            "latest_build": latest_build,
            "budget": self._budget_diagnostics(config),
            "thinking_complexity": policy.complexity.value,
        }

    @staticmethod
    def _budget_diagnostics(config: ContextBudgetConfig) -> dict[str, Any]:
        return {
            "max_context_tokens": config.max_context_tokens,
            "target_context_tokens": config.target_context_tokens,
            "fill_budget": config.fill_budget,
            "fill_ratio": config.fill_ratio,
            "source_fill_weights": {
                **{policy.source.value: policy.fill_weight for policy in DEFAULT_SOURCE_POLICIES},
                "background_material": BACKGROUND_FILL_WEIGHT,
            },
            "summary_output_limit": config.summary_output_limit,
            "summary_trigger_tokens": config.summary_trigger_tokens,
        }

    def project_message(
        self,
        message: ConversationMessage,
        *,
        include_sender: bool = False,
        media_projections: Sequence[dict[str, str]] = (),
        member_ref: str = "",
        identity_context: Any | None = None,
        identity_catalog: Any | None = None,
    ) -> str:
        text = self._project_message_text(
            message,
            identity_context=identity_context,
            identity_catalog=identity_catalog,
        )
        if message.components:
            text = self._with_component_projection(
                text,
                message.components,
                identity_context=identity_context,
                identity_catalog=identity_catalog,
            )
        if media_projections:
            text = self._with_media_projection(text, media_projections)
        if include_sender and str(message.role) == "user":
            text = self._with_sender_identity(text, message, member_ref=member_ref)
        memo = self._project_internal_memo(
            message,
            identity_context=identity_context,
            identity_catalog=identity_catalog,
        )
        if memo:
            text = self._with_internal_memo(text, memo)
        text = self._with_scene_narration(
            text,
            message,
            identity_context=identity_context,
            identity_catalog=identity_catalog,
        )
        if str(message.delivery_status or "").upper() == INTERRUPTED_UNSENT_STATUS:
            return f"[被打断，未发出] {text}".strip()
        return text

    @staticmethod
    def _with_scene_narration(
        text: str,
        message: ConversationMessage,
        *,
        identity_context: Any | None,
        identity_catalog: Any | None,
    ) -> str:
        metadata = getattr(message, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            return text

        def projected(field: str) -> list[str]:
            raw_values = metadata.get(field) or ()
            if isinstance(raw_values, str):
                raw_values = (raw_values,)
            values = [str(value or "").strip() for value in raw_values if str(value or "").strip()]
            if identity_context is not None:
                values = [
                    model_identity_projection(value, identity_context, identity_catalog)
                    for value in values
                ]
            return [f"【旁白】{value}" for value in values]

        before = projected("scene_narration_before")
        after = projected("scene_narration_after")
        return "\n".join([*before, *([text] if str(text).strip() else []), *after]).strip()

    @staticmethod
    def _project_message_text(
        message: ConversationMessage,
        *,
        identity_context: Any | None,
        identity_catalog: Any | None,
    ) -> str:
        template = str(getattr(message, "identity_template", "") or "").strip()
        text = str(message.plain_text or "").strip()
        if template and identity_context is not None:
            projected = model_identity_projection(template, identity_context, identity_catalog)
        elif not template:
            projected = escape_untrusted_identity_syntax(text)
        else:
            projected = text
        return projected

    @staticmethod
    def _project_internal_memo(
        message: ConversationMessage,
        *,
        identity_context: Any | None,
        identity_catalog: Any | None,
    ) -> str:
        memo = str(message.internal_memo or "").strip()
        if memo and identity_context is not None:
            return model_identity_projection(memo, identity_context, identity_catalog)
        return memo

    def interrupted_expression_messages(
        self, message: ConversationMessage
    ) -> tuple[ConversationMessage, ...]:
        """Project confirmed-unsent expressions as ordinary assistant timeline entries."""

        result: list[ConversationMessage] = []
        for expression in message.interrupted_expressions:
            content = str(expression.content or "").strip()
            result.append(
                ConversationMessage(
                    message_id=0,
                    profile_id=message.profile_id,
                    instance_id=message.instance_id,
                    direction=MessageDirection.OUTBOUND,
                    role="assistant",
                    internal_memo=str(expression.internal_memo or "").strip(),
                    expression_ordinal=int(expression.ordinal),
                    sender_id="soulcore",
                    sender_name="SoulCore",
                    plain_text=content
                    or self._interrupted_expression_placeholder(expression.expression_kind),
                    identity_template=content,
                    delivery_status=INTERRUPTED_UNSENT_STATUS,
                    occurred_at=message.occurred_at,
                    created_at=message.created_at,
                )
            )
        return tuple(result)

    @staticmethod
    def _interrupted_expression_placeholder(expression_kind: str) -> str:
        return {
            "IMAGE": "[原本想发送一张图片]",
            "STICKER": "[原本想发送一个表情包]",
            "FILE": "[原本想发送一个文件]",
        }.get(str(expression_kind or "").strip().upper(), "[原本还有一项表达]")

    @staticmethod
    def _with_internal_memo(
        text: str,
        memo: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Keep private continuity beside the exact visible message as one trim unit."""

        del metadata
        suffix = f"（留话：{str(memo).strip()}）"
        if not text:
            return suffix
        punctuation = "" if text.endswith(("。", "！", "？", ".", "!", "?")) else "。"
        return f"{text}{punctuation}{suffix}"

    def _with_component_projection(
        self,
        text: str,
        components: Sequence[dict[str, Any]],
        *,
        identity_context: Any | None = None,
        identity_catalog: Any | None = None,
    ) -> str:
        projected_components = [dict(item or {}) for item in components]
        if identity_context is not None:
            text = self._render_component_identities(
                text,
                projected_components,
                identity_context,
                identity_catalog,
            )
        non_text = [
            item
            for item in projected_components
            if str((item or {}).get("type") or (item or {}).get("kind") or "").lower()
            not in {"plain", "text"}
        ]
        projection_source = non_text if text else projected_components
        projected = str(self.projector.project(projection_source).content or "").strip()
        projected = self._missing_image_component_projection(text, projected)
        if not projected:
            return text
        return f"{text}\n{projected}".strip() if text else projected

    @staticmethod
    def _missing_image_component_projection(text: str, projected: str) -> str:
        """Avoid adding a second image marker already supplied by the adapter."""

        existing = len(_INBOUND_IMAGE_PLACEHOLDER.findall(str(text or "")))
        encountered = 0
        missing: list[str] = []
        for line in str(projected or "").splitlines():
            if _INBOUND_IMAGE_PLACEHOLDER.fullmatch(line.strip()):
                encountered += 1
                if encountered <= existing:
                    continue
            missing.append(line)
        return "\n".join(missing).strip()

    @staticmethod
    def _render_component_identities(
        text: str,
        components: Sequence[dict[str, Any]],
        identity_context: Any,
        identity_catalog: Any | None,
    ) -> str:
        participants = identity_context.participant_by_id
        for component in components:
            kind = str(component.get("type") or component.get("kind") or "").lower()
            participant_id = str(
                component.get("qq")
                or component.get("sender_id")
                or component.get("target_sender_id")
                or ""
            )
            participant = participants.get(participant_id)
            if kind == "at" and participant is not None:
                label = ConversationContextService._mention_identity_label(
                    participant_id,
                    participant,
                    identity_catalog,
                )
                text = text.replace("[提及一位群成员]", f"[提及{label}]", 1)
            if kind != "reply_reference":
                continue
            if str(component.get("target_role") or "").lower() == "assistant":
                component["target_sender_name"] = identity_context.character_name
            elif participant is not None:
                component["target_sender_name"] = participant.display_name
        return text

    @staticmethod
    def _mention_identity_label(
        participant_id: str,
        participant: Any,
        identity_catalog: Any | None,
    ) -> str:
        reference = (
            str(identity_catalog.group_participant_reference(participant_id) or "").strip()
            if identity_catalog is not None
            else ""
        )
        display_name = safe_model_identity(str(participant.display_name or ""))
        if reference and display_name:
            return f"{reference}（{display_name}）"
        return reference or display_name or "一位群成员"

    @staticmethod
    def _with_media_projection(text: str, media_projections: Sequence[dict[str, str]]) -> str:
        media_lines = [
            f"[图片内容] {item.get('text') or '图片描述暂不可用'}" for item in media_projections
        ]
        if not media_lines:
            return text

        remaining = iter(media_lines)
        replaced = 0

        def replace_placeholder(match: re.Match[str]) -> str:
            nonlocal replaced
            try:
                value = next(remaining)
            except StopIteration:
                return match.group(0)
            replaced += 1
            return value

        projected_text = _INBOUND_IMAGE_PLACEHOLDER.sub(replace_placeholder, text)
        media_text = "\n".join(media_lines[replaced:])
        if media_text not in text:
            return f"{projected_text}\n{media_text}".strip()
        return projected_text

    @staticmethod
    def _with_sender_identity(text: str, message: Any, *, member_ref: str = "") -> str:
        sender_name = str(message.sender_name or "").strip()
        if not sender_name:
            return text
        suffix = f" | member_ref：{member_ref}" if member_ref else ""
        return f"[发送者：{sender_name}{suffix}]\n{text}"

    @staticmethod
    def _current_player_text(run_prompt: str) -> str:
        text = str(run_prompt or "")
        trigger_prefix = "<本轮触发信息>\n[[内容]]: "
        if text.startswith(trigger_prefix):
            end = text.find("\n</本轮触发信息>")
            if end >= 0:
                return unescape(text[len(trigger_prefix) : end]).strip()
        markers = (
            "\n\n<主动联系缘由>",
            "\n\n<本轮表达边界>",
            "\n\n<本轮必须处理的成果>",
            "\n\n<本轮可提交检查的表情包来源>",
            "\n\n<当前消息的称呼与引用关系>",
        )
        indexes = [index for marker in markers if (index := text.find(marker)) >= 0]
        return text[: min(indexes)].strip() if indexes else text.strip()


__all__ = [
    "ConversationContextService",
    "ConversationHistoryPage",
    "PreparedMainCoreContext",
]

"""Ordered action execution support for the Main Core text loop."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..ai.service import (
    CommandExecutionResult,
    CommandProtocolError,
    MainCoreCommandRegistry,
    execute_nonterminal_batch,
    register_result_references,
    terminal_decision,
)
from .agent_protocol import command_results_text
from .command_context import _active
from .roleplay_prompt import ExecutionRound
from .terminal_decision import validate_main_core_response_segment
from .text_command_runtime import NO_NEW_MAIN_CORE_ACTION


@dataclass(slots=True)
class _OrderedBatchStateSnapshot:
    collector_values: dict[str, Any]
    completed_nonterminal_results: dict[str, CommandExecutionResult]
    sticker_context: Any | None
    sticker_state: Any | None
    timer_context: Any | None
    timer_values: dict[str, Any]
    player_target_values: tuple[tuple[Any, dict[str, Any]], ...]


class OrderedCommandExecutionMixin:
    """Execute one action batch after its terminal boundary has been normalized."""

    def _without_replayed_nonterminal_commands(
        self,
        validated: list[Any],
        commands: Any,
    ) -> tuple[list[Any], tuple[Any, ...], list[CommandExecutionResult]]:
        """Consume deterministic no-op replays without another tool call."""

        replayed_ordinals: set[int] = set()
        results: list[CommandExecutionResult] = []
        fresh: list[Any] = []
        pending_fingerprints: set[str] = set()
        for item in validated:
            if item.spec.terminal:
                fresh.append(item)
                continue
            fingerprint = self._nonterminal_command_fingerprint(item)
            previous = self._completed_nonterminal_results.get(fingerprint)
            if previous is not None:
                replayed_ordinals.add(item.parsed.ordinal)
                results.append(
                    CommandExecutionResult(
                        item.parsed.ordinal,
                        item.spec.name,
                        previous.ok,
                        (
                            "这项行动已经在本次运行中执行过；系统沿用此前完整结果，"
                            "没有重复调用。\n"
                            f"{previous.content}"
                        ).strip(),
                        previous.media_asset_ids,
                        previous.references,
                        {**dict(previous.diagnostic), "replayed": True},
                        previous.public_references,
                        previous.model_input_images,
                    )
                )
                continue
            if fingerprint in pending_fingerprints:
                replayed_ordinals.add(item.parsed.ordinal)
                results.append(
                    CommandExecutionResult(
                        item.parsed.ordinal,
                        item.spec.name,
                        True,
                        "同批中完全相同的行动已合并；请使用本批第一次调用返回的结果。",
                        diagnostic={"replayed": True},
                    )
                )
                continue
            pending_fingerprints.add(fingerprint)
            fresh.append(item)
        return (
            fresh,
            tuple(item for item in commands if item.ordinal not in replayed_ordinals),
            results,
        )

    @staticmethod
    def _nonterminal_command_fingerprint(item: Any) -> str:
        payload = {
            "command": str(item.spec.internal_name or item.spec.name),
            "arguments": dict(item.arguments),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _remember_nonterminal_results(
        self,
        segment: list[Any],
        results: tuple[CommandExecutionResult, ...],
    ) -> None:
        by_ordinal = {item.ordinal: item for item in results}
        for item in segment:
            result = by_ordinal.get(item.parsed.ordinal)
            if result is None:
                continue
            self._completed_nonterminal_results[self._nonterminal_command_fingerprint(item)] = (
                result
            )

    async def _execute_validated_model_turn(
        self,
        *,
        parsed: Any,
        commands: Any,
        registry: MainCoreCommandRegistry,
        validated_by_ordinal: dict[int, Any],
        validation_results: Any,
        calls: tuple[str, ...],
        reference_map: dict[str, Any],
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        round_number: int,
        raw_text: str,
        command_set: Any,
        event: Any,
        max_parallel: int,
        model_gateway: Any,
        run_id: int,
        runtime_gate: Any | None,
        profile_id: str,
        instance_id: str,
    ) -> bool:
        has_nonterminal = self._has_nonterminal_command(commands, registry)
        snapshot = self._snapshot_ordered_batch_state()
        results = list(validation_results)
        executed, completed = await self._execute_ordered_segments(
            commands=commands,
            registry=registry,
            reference_map=reference_map,
            validated_by_ordinal=validated_by_ordinal,
            results=results,
            snapshot=snapshot,
            has_nonterminal=has_nonterminal,
            command_set=command_set,
            event=event,
            max_parallel=max_parallel,
            model_gateway=model_gateway,
            run_id=run_id,
            round_number=round_number,
            runtime_gate=runtime_gate,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        self._append_execution_round(
            parsed=parsed,
            calls=calls,
            results=results,
            executed=executed,
            reference_map=reference_map,
            rounds=rounds,
            rejection_errors=rejection_errors,
            round_number=round_number,
            raw_text=raw_text,
        )
        return completed

    @staticmethod
    def _has_nonterminal_command(
        commands: Any,
        registry: MainCoreCommandRegistry,
    ) -> bool:
        return any(
            not bool((spec := registry.get(item.name)) and spec.terminal) for item in commands
        )

    async def _execute_ordered_segments(
        self,
        *,
        commands: Any,
        registry: MainCoreCommandRegistry,
        reference_map: dict[str, Any],
        validated_by_ordinal: dict[int, Any],
        results: list[CommandExecutionResult],
        snapshot: _OrderedBatchStateSnapshot,
        has_nonterminal: bool,
        command_set: Any,
        event: Any,
        max_parallel: int,
        model_gateway: Any,
        run_id: int,
        round_number: int,
        runtime_gate: Any | None,
        profile_id: str,
        instance_id: str,
    ) -> tuple[bool, bool]:
        executed = False
        completed = False
        for is_terminal, ordinals in self._ordered_command_segments(commands, registry):
            segment = self._validated_segment(ordinals, validated_by_ordinal)
            if not segment:
                continue
            if is_terminal:
                outcome = await self._execute_terminal_segment(
                    segment=segment,
                    snapshot=snapshot,
                    has_nonterminal=has_nonterminal,
                    command_set=command_set,
                    event=event,
                    model_gateway=model_gateway,
                    run_id=run_id,
                    round_number=round_number,
                    runtime_gate=runtime_gate,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    reference_map=reference_map,
                )
                executed = executed or outcome[0]
                completed = outcome[1]
                if outcome[2]:
                    results.append(outcome[2])
                    break
                if outcome[3]:
                    results.append(outcome[3])
                continue
            await self._execute_nonterminal_segment(
                segment=segment,
                results=results,
                snapshot=snapshot,
                event=event,
                max_parallel=max_parallel,
                model_gateway=model_gateway,
                run_id=run_id,
                round_number=round_number,
                runtime_gate=runtime_gate,
                profile_id=profile_id,
                instance_id=instance_id,
            )
            executed = True
        return executed, completed

    @staticmethod
    def _validated_segment(
        ordinals: list[int],
        validated_by_ordinal: dict[int, Any],
    ) -> list[Any]:
        return [
            validated_by_ordinal[ordinal] for ordinal in ordinals if ordinal in validated_by_ordinal
        ]

    async def _execute_nonterminal_segment(
        self,
        *,
        segment: list[Any],
        results: list[CommandExecutionResult],
        snapshot: _OrderedBatchStateSnapshot,
        event: Any,
        max_parallel: int,
        model_gateway: Any,
        run_id: int,
        round_number: int,
        runtime_gate: Any | None,
        profile_id: str,
        instance_id: str,
    ) -> None:
        try:
            executed_results = await execute_nonterminal_batch(
                segment,
                event=event,
                max_parallel=max_parallel,
                scope_factory=lambda item: self._command_trace_scope(
                    model_gateway,
                    item,
                    run_id=run_id,
                    round_number=round_number,
                    runtime_gate=runtime_gate,
                    profile_id=profile_id,
                    instance_id=instance_id,
                ),
            )
            self._remember_nonterminal_results(segment, executed_results)
            results.extend(executed_results)
        except BaseException:
            self._restore_ordered_batch_state(snapshot)
            raise

    async def _execute_terminal_segment(
        self,
        *,
        segment: list[Any],
        snapshot: _OrderedBatchStateSnapshot,
        has_nonterminal: bool,
        command_set: Any,
        event: Any,
        model_gateway: Any,
        run_id: int,
        round_number: int,
        runtime_gate: Any | None,
        profile_id: str,
        instance_id: str,
        reference_map: dict[str, Any],
    ) -> tuple[
        bool,
        bool,
        CommandExecutionResult | None,
        CommandExecutionResult | None,
    ]:
        if has_nonterminal:
            raise CommandProtocolError("最终表达不能与非终止行动出现在同一个通道")
        decision = terminal_decision(segment, reference_map)
        try:
            commit_error = await self._commit_terminal_segment(
                terminal=segment,
                decision=decision,
                command_set=command_set,
                event=event,
                model_gateway=model_gateway,
                run_id=run_id,
                round_number=round_number,
                runtime_gate=runtime_gate,
                profile_id=profile_id,
                instance_id=instance_id,
            )
        except BaseException:
            self._restore_ordered_batch_state(snapshot)
            raise
        if commit_error:
            self._restore_ordered_batch_state(snapshot)
            return (
                False,
                False,
                CommandExecutionResult(
                    segment[0].parsed.ordinal,
                    "表达提交",
                    False,
                    commit_error,
                ),
                None,
            )
        return True, True, None, None

    def _append_execution_round(
        self,
        *,
        parsed: Any,
        calls: tuple[str, ...],
        results: list[CommandExecutionResult],
        executed: bool,
        reference_map: dict[str, Any],
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        round_number: int,
        raw_text: str,
    ) -> None:
        results.sort(key=lambda item: item.ordinal)
        normalized_results = tuple(register_result_references(results, reference_map))
        rejection = ""
        if executed:
            rejection_errors.clear()
        else:
            new_failures = tuple(
                item.content
                for item in normalized_results
                if not item.ok and not bool(item.diagnostic.get("replayed"))
            )
            if new_failures:
                rejection = "；".join(new_failures)
            elif any(item.diagnostic.get("replayed") for item in normalized_results):
                rejection = NO_NEW_MAIN_CORE_ACTION
            else:
                rejection = (
                    "；".join(item.content for item in normalized_results)
                    or "本步骤没有可执行的合法指令。"
                )
        rounds.append(
            ExecutionRound(
                round_number,
                parsed.working_text,
                calls,
                normalized_results,
                rejection=rejection,
                raw_text=raw_text,
                result_text=(
                    command_results_text(normalized_results)
                    or (
                        "最终表达：已通过预检\n将作为本次行动的最终提交。"
                        if executed
                        else rejection
                    )
                ),
            )
        )
        if rejection:
            self._reject(rejection_errors, rejection, rounds)

    def _ordered_batch_collector(self) -> Any:
        collector = getattr(self, "_batch_collector", None)
        return collector if collector is not None else _active()

    def _snapshot_ordered_batch_state(self) -> _OrderedBatchStateSnapshot:
        collector = self._ordered_batch_collector()
        collector_values = {
            name: deepcopy(getattr(collector, name))
            for name in (
                "selected_media_asset_ids",
                "selected_sticker_ref_ids",
                "sticker_reinforcements",
                "file_generation_requests",
                "selected_important_todo_refs",
                "player_profile_mutations",
                "player_profile_mutation_fingerprints",
            )
            if hasattr(collector, name)
        }
        sticker_context = getattr(collector, "sticker_command_context", None)
        sticker_state = (
            sticker_context.snapshot_batch_state()
            if sticker_context is not None
            and callable(getattr(sticker_context, "snapshot_batch_state", None))
            else None
        )
        timer_context = getattr(collector, "timer_command_context", None)
        timer_values = {
            name: deepcopy(getattr(timer_context, name))
            for name in (
                "creations",
                "managements",
                "revisions",
                "_creation_fingerprints",
                "_management_fingerprints",
                "_revision_fingerprints",
            )
            if timer_context is not None and hasattr(timer_context, name)
        }
        player_target_values = tuple(
            (target, self._snapshot_player_target(target))
            for target in dict(getattr(collector, "player_profile_targets", {}) or {}).values()
        )
        return _OrderedBatchStateSnapshot(
            collector_values=collector_values,
            completed_nonterminal_results=deepcopy(self._completed_nonterminal_results),
            sticker_context=sticker_context,
            sticker_state=sticker_state,
            timer_context=timer_context,
            timer_values=timer_values,
            player_target_values=player_target_values,
        )

    @staticmethod
    def _snapshot_player_target(target: Any) -> dict[str, Any]:
        return {
            name: deepcopy(getattr(target, name))
            for name in (
                "virtual_snapshot",
                "virtual_profile_version",
                "virtual_entry_versions",
                "entry_ref_map",
            )
            if hasattr(target, name)
        }

    def _restore_ordered_batch_state(
        self,
        snapshot: _OrderedBatchStateSnapshot,
    ) -> None:
        collector = self._ordered_batch_collector()
        self._completed_nonterminal_results = deepcopy(snapshot.completed_nonterminal_results)
        for name, value in snapshot.collector_values.items():
            setattr(collector, name, deepcopy(value))
        if snapshot.sticker_context is not None and snapshot.sticker_state is not None:
            snapshot.sticker_context.restore_batch_state(snapshot.sticker_state)
        if snapshot.timer_context is not None:
            for name, value in snapshot.timer_values.items():
                setattr(snapshot.timer_context, name, deepcopy(value))
        for target, values in snapshot.player_target_values:
            for name, value in values.items():
                setattr(target, name, deepcopy(value))

    async def _commit_terminal_segment(
        self,
        *,
        terminal: list[Any],
        decision: dict[str, Any],
        command_set: Any,
        event: Any,
        model_gateway: Any,
        run_id: int,
        round_number: int,
        runtime_gate: Any | None,
        profile_id: str,
        instance_id: str,
    ) -> str:
        await self._require_enabled(runtime_gate, profile_id, instance_id)
        async with self._terminal_trace_scope(
            model_gateway,
            terminal,
            run_id=run_id,
            round_number=round_number,
        ) as trace_state:
            try:
                commit_result = await self._commit_terminal(
                    command_set,
                    event,
                    decision,
                )
            except (CommandProtocolError, TypeError, ValueError) as exc:
                commit_result = f"error: {exc}"
            trace_state["ok"] = not bool(commit_result)
            trace_state["result"] = str(commit_result or "已原子写入表达时间线")
        return str(commit_result or "").removeprefix("error:").strip()

    @staticmethod
    def _ordered_command_segments(
        commands: Any,
        registry: MainCoreCommandRegistry,
    ) -> list[tuple[bool, list[int]]]:
        segments: list[tuple[bool, list[int]]] = []
        for item in commands:
            spec = registry.get(item.name)
            terminal = bool(spec is not None and spec.terminal)
            if not segments or segments[-1][0] != terminal:
                segments.append((terminal, []))
            segments[-1][1].append(item.ordinal)
        return segments

    @staticmethod
    async def _terminal_preflight_error(
        commands: Any,
        registry: MainCoreCommandRegistry,
        validated_by_ordinal: dict[int, Any],
        *,
        reference_map: dict[str, Any],
    ) -> str:
        for is_terminal, ordinals in OrderedCommandExecutionMixin._ordered_command_segments(
            commands,
            registry,
        ):
            if is_terminal:
                error = await OrderedCommandExecutionMixin._terminal_segment_preflight_error(
                    ordinals=ordinals,
                    validated_by_ordinal=validated_by_ordinal,
                    reference_map=reference_map,
                )
                if error:
                    return error
        return ""

    @staticmethod
    async def _terminal_segment_preflight_error(
        *,
        ordinals: list[int],
        validated_by_ordinal: dict[int, Any],
        reference_map: dict[str, Any],
    ) -> str:
        segment = OrderedCommandExecutionMixin._validated_segment(
            ordinals,
            validated_by_ordinal,
        )
        # Terminal-only expressions with an unusable visible send are rejected
        # before preflight.  An empty segment here can still represent a failed
        # non-send terminal action, which normal round settlement reports.
        if not segment:
            return ""
        try:
            decision = terminal_decision(segment, reference_map)
            error = await validate_main_core_response_segment(**decision)
            if error:
                return str(error).removeprefix("error:").strip()
        except (CommandProtocolError, TypeError, ValueError) as exc:
            return str(exc)
        return ""


__all__ = ["OrderedCommandExecutionMixin"]

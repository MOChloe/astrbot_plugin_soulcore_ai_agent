from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .prompt_debug import prompt_jsonable

T = TypeVar("T")
logger = logging.getLogger(__name__)


class StructuredOutputRejectedThreeTimes(RuntimeError):
    code = "model_output_rejected_three_times"

    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__(self.code)
        self.errors = errors


@runtime_checkable
class _ExchangeAnnotationPort(Protocol):
    async def annotate_model_exchange(
        self,
        invocation_id: str,
        *,
        round_no: int,
        processing: Mapping[str, object],
    ) -> None: ...


@runtime_checkable
class _CompletionTrace(Protocol):
    invocation_id: str


@dataclass(frozen=True, slots=True)
class ValidatedText(Generic[T]):
    completion: Any
    value: T
    rounds: int
    rejections: tuple[str, ...]


async def run_structured_text_session(
    *,
    model_gateway: Any,
    invoke: Callable[[int, str], Awaitable[Any]],
    validate: Callable[[str], T],
    maximum_rounds: int = 3,
) -> ValidatedText[T]:
    """Retry model-visible validation, never transport failures, at most three rounds."""

    errors: list[str] = []
    rounds = max(1, min(3, int(maximum_rounds)))
    for round_no in range(1, rounds + 1):
        feedback = _feedback(errors[-1]) if errors else ""
        completion = await invoke(round_no, feedback)
        try:
            value = validate(str(completion.text or ""))
        except (TypeError, ValueError) as exc:
            error = str(exc).strip() or type(exc).__name__
            errors.append(error)
            await record_structured_rejection(
                model_gateway=model_gateway,
                completion=completion,
                round_no=round_no,
                error=error,
                terminal=round_no == rounds,
            )
            if round_no == rounds:
                raise StructuredOutputRejectedThreeTimes(tuple(errors)) from exc
            continue
        await record_structured_acceptance(
            model_gateway=model_gateway,
            completion=completion,
            round_no=round_no,
            value=value,
        )
        return ValidatedText(
            completion=completion,
            value=value,
            rounds=round_no,
            rejections=tuple(errors),
        )
    raise AssertionError("structured validation loop did not terminate")


async def record_structured_rejection(
    *,
    model_gateway: Any,
    completion: Any,
    round_no: int,
    error: str,
    terminal: bool,
    extra_processing: Mapping[str, object] | None = None,
) -> None:
    if not (
        isinstance(model_gateway, _ExchangeAnnotationPort)
        and isinstance(completion, _CompletionTrace)
    ):
        return
    try:
        processing: dict[str, object] = {
            "rejection": str(error),
            "terminal_rejection": bool(terminal),
            "validation_round": max(1, int(round_no)),
        }
        processing.update(dict(extra_processing or {}))
        await model_gateway.annotate_model_exchange(
            str(completion.invocation_id or ""),
            round_no=max(1, int(round_no)),
            processing=processing,
        )
    except Exception:
        logger.exception("failed to record structured-output rejection")


async def record_structured_acceptance(
    *,
    model_gateway: Any,
    completion: Any,
    round_no: int,
    value: Any,
    normalizations: tuple[str, ...] = (),
    extra_processing: Mapping[str, object] | None = None,
) -> None:
    if not (
        isinstance(model_gateway, _ExchangeAnnotationPort)
        and isinstance(completion, _CompletionTrace)
    ):
        return
    try:
        processing: dict[str, object] = {
            "accepted": True,
            "validation_status": "ACCEPTED",
            "validation_round": max(1, int(round_no)),
            "validated_output": prompt_jsonable(value),
        }
        if normalizations:
            processing["normalizations"] = [
                {"action": str(action)} for action in dict.fromkeys(normalizations)
            ]
        processing.update(dict(extra_processing or {}))
        await model_gateway.annotate_model_exchange(
            str(completion.invocation_id or ""),
            round_no=max(1, int(round_no)),
            processing=processing,
        )
    except Exception:
        logger.exception("failed to record structured-output acceptance")


def _feedback(error: str) -> str:
    return f"上次写出的内容不符合要求，不能沿用。请根据下面的问题重新写出完整结果：\n{error}"


__all__ = [
    "StructuredOutputRejectedThreeTimes",
    "ValidatedText",
    "record_structured_acceptance",
    "record_structured_rejection",
    "run_structured_text_session",
]

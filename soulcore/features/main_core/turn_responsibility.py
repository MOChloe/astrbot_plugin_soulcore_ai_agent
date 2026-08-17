"""Model-visible situation classification for one Main Core invocation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...contracts.models import CoreWakeRequest, WakeSource
from ...shared.contact_runtime import is_proactive_contact_request
from ..files.service import is_file_recovery_wake


class MainCoreTurnKind(StrEnum):
    MESSAGE = "MESSAGE"
    SELF_INITIATED = "SELF_INITIATED"
    SCHEDULED = "SCHEDULED"
    RESUMED = "RESUMED"


@dataclass(frozen=True, slots=True)
class MainCoreTurnResponsibility:
    kind: MainCoreTurnKind

    @property
    def has_current_message(self) -> bool:
        return self.kind is MainCoreTurnKind.MESSAGE

    @property
    def self_initiated(self) -> bool:
        return self.kind is MainCoreTurnKind.SELF_INITIATED

    @property
    def uses_self_initiated_mode(self) -> bool:
        return self.kind in {
            MainCoreTurnKind.SELF_INITIATED,
            MainCoreTurnKind.SCHEDULED,
        }

    @property
    def scheduled(self) -> bool:
        return self.kind is MainCoreTurnKind.SCHEDULED

    @property
    def resumed(self) -> bool:
        return self.kind is MainCoreTurnKind.RESUMED


DEFAULT_MESSAGE_RESPONSIBILITY = MainCoreTurnResponsibility(
    kind=MainCoreTurnKind.MESSAGE,
)


def resolve_turn_responsibility(request: CoreWakeRequest) -> MainCoreTurnResponsibility:
    """Resolve transport/runtime signals once, before any Prompt is composed."""

    foreground_message = request.source in {
        WakeSource.FOREGROUND_MESSAGE,
        WakeSource.DEFERRED_MESSAGE,
    }
    if request.source is WakeSource.TIMER:
        kind = MainCoreTurnKind.SCHEDULED
    elif is_file_recovery_wake(request.source, request.metadata):
        kind = MainCoreTurnKind.RESUMED
    elif is_proactive_contact_request(request.metadata):
        kind = MainCoreTurnKind.SELF_INITIATED
    elif foreground_message:
        kind = MainCoreTurnKind.MESSAGE
    else:
        kind = MainCoreTurnKind.RESUMED
    return MainCoreTurnResponsibility(kind=kind)


__all__ = [
    "DEFAULT_MESSAGE_RESPONSIBILITY",
    "MainCoreTurnKind",
    "MainCoreTurnResponsibility",
    "resolve_turn_responsibility",
]

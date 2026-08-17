"""Pure domain surface for SoulCore's standalone Timer feature."""

from .constants import (
    MAX_CREATE_ACTIONS_PER_RUN,
    MAX_MANAGE_ACTIONS_PER_RUN,
    MAX_NONTERMINAL_OCCURRENCES_PER_INSTANCE,
    MAX_NONTERMINAL_RULES_PER_INSTANCE,
    MAX_PROMPT_CHARS,
)
from .contracts import (
    CreateTimerCommand,
    CreateTimerOutcome,
    CreateTimerResult,
    ManageTimerAction,
    ManageTimerCommand,
    ManageTimerOutcome,
    ManageTimerResult,
    PreparedTimerCreation,
    ReviseTimerCommand,
    ReviseTimerResult,
    RollOccurrenceCommand,
)
from .domain import (
    AbsoluteTimerRule,
    DeliveryAssociationRef,
    ExecutionEnvelopeRef,
    IdempotencyKey,
    OccurrenceStableRef,
    OpaqueTimerRef,
    RelativeTimerRule,
    SourceMessageRef,
    SourceRunRef,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerOccurrenceStatus,
    TimerRule,
    TimerRuleId,
    TimerRuleKind,
    TimerRuleRevision,
    TimerRuleStatus,
    TimerScope,
    WeeklyTimerRule,
    YearlyTimerRule,
    normalize_prompt,
)
from .errors import TimerDomainError, TimerErrorCode
from .natural_time import (
    ArrangementChangeKind,
    ArrangementChangeResolution,
    NaturalTimeCandidate,
    NaturalTimeResolution,
    NaturalTimeStatus,
    interpret_arrangement_change,
    interpret_natural_time,
)
from .projection import (
    TimerCandidateProjection,
    TimerProjectionSource,
    TimerRefTarget,
    project_candidates,
)
from .repository import (
    AdvanceOccurrenceCommand,
    InstanceOccupancy,
    InstanceOccupancyKind,
    InstanceOccupancyStatus,
    MutateClaimedOccurrenceCommand,
    OccurrenceMutationResult,
    OccurrencePage,
    RulePage,
)
from .rules import (
    OccurrenceRollPlan,
    canonical_rule,
    exact_timer_fingerprint,
    next_occurrence,
    parse_and_normalize_rule,
    plan_occurrence_roll,
    resolve_wall_time,
)
from .transitions import (
    OccurrenceAction,
    RuleAction,
    transition_occurrence,
    transition_rule,
)

__all__ = (
    ("AbsoluteTimerRule", "AdvanceOccurrenceCommand", "ArrangementChangeKind")
    + ("ArrangementChangeResolution",)
    + ("CreateTimerCommand", "CreateTimerOutcome", "CreateTimerResult")
    + ("DeliveryAssociationRef", "ExecutionEnvelopeRef", "IdempotencyKey")
    + ("InstanceOccupancy", "InstanceOccupancyKind")
    + ("InstanceOccupancyStatus", "MAX_CREATE_ACTIONS_PER_RUN")
    + ("MAX_MANAGE_ACTIONS_PER_RUN",)
    + ("MAX_NONTERMINAL_OCCURRENCES_PER_INSTANCE", "MAX_NONTERMINAL_RULES_PER_INSTANCE")
    + ("MAX_PROMPT_CHARS", "ManageTimerAction", "ManageTimerCommand", "ManageTimerOutcome")
    + ("ManageTimerResult", "MutateClaimedOccurrenceCommand", "OccurrenceAction")
    + ("OccurrenceMutationResult", "OccurrencePage", "OccurrenceRollPlan")
    + ("OccurrenceStableRef",)
    + ("OpaqueTimerRef", "PreparedTimerCreation", "RelativeTimerRule", "RollOccurrenceCommand")
    + ("ReviseTimerCommand", "ReviseTimerResult")
    + ("RuleAction", "RulePage")
    + ("SourceMessageRef", "SourceRunRef")
    + ("TimerCandidateProjection", "TimerDomainError", "TimerErrorCode")
    + ("TimerOccurrence", "TimerOccurrenceId", "TimerOccurrenceStatus", "TimerProjectionSource")
    + ("TimerRefTarget", "TimerRule", "TimerRuleId", "TimerRuleKind", "TimerRuleRevision")
    + ("TimerRuleStatus",)
    + ("TimerScope", "WeeklyTimerRule", "YearlyTimerRule", "canonical_rule")
    + ("exact_timer_fingerprint", "next_occurrence", "normalize_prompt")
    + ("parse_and_normalize_rule", "plan_occurrence_roll")
    + ("project_candidates", "resolve_wall_time", "transition_occurrence", "transition_rule")
    + ("NaturalTimeCandidate", "NaturalTimeResolution", "NaturalTimeStatus")
    + ("interpret_arrangement_change", "interpret_natural_time")
)

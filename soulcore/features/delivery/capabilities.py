"""Platform-neutral delivery capability facts used by the QPM coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .qpm import QPMBucketKey, QPMBucketLimit, QPMBucketScope

DEFAULT_GROUP_QPM = 20
QQ_OFFICIAL_FORMAL_GROUP_QPM = 20
QQ_OFFICIAL_UNCERTIFIED_ACCOUNT_QPM = 30
QQ_OFFICIAL_CERTIFIED_ACCOUNT_QPM = 60


class QQEnvironment(StrEnum):
    FORMAL = "formal"
    SANDBOX = "sandbox"


class QQAccountTier(StrEnum):
    UNCERTIFIED = "uncertified"
    CERTIFIED = "certified"


@dataclass(frozen=True, slots=True)
class PhysicalDeliveryReceipt:
    """An adapter/API acceptance receipt for one physical message fragment.

    A receipt proves only that the platform returned an addressable message
    identifier.  It deliberately does not claim client delivery.
    """

    platform_message_id: str
    fragment_ordinal: int = 0
    platform_id: str = ""
    adapter_name: str = ""
    accepted_unconfirmed: bool = True
    # QQ Official separates the addressable message id (retraction/passive
    # authorization) from its REFIDX reference index (native quoting).
    platform_reference_id: str = ""
    native_reply_supported: bool = False
    member_mention_supported: bool = False
    self_retraction_supported: bool = False
    returns_platform_message_id: bool = False
    retractable_for_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class DeliveryCapability:
    platform_id: str
    adapter_name: str
    qq_official: bool
    onebot: bool
    personal_wechat: bool = False
    qq_environment: QQEnvironment = QQEnvironment.FORMAL
    qq_account_tier: QQAccountTier = QQAccountTier.UNCERTIFIED
    qq_account_identity: str = ""
    account_identity_confirmed: bool = False
    quote: bool = False
    mention: bool = False
    retract_self: bool = False
    returns_id: bool = False
    inbound_recall_notice: bool = False

    @property
    def sandbox(self) -> bool:
        return self.qq_official and self.qq_environment is QQEnvironment.SANDBOX

    @property
    def autonomous_contact_allowed(self) -> bool:
        """Unknown adapters keep passive replies but deny autonomous sends."""

        return self.qq_official or self.onebot or self.personal_wechat

    def supports_route_kind(self, route_kind: str) -> bool:
        """Return whether this adapter can address the normalized route kind."""

        return not self.personal_wechat or str(route_kind) == "friend"

    def receipt(
        self,
        platform_message_id: str,
        fragment_ordinal: int = 0,
        *,
        platform_reference_id: str = "",
        accepted_unconfirmed: bool = True,
    ) -> PhysicalDeliveryReceipt:
        """Bind an adapter receipt to capabilities detected from the live adapter."""

        reference_id = str(platform_reference_id or "")
        return PhysicalDeliveryReceipt(
            platform_message_id=str(platform_message_id),
            fragment_ordinal=max(0, int(fragment_ordinal)),
            platform_id=self.platform_id,
            adapter_name=self.adapter_name,
            accepted_unconfirmed=bool(accepted_unconfirmed),
            platform_reference_id=reference_id,
            native_reply_supported=bool(self.quote and (not self.qq_official or reference_id)),
            member_mention_supported=bool(self.mention),
            self_retraction_supported=bool(self.retract_self),
            returns_platform_message_id=bool(self.returns_id),
            retractable_for_seconds=(120 if self.qq_official and self.retract_self else None),
        )

    def effective_group_qpm(self, configured: int = DEFAULT_GROUP_QPM) -> int:
        requested = max(1, int(configured))
        if self.qq_official and not self.sandbox:
            return min(requested, QQ_OFFICIAL_FORMAL_GROUP_QPM)
        return requested

    def group_bucket(self, target_id: str, configured: int) -> QPMBucketLimit:
        # The profile id is intentionally absent: all profiles addressing the
        # same AstrBot connection and physical target share this ledger.
        identity = f"{self.adapter_name}:{self.platform_id}:group:{target_id}"
        return QPMBucketLimit(
            QPMBucketKey(QPMBucketScope.GROUP, identity),
            self.effective_group_qpm(configured),
        )

    def proactive_account_bucket(self) -> QPMBucketLimit | None:
        if not self.qq_official or self.sandbox:
            return None
        limit = (
            QQ_OFFICIAL_CERTIFIED_ACCOUNT_QPM
            if self.qq_account_tier is QQAccountTier.CERTIFIED
            else QQ_OFFICIAL_UNCERTIFIED_ACCOUNT_QPM
        )
        # An account-wide platform quota must never be split merely because
        # AstrBot exposes the same QQ account through multiple connections.
        # Missing app identity therefore falls back to one conservative shared
        # bucket instead of an unsafe per-connection bucket.
        identity = self.qq_account_identity or "qq-official:unconfirmed-account"
        return QPMBucketLimit(
            QPMBucketKey(QPMBucketScope.QQ_ACCOUNT_PROACTIVE, identity),
            limit,
        )

"""Evidence taxonomy.

Every exhibit in a dispute is reduced to a common shape so the ledger can treat
merchant fulfilment logs and card-member photographs with the same arithmetic.

The three quantities that matter per exhibit:

    lambda_lr  log-likelihood ratio  log[ P(e|H_CM) / P(e|H_M) ]
    quality    authenticity in [0,1] — can we trust this artefact at all
    weight     reason-code-specific relevance multiplier

Their product is the exhibit's signed contribution to the posterior, in nats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Party(str, Enum):
    """Who filed an exhibit, and who a verdict can favour."""

    CARD_MEMBER = "card_member"
    MERCHANT = "merchant"
    NETWORK = "network"
    """Network-sourced facts (auth logs, AVS, carrier APIs) belong to neither side."""

    @property
    def opponent(self) -> "Party":
        if self is Party.CARD_MEMBER:
            return Party.MERCHANT
        if self is Party.MERCHANT:
            return Party.CARD_MEMBER
        return Party.NETWORK


class EvidenceType(str, Enum):
    """Canonical exhibit kinds.

    Grouped by who normally produces them, though `Evidence.party` is what actually
    binds an exhibit to a filer — a card member can produce a delivery receipt too.
    """

    # --- fulfilment / delivery ---
    CARRIER_TRACKING = "carrier_tracking"
    DELIVERY_CONFIRMATION = "delivery_confirmation"
    SIGNATURE_PROOF = "signature_proof"
    SHIPPING_LABEL = "shipping_label"

    # --- commercial record ---
    INVOICE = "invoice"
    RECEIPT = "receipt"
    ORDER_CONFIRMATION = "order_confirmation"
    TERMS_OF_SERVICE = "terms_of_service"
    REFUND_POLICY = "refund_policy"
    CANCELLATION_POLICY = "cancellation_policy"

    # --- money movement ---
    REFUND_RECORD = "refund_record"
    CREDIT_NOTE = "credit_note"
    PRIOR_SETTLEMENT = "prior_settlement"

    # --- authorisation / network facts ---
    AUTH_LOG = "auth_log"
    AVS_MATCH = "avs_match"
    CVV_MATCH = "cvv_match"
    THREE_DS_RESULT = "three_ds_result"
    DEVICE_FINGERPRINT = "device_fingerprint"
    IP_GEOLOCATION = "ip_geolocation"

    # --- card-member exhibits ---
    CM_NARRATIVE = "cm_narrative"
    PHOTO_OF_ITEM = "photo_of_item"
    RETURN_TRACKING = "return_tracking"
    RETURN_RECEIPT = "return_receipt"
    CHAT_LOG = "chat_log"
    EMAIL_THREAD = "email_thread"
    CANCELLATION_REQUEST = "cancellation_request"
    BANK_STATEMENT = "bank_statement"

    # --- merchant exhibits ---
    MERCHANT_REBUTTAL = "merchant_rebuttal"
    CRM_LOG = "crm_log"
    USAGE_LOG = "usage_log"
    """Service actually consumed — logins, streams, seat check-ins."""
    CATALOG_IMAGE = "catalog_image"
    PRODUCT_DESCRIPTION = "product_description"

    # --- derived / computed ---
    VISUAL_SIMILARITY = "visual_similarity"
    DUPLICATE_TXN_MATCH = "duplicate_txn_match"
    POLICY_WINDOW_CHECK = "policy_window_check"


#: Which side an exhibit type *tends* to support, absent contrary content.
#:
#: This is a prior over exhibit semantics, not a verdict: a `DELIVERY_CONFIRMATION`
#: normally helps the merchant, but the NLI verifier can still find that it names a
#: different address and flip its sign. Values are in [-1, 1] where +1 favours the
#: card member and -1 favours the merchant.
_POLARITY: dict[EvidenceType, float] = {
    # merchant-favouring fulfilment proof
    EvidenceType.CARRIER_TRACKING: -0.8,
    EvidenceType.DELIVERY_CONFIRMATION: -1.0,
    EvidenceType.SIGNATURE_PROOF: -1.0,
    EvidenceType.SHIPPING_LABEL: -0.4,
    EvidenceType.INVOICE: -0.3,
    EvidenceType.RECEIPT: -0.3,
    EvidenceType.ORDER_CONFIRMATION: -0.2,
    EvidenceType.TERMS_OF_SERVICE: -0.5,
    EvidenceType.REFUND_POLICY: -0.4,
    EvidenceType.CANCELLATION_POLICY: -0.4,
    EvidenceType.CRM_LOG: -0.5,
    EvidenceType.USAGE_LOG: -0.9,
    EvidenceType.CATALOG_IMAGE: -0.2,
    EvidenceType.PRODUCT_DESCRIPTION: -0.3,
    EvidenceType.MERCHANT_REBUTTAL: -0.4,
    # network facts lean merchant when they match, but are near-neutral by default
    EvidenceType.AUTH_LOG: -0.3,
    EvidenceType.AVS_MATCH: -0.6,
    EvidenceType.CVV_MATCH: -0.5,
    EvidenceType.THREE_DS_RESULT: -0.7,
    EvidenceType.DEVICE_FINGERPRINT: -0.4,
    EvidenceType.IP_GEOLOCATION: -0.2,
    # card-member-favouring exhibits
    EvidenceType.CM_NARRATIVE: 0.3,
    EvidenceType.PHOTO_OF_ITEM: 0.6,
    EvidenceType.RETURN_TRACKING: 0.9,
    EvidenceType.RETURN_RECEIPT: 0.9,
    EvidenceType.CHAT_LOG: 0.4,
    EvidenceType.EMAIL_THREAD: 0.4,
    EvidenceType.CANCELLATION_REQUEST: 0.8,
    EvidenceType.BANK_STATEMENT: 0.5,
    EvidenceType.REFUND_RECORD: -0.9,
    EvidenceType.CREDIT_NOTE: -0.8,
    EvidenceType.PRIOR_SETTLEMENT: -0.6,
    # derived signals are computed, so they start neutral and take their sign
    # from the measurement itself
    EvidenceType.VISUAL_SIMILARITY: 0.0,
    EvidenceType.DUPLICATE_TXN_MATCH: 0.0,
    EvidenceType.POLICY_WINDOW_CHECK: 0.0,
}


def evidence_polarity(etype: EvidenceType) -> float:
    """Default lean of an exhibit type, in [-1, 1]. Positive favours the card member."""
    return _POLARITY.get(etype, 0.0)


#: Exhibit types whose content is machine-verifiable against an external source
#: (carrier API, auth switch, ledger). These can earn a high `quality`; everything
#: else is capped in `Evidence.effective_quality`.
VERIFIABLE_TYPES: frozenset[EvidenceType] = frozenset(
    {
        EvidenceType.CARRIER_TRACKING,
        EvidenceType.DELIVERY_CONFIRMATION,
        EvidenceType.SIGNATURE_PROOF,
        EvidenceType.RETURN_TRACKING,
        EvidenceType.AUTH_LOG,
        EvidenceType.AVS_MATCH,
        EvidenceType.CVV_MATCH,
        EvidenceType.THREE_DS_RESULT,
        EvidenceType.REFUND_RECORD,
        EvidenceType.CREDIT_NOTE,
        EvidenceType.USAGE_LOG,
        EvidenceType.DUPLICATE_TXN_MATCH,
    }
)

#: Ceiling on `quality` for self-attested artefacts. A screenshot or a narrative can
#: never be as trustworthy as a carrier API response, so unverified exhibits are
#: clamped. This is a fairness control as much as a security one — it stops a party
#: from winning by simply uploading more unverifiable material.
UNVERIFIED_QUALITY_CEILING: float = 0.65


@dataclass(slots=True)
class Evidence:
    """A single exhibit in a dispute.

    Attributes:
        etype: canonical exhibit kind.
        party: who filed it.
        content: extracted text, or a short machine description for binary artefacts.
        quality: raw authenticity in [0,1] from the provenance checks (EXIF, carrier
            lookup, hash match). Defaults to 0.5 — "submitted, unverified".
        verified: True when an external source confirmed the artefact.
        blob_sha256: content hash of the underlying file, when there is one.
        occurred_at: the timestamp the exhibit *describes* (delivery time, ship date),
            which is distinct from when it was filed.
        metadata: structured fields pulled out by the parsers (tracking number,
            address, amounts, policy windows).
    """

    etype: EvidenceType
    party: Party
    content: str = ""
    quality: float = 0.5
    verified: bool = False
    blob_sha256: str | None = None
    occurred_at: datetime | None = None
    filed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0,1], got {self.quality}")
        if self.evidence_id is None:
            self.evidence_id = self._derive_id()

    def _derive_id(self) -> str:
        basis = f"{self.etype.value}|{self.party.value}|{self.blob_sha256 or self.content[:256]}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    @property
    def effective_quality(self) -> float:
        """Authenticity after the unverified ceiling is applied.

        An exhibit only escapes `UNVERIFIED_QUALITY_CEILING` if it is both a
        machine-verifiable type *and* actually verified against its source.
        """
        if self.verified and self.etype in VERIFIABLE_TYPES:
            return self.quality
        return min(self.quality, UNVERIFIED_QUALITY_CEILING)

    @property
    def is_network_sourced(self) -> bool:
        """Network facts carry no filer bias, which the ledger rewards."""
        return self.party is Party.NETWORK

    def describe(self) -> str:
        """Human label used in the verdict narration and the ledger waterfall."""
        label = self.etype.value.replace("_", " ")
        mark = "verified" if self.verified else "unverified"
        return f"{label} ({self.party.value.replace('_', ' ')}, {mark})"

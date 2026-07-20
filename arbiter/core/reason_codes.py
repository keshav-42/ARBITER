"""AMEX reason-code registry — the statute.

Each reason code carries the thing that makes ARBITER different from a classifier:
an explicit **burden of proof**, expressed as a prior in log-odds.

The AMEX Chargeback Code Guide already says who must prove what. Under C08 the
merchant must produce proof of delivery, so an unrebutted C08 claim favours the card
member. Under C31 the card member asserts the goods differ from the description, so
the burden sits with them. Encoding that as `prior_logodds` means the arbitration
starts from the regulation rather than from a learned bias.

Sign convention throughout ARBITER:

    positive log-odds  →  favours the CARD MEMBER
    negative log-odds  →  favours the MERCHANT

Reference: AMEX Chargeback Code Guide
https://www.americanexpress.com/content/dam/amex/au/en/merchant/static/chargebackcodeguide.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from arbiter.core.evidence import EvidenceType


class ReasonCode(str, Enum):
    """AMEX dispute reason codes, grouped by the guide's own chapters."""

    # --- C: Card Member disputes (service/merchandise) ---
    C02 = "C02"  # Credit not processed
    C04 = "C04"  # Goods/services returned or refused
    C05 = "C05"  # Goods/services cancelled
    C08 = "C08"  # Goods/services not received or only partially received
    C14 = "C14"  # Paid by other means
    C18 = "C18"  # "No Show" or CARDeposit cancelled
    C28 = "C28"  # Cancelled recurring billing
    C31 = "C31"  # Goods/services not as described
    C32 = "C32"  # Goods/services damaged or defective

    # --- F: Fraud-adjacent, but resolved as disputes ---
    F10 = "F10"  # Missing imprint
    F14 = "F14"  # Missing signature
    F24 = "F24"  # No Card Member authorisation
    F29 = "F29"  # Card not present
    F30 = "F30"  # EMV counterfeit
    F31 = "F31"  # EMV lost/stolen/non-received

    # --- P: Processing errors ---
    P01 = "P01"  # Unassigned Card number
    P03 = "P03"  # Credit processed as charge
    P04 = "P04"  # Charge processed as credit
    P05 = "P05"  # Incorrect charge amount
    P07 = "P07"  # Late submission
    P08 = "P08"  # Duplicate charge
    P22 = "P22"  # Non-matching Card number
    P23 = "P23"  # Currency discrepancy

    # --- R: Authorisation / retrieval ---
    R03 = "R03"  # Insufficient reply
    R13 = "R13"  # No reply
    M10 = "M10"  # Vehicle rental — capital damages
    M49 = "M49"  # Vehicle rental — theft or loss of use


class BurdenOfProof(str, Enum):
    """Which party the guide requires to substantiate the claim."""

    MERCHANT = "merchant"
    """Merchant must produce compelling evidence; silence loses."""

    CARD_MEMBER = "card_member"
    """Card member asserts a subjective condition and must substantiate it."""

    BALANCED = "balanced"
    """Neither side is privileged; the record decides."""


def _prob_to_logodds(p: float) -> float:
    """Convert a probability that the card member prevails into log-odds."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0,1), got {p}")
    return math.log(p / (1.0 - p))


@dataclass(frozen=True, slots=True)
class ReasonCodeSpec:
    """The statutory profile of one reason code.

    Attributes:
        code: the reason code itself.
        title: the guide's short name.
        description: what the card member is actually claiming.
        burden: who must substantiate.
        prior_cm_win: probability the card member prevails on an empty record, i.e.
            with the claim filed and nothing rebutting it. This is the burden of
            proof made numeric.
        required_merchant_evidence: exhibits the guide expects the merchant to
            produce. Absence of *all* of these is dispositive against them.
        required_cm_evidence: exhibits the card member must produce when the burden
            is theirs.
        evidence_weights: per-exhibit relevance multipliers for this code. An
            exhibit not listed falls back to 1.0. This is what makes a delivery
            confirmation decisive for C08 but nearly irrelevant for P08.
        representment_window_days: merchant's response window under the guide.
        supports_settlement: whether a partial split is a sensible disposition.
            Binary facts (duplicate charge, wrong currency) are all-or-nothing.
        guide_reference: citation surfaced in the verdict.
    """

    code: ReasonCode
    title: str
    description: str
    burden: BurdenOfProof
    prior_cm_win: float
    required_merchant_evidence: tuple[EvidenceType, ...] = ()
    required_cm_evidence: tuple[EvidenceType, ...] = ()
    evidence_weights: dict[EvidenceType, float] = field(default_factory=dict)
    representment_window_days: int = 20
    supports_settlement: bool = True
    guide_reference: str = ""

    @property
    def prior_logodds(self) -> float:
        """Burden of proof as a prior in log-odds. Positive favours the card member."""
        return _prob_to_logodds(self.prior_cm_win)

    def weight_for(self, etype: EvidenceType) -> float:
        """Relevance multiplier of an exhibit type under this code."""
        return self.evidence_weights.get(etype, 1.0)


# --------------------------------------------------------------------------------------
# The registry.
#
# Priors are set from the guide's allocation of burden, then sanity-checked against
# published network win-rate ranges. They are deliberately moderate: a prior should
# tilt the starting point, never decide the case on its own. Strong claims are made
# by the statute layer (dispositive rules), not by these numbers.
# --------------------------------------------------------------------------------------

_SPECS: tuple[ReasonCodeSpec, ...] = (
    ReasonCodeSpec(
        code=ReasonCode.C08,
        title="Goods/Services Not Received",
        description=(
            "The Card Member states the goods or services paid for were never received, "
            "or were only partially received."
        ),
        burden=BurdenOfProof.MERCHANT,
        # The merchant must evidence delivery. An empty record loses for them.
        prior_cm_win=0.72,
        required_merchant_evidence=(
            EvidenceType.DELIVERY_CONFIRMATION,
            EvidenceType.CARRIER_TRACKING,
            EvidenceType.SIGNATURE_PROOF,
        ),
        evidence_weights={
            EvidenceType.DELIVERY_CONFIRMATION: 2.2,
            EvidenceType.SIGNATURE_PROOF: 2.0,
            EvidenceType.CARRIER_TRACKING: 1.8,
            EvidenceType.USAGE_LOG: 1.6,
            EvidenceType.CM_NARRATIVE: 0.8,
            EvidenceType.CATALOG_IMAGE: 0.2,
            EvidenceType.VISUAL_SIMILARITY: 0.1,
        },
        guide_reference="Chargeback Code Guide — C08 Goods/Services Not Received",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C02,
        title="Credit Not Processed",
        description=(
            "The Card Member was promised a credit or refund by the merchant that was "
            "never applied to the account."
        ),
        burden=BurdenOfProof.MERCHANT,
        # If a credit was promised, the merchant must show it was issued.
        prior_cm_win=0.70,
        required_merchant_evidence=(
            EvidenceType.REFUND_RECORD,
            EvidenceType.CREDIT_NOTE,
        ),
        required_cm_evidence=(EvidenceType.CM_NARRATIVE,),
        evidence_weights={
            EvidenceType.REFUND_RECORD: 2.5,
            EvidenceType.CREDIT_NOTE: 2.2,
            EvidenceType.PRIOR_SETTLEMENT: 1.8,
            EvidenceType.CHAT_LOG: 1.5,
            EvidenceType.EMAIL_THREAD: 1.5,
            EvidenceType.BANK_STATEMENT: 1.4,
            EvidenceType.REFUND_POLICY: 1.2,
        },
        guide_reference="Chargeback Code Guide — C02 Credit Not Processed",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C04,
        title="Goods/Services Returned or Refused",
        description=(
            "The Card Member returned the merchandise or refused delivery, and no credit "
            "was received."
        ),
        burden=BurdenOfProof.CARD_MEMBER,
        # The card member must show the return actually happened and was timely.
        prior_cm_win=0.52,
        required_cm_evidence=(
            EvidenceType.RETURN_TRACKING,
            EvidenceType.RETURN_RECEIPT,
        ),
        required_merchant_evidence=(EvidenceType.REFUND_POLICY,),
        evidence_weights={
            EvidenceType.RETURN_TRACKING: 2.4,
            EvidenceType.RETURN_RECEIPT: 2.2,
            EvidenceType.POLICY_WINDOW_CHECK: 2.0,
            EvidenceType.REFUND_POLICY: 1.6,
            EvidenceType.CANCELLATION_POLICY: 1.2,
            EvidenceType.REFUND_RECORD: 1.8,
        },
        guide_reference="Chargeback Code Guide — C04 Goods/Services Returned",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C05,
        title="Goods/Services Cancelled",
        description="The Card Member cancelled the order or service before fulfilment.",
        burden=BurdenOfProof.CARD_MEMBER,
        prior_cm_win=0.54,
        required_cm_evidence=(EvidenceType.CANCELLATION_REQUEST,),
        required_merchant_evidence=(EvidenceType.CANCELLATION_POLICY,),
        evidence_weights={
            EvidenceType.CANCELLATION_REQUEST: 2.4,
            EvidenceType.CANCELLATION_POLICY: 1.8,
            EvidenceType.POLICY_WINDOW_CHECK: 1.9,
            EvidenceType.CHAT_LOG: 1.4,
            EvidenceType.EMAIL_THREAD: 1.4,
            EvidenceType.USAGE_LOG: 1.7,
        },
        guide_reference="Chargeback Code Guide — C05 Goods/Services Cancelled",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C31,
        title="Goods/Services Not As Described",
        description=(
            "The Card Member received goods or services materially different from the "
            "merchant's description."
        ),
        burden=BurdenOfProof.CARD_MEMBER,
        # A subjective assertion: the card member must substantiate the discrepancy.
        prior_cm_win=0.44,
        required_cm_evidence=(
            EvidenceType.PHOTO_OF_ITEM,
            EvidenceType.CM_NARRATIVE,
        ),
        required_merchant_evidence=(
            EvidenceType.PRODUCT_DESCRIPTION,
            EvidenceType.CATALOG_IMAGE,
        ),
        evidence_weights={
            EvidenceType.VISUAL_SIMILARITY: 2.0,
            EvidenceType.PHOTO_OF_ITEM: 1.7,
            EvidenceType.PRODUCT_DESCRIPTION: 1.6,
            EvidenceType.CATALOG_IMAGE: 1.4,
            EvidenceType.RETURN_TRACKING: 1.5,
            EvidenceType.CHAT_LOG: 1.2,
            EvidenceType.DELIVERY_CONFIRMATION: 0.3,
        },
        guide_reference="Chargeback Code Guide — C31 Not As Described",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C32,
        title="Goods/Services Damaged or Defective",
        description="The Card Member received merchandise that arrived damaged or defective.",
        burden=BurdenOfProof.CARD_MEMBER,
        prior_cm_win=0.46,
        required_cm_evidence=(EvidenceType.PHOTO_OF_ITEM,),
        required_merchant_evidence=(EvidenceType.SHIPPING_LABEL,),
        evidence_weights={
            EvidenceType.PHOTO_OF_ITEM: 2.0,
            EvidenceType.VISUAL_SIMILARITY: 1.8,
            EvidenceType.RETURN_TRACKING: 1.5,
            EvidenceType.CHAT_LOG: 1.2,
            EvidenceType.DELIVERY_CONFIRMATION: 0.3,
        },
        guide_reference="Chargeback Code Guide — C32 Damaged or Defective",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C28,
        title="Cancelled Recurring Billing",
        description=(
            "The Card Member cancelled a recurring subscription and was billed after "
            "the cancellation took effect."
        ),
        burden=BurdenOfProof.MERCHANT,
        # The merchant controls the subscription ledger and must show consent persisted.
        prior_cm_win=0.68,
        required_cm_evidence=(EvidenceType.CANCELLATION_REQUEST,),
        required_merchant_evidence=(
            EvidenceType.CRM_LOG,
            EvidenceType.USAGE_LOG,
        ),
        evidence_weights={
            EvidenceType.CANCELLATION_REQUEST: 2.3,
            EvidenceType.CRM_LOG: 1.9,
            EvidenceType.USAGE_LOG: 1.8,
            EvidenceType.CANCELLATION_POLICY: 1.5,
            EvidenceType.EMAIL_THREAD: 1.4,
        },
        guide_reference="Chargeback Code Guide — C28 Cancelled Recurring Billing",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C14,
        title="Paid by Other Means",
        description="The Card Member paid for the transaction by an alternative method.",
        burden=BurdenOfProof.CARD_MEMBER,
        prior_cm_win=0.50,
        required_cm_evidence=(EvidenceType.BANK_STATEMENT, EvidenceType.RECEIPT),
        evidence_weights={
            EvidenceType.BANK_STATEMENT: 2.4,
            EvidenceType.RECEIPT: 2.0,
            EvidenceType.DUPLICATE_TXN_MATCH: 2.2,
        },
        supports_settlement=False,
        guide_reference="Chargeback Code Guide — C14 Paid by Other Means",
    ),
    ReasonCodeSpec(
        code=ReasonCode.C18,
        title="No Show or CARDeposit Cancelled",
        description="A reservation deposit was charged despite cancellation or no-show terms.",
        burden=BurdenOfProof.MERCHANT,
        prior_cm_win=0.60,
        required_merchant_evidence=(EvidenceType.CANCELLATION_POLICY,),
        required_cm_evidence=(EvidenceType.CANCELLATION_REQUEST,),
        evidence_weights={
            EvidenceType.CANCELLATION_POLICY: 2.0,
            EvidenceType.CANCELLATION_REQUEST: 2.0,
            EvidenceType.POLICY_WINDOW_CHECK: 1.8,
        },
        guide_reference="Chargeback Code Guide — C18 No Show / CARDeposit",
    ),
    ReasonCodeSpec(
        code=ReasonCode.P08,
        title="Duplicate Charge",
        description="The same transaction was billed to the account more than once.",
        burden=BurdenOfProof.MERCHANT,
        # Near-deterministic: the ledger either shows two charges or it does not.
        prior_cm_win=0.80,
        required_merchant_evidence=(EvidenceType.AUTH_LOG, EvidenceType.INVOICE),
        evidence_weights={
            EvidenceType.DUPLICATE_TXN_MATCH: 3.0,
            EvidenceType.AUTH_LOG: 2.2,
            EvidenceType.INVOICE: 1.6,
            EvidenceType.RECEIPT: 1.4,
            EvidenceType.PHOTO_OF_ITEM: 0.1,
            EvidenceType.VISUAL_SIMILARITY: 0.1,
        },
        supports_settlement=False,
        guide_reference="Chargeback Code Guide — P08 Duplicate Charge",
    ),
    ReasonCodeSpec(
        code=ReasonCode.P05,
        title="Incorrect Charge Amount",
        description="The amount billed differs from the amount the Card Member authorised.",
        burden=BurdenOfProof.MERCHANT,
        prior_cm_win=0.66,
        required_merchant_evidence=(EvidenceType.INVOICE, EvidenceType.RECEIPT),
        evidence_weights={
            EvidenceType.INVOICE: 2.3,
            EvidenceType.RECEIPT: 2.1,
            EvidenceType.ORDER_CONFIRMATION: 1.9,
            EvidenceType.AUTH_LOG: 1.7,
        },
        # A price discrepancy has a natural partial remedy: refund the difference.
        supports_settlement=True,
        guide_reference="Chargeback Code Guide — P05 Incorrect Charge Amount",
    ),
    ReasonCodeSpec(
        code=ReasonCode.P07,
        title="Late Submission",
        description="The charge was submitted outside the permitted settlement window.",
        burden=BurdenOfProof.MERCHANT,
        prior_cm_win=0.64,
        required_merchant_evidence=(EvidenceType.AUTH_LOG,),
        evidence_weights={
            EvidenceType.AUTH_LOG: 2.6,
            EvidenceType.POLICY_WINDOW_CHECK: 2.2,
        },
        supports_settlement=False,
        guide_reference="Chargeback Code Guide — P07 Late Submission",
    ),
    ReasonCodeSpec(
        code=ReasonCode.P03,
        title="Credit Processed as Charge",
        description="A credit was submitted as a debit against the account.",
        burden=BurdenOfProof.MERCHANT,
        prior_cm_win=0.78,
        required_merchant_evidence=(EvidenceType.AUTH_LOG, EvidenceType.CREDIT_NOTE),
        evidence_weights={
            EvidenceType.AUTH_LOG: 2.5,
            EvidenceType.CREDIT_NOTE: 2.3,
            EvidenceType.REFUND_RECORD: 2.0,
        },
        supports_settlement=False,
        guide_reference="Chargeback Code Guide — P03 Credit Processed as Charge",
    ),
    ReasonCodeSpec(
        code=ReasonCode.P23,
        title="Currency Discrepancy",
        description="The transaction currency differs from the currency presented at purchase.",
        burden=BurdenOfProof.MERCHANT,
        prior_cm_win=0.68,
        required_merchant_evidence=(EvidenceType.INVOICE, EvidenceType.AUTH_LOG),
        evidence_weights={
            EvidenceType.INVOICE: 2.2,
            EvidenceType.AUTH_LOG: 2.0,
            EvidenceType.ORDER_CONFIRMATION: 1.8,
        },
        supports_settlement=True,
        guide_reference="Chargeback Code Guide — P23 Currency Discrepancy",
    ),
    ReasonCodeSpec(
        code=ReasonCode.F24,
        title="No Card Member Authorisation",
        description=(
            "The Card Member states they did not authorise or participate in the charge. "
            "Resolved as a dispute over the authorisation record, not as fraud detection."
        ),
        burden=BurdenOfProof.MERCHANT,
        prior_cm_win=0.66,
        required_merchant_evidence=(
            EvidenceType.AUTH_LOG,
            EvidenceType.AVS_MATCH,
            EvidenceType.THREE_DS_RESULT,
        ),
        evidence_weights={
            EvidenceType.THREE_DS_RESULT: 2.4,
            EvidenceType.AVS_MATCH: 2.0,
            EvidenceType.CVV_MATCH: 1.8,
            EvidenceType.DEVICE_FINGERPRINT: 1.7,
            EvidenceType.SIGNATURE_PROOF: 1.9,
            EvidenceType.IP_GEOLOCATION: 1.3,
            EvidenceType.USAGE_LOG: 1.6,
        },
        supports_settlement=False,
        guide_reference="Chargeback Code Guide — F24 No Card Member Authorisation",
    ),
    ReasonCodeSpec(
        code=ReasonCode.F29,
        title="Card Not Present",
        description="A card-not-present charge the Card Member states they did not make.",
        burden=BurdenOfProof.MERCHANT,
        prior_cm_win=0.70,
        required_merchant_evidence=(
            EvidenceType.AVS_MATCH,
            EvidenceType.THREE_DS_RESULT,
        ),
        evidence_weights={
            EvidenceType.THREE_DS_RESULT: 2.5,
            EvidenceType.AVS_MATCH: 2.1,
            EvidenceType.CVV_MATCH: 1.9,
            EvidenceType.DEVICE_FINGERPRINT: 1.7,
            EvidenceType.DELIVERY_CONFIRMATION: 1.8,
            EvidenceType.IP_GEOLOCATION: 1.4,
        },
        supports_settlement=False,
        guide_reference="Chargeback Code Guide — F29 Card Not Present",
    ),
    ReasonCodeSpec(
        code=ReasonCode.R13,
        title="No Reply",
        description="The merchant did not respond to a support-documentation request in time.",
        burden=BurdenOfProof.MERCHANT,
        # Procedural default: silence is dispositive.
        prior_cm_win=0.88,
        required_merchant_evidence=(EvidenceType.INVOICE, EvidenceType.RECEIPT),
        evidence_weights={
            EvidenceType.INVOICE: 2.0,
            EvidenceType.RECEIPT: 2.0,
        },
        supports_settlement=False,
        guide_reference="Chargeback Code Guide — R13 No Reply",
    ),
)

#: Registry keyed by code.
REASON_CODES: dict[ReasonCode, ReasonCodeSpec] = {spec.code: spec for spec in _SPECS}


def get_spec(code: ReasonCode | str) -> ReasonCodeSpec:
    """Look up a reason-code spec, accepting either the enum or its string form.

    Raises:
        KeyError: if the code has no registered spec.
    """
    if isinstance(code, str):
        code = ReasonCode(code)
    try:
        return REASON_CODES[code]
    except KeyError as exc:
        raise KeyError(f"no spec registered for reason code {code.value}") from exc


def supported_codes() -> tuple[ReasonCode, ...]:
    """Reason codes the engine can currently adjudicate."""
    return tuple(REASON_CODES)

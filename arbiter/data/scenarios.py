"""Scenario definitions — the causal ground truth behind each synthetic dispute.

The central design decision of the corpus: **generate the world first, then derive the
evidence from it.** A scenario says what actually happened — the parcel was delivered,
or it was not; the return was posted on day 6, or on day 19 — and the exhibits are
rendered as noisy, partial observations of that world.

This inverts the usual synthetic-data mistake of writing a narrative and then labelling
it, which teaches a model to read surface cues rather than reason about facts. Here the
label is causally upstream of the text, so a model that learns to predict it has to
learn the relationship between evidence and outcome.

It is also why CFPB is the wrong corpus for this task. CFPB narratives are grievances
*about banks* with no ground-truth verdict attached. What ARBITER needs is
(evidence bundle -> who should win), and that pairing has to be constructed.

Ground truth here is deliberately **not** the ledger's output. It is what the world
did. When the ledger disagrees, that is a genuine error to measure — which is exactly
what Stage 5 calibrates against and Stage 9 audits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from arbiter.core.reason_codes import ReasonCode


class GroundTruth(str, Enum):
    """What actually happened, independent of what the evidence shows."""

    CARD_MEMBER_RIGHT = "card_member"
    """The Card Member's account of events is factually correct."""

    MERCHANT_RIGHT = "merchant"
    """The Merchant's account is factually correct."""

    GENUINELY_AMBIGUOUS = "ambiguous"
    """Both accounts are defensible — a subjective quality judgement, a courier
    marked delivered but left unattended, a policy the parties read differently.
    These are the cases a calibrated system should abstain on rather than guess."""


@dataclass(frozen=True, slots=True)
class Scenario:
    """One archetypal way a dispute arises.

    Attributes:
        scenario_id: stable identifier, used in the corpus manifest and eval slices.
        reason_code: the AMEX code the claim is filed under.
        truth: what actually happened.
        summary: one line describing the underlying situation.
        weight: relative sampling frequency within its reason code.
        facts: structured world-state consumed by the evidence renderer. Keys are
            interpreted by `generator.py`; see that module for the vocabulary.
        difficulty: 0 = clear-cut, 1 = maximally hard. Drives how much noise the
            renderer injects and provides the eval harness a stratification axis.
    """

    scenario_id: str
    reason_code: ReasonCode
    truth: GroundTruth
    summary: str
    weight: float = 1.0
    facts: dict[str, object] = field(default_factory=dict)
    difficulty: float = 0.3


# --------------------------------------------------------------------------------------
# C08 — goods not received. The highest-volume dispute family.
# --------------------------------------------------------------------------------------

_C08: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="C08.never_shipped",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Merchant charged the card but never dispatched the order.",
        weight=2.0,
        difficulty=0.1,
        facts={"shipped": False, "delivered": False, "merchant_responds": True},
    ),
    Scenario(
        scenario_id="C08.lost_in_transit",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Parcel entered the carrier network and was lost before delivery.",
        weight=1.5,
        difficulty=0.35,
        facts={
            "shipped": True,
            "delivered": False,
            "tracking_exists": True,
            "last_scan": "in_transit",
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C08.delivered_correctly",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Parcel was delivered to the Card Member's address and signed for.",
        weight=2.0,
        difficulty=0.1,
        facts={
            "shipped": True,
            "delivered": True,
            "tracking_exists": True,
            "signature": True,
            "address_match": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C08.wrong_address",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Carrier delivered, but to an address that is not the Card Member's.",
        weight=1.0,
        difficulty=0.4,
        facts={
            "shipped": True,
            "delivered": True,
            "tracking_exists": True,
            "address_match": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C08.porch_theft",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.GENUINELY_AMBIGUOUS,
        summary=(
            "Carrier marked delivered and left the parcel unattended. It may have been "
            "stolen after delivery. Neither party is at fault and neither can prove it."
        ),
        weight=1.2,
        difficulty=0.9,
        facts={
            "shipped": True,
            "delivered": True,
            "tracking_exists": True,
            "signature": False,
            "address_match": True,
            "left_unattended": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C08.merchant_silent",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Merchant never responded to the documentation request.",
        weight=0.8,
        difficulty=0.05,
        facts={"shipped": False, "delivered": False, "merchant_responds": False},
    ),
    Scenario(
        scenario_id="C08.partial_shipment",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Only part of a multi-item order arrived; the balance was never sent.",
        weight=1.0,
        difficulty=0.55,
        facts={
            "shipped": True,
            "delivered": True,
            "tracking_exists": True,
            "partial": True,
            "address_match": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C08.digital_delivered",
        reason_code=ReasonCode.C08,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Digital goods were delivered and the account shows sustained usage.",
        weight=1.0,
        difficulty=0.25,
        facts={
            "digital": True,
            "delivered": True,
            "usage_log": True,
            "merchant_responds": True,
        },
    ),
)

# --------------------------------------------------------------------------------------
# C02 — credit not processed.
# --------------------------------------------------------------------------------------

_C02: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="C02.promised_never_issued",
        reason_code=ReasonCode.C02,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Merchant promised a refund in writing and never issued it.",
        weight=2.0,
        difficulty=0.2,
        facts={
            "refund_promised": True,
            "refund_issued": False,
            "written_promise": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C02.already_refunded",
        reason_code=ReasonCode.C02,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="The refund was issued and has posted; the claim is moot.",
        weight=1.5,
        difficulty=0.05,
        facts={
            "refund_promised": True,
            "refund_issued": True,
            "refund_posted": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C02.refund_in_flight",
        reason_code=ReasonCode.C02,
        truth=GroundTruth.GENUINELY_AMBIGUOUS,
        summary=(
            "Refund was issued but has not yet settled. The Card Member filed before "
            "it appeared; both accounts are accurate as of the filing date."
        ),
        weight=1.0,
        difficulty=0.85,
        facts={
            "refund_promised": True,
            "refund_issued": True,
            "refund_posted": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C02.no_promise_made",
        reason_code=ReasonCode.C02,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="No refund was ever agreed; the Card Member misread the policy.",
        weight=1.2,
        difficulty=0.5,
        facts={
            "refund_promised": False,
            "refund_issued": False,
            "merchant_responds": True,
        },
    ),
)

# --------------------------------------------------------------------------------------
# C04 — returned goods. Turns on the policy window, which the statute layer decides.
# --------------------------------------------------------------------------------------

_C04: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="C04.timely_return_no_credit",
        reason_code=ReasonCode.C04,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Return shipped well inside the window, carrier-confirmed, no credit issued.",
        weight=2.0,
        difficulty=0.15,
        facts={
            "returned": True,
            "return_day": 6,
            "return_window": 14,
            "return_tracked": True,
            "refund_issued": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C04.late_return",
        reason_code=ReasonCode.C04,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Return shipped after the published window had closed.",
        weight=1.5,
        difficulty=0.1,
        facts={
            "returned": True,
            "return_day": 19,
            "return_window": 14,
            "return_tracked": True,
            "refund_issued": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C04.boundary_return",
        reason_code=ReasonCode.C04,
        truth=GroundTruth.GENUINELY_AMBIGUOUS,
        summary=(
            "Return posted on the final day of the window. Whether it counts depends on "
            "whether the policy measures postmark or receipt."
        ),
        weight=0.8,
        difficulty=0.95,
        facts={
            "returned": True,
            "return_day": 14,
            "return_window": 14,
            "return_tracked": True,
            "refund_issued": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C04.never_returned",
        reason_code=ReasonCode.C04,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Card Member claims a return but no shipment was ever tendered.",
        weight=1.2,
        difficulty=0.3,
        facts={
            "returned": False,
            "return_tracked": False,
            "refund_issued": False,
            "merchant_responds": True,
        },
    ),
)

# --------------------------------------------------------------------------------------
# C31 / C32 — not as described, damaged. The subjective family.
# --------------------------------------------------------------------------------------

_C31: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="C31.materially_different",
        reason_code=ReasonCode.C31,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Item received is a different model from the one advertised.",
        weight=1.8,
        difficulty=0.4,
        facts={
            "delivered": True,
            "visual_similarity": 0.31,
            "description_match": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C31.as_described",
        reason_code=ReasonCode.C31,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Item matches the listing; the Card Member simply disliked it.",
        weight=1.6,
        difficulty=0.35,
        facts={
            "delivered": True,
            "visual_similarity": 0.93,
            "description_match": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C31.subjective_quality",
        reason_code=ReasonCode.C31,
        truth=GroundTruth.GENUINELY_AMBIGUOUS,
        summary=(
            "Item broadly matches the listing but falls short on finish or colour. "
            "Reasonable people would disagree about whether it was misdescribed."
        ),
        weight=1.4,
        difficulty=0.95,
        facts={
            "delivered": True,
            "visual_similarity": 0.68,
            "description_match": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C31.counterfeit",
        reason_code=ReasonCode.C31,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Item is a counterfeit of the branded product advertised.",
        weight=0.8,
        difficulty=0.5,
        facts={
            "delivered": True,
            "visual_similarity": 0.55,
            "description_match": False,
            "counterfeit": True,
            "merchant_responds": True,
        },
    ),
)

_C32: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="C32.arrived_damaged",
        reason_code=ReasonCode.C32,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Item arrived broken; damage is visible in the photographs.",
        weight=1.8,
        difficulty=0.3,
        facts={
            "delivered": True,
            "damaged": True,
            "photo_evidence": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C32.damaged_after_use",
        reason_code=ReasonCode.C32,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Item functioned for weeks before failing; damage is consistent with use.",
        weight=1.2,
        difficulty=0.7,
        facts={
            "delivered": True,
            "damaged": True,
            "photo_evidence": True,
            "days_before_report": 45,
            "merchant_responds": True,
        },
    ),
)

# --------------------------------------------------------------------------------------
# C05 / C28 — cancellation and recurring billing.
# --------------------------------------------------------------------------------------

_C05: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="C05.cancelled_in_time",
        reason_code=ReasonCode.C05,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Order was cancelled before dispatch but charged anyway.",
        weight=1.6,
        difficulty=0.25,
        facts={
            "cancelled": True,
            "cancel_day": 1,
            "cancel_window": 3,
            "shipped": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C05.cancelled_too_late",
        reason_code=ReasonCode.C05,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Cancellation was requested after the order had already shipped.",
        weight=1.3,
        difficulty=0.35,
        facts={
            "cancelled": True,
            "cancel_day": 5,
            "cancel_window": 3,
            "shipped": True,
            "merchant_responds": True,
        },
    ),
)

_C28: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="C28.billed_after_cancellation",
        reason_code=ReasonCode.C28,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Subscription was cancelled but billing continued for further cycles.",
        weight=2.0,
        difficulty=0.25,
        facts={
            "cancelled": True,
            "billed_after_cancel": True,
            "usage_after_cancel": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C28.still_using_service",
        reason_code=ReasonCode.C28,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Card Member claims cancellation but the account shows continued use.",
        weight=1.3,
        difficulty=0.4,
        facts={
            "cancelled": False,
            "billed_after_cancel": False,
            "usage_after_cancel": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="C28.cancellation_not_processed",
        reason_code=ReasonCode.C28,
        truth=GroundTruth.GENUINELY_AMBIGUOUS,
        summary=(
            "Card Member cancelled through a channel the merchant does not treat as "
            "binding. Both parties acted in good faith under different rules."
        ),
        weight=1.0,
        difficulty=0.9,
        facts={
            "cancelled": True,
            "cancel_channel": "chat",
            "billed_after_cancel": True,
            "usage_after_cancel": False,
            "merchant_responds": True,
        },
    ),
)

# --------------------------------------------------------------------------------------
# P-family — processing errors. Mostly deterministic from the ledger.
# --------------------------------------------------------------------------------------

_P: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="P08.true_duplicate",
        reason_code=ReasonCode.P08,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Terminal submitted the same charge twice within minutes.",
        weight=2.0,
        difficulty=0.05,
        facts={"duplicate_confirmed": True, "merchant_responds": True},
    ),
    Scenario(
        scenario_id="P08.two_genuine_purchases",
        reason_code=ReasonCode.P08,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Two separate purchases of the same value on the same day.",
        weight=1.2,
        difficulty=0.45,
        facts={"duplicate_confirmed": False, "merchant_responds": True},
    ),
    Scenario(
        scenario_id="P05.overcharged",
        reason_code=ReasonCode.P05,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Amount billed exceeds the amount on the order confirmation.",
        weight=1.5,
        difficulty=0.2,
        facts={"amount_mismatch": True, "merchant_responds": True},
    ),
    Scenario(
        scenario_id="P05.tip_or_surcharge",
        reason_code=ReasonCode.P05,
        truth=GroundTruth.GENUINELY_AMBIGUOUS,
        summary=(
            "Final amount exceeds the quote because of a gratuity or service charge "
            "that was disclosed only in small print."
        ),
        weight=1.0,
        difficulty=0.85,
        facts={"amount_mismatch": True, "disclosed_surcharge": True, "merchant_responds": True},
    ),
    Scenario(
        scenario_id="P07.late_submission",
        reason_code=ReasonCode.P07,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Charge was submitted months after authorisation.",
        weight=1.0,
        difficulty=0.15,
        facts={"submission_delay_days": 95, "merchant_responds": True},
    ),
)

# --------------------------------------------------------------------------------------
# F-family — authorisation disputes. Resolved on the auth record, not fraud scoring.
# --------------------------------------------------------------------------------------

_F: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="F29.unauthorised_cnp",
        reason_code=ReasonCode.F29,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Card-not-present charge with no authentication and a mismatched address.",
        weight=1.5,
        difficulty=0.3,
        facts={
            "three_ds": False,
            "avs_match": False,
            "delivered": False,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="F29.authenticated_purchase",
        reason_code=ReasonCode.F29,
        truth=GroundTruth.MERCHANT_RIGHT,
        summary="Charge carries a successful 3-D Secure authentication and matching AVS.",
        weight=1.5,
        difficulty=0.2,
        facts={
            "three_ds": True,
            "avs_match": True,
            "delivered": True,
            "merchant_responds": True,
        },
    ),
    Scenario(
        scenario_id="F29.family_member_used_card",
        reason_code=ReasonCode.F29,
        truth=GroundTruth.GENUINELY_AMBIGUOUS,
        summary=(
            "Purchase was made from the Card Member's own device and address by someone "
            "in the household. Authorised in fact, unauthorised in the Card Member's view."
        ),
        weight=1.0,
        difficulty=0.95,
        facts={
            "three_ds": False,
            "avs_match": True,
            "device_match": True,
            "delivered": True,
            "merchant_responds": True,
        },
    ),
)

# --------------------------------------------------------------------------------------
# R13 — procedural default.
# --------------------------------------------------------------------------------------

_R: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="R13.no_reply",
        reason_code=ReasonCode.R13,
        truth=GroundTruth.CARD_MEMBER_RIGHT,
        summary="Merchant did not reply to the documentation request within the window.",
        weight=1.0,
        difficulty=0.05,
        facts={"merchant_responds": False},
    ),
)


_ALL: tuple[Scenario, ...] = (
    *_C08,
    *_C02,
    *_C04,
    *_C31,
    *_C32,
    *_C05,
    *_C28,
    *_P,
    *_F,
    *_R,
)

#: Every scenario, keyed by id.
SCENARIOS: dict[str, Scenario] = {s.scenario_id: s for s in _ALL}


def scenarios_for(code: ReasonCode) -> tuple[Scenario, ...]:
    """All scenarios defined for a reason code."""
    return tuple(s for s in _ALL for _ in (1,) if s.reason_code is code)


def covered_codes() -> tuple[ReasonCode, ...]:
    """Reason codes with at least one scenario."""
    seen: dict[ReasonCode, None] = {}
    for s in _ALL:
        seen.setdefault(s.reason_code, None)
    return tuple(seen)


def all_scenarios() -> tuple[Scenario, ...]:
    """The full scenario catalogue."""
    return _ALL

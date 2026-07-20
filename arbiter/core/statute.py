"""The statute layer — deterministic rules that override the probabilistic ledger.

Some facts are not evidence to be weighed; they are constraints. A return shipped on
day 16 under a 14-day policy does not make the merchant *more likely* to be right — it
settles the question. Encoding these as hard rules is what makes ARBITER neuro-symbolic:
symbolic law on top, learned likelihoods underneath.

A rule may be:

    DISPOSITIVE  clamps the posterior to ±infinity — the case is decided
    STRONG       injects a large fixed log-odds shift, still rebuttable
    ADVISORY     records a finding for the narration without moving the posterior

Every firing is recorded with its guide citation so the verdict can cite chapter and
verse rather than gesturing at a model score.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.reason_codes import BurdenOfProof, ReasonCode, ReasonCodeSpec

#: Log-odds magnitude representing a decided case. Finite so downstream arithmetic
#: (sigmoid, settlement) stays numerically well behaved, but far beyond any
#: accumulation of ordinary evidence.
DISPOSITIVE_LOGODDS: float = 12.0

#: Magnitude of a STRONG rule — heavy, but a determined counter-record can overcome it.
STRONG_LOGODDS: float = 2.5


class RuleForce(str, Enum):
    """How hard a rule pushes."""

    DISPOSITIVE = "dispositive"
    STRONG = "strong"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class RuleFinding:
    """A record of one rule firing, surfaced in the verdict and stored for audit."""

    rule_id: str
    force: RuleForce
    favours: Party
    rationale: str
    guide_reference: str
    logodds_delta: float = 0.0

    @property
    def is_dispositive(self) -> bool:
        return self.force is RuleForce.DISPOSITIVE


@dataclass(frozen=True, slots=True)
class StatuteContext:
    """Everything a rule may inspect.

    Kept explicit rather than passing a mutable case object, so rules stay pure and
    individually testable.
    """

    spec: ReasonCodeSpec
    evidence: tuple[Evidence, ...]
    #: Days between the merchant's documentation request and their reply, if any.
    merchant_response_days: int | None = None
    #: Days between purchase/delivery and the card member's return shipment.
    return_shipped_day: int | None = None
    #: The merchant's stated return window, in days.
    return_window_days: int | None = None
    #: Day the Card Member requested cancellation, relative to the order.
    cancel_day: int | None = None
    #: The merchant's stated cancellation window, in days.
    cancel_window_days: int | None = None
    #: Days between the transaction and the dispute being filed.
    days_since_transaction: int | None = None
    #: Days between authorisation and the merchant submitting the charge for
    #: settlement. Distinct from `days_since_transaction`, which measures the Card
    #: Member's filing delay — conflating the two inverts P07.
    submission_delay_days: int | None = None
    #: Whether the ledger shows a second charge matching amount + merchant + window.
    duplicate_confirmed: bool = False
    #: Whether a refund for the disputed amount is already posted.
    refund_already_posted: bool = False

    def by_type(self, etype: EvidenceType) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if e.etype is etype)

    def has(self, etype: EvidenceType, *, verified_only: bool = False) -> bool:
        items = self.by_type(etype)
        if verified_only:
            return any(e.verified for e in items)
        return bool(items)

    def has_any(
        self, etypes: Sequence[EvidenceType], *, verified_only: bool = False
    ) -> bool:
        return any(self.has(t, verified_only=verified_only) for t in etypes)

    def from_party(self, party: Party) -> tuple[Evidence, ...]:
        return tuple(e for e in self.evidence if e.party is party)


#: A rule is a pure predicate returning a finding, or None when it does not apply.
Rule = Callable[[StatuteContext], RuleFinding | None]

_REGISTRY: list[Rule] = []


def rule(fn: Rule) -> Rule:
    """Register a statute rule. Order of registration is not significant."""
    _REGISTRY.append(fn)
    return fn


# --------------------------------------------------------------------------------------
# Procedural rules — these apply across reason codes.
# --------------------------------------------------------------------------------------


@rule
def merchant_no_reply(ctx: StatuteContext) -> RuleFinding | None:
    """Merchant silence past the representment window is a procedural default.

    The guide gives the merchant a fixed window to respond. Missing it decides the
    case regardless of what the evidence might have shown.
    """
    days = ctx.merchant_response_days
    if days is None:
        return None
    if days <= ctx.spec.representment_window_days:
        return None
    return RuleFinding(
        rule_id="PROC.NO_REPLY",
        force=RuleForce.DISPOSITIVE,
        favours=Party.CARD_MEMBER,
        rationale=(
            f"Merchant responded after {days} days, exceeding the "
            f"{ctx.spec.representment_window_days}-day representment window for "
            f"{ctx.spec.code.value}. Resolved in favour of the Card Member by default."
        ),
        guide_reference="Chargeback Code Guide — representment time limits",
        logodds_delta=DISPOSITIVE_LOGODDS,
    )


@rule
def refund_already_posted(ctx: StatuteContext) -> RuleFinding | None:
    """A posted refund for the disputed amount moots the dispute."""
    if not ctx.refund_already_posted:
        return None
    return RuleFinding(
        rule_id="PROC.REFUND_POSTED",
        force=RuleForce.DISPOSITIVE,
        favours=Party.MERCHANT,
        rationale=(
            "A credit for the disputed amount is already posted to the account. "
            "No chargeback is warranted; the claim is resolved as satisfied."
        ),
        guide_reference="Chargeback Code Guide — credit already processed",
        logodds_delta=-DISPOSITIVE_LOGODDS,
    )


#: Exhibits that turn a bare assertion into a substantiated allegation. A narrative is
#: excluded on purpose — anyone can write one, so it cannot shift a burden by itself.
_SUBSTANTIATING: frozenset[EvidenceType] = frozenset(
    {
        EvidenceType.EMAIL_THREAD,
        EvidenceType.CHAT_LOG,
        EvidenceType.RETURN_TRACKING,
        EvidenceType.RETURN_RECEIPT,
        EvidenceType.CANCELLATION_REQUEST,
        EvidenceType.BANK_STATEMENT,
        EvidenceType.ORDER_CONFIRMATION,
        EvidenceType.RECEIPT,
        EvidenceType.PHOTO_OF_ITEM,
        EvidenceType.DUPLICATE_TXN_MATCH,
    }
)


def _cm_claim_substantiated(ctx: StatuteContext) -> bool:
    """Whether the Card Member offered anything beyond their own account.

    Structured facts count too: a confirmed duplicate in the authorisation ledger
    substantiates a P08 claim even if the Card Member filed no documents.
    """
    if ctx.duplicate_confirmed:
        return True
    return any(
        e.etype in _SUBSTANTIATING
        for e in ctx.from_party(Party.CARD_MEMBER)
    )


@rule
def unmet_card_member_burden(ctx: StatuteContext) -> RuleFinding | None:
    """A Card Member who files none of the evidence their own claim requires loses ground.

    This is the other half of the burden of proof, and omitting it was a real modelling
    error: without it, absence of evidence read as neutral. A claim that a return was
    shipped, a cancellation was requested, or a second charge exists produces *no*
    exhibit when the underlying event never happened — so nothing pushed toward the
    merchant and the prior sat unopposed.

    Under a Card-Member-burden code, failing to produce the substantiating exhibit is
    itself evidence. Deliberately STRONG rather than dispositive: evidence can be lost,
    and a merchant's own records may still corroborate the claim.
    """
    if ctx.spec.burden is not BurdenOfProof.CARD_MEMBER:
        return None
    required = ctx.spec.required_cm_evidence
    if not required:
        return None
    # A narrative alone never discharges a burden — anyone can assert anything.
    substantive = tuple(t for t in required if t is not EvidenceType.CM_NARRATIVE)
    if not substantive or ctx.has_any(substantive):
        return None
    missing = ", ".join(t.value.replace("_", " ") for t in substantive)
    return RuleFinding(
        rule_id="BURDEN.CM_UNMET",
        force=RuleForce.STRONG,
        favours=Party.MERCHANT,
        rationale=(
            f"Under AMEX Code {ctx.spec.code.value} the Card Member must substantiate "
            f"the claim, but filed none of the required evidence ({missing})."
        ),
        guide_reference=f"Chargeback Code Guide — {ctx.spec.code.value} evidence requirements",
        logodds_delta=-STRONG_LOGODDS,
    )


@rule
def unmet_merchant_burden(ctx: StatuteContext) -> RuleFinding | None:
    """A responsive merchant who files nothing substantive fails their burden.

    Complements `merchant_no_reply`, which handles silence. This covers the merchant
    who replies on time but produces no records. C08 has its own dispositive rule, so
    it is excluded here to avoid double-counting.
    """
    if ctx.spec.burden is not BurdenOfProof.MERCHANT:
        return None
    if ctx.spec.code is ReasonCode.C08:
        return None
    required = ctx.spec.required_merchant_evidence
    if not required or ctx.has_any(required):
        return None
    # Only applies once the merchant has actually engaged; silence is handled elsewhere.
    if ctx.merchant_response_days is None:
        return None
    if ctx.merchant_response_days > ctx.spec.representment_window_days:
        return None

    # A merchant cannot document an event that never occurred. Where the Card Member
    # alleges something the merchant did — promised a credit, billed twice — the claim
    # must carry some substantiation before the merchant's silence counts against them.
    # Without this guard the rule punished merchants for being correct: in
    # "no refund was ever agreed" there is simply no credit note to produce.
    if not _cm_claim_substantiated(ctx):
        return None
    missing = ", ".join(t.value.replace("_", " ") for t in required)
    return RuleFinding(
        rule_id="BURDEN.MERCHANT_UNMET",
        force=RuleForce.STRONG,
        favours=Party.CARD_MEMBER,
        rationale=(
            f"Under AMEX Code {ctx.spec.code.value} the Merchant must substantiate the "
            f"charge, but produced none of the required records ({missing})."
        ),
        guide_reference=f"Chargeback Code Guide — {ctx.spec.code.value} evidence requirements",
        logodds_delta=STRONG_LOGODDS,
    )


# --------------------------------------------------------------------------------------
# Fulfilment rules — C08 and neighbours.
# --------------------------------------------------------------------------------------


@rule
def no_delivery_proof(ctx: StatuteContext) -> RuleFinding | None:
    """Under C08 the merchant must evidence delivery; producing nothing loses.

    Deliberately requires *verified* proof: an unverified screenshot claiming
    delivery does not discharge the burden.
    """
    if ctx.spec.code is not ReasonCode.C08:
        return None
    proof = (
        EvidenceType.DELIVERY_CONFIRMATION,
        EvidenceType.SIGNATURE_PROOF,
        EvidenceType.CARRIER_TRACKING,
        EvidenceType.USAGE_LOG,
    )
    if ctx.has_any(proof, verified_only=True):
        return None
    if ctx.has_any(proof):
        # Something was filed but nothing could be verified — strong, not dispositive.
        return RuleFinding(
            rule_id="C08.UNVERIFIED_DELIVERY",
            force=RuleForce.STRONG,
            favours=Party.CARD_MEMBER,
            rationale=(
                "Merchant submitted delivery documentation, but none of it could be "
                "verified against the carrier record."
            ),
            guide_reference="Chargeback Code Guide — C08 proof of delivery",
            logodds_delta=STRONG_LOGODDS,
        )
    return RuleFinding(
        rule_id="C08.NO_DELIVERY_PROOF",
        force=RuleForce.DISPOSITIVE,
        favours=Party.CARD_MEMBER,
        rationale=(
            "AMEX Code C08 requires valid carrier delivery confirmation. The merchant "
            "produced no proof of delivery, signature, or service usage."
        ),
        guide_reference="Chargeback Code Guide — C08 requires proof of delivery",
        logodds_delta=DISPOSITIVE_LOGODDS,
    )


@rule
def delivery_address_mismatch(ctx: StatuteContext) -> RuleFinding | None:
    """Delivery to an address other than the Card Member's does not discharge proof.

    The parsers record `address_match` on delivery exhibits; an explicit False is a
    substantive failure of the merchant's evidence.
    """
    for ev in ctx.by_type(EvidenceType.DELIVERY_CONFIRMATION):
        if ev.metadata.get("address_match") is False:
            return RuleFinding(
                rule_id="C08.ADDRESS_MISMATCH",
                force=RuleForce.STRONG,
                favours=Party.CARD_MEMBER,
                rationale=(
                    "Carrier delivery confirmation names an address that does not match "
                    "the Card Member's billing or registered shipping address."
                ),
                guide_reference="Chargeback Code Guide — delivery to Card Member address",
                logodds_delta=STRONG_LOGODDS,
            )
    return None


# --------------------------------------------------------------------------------------
# Policy-window rules — returns and cancellations.
# --------------------------------------------------------------------------------------


@rule
def return_outside_window(ctx: StatuteContext) -> RuleFinding | None:
    """A return shipped after the stated window is a policy violation, not a judgement call.

    This is the canonical example of why the statute layer exists: the arithmetic is
    exact, so no model should be asked to estimate it.
    """
    if ctx.spec.code not in (ReasonCode.C04, ReasonCode.C31, ReasonCode.C32):
        return None
    shipped, window = ctx.return_shipped_day, ctx.return_window_days
    if shipped is None or window is None:
        return None
    if shipped <= window:
        return None
    # Damaged/defective goods are not bound by an ordinary return window in the same
    # way — treat lateness as strong evidence rather than dispositive.
    force = (
        RuleForce.STRONG if ctx.spec.code is ReasonCode.C32 else RuleForce.DISPOSITIVE
    )
    delta = STRONG_LOGODDS if force is RuleForce.STRONG else DISPOSITIVE_LOGODDS
    return RuleFinding(
        rule_id="POLICY.RETURN_LATE",
        force=force,
        favours=Party.MERCHANT,
        rationale=(
            f"Card Member shipped the return on day {shipped}, outside the merchant's "
            f"published {window}-day return window."
        ),
        guide_reference="Merchant return policy — published window",
        logodds_delta=-delta,
    )


@rule
def return_within_window_unrefunded(ctx: StatuteContext) -> RuleFinding | None:
    """A timely, verified return with no credit issued strongly favours the Card Member."""
    if ctx.spec.code is not ReasonCode.C04:
        return None
    shipped, window = ctx.return_shipped_day, ctx.return_window_days
    if shipped is None or window is None or shipped > window:
        return None
    if not ctx.has(EvidenceType.RETURN_TRACKING, verified_only=True):
        return None
    if ctx.has(EvidenceType.REFUND_RECORD, verified_only=True):
        return None
    return RuleFinding(
        rule_id="POLICY.TIMELY_RETURN_NO_CREDIT",
        force=RuleForce.STRONG,
        favours=Party.CARD_MEMBER,
        rationale=(
            f"Return was shipped on day {shipped}, within the merchant's {window}-day "
            "window, and carrier-verified, yet no credit has been issued."
        ),
        guide_reference="Chargeback Code Guide — C04 returned goods",
        logodds_delta=STRONG_LOGODDS,
    )


@rule
def cancellation_outside_window(ctx: StatuteContext) -> RuleFinding | None:
    """A cancellation requested after the order shipped comes too late.

    Parallel to `return_outside_window`, but for C05/C18: once fulfilment has begun
    the merchant has already incurred the cost.
    """
    if ctx.spec.code not in (ReasonCode.C05, ReasonCode.C18):
        return None
    day, window = ctx.cancel_day, ctx.cancel_window_days
    if day is None or window is None or day <= window:
        return None
    return RuleFinding(
        rule_id="POLICY.CANCEL_LATE",
        force=RuleForce.STRONG,
        favours=Party.MERCHANT,
        rationale=(
            f"Cancellation was requested on day {day}, after the merchant's "
            f"{window}-day cancellation window had closed."
        ),
        guide_reference="Merchant cancellation policy — published window",
        logodds_delta=-STRONG_LOGODDS,
    )


@rule
def filed_outside_dispute_window(ctx: StatuteContext) -> RuleFinding | None:
    """Claims filed long after the transaction fall outside the guide's filing window."""
    days = ctx.days_since_transaction
    if days is None or days <= 120:
        return None
    return RuleFinding(
        rule_id="PROC.FILED_LATE",
        force=RuleForce.STRONG,
        favours=Party.MERCHANT,
        rationale=(
            f"The dispute was filed {days} days after the transaction, beyond the "
            "120-day filing window contemplated by the guide."
        ),
        guide_reference="Chargeback Code Guide — Card Member filing time limits",
        logodds_delta=-STRONG_LOGODDS,
    )


# --------------------------------------------------------------------------------------
# Ledger-fact rules — duplicates and authorisation.
# --------------------------------------------------------------------------------------


@rule
def late_settlement_submission(ctx: StatuteContext) -> RuleFinding | None:
    """A charge submitted long after authorisation is a P07 processing violation.

    Note this measures the *merchant's* settlement delay, which is the opposite
    direction from `filed_outside_dispute_window` — that one measures how long the
    Card Member waited to complain. Both are "late", and confusing them inverts the
    verdict, so they read separate fields.
    """
    if ctx.spec.code is not ReasonCode.P07:
        return None
    delay = ctx.submission_delay_days
    if delay is None or delay <= 30:
        return None
    return RuleFinding(
        rule_id="P07.LATE_SUBMISSION",
        force=RuleForce.DISPOSITIVE if delay > 60 else RuleForce.STRONG,
        favours=Party.CARD_MEMBER,
        rationale=(
            f"The Merchant submitted the charge for settlement {delay} days after "
            "authorisation, outside the permitted submission window."
        ),
        guide_reference="Chargeback Code Guide — P07 Late Submission",
        logodds_delta=DISPOSITIVE_LOGODDS if delay > 60 else STRONG_LOGODDS,
    )


@rule
def duplicate_confirmed(ctx: StatuteContext) -> RuleFinding | None:
    """A confirmed duplicate in the authorisation ledger settles P08 outright."""
    if ctx.spec.code is not ReasonCode.P08:
        return None
    if not ctx.duplicate_confirmed:
        return None
    return RuleFinding(
        rule_id="P08.DUPLICATE_CONFIRMED",
        force=RuleForce.DISPOSITIVE,
        favours=Party.CARD_MEMBER,
        rationale=(
            "The authorisation ledger contains two settled charges with identical "
            "amount, merchant, and terminal within the duplicate window."
        ),
        guide_reference="Chargeback Code Guide — P08 Duplicate Charge",
        logodds_delta=DISPOSITIVE_LOGODDS,
    )


@rule
def strong_authentication_present(ctx: StatuteContext) -> RuleFinding | None:
    """A successful 3-D Secure authentication shifts authorisation liability.

    Only applies to the authorisation-dispute codes; it says nothing about whether
    goods arrived.
    """
    if ctx.spec.code not in (ReasonCode.F24, ReasonCode.F29):
        return None
    for ev in ctx.by_type(EvidenceType.THREE_DS_RESULT):
        if ev.verified and ev.metadata.get("authenticated") is True:
            return RuleFinding(
                rule_id="AUTH.3DS_SUCCESS",
                force=RuleForce.STRONG,
                favours=Party.MERCHANT,
                rationale=(
                    "The transaction carries a successful 3-D Secure authentication, "
                    "shifting authorisation liability away from the merchant."
                ),
                guide_reference="Chargeback Code Guide — authenticated transactions",
                logodds_delta=-STRONG_LOGODDS,
            )
    return None


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatuteOutcome:
    """Aggregate result of running the statute over a case."""

    findings: tuple[RuleFinding, ...]
    #: Set when a dispositive rule fired; the ledger clamps to this value.
    clamp: float | None
    #: Total shift contributed by non-dispositive rules.
    logodds_delta: float

    @property
    def is_decided(self) -> bool:
        return self.clamp is not None

    @property
    def dispositive(self) -> RuleFinding | None:
        return next((f for f in self.findings if f.is_dispositive), None)


def evaluate(ctx: StatuteContext) -> StatuteOutcome:
    """Run every registered rule against a case.

    When dispositive rules conflict — a merchant default *and* a posted refund, say —
    the one with the larger magnitude wins, and ties resolve toward the merchant, since
    a clamp against the party who did not default would be the harsher error. Both
    findings are still reported so the conflict is visible in the audit trail.
    """
    findings = [f for f in (r(ctx) for r in _REGISTRY) if f is not None]

    dispositive = [f for f in findings if f.is_dispositive]
    clamp: float | None = None
    if dispositive:
        winner = max(
            dispositive,
            key=lambda f: (abs(f.logodds_delta), f.favours is Party.MERCHANT),
        )
        clamp = math.copysign(DISPOSITIVE_LOGODDS, winner.logodds_delta)

    delta = sum(f.logodds_delta for f in findings if f.force is RuleForce.STRONG)

    return StatuteOutcome(findings=tuple(findings), clamp=clamp, logodds_delta=delta)


def registered_rule_count() -> int:
    """Number of statute rules currently registered — used in the eval report."""
    return len(_REGISTRY)

"""Counterfactual recourse — what would change this outcome.

Either party clicks "Challenge the Verdict" and is told, precisely, what would flip it:
*"Provide a signed delivery confirmation and the verdict reverses."* Regulators require
actionable recourse, and it is almost never built.

In log-odds space this is arithmetic rather than search. The verdict flips when the
posterior crosses zero, so the losing party needs evidence whose combined contribution
exceeds the current margin |Lambda|. Two kinds of recourse follow:

    additive    a piece of evidence the party has not yet filed. Its expected
                contribution is w * lambda * q for the reason code; if that clears the
                margin, filing it is sufficient to flip the outcome.

    subtractive an exhibit currently counting against the party that is unverified. If
                verifying it fails — or it is shown to be inauthentic — its contribution
                is removed, which may itself close the margin.

Because the ledger is additive, each option's effect is exact: the recourse is not a
model prediction but a statement about the arithmetic. That is what makes it safe to
show a user as a promise rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arbiter.core.evidence import EvidenceType, Party, evidence_polarity
from arbiter.core.ledger import (
    MAX_EXHIBIT_CONTRIBUTION,
    NATS_TO_DECIBANS,
    Adjudication,
)
from arbiter.core.reason_codes import ReasonCodeSpec, get_spec

#: Assumed authenticity of newly filed evidence when projecting its effect. Deliberately
#: below 1.0: a party cannot promise their future exhibit will verify, so the recourse
#: is quoted at a realistic, not best-case, strength.
ASSUMED_NEW_QUALITY: float = 0.7

#: Exhibit types it is meaningful to *ask a party to provide*. Computed and
#: network-sourced types are excluded — a card member cannot furnish an AVS result, and
#: "provide a visual similarity score" is not an action a person can take.
_ACTIONABLE: dict[Party, tuple[EvidenceType, ...]] = {
    Party.CARD_MEMBER: (
        EvidenceType.RETURN_TRACKING,
        EvidenceType.RETURN_RECEIPT,
        EvidenceType.CANCELLATION_REQUEST,
        EvidenceType.PHOTO_OF_ITEM,
        EvidenceType.EMAIL_THREAD,
        EvidenceType.CHAT_LOG,
        EvidenceType.BANK_STATEMENT,
        EvidenceType.ORDER_CONFIRMATION,
        EvidenceType.RECEIPT,
    ),
    Party.MERCHANT: (
        EvidenceType.DELIVERY_CONFIRMATION,
        EvidenceType.SIGNATURE_PROOF,
        EvidenceType.CARRIER_TRACKING,
        EvidenceType.USAGE_LOG,
        EvidenceType.REFUND_RECORD,
        EvidenceType.CREDIT_NOTE,
        EvidenceType.INVOICE,
        EvidenceType.PRODUCT_DESCRIPTION,
        EvidenceType.CRM_LOG,
    ),
}


def _readable(etype: EvidenceType) -> str:
    return etype.value.replace("_", " ")


@dataclass(frozen=True, slots=True)
class RecourseOption:
    """One concrete action a party could take to change the outcome."""

    party: Party
    kind: str  # "provide" | "verify" | "discredit"
    evidence_type: EvidenceType | None
    #: Signed change to the posterior this action would produce, in nats.
    logodds_delta: float
    #: Whether the action alone would flip the verdict.
    sufficient: bool
    description: str

    @property
    def decibans(self) -> float:
        return self.logodds_delta * NATS_TO_DECIBANS


@dataclass(frozen=True, slots=True)
class Recourse:
    """The recourse offered to the party that did not prevail."""

    losing_party: Party
    margin: float
    """How far the posterior sits from a flipped verdict, in nats (always positive)."""
    options: tuple[RecourseOption, ...] = field(default_factory=tuple)

    @property
    def has_path(self) -> bool:
        """True when at least one single action would flip the verdict."""
        return any(o.sufficient for o in self.options)

    @property
    def margin_decibans(self) -> float:
        return self.margin * NATS_TO_DECIBANS

    def best(self) -> RecourseOption | None:
        """The most efficient sufficient option, or the strongest if none suffices."""
        sufficient = [o for o in self.options if o.sufficient]
        pool = sufficient or list(self.options)
        return min(pool, key=lambda o: abs(o.logodds_delta)) if sufficient else (
            max(pool, key=lambda o: abs(o.logodds_delta)) if pool else None
        )

    def explain(self) -> str:
        """One line for the "Challenge the Verdict" panel."""
        who = "Card Member" if self.losing_party is Party.CARD_MEMBER else "Merchant"
        top = self.best()
        if top is None:
            return f"No single action would change this outcome for the {who}."
        if top.sufficient:
            return f"To reverse this outcome, the {who} could: {top.description}"
        return (
            f"No single action would fully reverse this outcome for the {who}. The "
            f"strongest available step: {top.description}"
        )


def _already_filed(adj: Adjudication, party: Party) -> set[EvidenceType]:
    return {e.etype for e in adj.entries if e.party is party}


def _projected_contribution(
    etype: EvidenceType, party: Party, spec: ReasonCodeSpec
) -> float:
    """Signed contribution a freshly filed exhibit of this type would add, in nats.

    Uses the type's default polarity as the likelihood ratio (the party has not filed
    it yet, so there is no NLI signal), the reason-code weight, and a conservative
    assumed authenticity. Sign follows the ledger convention: positive favours the
    Card Member.
    """
    lam = evidence_polarity(etype) * 1.2
    weight = spec.weight_for(etype)
    raw = lam * ASSUMED_NEW_QUALITY * weight
    return max(-MAX_EXHIBIT_CONTRIBUTION, min(MAX_EXHIBIT_CONTRIBUTION, raw))


def counterfactual_recourse(adj: Adjudication) -> Recourse:
    """Compute what the losing party could do to change the outcome.

    A contested verdict has no losing party, so recourse is offered to whichever side
    the posterior currently disfavours — the one who would lose if forced to a ruling.
    """
    if adj.decided_by_statute:
        # A dispositive determination is procedural; the recourse is to contest the
        # underlying fact, not to add evidence, so there is no arithmetic path.
        losing = Party.MERCHANT if adj.posterior_logodds > 0 else Party.CARD_MEMBER
        return Recourse(losing_party=losing, margin=abs(adj.posterior_logodds))

    losing = Party.MERCHANT if adj.posterior_logodds > 0 else Party.CARD_MEMBER
    margin = abs(adj.posterior_logodds)
    spec = get_spec(adj.reason_code)

    # The losing party needs to move the posterior toward their own side by `margin`.
    # Card member gains are positive; merchant gains are negative.
    needed_sign = 1.0 if losing is Party.CARD_MEMBER else -1.0

    options: list[RecourseOption] = []

    # --- additive: file a helpful exhibit not yet on the record ---
    filed = _already_filed(adj, losing)
    for etype in _ACTIONABLE[losing]:
        if etype in filed:
            continue
        delta = _projected_contribution(etype, losing, spec)
        # Keep only exhibits that actually help the losing party.
        if delta * needed_sign <= 0:
            continue
        sufficient = abs(delta) >= margin
        options.append(
            RecourseOption(
                party=losing,
                kind="provide",
                evidence_type=etype,
                logodds_delta=delta,
                sufficient=sufficient,
                description=(
                    f"provide {_readable(etype)}"
                    + (
                        " — this alone would reverse the outcome"
                        if sufficient
                        else f" (worth about {abs(delta * NATS_TO_DECIBANS):.1f} "
                        "decibans toward their side)"
                    )
                ),
            )
        )

    # --- subtractive: discredit an opposing unverified exhibit ---
    for entry in adj.entries:
        if entry.party is losing:
            continue
        # Network-sourced and computed exhibits (AVS results, visual similarity, auth
        # logs) are measurements, not a party's account. You cannot challenge the
        # "authenticity" of a cosine distance, so they are never discreditable
        # regardless of their quality score.
        if entry.party is Party.NETWORK:
            continue
        # An exhibit counting against the losing party, that is not verified, could be
        # challenged. Removing it moves the posterior by exactly its contribution.
        if entry.contribution * needed_sign >= 0:
            continue
        if entry.quality >= 0.85:
            continue  # verified exhibits are not realistically discreditable
        delta = -entry.contribution
        sufficient = abs(delta) >= margin
        options.append(
            RecourseOption(
                party=losing,
                kind="discredit",
                evidence_type=None,
                logodds_delta=delta,
                sufficient=sufficient,
                description=(
                    f"challenge the authenticity of the opposing "
                    f"{entry.label.split(' (')[0]}"
                    + (
                        " — removing it would reverse the outcome"
                        if sufficient
                        else ""
                    )
                ),
            )
        )

    options.sort(key=lambda o: (not o.sufficient, -abs(o.logodds_delta)))
    return Recourse(losing_party=losing, margin=margin, options=tuple(options))

"""The Bayesian Evidence Ledger — ARBITER's arbitration engine.

The posterior log-odds that the Card Member prevails:

    Lambda = Lambda_0(code)  +  sum_i  w_i * lambda_i * q_i  +  reputation

        Lambda_0   burden of proof from the reason-code spec
        lambda_i   log-likelihood ratio of exhibit i
        q_i        authenticity of exhibit i, in [0,1]
        w_i        reason-code-specific relevance weight

Then the statute layer may clamp the result outright.

Because the model is additive in log-space, every exhibit carries an exact signed
contribution. The explanation is not narrated after the decision — it *is* the
decision, read back term by term. That property is the whole point: a verdict here
can be audited by re-adding the numbers.

Contributions are reported in **decibans** (10 * log10(odds ratio)) because a base-10
log scale is far easier to read on a waterfall chart than nats, and it is the
conventional unit for weight of evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from arbiter.core.evidence import Evidence, EvidenceType, Party, evidence_polarity
from arbiter.core.reason_codes import ReasonCode, ReasonCodeSpec, get_spec
from arbiter.core.reputation import ReputationState, combined_contribution
from arbiter.core.statute import RuleFinding, StatuteContext, StatuteOutcome
from arbiter.core.statute import evaluate as evaluate_statute

#: nats -> decibans
NATS_TO_DECIBANS: float = 10.0 / math.log(10.0)

#: Evidence types whose meaning is symmetric between the parties: either side could
#: file them with the same force, so their polarity should follow the filer rather than
#: a fixed lean. A refund-promise email helps whoever produces it. Directional types
#: (delivery confirmation, photos, AVS) are excluded — who files them cannot change
#: what they attest. Keeping this list here, beside the lambda logic, so the two stay
#: in sync; the fairness audit uses the same set to test symmetry.
FILER_ORIENTED_TYPES: frozenset[EvidenceType] = frozenset(
    {
        EvidenceType.EMAIL_THREAD,
        EvidenceType.CHAT_LOG,
        EvidenceType.RECEIPT,
        EvidenceType.INVOICE,
        EvidenceType.ORDER_CONFIRMATION,
    }
)

#: Posterior magnitude below which the case is too close to call on the evidence.
#: Cases inside this band are candidates for settlement rather than a binary verdict.
#: Stage 5 replaces this heuristic with a conformal threshold calibrated to a target
#: coverage; it remains as the fallback when no calibration is loaded.
CONTESTED_BAND: float = 0.85

#: Ceiling on the magnitude a single exhibit may contribute, in nats. Prevents one
#: overconfident model output from dominating a case that should turn on the record
#: as a whole.
MAX_EXHIBIT_CONTRIBUTION: float = 3.0

#: Naive Bayes assumes exhibits are conditionally independent. They are not: a
#: delivery confirmation and a signature proof are usually two views of the same
#: carrier record, so adding both at full strength double-counts one fact and drives
#: the posterior to false certainty.
#:
#: Within a correlation group, the k-th strongest exhibit is discounted by
#: `CORRELATION_DECAY ** (k-1)`. The strongest still counts fully; corroboration adds
#: real but diminishing weight. This is a deliberately simple stand-in for a full
#: covariance model, and it is what keeps confidences honest enough for the Stage 5
#: conformal layer to calibrate against.
CORRELATION_DECAY: float = 0.55

#: Ceiling on the total magnitude the evidence sum may reach, in nats. Roughly
#: P(win) = 0.995 — beyond this the posterior claims a precision no dispute record
#: with human-supplied exhibits can support.
MAX_EVIDENCE_LOGODDS: float = 5.3

#: How strongly NLI net-support modulates the type prior. At 0.8 a fully entailed,
#: fully committed exhibit is worth ~1.8x its type prior, and a weakly supported one
#: shrinks toward zero. Kept below 1.0 so the layer adjusts rather than dominates.
NLI_SUPPORT_GAIN: float = 0.8

#: Contradiction probability required before an exhibit is treated as self-defeating
#: and its sign flipped. Set high because flipping an exhibit against the party who
#: filed it is a strong claim, and MNLI models emit low-confidence contradictions
#: readily on terse documentary text.
NLI_CONTRADICTION_FLOOR: float = 0.45

#: Magnitude assigned to a contradicted exhibit, in nats. Comparable to a strong type
#: prior, so a self-defeating document counts against its filer about as much as a
#: solid one would have counted for them.
NLI_MAX_SWING: float = 1.4


#: Exhibit types that attest substantially the same underlying fact. Exhibits sharing
#: a group *and* a filing party are treated as correlated and damped.
_CORRELATION_GROUPS: dict[str, frozenset[EvidenceType]] = {
    "delivery": frozenset(
        {
            EvidenceType.DELIVERY_CONFIRMATION,
            EvidenceType.SIGNATURE_PROOF,
            EvidenceType.CARRIER_TRACKING,
            EvidenceType.SHIPPING_LABEL,
        }
    ),
    "commercial_record": frozenset(
        {
            EvidenceType.INVOICE,
            EvidenceType.RECEIPT,
            EvidenceType.ORDER_CONFIRMATION,
        }
    ),
    "policy": frozenset(
        {
            EvidenceType.TERMS_OF_SERVICE,
            EvidenceType.REFUND_POLICY,
            EvidenceType.CANCELLATION_POLICY,
        }
    ),
    "authentication": frozenset(
        {
            EvidenceType.AVS_MATCH,
            EvidenceType.CVV_MATCH,
            EvidenceType.THREE_DS_RESULT,
            EvidenceType.DEVICE_FINGERPRINT,
            EvidenceType.IP_GEOLOCATION,
        }
    ),
    "correspondence": frozenset(
        {
            EvidenceType.CHAT_LOG,
            EvidenceType.EMAIL_THREAD,
            EvidenceType.CM_NARRATIVE,
            EvidenceType.MERCHANT_REBUTTAL,
        }
    ),
    "credit": frozenset(
        {
            EvidenceType.REFUND_RECORD,
            EvidenceType.CREDIT_NOTE,
            EvidenceType.PRIOR_SETTLEMENT,
        }
    ),
    "return": frozenset(
        {
            EvidenceType.RETURN_TRACKING,
            EvidenceType.RETURN_RECEIPT,
        }
    ),
    "item_condition": frozenset(
        {
            EvidenceType.PHOTO_OF_ITEM,
            EvidenceType.VISUAL_SIMILARITY,
        }
    ),
}


def correlation_group(etype: EvidenceType) -> str | None:
    """Name of the correlation group an exhibit type belongs to, if any."""
    for name, members in _CORRELATION_GROUPS.items():
        if etype in members:
            return name
    return None


class Verdict(str, Enum):
    """Disposition of a dispute."""

    CARD_MEMBER = "card_member"
    MERCHANT = "merchant"
    CONTESTED = "contested"
    """Genuinely balanced — route to settlement or human review."""


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One line of the ledger: an exhibit and its exact effect on the posterior.

    This is the unit the waterfall chart renders and the audit trail stores.
    """

    evidence_id: str
    label: str
    party: Party
    #: The exhibit's canonical type, carried so downstream consumers (counterfactual
    #: recourse, the fairness audit) can reason about what was filed without re-reading
    #: the original evidence list.
    etype: EvidenceType
    #: Log-likelihood ratio before weighting or quality discount, in nats.
    lambda_lr: float
    #: Authenticity actually applied, after the unverified ceiling.
    quality: float
    #: Reason-code relevance multiplier.
    weight: float
    #: Signed contribution in nats: w * lambda * q * damping, after clamping.
    contribution: float
    #: Whether the raw product hit `MAX_EXHIBIT_CONTRIBUTION`.
    clamped: bool = False
    #: Correlation discount applied, in (0,1]. Below 1 means a stronger exhibit in the
    #: same group already attested this fact.
    damping: float = 1.0
    #: Correlation group this exhibit was damped within, if any.
    group: str | None = None

    @property
    def decibans(self) -> float:
        """Contribution in decibans — the unit shown to users."""
        return self.contribution * NATS_TO_DECIBANS

    @property
    def favours(self) -> Party:
        return Party.CARD_MEMBER if self.contribution >= 0 else Party.MERCHANT

    def explain(self) -> str:
        """One-line natural-language account of this entry's effect."""
        direction = "Card Member" if self.contribution >= 0 else "Merchant"
        detail = (
            f"strength {self.lambda_lr:+.2f}, authenticity {self.quality:.2f}, "
            f"relevance {self.weight:.1f}x"
        )
        if self.damping < 1.0:
            detail += f", corroborative discount {self.damping:.2f}x"
        return f"{self.label}: {abs(self.decibans):.1f} db toward the {direction} ({detail})"


@dataclass(frozen=True, slots=True)
class Adjudication:
    """The complete, auditable result of arbitrating one dispute."""

    reason_code: ReasonCode
    verdict: Verdict
    #: Final posterior log-odds. Positive favours the Card Member.
    posterior_logodds: float
    #: Burden-of-proof prior that opened the ledger.
    prior_logodds: float
    #: Per-exhibit lines, ordered as supplied.
    entries: tuple[LedgerEntry, ...]
    #: Statute rules that fired.
    findings: tuple[RuleFinding, ...]
    #: Reputation adjustment applied, in nats.
    reputation_logodds: float = 0.0
    #: Set when a dispositive rule decided the case, bypassing the evidence sum.
    decided_by_statute: bool = False

    @property
    def p_card_member(self) -> float:
        """Posterior probability the Card Member prevails."""
        return 1.0 / (1.0 + math.exp(-self.posterior_logodds))

    @property
    def confidence(self) -> float:
        """Probability assigned to the side that won, in [0.5, 1]."""
        p = self.p_card_member
        return max(p, 1.0 - p)

    @property
    def evidence_logodds(self) -> float:
        """Aggregate evidence term, after the `MAX_EVIDENCE_LOGODDS` ceiling.

        This is the value that actually entered the posterior. Compare with
        `raw_evidence_logodds` to see whether the ceiling bound.
        """
        raw = self.raw_evidence_logodds
        return math.copysign(min(abs(raw), MAX_EVIDENCE_LOGODDS), raw)

    @property
    def raw_evidence_logodds(self) -> float:
        """Uncapped sum of exhibit contributions."""
        return sum(e.contribution for e in self.entries)

    @property
    def evidence_capped(self) -> bool:
        """True when the aggregate evidence ceiling bound the posterior."""
        return abs(self.raw_evidence_logodds) > MAX_EVIDENCE_LOGODDS

    @property
    def is_contested(self) -> bool:
        return self.verdict is Verdict.CONTESTED

    def entries_favouring(self, party: Party) -> tuple[LedgerEntry, ...]:
        return tuple(e for e in self.entries if e.favours is party)

    def decisive_entry(self) -> LedgerEntry | None:
        """The single exhibit that moved the posterior furthest.

        Drives the "what hurt me most" panel in the recourse UI.
        """
        return max(self.entries, key=lambda e: abs(e.contribution), default=None)

    def audit_sum(self) -> float:
        """Recompute the posterior from its parts.

        Should equal `posterior_logodds` unless the statute clamped the case. The eval
        harness asserts this identity — it is what makes the ledger auditable.
        """
        return (
            self.prior_logodds
            + self.evidence_logodds
            + self.reputation_logodds
            + sum(f.logodds_delta for f in self.findings if not f.is_dispositive)
        )


def compute_lambda(evidence: Evidence) -> float:
    """Log-likelihood ratio for an exhibit, in nats.

    Resolution order:

    1. `metadata['lambda_lr']` — an exact, computed value (visual similarity,
       duplicate matching). Trusted outright.
    2. `metadata['entail'] / ['contradict']` — NLI probabilities, used to *modulate*
       the type prior rather than replace it. See below.
    3. The exhibit type's default polarity alone.

    **Why NLI modulates rather than replaces.** Measured on 800 cases, letting the
    entail/contradict ratio stand as lambda dropped ledger agreement from 88.8% to
    77.8% with a real DeBERTa-MNLI model — worse than using no NLI at all. The cause
    was visible in the label distribution: 2376 of 2888 exhibits scored *neutral*.

    That is the model behaving correctly. A terse record like "Delivery confirmation.
    Delivered 09 Mar." genuinely does not *entail* "the cardholder received the
    goods" — entailment is a strict relation, and a document stub rarely satisfies
    it. But neutral still yields log(0.15/0.03) ~ +1.6 nats, so every unremarkable
    exhibit was injecting a confident-looking value derived from nothing.

    So the type prior stays the backbone — it encodes what an exhibit *is* — and NLI
    supplies a signed adjustment for what the exhibit *says*. Contradiction can still
    flip the sign outright, which is the property that makes the layer worth having:
    a delivery confirmation naming the wrong address should help the Card Member.
    """
    meta = evidence.metadata

    explicit = meta.get("lambda_lr")
    if explicit is not None:
        return float(explicit)

    # For filer-symmetric types (correspondence, receipts, order records) the type's
    # default polarity encodes a spurious lean — it assumes the Card Member is the one
    # filing. Orient it by the actual filer instead, so the same record helps whoever
    # produced it. Directional types (delivery proof, photos) keep their intrinsic
    # polarity, since who files them does not change what they mean. This is what makes
    # the fairness party-swap test exact: a symmetric exhibit's contribution negates
    # cleanly when the filer flips.
    if evidence.etype in FILER_ORIENTED_TYPES:
        magnitude = abs(evidence_polarity(evidence.etype)) or 0.35
        signed = magnitude if evidence.party is Party.CARD_MEMBER else -magnitude
        prior = signed * 1.2
    else:
        prior = evidence_polarity(evidence.etype) * 1.2

    entail, contradict = meta.get("entail"), meta.get("contradict")
    if entail is None or contradict is None:
        return prior

    e, c = float(entail), float(contradict)

    # Net support in [-1, 1] for the hypothesis, which is always phrased in the
    # filer's favour. Neutral mass is excluded from the numerator, so an exhibit the
    # model cannot resolve contributes nothing rather than a spurious positive.
    net = (e - c) / max(e + c, 1e-6)
    decisive = e + c  # how much probability mass the model committed at all

    # Contradiction overrides the type prior: the exhibit undermines its own purpose.
    if c > e and c >= NLI_CONTRADICTION_FLOOR:
        magnitude = min(1.0, (c - e)) * NLI_MAX_SWING
        return -math.copysign(magnitude, prior) if prior else -magnitude

    # Otherwise scale the prior by the model's net support. `decisive` keeps
    # low-commitment (mostly-neutral) judgements close to the prior.
    return prior * (1.0 + NLI_SUPPORT_GAIN * net * decisive)


def build_entry(
    evidence: Evidence, spec: ReasonCodeSpec, *, damping: float = 1.0
) -> LedgerEntry:
    """Reduce one exhibit to a ledger line under a given reason code.

    Args:
        evidence: the exhibit.
        spec: reason-code spec supplying the relevance weight.
        damping: correlation discount in (0,1], normally supplied by `build_entries`.
    """
    lam = compute_lambda(evidence)
    quality = evidence.effective_quality
    weight = spec.weight_for(evidence.etype)

    raw = lam * quality * weight * damping
    clamped = abs(raw) > MAX_EXHIBIT_CONTRIBUTION
    contribution = math.copysign(min(abs(raw), MAX_EXHIBIT_CONTRIBUTION), raw)

    return LedgerEntry(
        evidence_id=evidence.evidence_id or "",
        label=evidence.describe(),
        party=evidence.party,
        etype=evidence.etype,
        lambda_lr=lam,
        quality=quality,
        weight=weight,
        contribution=contribution,
        clamped=clamped,
        damping=damping,
        group=correlation_group(evidence.etype),
    )


def build_entries(
    evidence: list[Evidence] | tuple[Evidence, ...], spec: ReasonCodeSpec
) -> tuple[LedgerEntry, ...]:
    """Build ledger lines for a full record, damping correlated corroboration.

    Exhibits are grouped by (correlation group, filing party). Within each group the
    strongest exhibit keeps full weight and each subsequent one is discounted
    geometrically, so piling on redundant documents cannot manufacture certainty.

    Network-sourced exhibits are exempt: they are independent measurements rather
    than a party's own account of events.

    Returns entries in the original evidence order, so the waterfall reads in the
    order exhibits were filed.
    """
    provisional = [(i, e, build_entry(e, spec)) for i, e in enumerate(evidence)]

    # Rank within each (group, party) bucket by undamped magnitude.
    buckets: dict[tuple[str, Party], list[int]] = {}
    for i, ev, entry in provisional:
        if ev.is_network_sourced or entry.group is None:
            continue
        buckets.setdefault((entry.group, ev.party), []).append(i)

    damping: dict[int, float] = {}
    strength = {i: abs(entry.contribution) for i, _, entry in provisional}
    for members in buckets.values():
        ordered = sorted(members, key=lambda i: strength[i], reverse=True)
        for rank, i in enumerate(ordered):
            damping[i] = CORRELATION_DECAY**rank

    return tuple(
        build_entry(ev, spec, damping=damping.get(i, 1.0))
        for i, ev, _ in provisional
    )


@dataclass(slots=True)
class DisputeCase:
    """Everything needed to arbitrate one dispute."""

    reason_code: ReasonCode
    evidence: list[Evidence] = field(default_factory=list)
    amount: float = 0.0
    merchant_reputation: ReputationState | None = None
    card_member_reputation: ReputationState | None = None

    # Structured facts consumed by the statute layer.
    merchant_response_days: int | None = None
    return_shipped_day: int | None = None
    return_window_days: int | None = None
    cancel_day: int | None = None
    cancel_window_days: int | None = None
    days_since_transaction: int | None = None
    submission_delay_days: int | None = None
    duplicate_confirmed: bool = False
    refund_already_posted: bool = False

    def statute_context(self, spec: ReasonCodeSpec) -> StatuteContext:
        return StatuteContext(
            spec=spec,
            evidence=tuple(self.evidence),
            merchant_response_days=self.merchant_response_days,
            return_shipped_day=self.return_shipped_day,
            return_window_days=self.return_window_days,
            cancel_day=self.cancel_day,
            cancel_window_days=self.cancel_window_days,
            days_since_transaction=self.days_since_transaction,
            submission_delay_days=self.submission_delay_days,
            duplicate_confirmed=self.duplicate_confirmed,
            refund_already_posted=self.refund_already_posted,
        )


def adjudicate(
    case: DisputeCase,
    *,
    contested_band: float = CONTESTED_BAND,
    apply_reputation: bool = True,
) -> Adjudication:
    """Arbitrate a dispute and return a fully auditable result.

    Args:
        case: the dispute, its exhibits, and its structured facts.
        contested_band: posterior magnitude below which the verdict is CONTESTED.
            Stage 5 supplies a conformally calibrated value here.
        apply_reputation: set False for the fairness audit, which measures how much
            the verdict depends on reputation by re-running without it.

    Returns:
        An `Adjudication` whose `audit_sum()` reproduces its own posterior.
    """
    spec = get_spec(case.reason_code)

    entries = build_entries(case.evidence, spec)

    statute: StatuteOutcome = evaluate_statute(case.statute_context(spec))

    reputation = 0.0
    if apply_reputation:
        reputation = combined_contribution(
            case.merchant_reputation, case.card_member_reputation
        )

    # Cap the aggregate evidence term. Individual exhibits are already clamped and
    # correlated ones damped, but a long one-sided record can still accumulate past
    # what any human-supplied evidence bundle justifies.
    raw_evidence = sum(e.contribution for e in entries)
    evidence_sum = math.copysign(
        min(abs(raw_evidence), MAX_EVIDENCE_LOGODDS), raw_evidence
    )

    prior = spec.prior_logodds
    posterior = prior + evidence_sum + reputation + statute.logodds_delta

    decided_by_statute = statute.is_decided
    if decided_by_statute:
        # A dispositive rule settles the matter; the evidence sum is retained in the
        # entries for transparency but does not move the outcome.
        posterior = statute.clamp  # type: ignore[assignment]

    if decided_by_statute or abs(posterior) >= contested_band:
        verdict = Verdict.CARD_MEMBER if posterior > 0 else Verdict.MERCHANT
    else:
        verdict = Verdict.CONTESTED

    return Adjudication(
        reason_code=case.reason_code,
        verdict=verdict,
        posterior_logodds=posterior,
        prior_logodds=prior,
        entries=entries,
        findings=statute.findings,
        reputation_logodds=reputation,
        decided_by_statute=decided_by_statute,
    )


def waterfall(adj: Adjudication) -> list[dict[str, object]]:
    """Ledger as an ordered waterfall, in decibans, for the UI chart.

    Each step carries the running total so the front end can draw the bar without
    recomputing anything — the chart is a direct rendering of the arithmetic.
    """
    steps: list[dict[str, object]] = []
    running = adj.prior_logodds

    steps.append(
        {
            "label": f"Burden of proof ({adj.reason_code.value})",
            "kind": "prior",
            "delta_db": adj.prior_logodds * NATS_TO_DECIBANS,
            "running_db": running * NATS_TO_DECIBANS,
        }
    )

    for entry in adj.entries:
        running += entry.contribution
        steps.append(
            {
                "label": entry.label,
                "kind": "evidence",
                "evidence_id": entry.evidence_id,
                "party": entry.party.value,
                "delta_db": entry.decibans,
                "running_db": running * NATS_TO_DECIBANS,
                "explain": entry.explain(),
            }
        )

    if adj.evidence_capped:
        correction = adj.evidence_logodds - adj.raw_evidence_logodds
        running += correction
        steps.append(
            {
                "label": "Aggregate evidence ceiling",
                "kind": "cap",
                "delta_db": correction * NATS_TO_DECIBANS,
                "running_db": running * NATS_TO_DECIBANS,
                "explain": (
                    "The combined weight of evidence was capped to avoid claiming more "
                    "certainty than a documentary record can support."
                ),
            }
        )

    for finding in adj.findings:
        if finding.is_dispositive or finding.logodds_delta == 0.0:
            continue
        running += finding.logodds_delta
        steps.append(
            {
                "label": finding.rule_id,
                "kind": "statute",
                "delta_db": finding.logodds_delta * NATS_TO_DECIBANS,
                "running_db": running * NATS_TO_DECIBANS,
                "explain": finding.rationale,
            }
        )

    if adj.reputation_logodds:
        running += adj.reputation_logodds
        steps.append(
            {
                "label": "Reputation adjustment (capped)",
                "kind": "reputation",
                "delta_db": adj.reputation_logodds * NATS_TO_DECIBANS,
                "running_db": running * NATS_TO_DECIBANS,
            }
        )

    if adj.decided_by_statute:
        dispositive = next((f for f in adj.findings if f.is_dispositive), None)
        steps.append(
            {
                "label": f"Dispositive rule: {dispositive.rule_id if dispositive else 'statute'}",
                "kind": "clamp",
                "delta_db": (adj.posterior_logodds * NATS_TO_DECIBANS)
                - (running * NATS_TO_DECIBANS),
                "running_db": adj.posterior_logodds * NATS_TO_DECIBANS,
                "explain": dispositive.rationale if dispositive else "",
            }
        )

    return steps

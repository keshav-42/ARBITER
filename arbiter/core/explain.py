"""Verdict narration — template-constrained, derived strictly from the ledger.

The failure mode this module exists to avoid: a language model reading a decision and
inventing a plausible-sounding justification for it. That is rationalisation, not
explanation, and it is worse than no explanation at all because it is convincing.

Here the prose is assembled from the same numbers that produced the verdict. Every
sentence is backed by a `LedgerEntry` or a `RuleFinding`. Nothing is generated, so
nothing can be fabricated. An LLM may later restyle this text, but it may not
introduce a reason that is not already in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

from arbiter.core.evidence import Party
from arbiter.core.ledger import Adjudication, LedgerEntry, Verdict
from arbiter.core.reason_codes import BurdenOfProof, get_spec

#: Confidence bands used in the prose.
_BANDS: tuple[tuple[float, str], ...] = (
    (0.95, "decisive"),
    (0.85, "strong"),
    (0.70, "moderate"),
    (0.0, "narrow"),
)


def _band(confidence: float) -> str:
    return next(label for threshold, label in _BANDS if confidence >= threshold)


def _party_name(party: Party) -> str:
    return {
        Party.CARD_MEMBER: "the Card Member",
        Party.MERCHANT: "the Merchant",
        Party.NETWORK: "the network record",
    }[party]


@dataclass(frozen=True, slots=True)
class VerdictCard:
    """Structured explanation rendered by the UI and stored in the audit trail."""

    headline: str
    burden_statement: str
    citation: str
    reasoning: tuple[str, ...]
    decisive_factor: str | None
    confidence_label: str
    confidence: float
    counterfactual: str | None = None

    def as_text(self) -> str:
        """Flat rendering for logs, CLI output, and the audit record."""
        parts = [self.headline, "", self.burden_statement, ""]
        parts.extend(f"- {line}" for line in self.reasoning)
        if self.decisive_factor:
            parts += ["", f"Decisive factor: {self.decisive_factor}"]
        if self.counterfactual:
            parts += ["", f"To change this outcome: {self.counterfactual}"]
        parts += ["", f"Citation: {self.citation}"]
        return "\n".join(parts)


def _burden_statement(adj: Adjudication) -> str:
    spec = get_spec(adj.reason_code)
    prior_db = adj.prior_logodds * 10.0 / 2.302585092994046

    if spec.burden is BurdenOfProof.MERCHANT:
        who = (
            "Under AMEX Code {code} the Merchant carries the burden of proof: they must "
            "substantiate the charge with compelling evidence."
        )
    elif spec.burden is BurdenOfProof.CARD_MEMBER:
        who = (
            "Under AMEX Code {code} the Card Member carries the burden of proof: the "
            "claim asserts a condition they must substantiate."
        )
    else:
        who = "Under AMEX Code {code} neither party is privileged; the record decides."

    return (
        who.format(code=adj.reason_code.value)
        + f" The case therefore opens at {prior_db:+.1f} decibans."
    )


def _entry_sentence(entry: LedgerEntry) -> str:
    direction = _party_name(entry.favours)
    strength = abs(entry.decibans)

    if entry.quality < 0.5:
        caveat = " Its authenticity could not be established, so its weight is reduced."
    elif entry.quality >= 0.9:
        caveat = " It was verified against its source."
    else:
        caveat = ""

    return (
        f"{entry.label.capitalize()} contributes {strength:.1f} decibans toward "
        f"{direction}.{caveat}"
    )


def build_verdict_card(
    adj: Adjudication,
    *,
    counterfactual: str | None = None,
) -> VerdictCard:
    """Assemble the explanation for an adjudication.

    Args:
        adj: the completed adjudication.
        counterfactual: optional recourse sentence from Stage 6, describing what
            would flip the outcome.
    """
    spec = get_spec(adj.reason_code)
    confidence = adj.confidence
    band = _band(confidence)

    # --- headline ---
    if adj.verdict is Verdict.CONTESTED:
        headline = (
            f"Contested — the evidence on {adj.reason_code.value} is too finely "
            f"balanced to rule ({confidence:.0%} confidence). Routing to settlement."
        )
    else:
        winner = (
            "the Card Member"
            if adj.verdict is Verdict.CARD_MEMBER
            else "the Merchant"
        )
        headline = (
            f"Resolved in favour of {winner} — {band} ({confidence:.0%} confidence) "
            f"under AMEX Code {adj.reason_code.value}."
        )

    reasoning: list[str] = []

    # --- statute first: dispositive rules outrank everything ---
    dispositive = next((f for f in adj.findings if f.is_dispositive), None)
    if dispositive is not None:
        reasoning.append(f"{dispositive.rationale} ({dispositive.guide_reference})")

    for finding in adj.findings:
        if finding.is_dispositive or finding.logodds_delta == 0.0:
            continue
        reasoning.append(finding.rationale)

    # --- evidence, strongest first ---
    ranked = sorted(adj.entries, key=lambda e: abs(e.contribution), reverse=True)
    for entry in ranked[:5]:
        if abs(entry.decibans) < 0.05:
            continue
        reasoning.append(_entry_sentence(entry))

    if not adj.entries:
        reasoning.append(
            "No exhibits were filed, so the outcome rests entirely on the burden "
            "of proof and the procedural record."
        )

    if adj.reputation_logodds:
        side = _party_name(
            Party.CARD_MEMBER if adj.reputation_logodds > 0 else Party.MERCHANT
        )
        reasoning.append(
            f"Historical dispute records shift the balance {abs(adj.reputation_logodds) * 4.34:.1f} "
            f"decibans toward {side}. This adjustment is capped and cannot override evidence."
        )

    # --- decisive factor ---
    decisive: str | None = None
    if dispositive is not None:
        decisive = dispositive.rationale
    else:
        top = adj.decisive_entry()
        if top is not None and abs(top.decibans) >= 0.05:
            decisive = top.explain()

    return VerdictCard(
        headline=headline,
        burden_statement=_burden_statement(adj),
        citation=spec.guide_reference,
        reasoning=tuple(reasoning),
        decisive_factor=decisive,
        confidence_label=band,
        confidence=confidence,
        counterfactual=counterfactual,
    )

"""Reputation as a decaying Beta posterior.

A merchant with four thousand clean deliveries deserves more credence than one whose
disputes are upheld half the time. But reputation is also the most dangerous signal in
the system: left unchecked it becomes a self-reinforcing loop where small merchants
lose because they are small, and large merchants win because they are large.

Two controls keep it honest:

    CAP     the contribution is clamped to +/- REPUTATION_CAP log-odds, so reputation
            can nudge a close case but can never overcome evidence. This is a fairness
            constraint, and the eval harness asserts it.

    DECAY   counts decay geometrically, so a merchant who cleans up their operation
            recovers, and one who coasts on an old record does not.

Both parties are modelled symmetrically. A card member who disputes constantly and
loses accrues the same kind of record a merchant does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from arbiter.core.evidence import Party

#: Maximum absolute log-odds reputation may contribute. Chosen so that reputation
#: cannot by itself move a neutral case past the auto-resolve threshold — it takes
#: real evidence to decide a dispute.
REPUTATION_CAP: float = 0.40

#: Monthly geometric decay applied to accumulated counts.
DECAY_PER_MONTH: float = 0.98

#: Beta(2,2) — a weak symmetric prior. Neither party is presumed good or bad, and a
#: handful of cases will not swing the estimate hard.
PRIOR_ALPHA: float = 2.0
PRIOR_BETA: float = 2.0

#: Below this many effective observations the estimate is shrunk toward neutral,
#: preventing a brand-new merchant from being judged on three data points.
MIN_OBSERVATIONS: float = 10.0

_DAYS_PER_MONTH = 30.4375


@dataclass(slots=True)
class ReputationState:
    """Beta posterior over a party's rate of *losing* disputes.

    Attributes:
        party_id: merchant or card-member identifier.
        party_type: which side of the table this record belongs to.
        alpha: accumulated losses (disputes resolved against this party), plus prior.
        beta: accumulated wins (disputes resolved in this party's favour), plus prior.
        last_decay_at: timestamp decay was last applied through.
    """

    party_id: str
    party_type: Party
    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    last_decay_at: datetime | None = None

    @property
    def observations(self) -> float:
        """Effective sample size, excluding the prior."""
        return max(0.0, (self.alpha - PRIOR_ALPHA) + (self.beta - PRIOR_BETA))

    @property
    def loss_rate(self) -> float:
        """E[theta] — posterior mean probability this party loses a dispute."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def credible_interval(self) -> tuple[float, float]:
        """Approximate 95% central interval on the loss rate.

        Normal approximation to the Beta, which is adequate here because the value is
        only ever used for display and for the shrinkage decision.
        """
        n = self.alpha + self.beta
        mean = self.loss_rate
        sd = math.sqrt(mean * (1.0 - mean) / (n + 1.0))
        return (max(0.0, mean - 1.96 * sd), min(1.0, mean + 1.96 * sd))

    def decayed(self, now: datetime) -> "ReputationState":
        """Return a copy with geometric decay applied up to `now`.

        Decay pulls both counts toward the prior rather than toward zero, so a long
        quiet period returns a party to neutral rather than to an undefined state.
        """
        if self.last_decay_at is None:
            return ReputationState(
                party_id=self.party_id,
                party_type=self.party_type,
                alpha=self.alpha,
                beta=self.beta,
                last_decay_at=now,
            )
        months = (now - self.last_decay_at).total_seconds() / (_DAYS_PER_MONTH * 86400)
        if months <= 0:
            return self
        factor = DECAY_PER_MONTH**months
        return ReputationState(
            party_id=self.party_id,
            party_type=self.party_type,
            alpha=PRIOR_ALPHA + (self.alpha - PRIOR_ALPHA) * factor,
            beta=PRIOR_BETA + (self.beta - PRIOR_BETA) * factor,
            last_decay_at=now,
        )

    def record(self, *, lost: bool, weight: float = 1.0) -> "ReputationState":
        """Return a copy updated with one resolved dispute."""
        return ReputationState(
            party_id=self.party_id,
            party_type=self.party_type,
            alpha=self.alpha + (weight if lost else 0.0),
            beta=self.beta + (0.0 if lost else weight),
            last_decay_at=self.last_decay_at,
        )

    def logodds_contribution(self) -> float:
        """Capped, shrunk log-odds contribution. Positive favours the Card Member.

        A merchant who loses most disputes pushes the posterior toward the card member;
        a card member who loses most of theirs pushes it back toward the merchant. The
        magnitude is shrunk by sample size and hard-clamped by `REPUTATION_CAP`.
        """
        n = self.observations
        if n <= 0:
            return 0.0

        # Deviation from a neutral 50% loss rate, in log-odds.
        rate = min(max(self.loss_rate, 1e-6), 1.0 - 1e-6)
        raw = math.log(rate / (1.0 - rate))

        # Shrink toward zero until enough observations have accumulated.
        shrunk = raw * min(1.0, n / MIN_OBSERVATIONS)

        # A merchant's losses favour the card member (+); a card member's losses
        # favour the merchant (-).
        signed = shrunk if self.party_type is Party.MERCHANT else -shrunk

        return max(-REPUTATION_CAP, min(REPUTATION_CAP, signed))


def combined_contribution(
    merchant: ReputationState | None,
    card_member: ReputationState | None,
) -> float:
    """Total reputation adjustment from both parties, itself capped.

    Capping the sum as well as each term stops two mediocre records from compounding
    into a decisive prior.
    """
    total = 0.0
    if merchant is not None:
        total += merchant.logodds_contribution()
    if card_member is not None:
        total += card_member.logodds_contribution()
    return max(-REPUTATION_CAP, min(REPUTATION_CAP, total))

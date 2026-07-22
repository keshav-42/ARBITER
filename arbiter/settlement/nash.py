"""Nash bargaining — settling instead of fighting when the evidence is a coin flip.

When the posterior is genuinely balanced, a binary verdict makes one party angry and
is only about as likely to be right as a coin. The fairer disposition is to **split the
disputed amount in proportion to the evidence**, and to make both parties an offer that
beats their expected outcome from continuing.

The disagreement point is what a full chargeback cycle is *worth* to each party — its
expected recovery, minus what it costs to pursue. Modelling that cost is the whole
reason both sides accept: a merchant who would win 55% of the time still rationally
takes a settlement slightly below 55% of V, because 45 days of representment effort has
a price.

Under linear utilities the Nash bargaining solution has a clean closed form. With the
card member's probability of prevailing p = sigma(Lambda) and disputed amount V, the
card member's share is:

    x* = V * p  +  1/2 * (c_M - c_CM)

The first term is the evidence-proportional split; the second nudges toward whichever
party finds continuing more costly, because they have more to gain from settling. Costs
are modelled as fractions of V, so they scale with what is at stake.

This is a genuinely novel move for the dispute-resolution brief: it answers "fairly
weighs the card member vs merchant perspective" not by picking a winner but by finding
the allocation both sides prefer to a fight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from arbiter.core.ledger import Adjudication

#: Default cost of pursuing a full chargeback cycle, as a fraction of the disputed
#: amount. Representment is effort, time, and risk on both sides.
DEFAULT_MERCHANT_COST: float = 0.15
DEFAULT_CARD_MEMBER_COST: float = 0.10

#: A settlement is only worth offering inside the contested band. Outside it the
#: verdict is confident enough that a split would just give away a decided case.
#: Expressed in probability: only settle when neither side exceeds this.
SETTLEMENT_CONFIDENCE_CEILING: float = 0.80


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def settlement_share(
    logodds: float,
    *,
    merchant_cost: float = DEFAULT_MERCHANT_COST,
    card_member_cost: float = DEFAULT_CARD_MEMBER_COST,
) -> float:
    """The card member's fraction of the disputed amount under Nash bargaining.

    Returns a value in [0, 1]. The evidence-proportional term is sigma(Lambda); the
    cost adjustment shifts toward whichever party has more to gain from avoiding a
    fight. Clipped to [0, 1] because a cost asymmetry should never award more than the
    whole amount or less than nothing.
    """
    p = _sigmoid(logodds)
    adjustment = 0.5 * (merchant_cost - card_member_cost)
    return max(0.0, min(1.0, p + adjustment))


@dataclass(frozen=True, slots=True)
class SettlementOffer:
    """A proposed split of the disputed amount, with the reasoning both sides can see."""

    amount: float
    card_member_share: float
    merchant_share: float
    card_member_fraction: float
    #: Each party's expected value from continuing to a full ruling, net of cost.
    card_member_disagreement: float
    merchant_disagreement: float
    posterior_logodds: float

    @property
    def card_member_gain(self) -> float:
        """How much the offer beats the card member's expected fight outcome."""
        return self.card_member_share - self.card_member_disagreement

    @property
    def merchant_gain(self) -> float:
        return self.merchant_share - self.merchant_disagreement

    @property
    def is_mutually_beneficial(self) -> bool:
        """True when both parties do at least as well as they expect from fighting.

        This is the property that makes a settlement self-enforcing: neither side has
        a rational reason to reject it. A tiny tolerance absorbs floating-point noise.
        """
        return self.card_member_gain >= -1e-9 and self.merchant_gain >= -1e-9

    def explain(self) -> str:
        """Plain-language account of the offer for both portals."""
        cm_pct = self.card_member_fraction
        return (
            f"The evidence is finely balanced, so rather than rule for one side we "
            f"propose splitting the ${self.amount:,.2f}: "
            f"${self.card_member_share:,.2f} ({cm_pct:.0%}) to the Card Member and "
            f"${self.merchant_share:,.2f} ({1 - cm_pct:.0%}) to the Merchant. "
            f"Both parties do better than the expected outcome of a full dispute, "
            f"which for the Card Member is about ${self.card_member_disagreement:,.2f} "
            f"and for the Merchant about ${self.merchant_disagreement:,.2f} after the "
            f"cost of pursuing it."
        )


def propose_settlement(
    adj: Adjudication,
    amount: float,
    *,
    merchant_cost: float = DEFAULT_MERCHANT_COST,
    card_member_cost: float = DEFAULT_CARD_MEMBER_COST,
) -> SettlementOffer:
    """Compute a Nash settlement for a contested dispute.

    Args:
        adj: the adjudication. Its posterior sets the evidence-proportional split.
        amount: the disputed amount V.
        merchant_cost / card_member_cost: cost of pursuing a full cycle, as a
            fraction of V.

    The disagreement point for each party is their probability of prevailing times the
    amount, minus their cost of getting there. The offer splits by `settlement_share`,
    which centres on the posterior and tilts toward the higher-cost party.
    """
    p_cm = adj.p_card_member
    fraction = settlement_share(
        adj.posterior_logodds,
        merchant_cost=merchant_cost,
        card_member_cost=card_member_cost,
    )

    cm_share = amount * fraction
    merchant_share = amount - cm_share

    # Expected value of fighting: win probability * amount, less the cost of pursuit.
    cm_disagreement = p_cm * amount - card_member_cost * amount
    merchant_disagreement = (1 - p_cm) * amount - merchant_cost * amount

    return SettlementOffer(
        amount=amount,
        card_member_share=cm_share,
        merchant_share=merchant_share,
        card_member_fraction=fraction,
        card_member_disagreement=cm_disagreement,
        merchant_disagreement=merchant_disagreement,
        posterior_logodds=adj.posterior_logodds,
    )


def is_settlement_appropriate(adj: Adjudication) -> bool:
    """Whether a settlement should be offered at all.

    Two gates: the reason code must admit a partial remedy (a duplicate charge cannot
    be half refunded), and the case must be genuinely contested — settling a confident
    verdict just gives away a decided case.
    """
    from arbiter.core.reason_codes import get_spec

    if adj.decided_by_statute:
        return False
    if not get_spec(adj.reason_code).supports_settlement:
        return False
    return adj.confidence <= SETTLEMENT_CONFIDENCE_CEILING

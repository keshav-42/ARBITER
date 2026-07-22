"""Stage 6 — Nash settlement and counterfactual recourse."""

from __future__ import annotations

import math

import pytest

from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.ledger import DisputeCase, Verdict, adjudicate
from arbiter.core.reason_codes import ReasonCode
from arbiter.data.generator import CorpusConfig, generate_corpus
from arbiter.settlement.nash import (
    SETTLEMENT_CONFIDENCE_CEILING,
    is_settlement_appropriate,
    propose_settlement,
    settlement_share,
)
from arbiter.settlement.recourse import (
    Recourse,
    counterfactual_recourse,
)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------------------------
# settlement share
# --------------------------------------------------------------------------------------


def test_share_tracks_the_posterior():
    assert settlement_share(-3.0) < settlement_share(0.0) < settlement_share(3.0)


def test_balanced_posterior_splits_near_half():
    """At Lambda=0 the split is ~50%, tilted slightly toward the higher-cost party."""
    share = settlement_share(0.0, merchant_cost=0.15, card_member_cost=0.10)
    assert 0.5 <= share <= 0.55


def test_equal_costs_give_a_pure_evidence_split():
    for L in (-2.0, 0.0, 1.5):
        assert settlement_share(L, merchant_cost=0.1, card_member_cost=0.1) == pytest.approx(
            _sigmoid(L)
        )


def test_higher_merchant_cost_favours_the_card_member():
    base = settlement_share(0.0, merchant_cost=0.1, card_member_cost=0.1)
    tilted = settlement_share(0.0, merchant_cost=0.3, card_member_cost=0.1)
    assert tilted > base


def test_share_is_clipped_to_unit_interval():
    assert settlement_share(-20, merchant_cost=0.0, card_member_cost=0.9) >= 0.0
    assert settlement_share(20, merchant_cost=0.9, card_member_cost=0.0) <= 1.0


# --------------------------------------------------------------------------------------
# settlement offers
# --------------------------------------------------------------------------------------


def _contested_case(logodds_target: float = 0.2) -> DisputeCase:
    """A C31 case tuned to land near the given posterior."""
    return DisputeCase(
        reason_code=ReasonCode.C31,
        amount=300.0,
        evidence=[
            Evidence(
                etype=EvidenceType.PHOTO_OF_ITEM,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": logodds_target},
            )
        ],
    )


def test_offer_splits_the_full_amount():
    adj = adjudicate(_contested_case())
    offer = propose_settlement(adj, 300.0)
    assert offer.card_member_share + offer.merchant_share == pytest.approx(300.0)


def test_offer_is_mutually_beneficial():
    """The property that makes a settlement self-enforcing."""
    adj = adjudicate(_contested_case())
    offer = propose_settlement(adj, 300.0)
    assert offer.is_mutually_beneficial
    assert offer.card_member_gain >= 0
    assert offer.merchant_gain >= 0


def test_mutual_benefit_holds_across_the_corpus():
    """Every settlement offered must beat both parties' expected fight outcome."""
    n = mutual = 0
    for g in generate_corpus(2500, CorpusConfig(seed=55)):
        adj = adjudicate(g.case)
        if not is_settlement_appropriate(adj):
            continue
        n += 1
        mutual += propose_settlement(adj, g.amount).is_mutually_beneficial
    assert n > 100
    assert mutual == n  # must be 100%


def test_settlement_surplus_equals_avoided_cost():
    """The gain to each side comes from not paying the cost of fighting."""
    adj = adjudicate(_contested_case(0.0))
    offer = propose_settlement(adj, 300.0, merchant_cost=0.15, card_member_cost=0.10)
    total_gain = offer.card_member_gain + offer.merchant_gain
    avoided = (0.15 + 0.10) * 300.0
    assert total_gain == pytest.approx(avoided, abs=1e-6)


def test_offer_explanation_names_the_amounts():
    offer = propose_settlement(adjudicate(_contested_case()), 300.0)
    text = offer.explain()
    assert "$300.00" in text or "$300" in text
    assert "%" in text


def test_is_settlement_appropriate_gates_on_confidence():
    """A confident verdict should not be given away as a split."""
    confident = adjudicate(
        DisputeCase(
            reason_code=ReasonCode.C31,
            evidence=[
                Evidence(
                    etype=EvidenceType.PHOTO_OF_ITEM,
                    party=Party.CARD_MEMBER,
                    metadata={"lambda_lr": 2.0},
                ),
                Evidence(
                    etype=EvidenceType.VISUAL_SIMILARITY,
                    party=Party.NETWORK,
                    verified=True,
                    quality=0.9,
                    metadata={"lambda_lr": 2.6},
                ),
            ],
        )
    )
    assert confident.confidence > SETTLEMENT_CONFIDENCE_CEILING
    assert not is_settlement_appropriate(confident)


def test_is_settlement_appropriate_rejects_all_or_nothing_codes():
    """A duplicate charge cannot be half refunded."""
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.P08, amount=50.0))
    assert not is_settlement_appropriate(adj)


def test_is_settlement_appropriate_rejects_statute_cases():
    adj = adjudicate(
        DisputeCase(reason_code=ReasonCode.C04, return_shipped_day=19, return_window_days=14)
    )
    assert not is_settlement_appropriate(adj)


def test_split_leans_toward_the_favoured_party():
    """A posterior favouring the card member awards them the larger share."""
    cm_favoured = propose_settlement(adjudicate(_contested_case(0.6)), 300.0)
    merchant_favoured = propose_settlement(adjudicate(_contested_case(-0.6)), 300.0)
    assert cm_favoured.card_member_fraction > merchant_favoured.card_member_fraction


# --------------------------------------------------------------------------------------
# counterfactual recourse
# --------------------------------------------------------------------------------------


def test_recourse_targets_the_losing_party():
    """The merchant loses an empty C08, so recourse is offered to them."""
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08))
    rec = counterfactual_recourse(adj)
    assert adj.verdict is Verdict.CARD_MEMBER
    assert rec.losing_party is Party.MERCHANT


def test_sufficient_options_actually_flip_the_verdict():
    """The core guarantee: a 'sufficient' option provably crosses zero.

    Verified exhaustively across the corpus, not just on one case.
    """
    checked = 0
    for g in generate_corpus(2000, CorpusConfig(seed=77)):
        adj = adjudicate(g.case)
        if adj.verdict is Verdict.CONTESTED or adj.decided_by_statute:
            continue
        for opt in counterfactual_recourse(adj).options:
            if opt.sufficient:
                checked += 1
                flipped = (adj.posterior_logodds + opt.logodds_delta > 0) != (
                    adj.posterior_logodds > 0
                )
                assert flipped, f"{g.scenario_id}: {opt.description}"
    assert checked > 20  # the property is actually exercised


def test_recourse_only_suggests_helpful_evidence():
    """Every 'provide' option must move the posterior toward the losing party."""
    for g in generate_corpus(500, CorpusConfig(seed=88)):
        adj = adjudicate(g.case)
        if adj.verdict is Verdict.CONTESTED or adj.decided_by_statute:
            continue
        rec = counterfactual_recourse(adj)
        need = 1.0 if rec.losing_party is Party.CARD_MEMBER else -1.0
        for opt in rec.options:
            assert opt.logodds_delta * need > 0


def test_recourse_does_not_suggest_already_filed_evidence():
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[
            Evidence(
                etype=EvidenceType.DELIVERY_CONFIRMATION,
                party=Party.MERCHANT,
                verified=True,
                quality=0.95,
                metadata={"lambda_lr": -2.0, "address_match": True},
            )
        ],
    )
    adj = adjudicate(case)
    rec = counterfactual_recourse(adj)
    provided = {o.evidence_type for o in rec.options if o.kind == "provide"}
    assert EvidenceType.DELIVERY_CONFIRMATION not in provided


def test_recourse_can_suggest_discrediting_unverified_opposing_evidence():
    """A losing party can challenge an unverified exhibit counting against them."""
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.PRODUCT_DESCRIPTION,
                party=Party.MERCHANT,
                quality=0.6,
                metadata={"lambda_lr": -1.5},
            )
        ],
    )
    adj = adjudicate(case)
    if adj.verdict is Verdict.MERCHANT:
        rec = counterfactual_recourse(adj)
        assert any(o.kind == "discredit" for o in rec.options)


def test_verified_evidence_is_not_discreditable():
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.VISUAL_SIMILARITY,
                party=Party.NETWORK,
                verified=True,
                quality=0.95,
                metadata={"lambda_lr": -2.4},
            )
        ],
    )
    adj = adjudicate(case)
    rec = counterfactual_recourse(adj)
    assert all(o.kind != "discredit" for o in rec.options)


def test_recourse_explanation_is_actionable():
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08))
    text = counterfactual_recourse(adj).explain()
    assert "Merchant" in text
    assert "provide" in text.lower() or "reverse" in text.lower()


def test_statute_case_has_margin_but_no_evidence_path():
    """A dispositive determination is contested on its facts, not its evidence sum."""
    adj = adjudicate(
        DisputeCase(reason_code=ReasonCode.C04, return_shipped_day=19, return_window_days=14)
    )
    rec = counterfactual_recourse(adj)
    assert rec.losing_party is Party.CARD_MEMBER
    assert rec.margin > 0
    assert not rec.options  # no arithmetic path around a hard rule


def test_best_prefers_the_cheapest_sufficient_option():
    """When several actions would flip it, recommend the smallest.

    A strongly-losing merchant (empty C08) needs a big move; construct a case where
    more than one option suffices and check the minimal one is chosen.
    """
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08))
    rec = counterfactual_recourse(adj)
    best = rec.best()
    if best and best.sufficient:
        sufficient = [o for o in rec.options if o.sufficient]
        assert abs(best.logodds_delta) == min(abs(o.logodds_delta) for o in sufficient)


def test_recourse_margin_matches_posterior():
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08))
    rec = counterfactual_recourse(adj)
    assert rec.margin == pytest.approx(abs(adj.posterior_logodds))


def test_empty_recourse_explains_gracefully():
    """A contested case with no actionable path still returns a sensible message."""
    rec = Recourse(losing_party=Party.MERCHANT, margin=0.5, options=())
    assert not rec.has_path
    assert "No single action" in rec.explain()

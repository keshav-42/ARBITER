"""Stage 2 — the Bayesian Evidence Ledger, reputation, and the narration layer."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.explain import build_verdict_card
from arbiter.core.ledger import (
    CONTESTED_BAND,
    CORRELATION_DECAY,
    MAX_EVIDENCE_LOGODDS,
    MAX_EXHIBIT_CONTRIBUTION,
    NATS_TO_DECIBANS,
    DisputeCase,
    Verdict,
    adjudicate,
    compute_lambda,
    correlation_group,
    waterfall,
)
from arbiter.core.reason_codes import ReasonCode, get_spec
from arbiter.core.reputation import (
    PRIOR_ALPHA,
    PRIOR_BETA,
    REPUTATION_CAP,
    ReputationState,
    combined_contribution,
)

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _verified_delivery(**meta) -> Evidence:
    return Evidence(
        etype=EvidenceType.DELIVERY_CONFIRMATION,
        party=Party.MERCHANT,
        verified=True,
        quality=0.95,
        metadata=meta,
    )


# --------------------------------------------------------------------------------------
# lambda derivation
# --------------------------------------------------------------------------------------


def test_explicit_lambda_wins():
    ev = Evidence(
        etype=EvidenceType.RECEIPT, party=Party.MERCHANT, metadata={"lambda_lr": -1.7}
    )
    assert compute_lambda(ev) == -1.7


def test_nli_probabilities_convert_to_lambda():
    """Entailment supports the filer, so a merchant exhibit yields a negative LR."""
    ev = Evidence(
        etype=EvidenceType.CARRIER_TRACKING,
        party=Party.MERCHANT,
        metadata={"entail": 0.9, "contradict": 0.05},
    )
    assert compute_lambda(ev) < 0

    ev_cm = Evidence(
        etype=EvidenceType.RETURN_TRACKING,
        party=Party.CARD_MEMBER,
        metadata={"entail": 0.9, "contradict": 0.05},
    )
    assert compute_lambda(ev_cm) > 0


def test_contradicted_evidence_flips_sign():
    """A merchant exhibit the verifier contradicts should help the Card Member."""
    ev = Evidence(
        etype=EvidenceType.DELIVERY_CONFIRMATION,
        party=Party.MERCHANT,
        metadata={"entail": 0.03, "contradict": 0.94},
    )
    assert compute_lambda(ev) > 0


def test_lambda_falls_back_to_type_polarity():
    ev = Evidence(etype=EvidenceType.RETURN_TRACKING, party=Party.CARD_MEMBER)
    assert compute_lambda(ev) > 0


# --------------------------------------------------------------------------------------
# core arithmetic
# --------------------------------------------------------------------------------------


def test_empty_case_posterior_equals_prior_when_no_rules_fire():
    """With every rule inert, the posterior is the burden-of-proof prior alone.

    C31 no longer qualifies: since Stage 3 an empty record trips BURDEN.CM_UNMET,
    because failing to substantiate your own claim is itself evidence. C08 with
    verified delivery leaves no rule firing, so it isolates the prior.
    """
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[
            Evidence(
                etype=EvidenceType.DELIVERY_CONFIRMATION,
                party=Party.MERCHANT,
                verified=True,
                quality=0.9,
                metadata={"lambda_lr": 0.0, "address_match": True},
            )
        ],
    )
    adj = adjudicate(case)
    assert not adj.findings
    assert adj.posterior_logodds == pytest.approx(get_spec(ReasonCode.C08).prior_logodds)


def test_unmet_card_member_burden_moves_an_empty_c31_record():
    """The Stage 3 fix: absence of required evidence is not neutral."""
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C31))
    assert adj.posterior_logodds < get_spec(ReasonCode.C31).prior_logodds
    assert any(f.rule_id == "BURDEN.CM_UNMET" for f in adj.findings)


def test_audit_sum_reproduces_posterior():
    """The identity that makes the ledger auditable."""
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.PHOTO_OF_ITEM,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 1.1},
            ),
            Evidence(
                etype=EvidenceType.PRODUCT_DESCRIPTION,
                party=Party.MERCHANT,
                metadata={"lambda_lr": -0.7},
            ),
        ],
    )
    adj = adjudicate(case)
    assert adj.audit_sum() == pytest.approx(adj.posterior_logodds)


def test_audit_sum_holds_with_reputation_and_strong_rules():
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[_verified_delivery(address_match=False)],
        merchant_reputation=ReputationState(
            party_id="m1", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 30, beta=PRIOR_BETA + 5
        ),
    )
    adj = adjudicate(case)
    assert adj.audit_sum() == pytest.approx(adj.posterior_logodds)


def test_contribution_is_product_of_three_factors():
    ev = Evidence(
        etype=EvidenceType.DELIVERY_CONFIRMATION,
        party=Party.MERCHANT,
        verified=True,
        quality=0.9,
        metadata={"lambda_lr": -1.0},
    )
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08, evidence=[ev]))
    entry = adj.entries[0]
    expected = -1.0 * 0.9 * get_spec(ReasonCode.C08).weight_for(
        EvidenceType.DELIVERY_CONFIRMATION
    )
    # Clamped at MAX_EXHIBIT_CONTRIBUTION.
    assert entry.contribution == pytest.approx(
        math.copysign(min(abs(expected), MAX_EXHIBIT_CONTRIBUTION), expected)
    )


def test_single_exhibit_cannot_dominate():
    ev = Evidence(
        etype=EvidenceType.DELIVERY_CONFIRMATION,
        party=Party.MERCHANT,
        verified=True,
        quality=1.0,
        metadata={"lambda_lr": -50.0},
    )
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08, evidence=[ev]))
    assert abs(adj.entries[0].contribution) == pytest.approx(MAX_EXHIBIT_CONTRIBUTION)
    assert adj.entries[0].clamped


def test_unverified_evidence_contributes_less_than_verified():
    def contribution(verified: bool) -> float:
        ev = Evidence(
            etype=EvidenceType.CARRIER_TRACKING,
            party=Party.MERCHANT,
            verified=verified,
            quality=0.95,
            metadata={"lambda_lr": -0.5},
        )
        return abs(
            adjudicate(
                DisputeCase(reason_code=ReasonCode.C08, evidence=[ev])
            ).entries[0].contribution
        )

    assert contribution(True) > contribution(False)


def test_decibans_conversion():
    ev = Evidence(
        etype=EvidenceType.RECEIPT, party=Party.MERCHANT, metadata={"lambda_lr": -1.0}
    )
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C31, evidence=[ev]))
    e = adj.entries[0]
    assert e.decibans == pytest.approx(e.contribution * NATS_TO_DECIBANS)


# --------------------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------------------


def test_verified_delivery_defeats_c08_claim():
    """The headline scenario: merchant proves delivery, claim fails."""
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[
            _verified_delivery(address_match=True),
            Evidence(
                etype=EvidenceType.SIGNATURE_PROOF,
                party=Party.MERCHANT,
                verified=True,
                quality=0.92,
            ),
        ],
    )
    adj = adjudicate(case)
    assert adj.verdict is Verdict.MERCHANT
    assert adj.p_card_member < 0.5


def test_missing_delivery_proof_wins_c08_for_card_member():
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08))
    assert adj.verdict is Verdict.CARD_MEMBER
    assert adj.decided_by_statute


def test_balanced_evidence_is_contested():
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.PHOTO_OF_ITEM,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 0.25},
            ),
        ],
    )
    adj = adjudicate(case)
    assert adj.verdict is Verdict.CONTESTED
    assert abs(adj.posterior_logodds) < CONTESTED_BAND


def test_statute_clamp_overrides_evidence_sum():
    """Even with strong merchant evidence, a late reply is a procedural default."""
    spec = get_spec(ReasonCode.C31)
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.PRODUCT_DESCRIPTION,
                party=Party.MERCHANT,
                metadata={"lambda_lr": -2.0},
            )
        ],
        merchant_response_days=spec.representment_window_days + 5,
    )
    adj = adjudicate(case)
    assert adj.verdict is Verdict.CARD_MEMBER
    assert adj.decided_by_statute
    # Evidence is still recorded for transparency.
    assert adj.entries and adj.entries[0].contribution < 0


def test_confidence_is_symmetric_around_half():
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C31))
    assert 0.5 <= adj.confidence <= 1.0


def test_p_card_member_matches_posterior():
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C02))
    assert adj.p_card_member == pytest.approx(
        1 / (1 + math.exp(-adj.posterior_logodds))
    )


def test_decisive_entry_is_the_largest_mover():
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.CHAT_LOG,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 0.2},
            ),
            Evidence(
                etype=EvidenceType.PHOTO_OF_ITEM,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 1.5},
            ),
        ],
    )
    adj = adjudicate(case)
    assert adj.decisive_entry().label.startswith("photo of item")


# --------------------------------------------------------------------------------------
# correlation damping — guards against manufactured certainty
# --------------------------------------------------------------------------------------


def test_correlated_exhibits_are_damped():
    """Delivery confirmation and signature proof are two views of one carrier record."""
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[
            _verified_delivery(address_match=True),
            Evidence(
                etype=EvidenceType.SIGNATURE_PROOF,
                party=Party.MERCHANT,
                verified=True,
                quality=0.95,
            ),
            Evidence(
                etype=EvidenceType.CARRIER_TRACKING,
                party=Party.MERCHANT,
                verified=True,
                quality=0.95,
            ),
        ],
    )
    adj = adjudicate(case)
    dampings = sorted(e.damping for e in adj.entries)
    # Strongest keeps full weight; the rest are discounted geometrically.
    assert dampings[-1] == pytest.approx(1.0)
    assert dampings[-2] == pytest.approx(CORRELATION_DECAY)
    assert dampings[-3] == pytest.approx(CORRELATION_DECAY**2)


def test_damping_only_applies_within_a_party():
    """Opposing exhibits of the same kind are independent claims, not corroboration."""
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.CHAT_LOG,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 0.8},
            ),
            Evidence(
                etype=EvidenceType.MERCHANT_REBUTTAL,
                party=Party.MERCHANT,
                metadata={"lambda_lr": -0.8},
            ),
        ],
    )
    adj = adjudicate(case)
    assert all(e.damping == 1.0 for e in adj.entries)


def test_network_evidence_is_exempt_from_damping():
    """Network measurements are independent, not a party's own account."""
    case = DisputeCase(
        reason_code=ReasonCode.F29,
        evidence=[
            Evidence(
                etype=EvidenceType.AVS_MATCH, party=Party.NETWORK, verified=True
            ),
            Evidence(
                etype=EvidenceType.CVV_MATCH, party=Party.NETWORK, verified=True
            ),
        ],
    )
    adj = adjudicate(case)
    assert all(e.damping == 1.0 for e in adj.entries)


def test_ungrouped_exhibits_are_not_damped():
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[
            Evidence(etype=EvidenceType.USAGE_LOG, party=Party.MERCHANT, verified=True),
            Evidence(etype=EvidenceType.CRM_LOG, party=Party.MERCHANT),
        ],
    )
    adj = adjudicate(case)
    assert all(e.damping == 1.0 for e in adj.entries)


def test_piling_on_redundant_evidence_has_diminishing_returns():
    """Ten copies of the same fact must not beat two."""

    def posterior(n: int) -> float:
        ev = [_verified_delivery(address_match=True)]
        ev += [
            Evidence(
                etype=EvidenceType.CARRIER_TRACKING,
                party=Party.MERCHANT,
                verified=True,
                quality=0.95,
            )
            for _ in range(n)
        ]
        return adjudicate(DisputeCase(reason_code=ReasonCode.C08, evidence=ev)).posterior_logodds

    gain_early = abs(posterior(2) - posterior(1))
    gain_late = abs(posterior(9) - posterior(8))
    assert gain_late < gain_early


def test_aggregate_evidence_ceiling_binds():
    ev = [
        Evidence(
            etype=EvidenceType.USAGE_LOG,
            party=Party.MERCHANT,
            verified=True,
            quality=1.0,
            metadata={"lambda_lr": -3.0},
        )
        for _ in range(10)
    ]
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08, evidence=ev))
    assert adj.evidence_capped
    assert abs(adj.evidence_logodds) == pytest.approx(MAX_EVIDENCE_LOGODDS)


def test_confidence_stays_below_certainty_without_statute():
    """No evidence bundle alone should reach 100% — only the statute is that sure."""
    ev = [
        Evidence(
            etype=EvidenceType.USAGE_LOG,
            party=Party.MERCHANT,
            verified=True,
            quality=1.0,
            metadata={"lambda_lr": -5.0},
        )
        for _ in range(20)
    ]
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08, evidence=ev))
    assert not adj.decided_by_statute
    assert adj.confidence < 0.999


def test_audit_sum_holds_when_evidence_is_capped():
    ev = [
        Evidence(
            etype=EvidenceType.USAGE_LOG,
            party=Party.MERCHANT,
            verified=True,
            quality=1.0,
            metadata={"lambda_lr": -3.0},
        )
        for _ in range(8)
    ]
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08, evidence=ev))
    assert adj.audit_sum() == pytest.approx(adj.posterior_logodds)


def test_waterfall_shows_cap_step():
    ev = [
        Evidence(
            etype=EvidenceType.USAGE_LOG,
            party=Party.MERCHANT,
            verified=True,
            quality=1.0,
            metadata={"lambda_lr": -3.0},
        )
        for _ in range(8)
    ]
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08, evidence=ev))
    steps = waterfall(adj)
    assert any(s["kind"] == "cap" for s in steps)
    assert steps[-1]["running_db"] == pytest.approx(
        adj.posterior_logodds * NATS_TO_DECIBANS
    )


def test_damping_is_surfaced_in_the_explanation():
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[
            _verified_delivery(address_match=True),
            Evidence(
                etype=EvidenceType.CARRIER_TRACKING,
                party=Party.MERCHANT,
                verified=True,
                quality=0.95,
            ),
        ],
    )
    adj = adjudicate(case)
    damped = [e for e in adj.entries if e.damping < 1.0]
    assert damped and "corroborative discount" in damped[0].explain()


def test_correlation_group_lookup():
    assert correlation_group(EvidenceType.SIGNATURE_PROOF) == "delivery"
    assert correlation_group(EvidenceType.USAGE_LOG) is None


# --------------------------------------------------------------------------------------
# reputation
# --------------------------------------------------------------------------------------


def test_reputation_is_capped():
    awful = ReputationState(
        party_id="m", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 5000, beta=PRIOR_BETA
    )
    assert awful.logodds_contribution() <= REPUTATION_CAP


def test_reputation_sign_by_party():
    """A merchant's losses help the Card Member; a Card Member's losses help the merchant."""
    m = ReputationState(
        party_id="m", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 50, beta=PRIOR_BETA + 5
    )
    cm = ReputationState(
        party_id="c", party_type=Party.CARD_MEMBER, alpha=PRIOR_ALPHA + 50, beta=PRIOR_BETA + 5
    )
    assert m.logodds_contribution() > 0
    assert cm.logodds_contribution() < 0


def test_new_party_has_no_reputation_effect():
    fresh = ReputationState(party_id="new", party_type=Party.MERCHANT)
    assert fresh.logodds_contribution() == 0.0


def test_small_samples_are_shrunk():
    few = ReputationState(
        party_id="m", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 2, beta=PRIOR_BETA
    )
    many = ReputationState(
        party_id="m", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 40, beta=PRIOR_BETA
    )
    assert few.logodds_contribution() < many.logodds_contribution()


def test_combined_contribution_is_capped():
    m = ReputationState(
        party_id="m", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 900, beta=PRIOR_BETA
    )
    cm = ReputationState(
        party_id="c", party_type=Party.CARD_MEMBER, alpha=PRIOR_BETA, beta=PRIOR_BETA + 900
    )
    assert abs(combined_contribution(m, cm)) <= REPUTATION_CAP


def test_decay_pulls_toward_prior():
    now = datetime(2026, 1, 1)
    state = ReputationState(
        party_id="m",
        party_type=Party.MERCHANT,
        alpha=PRIOR_ALPHA + 100,
        beta=PRIOR_BETA,
        last_decay_at=now,
    )
    later = state.decayed(now + timedelta(days=365 * 3))
    assert later.alpha < state.alpha
    assert later.alpha > PRIOR_ALPHA


def test_record_updates_counts():
    s = ReputationState(party_id="m", party_type=Party.MERCHANT)
    assert s.record(lost=True).alpha == s.alpha + 1
    assert s.record(lost=False).beta == s.beta + 1


def test_reputation_cannot_flip_a_decided_case():
    """The fairness guarantee: reputation nudges, evidence decides."""
    strong = [
        _verified_delivery(address_match=True),
        Evidence(
            etype=EvidenceType.SIGNATURE_PROOF,
            party=Party.MERCHANT,
            verified=True,
            quality=0.95,
        ),
    ]
    terrible = ReputationState(
        party_id="m", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 999, beta=PRIOR_BETA
    )
    adj = adjudicate(
        DisputeCase(
            reason_code=ReasonCode.C08, evidence=strong, merchant_reputation=terrible
        )
    )
    assert adj.verdict is Verdict.MERCHANT


def test_apply_reputation_false_zeroes_the_term():
    rep = ReputationState(
        party_id="m", party_type=Party.MERCHANT, alpha=PRIOR_ALPHA + 50, beta=PRIOR_BETA
    )
    adj = adjudicate(
        DisputeCase(reason_code=ReasonCode.C31, merchant_reputation=rep),
        apply_reputation=False,
    )
    assert adj.reputation_logodds == 0.0


# --------------------------------------------------------------------------------------
# waterfall + narration
# --------------------------------------------------------------------------------------


def test_waterfall_starts_at_prior_and_ends_at_posterior():
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.PHOTO_OF_ITEM,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 0.9},
            )
        ],
    )
    adj = adjudicate(case)
    steps = waterfall(adj)
    assert steps[0]["kind"] == "prior"
    assert steps[-1]["running_db"] == pytest.approx(
        adj.posterior_logodds * NATS_TO_DECIBANS
    )


def test_waterfall_includes_clamp_step_when_statute_decides():
    steps = waterfall(adjudicate(DisputeCase(reason_code=ReasonCode.C08)))
    assert any(s["kind"] == "clamp" for s in steps)


def test_verdict_card_cites_the_guide():
    adj = adjudicate(DisputeCase(reason_code=ReasonCode.C08))
    card = build_verdict_card(adj)
    assert "C08" in card.headline or "C08" in card.burden_statement
    assert card.citation
    assert card.reasoning


def test_verdict_card_states_who_bears_the_burden():
    card = build_verdict_card(adjudicate(DisputeCase(reason_code=ReasonCode.C08)))
    assert "Merchant carries the burden" in card.burden_statement

    card31 = build_verdict_card(adjudicate(DisputeCase(reason_code=ReasonCode.C31)))
    assert "Card Member carries the burden" in card31.burden_statement


def test_verdict_card_reports_contested():
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.PHOTO_OF_ITEM,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 0.25},
            )
        ],
    )
    card = build_verdict_card(adjudicate(case))
    assert "Contested" in card.headline


def test_verdict_card_leads_with_dispositive_rule():
    card = build_verdict_card(adjudicate(DisputeCase(reason_code=ReasonCode.C08)))
    assert "C08" in card.reasoning[0] or "delivery" in card.reasoning[0].lower()


def test_verdict_card_notes_empty_record():
    card = build_verdict_card(adjudicate(DisputeCase(reason_code=ReasonCode.C31)))
    assert any("No exhibits" in line for line in card.reasoning)


def test_verdict_card_text_renders():
    adj = adjudicate(
        DisputeCase(
            reason_code=ReasonCode.C04,
            return_shipped_day=16,
            return_window_days=14,
        )
    )
    text = build_verdict_card(adj).as_text()
    assert "day 16" in text
    assert "14-day" in text

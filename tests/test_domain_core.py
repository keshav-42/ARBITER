"""Stage 1 — domain core: reason codes, evidence taxonomy, statute rules."""

from __future__ import annotations

import math

import pytest

from arbiter.core.evidence import (
    UNVERIFIED_QUALITY_CEILING,
    Evidence,
    EvidenceType,
    Party,
    evidence_polarity,
)
from arbiter.core.reason_codes import (
    REASON_CODES,
    BurdenOfProof,
    ReasonCode,
    get_spec,
    supported_codes,
)
from arbiter.core.ledger import DisputeCase, Verdict, adjudicate
from arbiter.core.statute import (
    DISPOSITIVE_LOGODDS,
    PRESUMPTION_LOGODDS,
    STRONG_LOGODDS,
    RuleForce,
    StatuteContext,
    evaluate,
    registered_rule_count,
)

# --------------------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------------------


def test_every_spec_is_keyed_by_its_own_code():
    for code, spec in REASON_CODES.items():
        assert spec.code is code


def test_priors_are_valid_probabilities():
    for spec in REASON_CODES.values():
        assert 0.0 < spec.prior_cm_win < 1.0


def test_prior_logodds_sign_follows_burden():
    """Merchant-burden codes must start favouring the Card Member, and vice versa."""
    for spec in REASON_CODES.values():
        if spec.burden is BurdenOfProof.MERCHANT:
            assert spec.prior_logodds > 0, f"{spec.code} should favour the Card Member"
        elif spec.burden is BurdenOfProof.CARD_MEMBER:
            # A card-member burden means they do not get a free head start.
            assert spec.prior_logodds < 0.25, f"{spec.code} gives CM too much prior"


def test_prior_logodds_roundtrips_to_probability():
    spec = get_spec(ReasonCode.C08)
    recovered = 1.0 / (1.0 + math.exp(-spec.prior_logodds))
    assert recovered == pytest.approx(spec.prior_cm_win)


def test_c08_favours_card_member_more_than_c31():
    """The burden asymmetry the whole design rests on."""
    assert get_spec(ReasonCode.C08).prior_logodds > get_spec(ReasonCode.C31).prior_logodds


def test_get_spec_accepts_string_and_enum():
    assert get_spec("C08") is get_spec(ReasonCode.C08)


def test_get_spec_rejects_unregistered_code():
    with pytest.raises(KeyError):
        get_spec(ReasonCode.M10)


def test_weight_for_defaults_to_one():
    spec = get_spec(ReasonCode.C08)
    assert spec.weight_for(EvidenceType.DELIVERY_CONFIRMATION) > 1.0
    assert spec.weight_for(EvidenceType.BANK_STATEMENT) == 1.0


def test_reason_code_weighting_is_context_sensitive():
    """A delivery confirmation decides C08 but is nearly irrelevant to P08."""
    assert get_spec(ReasonCode.C08).weight_for(EvidenceType.DELIVERY_CONFIRMATION) > 2.0
    assert get_spec(ReasonCode.P08).weight_for(EvidenceType.VISUAL_SIMILARITY) < 0.5


def test_supported_codes_non_empty():
    assert len(supported_codes()) >= 15


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------


def test_polarity_signs_are_sensible():
    assert evidence_polarity(EvidenceType.DELIVERY_CONFIRMATION) < 0  # merchant
    assert evidence_polarity(EvidenceType.RETURN_TRACKING) > 0  # card member
    assert evidence_polarity(EvidenceType.VISUAL_SIMILARITY) == 0  # computed


def test_unverified_evidence_is_capped():
    ev = Evidence(
        etype=EvidenceType.PHOTO_OF_ITEM,
        party=Party.CARD_MEMBER,
        quality=0.99,
    )
    assert ev.effective_quality == UNVERIFIED_QUALITY_CEILING


def test_verified_verifiable_evidence_escapes_the_cap():
    ev = Evidence(
        etype=EvidenceType.CARRIER_TRACKING,
        party=Party.MERCHANT,
        quality=0.95,
        verified=True,
    )
    assert ev.effective_quality == 0.95


def test_verified_but_unverifiable_type_still_capped():
    """Marking a narrative 'verified' must not buy trust it cannot earn."""
    ev = Evidence(
        etype=EvidenceType.CM_NARRATIVE,
        party=Party.CARD_MEMBER,
        quality=0.9,
        verified=True,
    )
    assert ev.effective_quality == UNVERIFIED_QUALITY_CEILING


def test_quality_out_of_range_rejected():
    with pytest.raises(ValueError):
        Evidence(etype=EvidenceType.RECEIPT, party=Party.MERCHANT, quality=1.5)


def test_evidence_id_is_deterministic():
    kw = dict(etype=EvidenceType.RECEIPT, party=Party.MERCHANT, content="order 123")
    assert Evidence(**kw).evidence_id == Evidence(**kw).evidence_id


def test_party_opponent():
    assert Party.CARD_MEMBER.opponent is Party.MERCHANT
    assert Party.MERCHANT.opponent is Party.CARD_MEMBER
    assert Party.NETWORK.opponent is Party.NETWORK


# --------------------------------------------------------------------------------------
# Statute
# --------------------------------------------------------------------------------------


def _ctx(code: ReasonCode, evidence=(), **kw) -> StatuteContext:
    return StatuteContext(spec=get_spec(code), evidence=tuple(evidence), **kw)


def test_rules_are_registered():
    assert registered_rule_count() >= 8


def test_c08_with_no_delivery_proof_raises_a_presumption_for_card_member():
    """Near-decisive, not dispositive: a merchant who filed nothing may still be
    right, and a clamp would report confidence 1.0 and foreclose abstention."""
    outcome = evaluate(_ctx(ReasonCode.C08))
    assert not outcome.is_decided
    finding = next(f for f in outcome.findings if f.rule_id == "C08.NO_DELIVERY_PROOF")
    assert finding.force is RuleForce.STRONG
    assert finding.favours is Party.CARD_MEMBER
    assert outcome.logodds_delta >= PRESUMPTION_LOGODDS - STRONG_LOGODDS


def test_c08_with_verified_delivery_is_not_dispositive():
    ev = Evidence(
        etype=EvidenceType.DELIVERY_CONFIRMATION,
        party=Party.MERCHANT,
        verified=True,
        quality=0.95,
    )
    outcome = evaluate(_ctx(ReasonCode.C08, [ev]))
    assert not outcome.is_decided


def test_c08_unverified_delivery_is_strong_not_dispositive():
    ev = Evidence(etype=EvidenceType.CARRIER_TRACKING, party=Party.MERCHANT, verified=False)
    outcome = evaluate(_ctx(ReasonCode.C08, [ev]))
    assert not outcome.is_decided
    assert outcome.logodds_delta > 0  # favours card member
    assert any(f.rule_id == "C08.UNVERIFIED_DELIVERY" for f in outcome.findings)


def test_address_mismatch_favours_card_member():
    ev = Evidence(
        etype=EvidenceType.DELIVERY_CONFIRMATION,
        party=Party.MERCHANT,
        verified=True,
        metadata={"address_match": False},
    )
    outcome = evaluate(_ctx(ReasonCode.C08, [ev]))
    assert any(f.rule_id == "C08.ADDRESS_MISMATCH" for f in outcome.findings)
    assert outcome.logodds_delta > 0


def test_late_return_is_dispositive_for_merchant():
    """The canonical statute case: day 16 against a 14-day window."""
    outcome = evaluate(
        _ctx(ReasonCode.C04, return_shipped_day=16, return_window_days=14)
    )
    assert outcome.is_decided
    assert outcome.clamp == -DISPOSITIVE_LOGODDS
    assert outcome.dispositive.favours is Party.MERCHANT


def test_return_on_final_day_is_timely():
    outcome = evaluate(
        _ctx(ReasonCode.C04, return_shipped_day=14, return_window_days=14)
    )
    assert not any(f.rule_id == "POLICY.RETURN_LATE" for f in outcome.findings)


def test_late_return_on_damaged_goods_is_only_strong():
    """C32 damage is not bound by an ordinary return window the same way."""
    outcome = evaluate(
        _ctx(ReasonCode.C32, return_shipped_day=30, return_window_days=14)
    )
    assert not outcome.is_decided
    assert outcome.logodds_delta < 0


def test_timely_verified_return_without_credit_favours_card_member():
    ev = Evidence(
        etype=EvidenceType.RETURN_TRACKING, party=Party.CARD_MEMBER, verified=True
    )
    outcome = evaluate(
        _ctx(ReasonCode.C04, [ev], return_shipped_day=5, return_window_days=14)
    )
    assert any(f.rule_id == "POLICY.TIMELY_RETURN_NO_CREDIT" for f in outcome.findings)


def test_merchant_no_reply_raises_a_rebuttable_presumption():
    """Silence normally loses, but is NOT a hard clamp.

    As a dispositive rule this had a 20.8% error rate against ground truth — the
    worst of any statute rule — because a merchant who was factually right but
    missed a deadline was clamped to a confident wrong verdict. A clamp also
    reports confidence 1.0, so the conformal layer could never abstain on exactly
    the cases where the system was most often wrong.
    """
    spec = get_spec(ReasonCode.C31)
    outcome = evaluate(
        _ctx(ReasonCode.C31, merchant_response_days=spec.representment_window_days + 1)
    )
    finding = next(f for f in outcome.findings if f.rule_id == "PROC.NO_REPLY")
    assert finding.force is RuleForce.STRONG
    assert not finding.is_dispositive
    assert finding.favours is Party.CARD_MEMBER
    assert outcome.clamp is None
    # Still heavy enough to decide an otherwise empty record.
    assert outcome.logodds_delta >= PRESUMPTION_LOGODDS - STRONG_LOGODDS


def test_silence_alone_still_loses_the_case():
    case = DisputeCase(reason_code=ReasonCode.C31, merchant_response_days=99)
    assert adjudicate(case).verdict is Verdict.CARD_MEMBER


def test_a_compelling_record_can_rebut_the_presumption():
    """The property the presumption exists to allow."""
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        merchant_response_days=99,
        evidence=[
            Evidence(
                etype=EvidenceType.VISUAL_SIMILARITY,
                party=Party.NETWORK,
                verified=True,
                quality=0.95,
                metadata={"lambda_lr": -2.4, "cosine": 0.97},
            ),
            Evidence(
                etype=EvidenceType.PRODUCT_DESCRIPTION,
                party=Party.MERCHANT,
                quality=0.6,
                metadata={"lambda_lr": -2.0},
            ),
        ],
    )
    assert adjudicate(case).verdict is not Verdict.CARD_MEMBER


def test_merchant_reply_on_deadline_is_timely():
    spec = get_spec(ReasonCode.C31)
    outcome = evaluate(
        _ctx(ReasonCode.C31, merchant_response_days=spec.representment_window_days)
    )
    assert not any(f.rule_id == "PROC.NO_REPLY" for f in outcome.findings)


def test_posted_refund_moots_the_dispute():
    outcome = evaluate(_ctx(ReasonCode.C02, refund_already_posted=True))
    assert outcome.is_decided
    assert outcome.clamp == -DISPOSITIVE_LOGODDS


def test_confirmed_duplicate_decides_p08():
    outcome = evaluate(_ctx(ReasonCode.P08, duplicate_confirmed=True))
    assert outcome.is_decided
    assert outcome.dispositive.favours is Party.CARD_MEMBER


def test_three_ds_success_favours_merchant_on_auth_codes():
    ev = Evidence(
        etype=EvidenceType.THREE_DS_RESULT,
        party=Party.NETWORK,
        verified=True,
        metadata={"authenticated": True},
    )
    outcome = evaluate(_ctx(ReasonCode.F29, [ev]))
    assert any(f.rule_id == "AUTH.3DS_SUCCESS" for f in outcome.findings)
    assert outcome.logodds_delta < 0


def test_three_ds_does_not_apply_to_fulfilment_codes():
    ev = Evidence(
        etype=EvidenceType.THREE_DS_RESULT,
        party=Party.NETWORK,
        verified=True,
        metadata={"authenticated": True},
    )
    outcome = evaluate(_ctx(ReasonCode.C08, [ev]))
    assert not any(f.rule_id == "AUTH.3DS_SUCCESS" for f in outcome.findings)


def test_conflicting_dispositive_rules_resolve_to_merchant():
    """Merchant default and a posted refund both fire; the tie breaks toward merchant."""
    outcome = evaluate(
        _ctx(
            ReasonCode.C02,
            merchant_response_days=99,
            refund_already_posted=True,
        )
    )
    assert outcome.is_decided
    assert outcome.clamp == -DISPOSITIVE_LOGODDS
    # Both findings stay visible for the audit trail.
    ids = {f.rule_id for f in outcome.findings}
    assert {"PROC.NO_REPLY", "PROC.REFUND_POSTED"} <= ids


def test_findings_carry_citations():
    outcome = evaluate(_ctx(ReasonCode.C08))
    assert all(f.guide_reference for f in outcome.findings)
    assert all(f.rationale for f in outcome.findings)


def test_advisory_rules_do_not_move_the_posterior():
    outcome = evaluate(_ctx(ReasonCode.C08))
    advisory = [f for f in outcome.findings if f.force is RuleForce.ADVISORY]
    assert all(f.logodds_delta == 0.0 for f in advisory)

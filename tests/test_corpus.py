"""Stage 3 — synthetic corpus generation and the burden rules it exposed."""

from __future__ import annotations

import json
import random

import pytest

from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.ledger import DisputeCase, Verdict, adjudicate
from arbiter.core.reason_codes import ReasonCode, get_spec
from arbiter.core.statute import StatuteContext, evaluate
from arbiter.data.generator import (
    CorpusConfig,
    generate_case,
    generate_corpus,
    write_corpus,
)
from arbiter.data.narratives import render_cm_narrative, render_merchant_rebuttal
from arbiter.data.scenarios import (
    SCENARIOS,
    GroundTruth,
    all_scenarios,
    covered_codes,
    scenarios_for,
)

# --------------------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------------------


def test_scenario_ids_are_unique():
    ids = [s.scenario_id for s in all_scenarios()]
    assert len(ids) == len(set(ids))


def test_every_scenario_has_a_registered_reason_code():
    for s in all_scenarios():
        get_spec(s.reason_code)  # raises if unregistered


def test_scenarios_cover_both_outcomes_and_ambiguity():
    truths = {s.truth for s in all_scenarios()}
    assert truths == set(GroundTruth)


def test_high_volume_codes_have_multiple_scenarios():
    for code in (ReasonCode.C08, ReasonCode.C02, ReasonCode.C04, ReasonCode.C31):
        assert len(scenarios_for(code)) >= 2


def test_ambiguous_scenarios_are_marked_hard():
    """If a case is genuinely undecidable, its difficulty should say so."""
    for s in all_scenarios():
        if s.truth is GroundTruth.GENUINELY_AMBIGUOUS:
            assert s.difficulty >= 0.7, s.scenario_id


def test_scenarios_for_filters_by_code():
    assert all(s.reason_code is ReasonCode.C08 for s in scenarios_for(ReasonCode.C08))


def test_covered_codes_non_empty():
    assert len(covered_codes()) >= 10


# --------------------------------------------------------------------------------------
# narratives
# --------------------------------------------------------------------------------------


def test_narratives_are_non_empty():
    rng = random.Random(0)
    ctx = {"item": "kettle", "amount": 40.0}
    for code in (ReasonCode.C08, ReasonCode.C02, ReasonCode.C31):
        assert render_cm_narrative(code, ctx, rng).strip()
        assert render_merchant_rebuttal(code, ctx, rng).strip()


def test_narratives_vary():
    """A classifier trained on three templates learns the templates."""
    rng = random.Random(1)
    ctx = {"item": "kettle", "amount": 40.0}
    seen = {render_cm_narrative(ReasonCode.C08, ctx, rng) for _ in range(40)}
    assert len(seen) > 20


def test_narrative_generation_is_reproducible():
    ctx = {"item": "kettle", "amount": 40.0}
    a = render_cm_narrative(ReasonCode.C08, ctx, random.Random(7))
    b = render_cm_narrative(ReasonCode.C08, ctx, random.Random(7))
    assert a == b


def test_zero_noise_leaves_text_clean():
    rng = random.Random(3)
    text = render_cm_narrative(ReasonCode.C08, {"item": "x"}, rng, noise=0.0)
    assert text == text.strip()


# --------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------


def test_corpus_is_reproducible():
    a = [g.case_id for g in generate_corpus(50, CorpusConfig(seed=11))]
    b = [g.case_id for g in generate_corpus(50, CorpusConfig(seed=11))]
    assert a == b


def test_different_seeds_differ():
    a = [g.case_id for g in generate_corpus(50, CorpusConfig(seed=1))]
    b = [g.case_id for g in generate_corpus(50, CorpusConfig(seed=2))]
    assert a != b


def test_generated_cases_are_adjudicable():
    for gen in generate_corpus(200, CorpusConfig(seed=5)):
        adj = adjudicate(gen.case)
        assert adj.verdict in set(Verdict)
        assert -20 <= adj.posterior_logodds <= 20


def test_every_case_has_a_card_member_narrative():
    for gen in generate_corpus(100, CorpusConfig(seed=6)):
        assert any(
            e.etype is EvidenceType.CM_NARRATIVE for e in gen.case.evidence
        )


def test_silent_merchants_file_nothing():
    """A merchant past the window should have no exhibits in the record."""
    found = False
    for gen in generate_corpus(400, CorpusConfig(seed=8)):
        spec = get_spec(gen.reason_code)
        if (gen.case.merchant_response_days or 0) > spec.representment_window_days:
            found = True
            assert not [e for e in gen.case.evidence if e.party is Party.MERCHANT]
    assert found, "no silent-merchant case generated"


def test_record_round_trips_to_json():
    gen = next(iter(generate_corpus(1, CorpusConfig(seed=9))))
    record = gen.to_record()
    assert json.loads(json.dumps(record))["case_id"] == gen.case_id


def test_record_carries_ground_truth_and_facts():
    gen = next(iter(generate_corpus(1, CorpusConfig(seed=10))))
    rec = gen.to_record()
    assert rec["truth"] in {t.value for t in GroundTruth}
    assert "submission_delay_days" in rec["facts"]
    assert "cancel_day" in rec["facts"]


def test_write_corpus_creates_jsonl(tmp_path):
    out = tmp_path / "corpus.jsonl"
    stats = write_corpus(out, 40, CorpusConfig(seed=12))
    assert stats["cases"] == 40
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 40
    assert json.loads(lines[0])["case_id"]


def test_amounts_within_configured_range():
    cfg = CorpusConfig(seed=13, amount_min=50, amount_max=100)
    for gen in generate_corpus(50, cfg):
        assert 50 <= gen.amount <= 100


def test_generate_corpus_rejects_empty_scenario_pool():
    with pytest.raises(ValueError):
        list(generate_corpus(5, CorpusConfig(), scenarios=[]))


def test_single_scenario_pool_is_honoured():
    scenario = SCENARIOS["C08.never_shipped"]
    gens = list(generate_corpus(20, CorpusConfig(seed=14), scenarios=[scenario]))
    assert all(g.scenario_id == "C08.never_shipped" for g in gens)


def test_generate_case_is_deterministic_for_a_seed():
    s = SCENARIOS["C31.materially_different"]
    a = generate_case(s, random.Random(2), CorpusConfig(), index=0)
    b = generate_case(s, random.Random(2), CorpusConfig(), index=0)
    assert a.cm_narrative == b.cm_narrative


# --------------------------------------------------------------------------------------
# burden rules — added after the corpus exposed absence-of-evidence as neutral
# --------------------------------------------------------------------------------------


def _ctx(code: ReasonCode, evidence=(), **kw) -> StatuteContext:
    return StatuteContext(spec=get_spec(code), evidence=tuple(evidence), **kw)


def test_card_member_filing_nothing_loses_ground_on_their_own_burden():
    """C04 puts the burden on the Card Member; a bare claim does not discharge it."""
    outcome = evaluate(_ctx(ReasonCode.C04, merchant_response_days=5))
    assert any(f.rule_id == "BURDEN.CM_UNMET" for f in outcome.findings)
    assert outcome.logodds_delta < 0


def test_a_narrative_alone_never_discharges_a_burden():
    ev = [Evidence(etype=EvidenceType.CM_NARRATIVE, party=Party.CARD_MEMBER)]
    outcome = evaluate(_ctx(ReasonCode.C04, ev, merchant_response_days=5))
    assert any(f.rule_id == "BURDEN.CM_UNMET" for f in outcome.findings)


def test_filing_the_required_exhibit_discharges_the_burden():
    ev = [Evidence(etype=EvidenceType.RETURN_TRACKING, party=Party.CARD_MEMBER, verified=True)]
    outcome = evaluate(_ctx(ReasonCode.C04, ev, merchant_response_days=5))
    assert not any(f.rule_id == "BURDEN.CM_UNMET" for f in outcome.findings)


def test_merchant_burden_needs_a_substantiated_claim_first():
    """A merchant cannot document an event that never happened.

    Without this guard the rule punished merchants for being right: in
    "no refund was ever agreed" there is no credit note to produce.
    """
    outcome = evaluate(_ctx(ReasonCode.C02, merchant_response_days=5))
    assert not any(f.rule_id == "BURDEN.MERCHANT_UNMET" for f in outcome.findings)


def test_substantiated_claim_activates_merchant_burden():
    ev = [Evidence(etype=EvidenceType.EMAIL_THREAD, party=Party.CARD_MEMBER)]
    outcome = evaluate(_ctx(ReasonCode.C02, ev, merchant_response_days=5))
    assert any(f.rule_id == "BURDEN.MERCHANT_UNMET" for f in outcome.findings)
    assert outcome.logodds_delta > 0


def test_merchant_burden_does_not_double_count_with_c08_rule():
    """C08 has its own dispositive rule; the generic one must stay out of the way."""
    outcome = evaluate(_ctx(ReasonCode.C08, merchant_response_days=5))
    assert not any(f.rule_id == "BURDEN.MERCHANT_UNMET" for f in outcome.findings)


def test_late_settlement_submission_favours_card_member():
    """P07 measures the merchant's delay, not the Card Member's."""
    outcome = evaluate(_ctx(ReasonCode.P07, submission_delay_days=95))
    finding = next(f for f in outcome.findings if f.rule_id == "P07.LATE_SUBMISSION")
    assert finding.favours is Party.CARD_MEMBER
    assert outcome.is_decided


def test_prompt_submission_does_not_fire_p07():
    outcome = evaluate(_ctx(ReasonCode.P07, submission_delay_days=10))
    assert not any(f.rule_id == "P07.LATE_SUBMISSION" for f in outcome.findings)


def test_submission_delay_is_distinct_from_filing_delay():
    """Conflating the two inverted P07; they must not be the same field."""
    case = DisputeCase(
        reason_code=ReasonCode.P07,
        submission_delay_days=95,
        days_since_transaction=20,
    )
    assert adjudicate(case).verdict is Verdict.CARD_MEMBER


def test_late_cancellation_favours_merchant():
    outcome = evaluate(_ctx(ReasonCode.C05, cancel_day=5, cancel_window_days=3))
    assert any(f.rule_id == "POLICY.CANCEL_LATE" for f in outcome.findings)
    assert outcome.logodds_delta < 0


def test_timely_cancellation_does_not_fire():
    outcome = evaluate(_ctx(ReasonCode.C05, cancel_day=1, cancel_window_days=3))
    assert not any(f.rule_id == "POLICY.CANCEL_LATE" for f in outcome.findings)


# --------------------------------------------------------------------------------------
# end-to-end quality gates
#
# These lock in the behaviour the corpus was built to measure. They are deliberately
# loose: they guard against regression, not against a specific tuned number.
# --------------------------------------------------------------------------------------

_TRUTH_TO_VERDICT = {
    GroundTruth.CARD_MEMBER_RIGHT: Verdict.CARD_MEMBER,
    GroundTruth.MERCHANT_RIGHT: Verdict.MERCHANT,
}


@pytest.fixture(scope="module")
def scored():
    return [
        (g.truth, g.difficulty, adjudicate(g.case))
        for g in generate_corpus(1500, CorpusConfig(seed=42))
    ]


def test_agreement_with_ground_truth_is_high(scored):
    decided = [
        (t, a) for t, _, a in scored
        if t is not GroundTruth.GENUINELY_AMBIGUOUS and a.verdict is not Verdict.CONTESTED
    ]
    hits = sum(a.verdict is _TRUTH_TO_VERDICT[t] for t, a in decided)
    assert hits / len(decided) >= 0.80


def test_easy_cases_are_resolved_reliably(scored):
    easy = [
        (t, a) for t, d, a in scored
        if d < 0.35 and t is not GroundTruth.GENUINELY_AMBIGUOUS
        and a.verdict is not Verdict.CONTESTED
    ]
    hits = sum(a.verdict is _TRUTH_TO_VERDICT[t] for t, a in easy)
    assert hits / len(easy) >= 0.88


def test_ambiguous_cases_abstain_more_than_clear_ones(scored):
    """The property Stage 5 calibrates: uncertainty is ranked correctly.

    The absolute abstention rate depends on CONTESTED_BAND, which is a heuristic
    until conformal calibration replaces it. What must hold now is the *ordering*.
    """
    def rate(pred):
        rows = [a for t, _, a in scored if pred(t)]
        return sum(a.verdict is Verdict.CONTESTED for a in rows) / len(rows)

    ambiguous = rate(lambda t: t is GroundTruth.GENUINELY_AMBIGUOUS)
    clear = rate(lambda t: t is not GroundTruth.GENUINELY_AMBIGUOUS)
    assert ambiguous > clear


def test_no_reason_code_collapses(scored):
    """Guards the failure this stage found: a whole code performing at chance."""
    by_code: dict[str, list[int]] = {}
    for truth, _, adj in scored:
        if truth is GroundTruth.GENUINELY_AMBIGUOUS or adj.verdict is Verdict.CONTESTED:
            continue
        pair = by_code.setdefault(adj.reason_code.value, [0, 0])
        pair[0] += adj.verdict is _TRUTH_TO_VERDICT[truth]
        pair[1] += 1

    for code, (hits, n) in by_code.items():
        if n >= 30:
            assert hits / n >= 0.60, f"{code} at {hits / n:.0%} over {n} cases"


def test_statute_decides_a_meaningful_share(scored):
    share = sum(a.decided_by_statute for _, _, a in scored) / len(scored)
    assert 0.10 <= share <= 0.60

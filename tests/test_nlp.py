"""Stage 4 — hypothesis templates, NLI verification, reason-code classification.

Every test here runs against the offline stub backend. Nothing downloads weights, so
the suite stays fast and works without a network.
"""

from __future__ import annotations

import math

import pytest

from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.ledger import (
    NLI_CONTRADICTION_FLOOR,
    DisputeCase,
    adjudicate,
    compute_lambda,
)
from arbiter.core.reason_codes import ReasonCode
from arbiter.data.generator import CorpusConfig, generate_corpus
from arbiter.nlp.classifier import (
    AMBIGUITY_THRESHOLD,
    CodePrediction,
    KeywordClassifier,
    TransformerClassifier,
    classify,
    evaluate_classifier,
)
from arbiter.nlp.hypotheses import (
    has_specific_hypothesis,
    hypothesis_for,
    supported_pairs,
)
from arbiter.nlp.verifier import (
    MAX_NLI_LAMBDA,
    EvidenceVerifier,
    NLIResult,
    StubNLIBackend,
    TransformerNLIBackend,
    summarise,
    verify_case,
)

# --------------------------------------------------------------------------------------
# hypotheses
# --------------------------------------------------------------------------------------


def test_keyed_hypothesis_beats_type_fallback():
    keyed = hypothesis_for(ReasonCode.C08, EvidenceType.DELIVERY_CONFIRMATION)
    generic = hypothesis_for(ReasonCode.P08, EvidenceType.DELIVERY_CONFIRMATION)
    assert keyed is not None and generic is not None
    assert keyed.text != generic.text
    assert has_specific_hypothesis(ReasonCode.C08, EvidenceType.DELIVERY_CONFIRMATION)
    assert not has_specific_hypothesis(ReasonCode.P08, EvidenceType.DELIVERY_CONFIRMATION)


def test_c08_delivery_hypothesis_demands_receipt_not_dispatch():
    """The distinction the whole code turns on."""
    hyp = hypothesis_for(ReasonCode.C08, EvidenceType.DELIVERY_CONFIRMATION)
    assert "received" in hyp.text.lower() or "cardholder's own address" in hyp.text.lower()
    assert hyp.dispositive_if_contradicted


def test_computed_types_have_no_hypothesis():
    """Visual similarity is measured, not inferred; NLI must not overwrite it."""
    assert hypothesis_for(ReasonCode.C31, EvidenceType.VISUAL_SIMILARITY) is None
    assert hypothesis_for(ReasonCode.P08, EvidenceType.DUPLICATE_TXN_MATCH) is None


def test_every_keyed_hypothesis_has_text():
    for code, etype in supported_pairs():
        hyp = hypothesis_for(code, etype)
        assert hyp and hyp.text.strip()


def test_keyed_hypotheses_carry_rationale_where_subtle():
    """The framing must be challengeable, not just the model's answer."""
    hyp = hypothesis_for(ReasonCode.C32, EvidenceType.PHOTO_OF_ITEM)
    assert hyp.rationale


# --------------------------------------------------------------------------------------
# NLI result arithmetic
# --------------------------------------------------------------------------------------


def test_lambda_sign_follows_entailment():
    assert NLIResult(entail=0.9, neutral=0.07, contradict=0.03).lambda_lr > 0
    assert NLIResult(entail=0.03, neutral=0.07, contradict=0.9).lambda_lr < 0


def test_lambda_is_clamped():
    extreme = NLIResult(entail=0.9999, neutral=0.0, contradict=0.0000001)
    assert abs(extreme.lambda_lr) <= MAX_NLI_LAMBDA


def test_verdict_labels():
    assert NLIResult(0.8, 0.1, 0.1).verdict == "entailment"
    assert NLIResult(0.1, 0.8, 0.1).verdict == "neutral"
    assert NLIResult(0.1, 0.1, 0.8).verdict == "contradiction"


def test_is_contradicted_requires_majority():
    assert NLIResult(0.1, 0.2, 0.7).is_contradicted
    assert not NLIResult(0.3, 0.4, 0.3).is_contradicted


# --------------------------------------------------------------------------------------
# stub backend
# --------------------------------------------------------------------------------------


def test_stub_is_deterministic():
    a = StubNLIBackend().score([("delivered to the address", "it was delivered")])
    b = StubNLIBackend().score([("delivered to the address", "it was delivered")])
    assert a[0].entail == b[0].entail


def test_stub_returns_one_result_per_pair():
    pairs = [("a", "b"), ("c", "d"), ("e", "f")]
    assert len(StubNLIBackend().score(pairs)) == 3


def test_stub_distribution_sums_to_one():
    r = StubNLIBackend().score([("delivered and signed", "it arrived")])[0]
    assert r.entail + r.neutral + r.contradict == pytest.approx(1.0)


def test_stub_separates_positive_from_negative_premises():
    pos = StubNLIBackend().score([("Delivered. Signed by resident.", "it arrived")])[0]
    neg = StubNLIBackend().score(
        [("Never delivered. Still in transit.", "it arrived")]
    )[0]
    assert pos.entail > neg.entail


def test_empty_batch_is_safe():
    assert StubNLIBackend().score([]) == []


# --------------------------------------------------------------------------------------
# verifier
# --------------------------------------------------------------------------------------


def _ev(etype: EvidenceType, party: Party, **meta) -> Evidence:
    return Evidence(etype=etype, party=party, content=meta.pop("content", ""), metadata=meta)


def test_verifier_annotates_metadata():
    ev = [_ev(EvidenceType.DELIVERY_CONFIRMATION, Party.MERCHANT, content="Delivered 3 Mar")]
    EvidenceVerifier().verify(ev, ReasonCode.C08)
    assert "entail" in ev[0].metadata
    assert "contradict" in ev[0].metadata
    assert ev[0].metadata["nli_hypothesis"]
    assert ev[0].metadata["nli_backend"] == "stub"


def test_verifier_skips_computed_types():
    """A measured value must survive verification untouched."""
    ev = [_ev(EvidenceType.VISUAL_SIMILARITY, Party.NETWORK, lambda_lr=0.7, cosine=0.4)]
    EvidenceVerifier().verify(ev, ReasonCode.C31)
    assert "entail" not in ev[0].metadata
    assert ev[0].metadata["lambda_lr"] == 0.7


def test_verifier_skips_explicit_lambda():
    ev = [_ev(EvidenceType.AVS_MATCH, Party.NETWORK, lambda_lr=-0.9)]
    EvidenceVerifier().verify(ev, ReasonCode.F29)
    assert "entail" not in ev[0].metadata


def test_verifier_caches_repeated_pairs():
    v = EvidenceVerifier()
    ev = [
        _ev(EvidenceType.RECEIPT, Party.MERCHANT, content="same text"),
        _ev(EvidenceType.RECEIPT, Party.MERCHANT, content="same text"),
    ]
    v.verify(ev, ReasonCode.P05)
    assert len(v.cache) == 1


def test_premise_excludes_provenance_boilerplate():
    """Provenance is applied as effective_quality; restating it double-counts.

    It also poisoned the lexical stub, whose negation cues fired on the phrase
    "could not be independently verified" and scored 4 exhibits in 5 as contradictions.
    """
    from arbiter.nlp.verifier import _premise

    text = _premise(
        Evidence(etype=EvidenceType.RECEIPT, party=Party.MERCHANT, verified=False)
    )
    assert "verified" not in text.lower()


def test_premise_folds_in_structured_metadata():
    from arbiter.nlp.verifier import _premise

    text = _premise(
        Evidence(
            etype=EvidenceType.DELIVERY_CONFIRMATION,
            party=Party.MERCHANT,
            metadata={"address_match": False},
        )
    )
    assert "does not match" in text.lower()


class _AlwaysContradicts:
    """Backend that always contradicts, to exercise the self-defeating path."""

    name = "always-contradicts"

    def score(self, pairs):
        return [
            NLIResult(entail=0.02, neutral=0.08, contradict=0.90, hypothesis=h, backend=self.name)
            for _, h in pairs
        ]


def test_self_defeating_flag_set_on_contradiction():
    """A merchant's own delivery proof, contradicted, is flagged for the verdict card."""
    ev = [
        Evidence(
            etype=EvidenceType.DELIVERY_CONFIRMATION,
            party=Party.MERCHANT,
            metadata={"address_match": False},
        )
    ]
    EvidenceVerifier(backend=_AlwaysContradicts()).verify(ev, ReasonCode.C08)
    assert ev[0].metadata["nli_verdict"] == "contradiction"
    assert ev[0].metadata.get("self_defeating")


def test_self_defeating_not_set_for_non_dispositive_hypotheses():
    """Only hypotheses marked dispositive_if_contradicted raise the flag."""
    ev = [Evidence(etype=EvidenceType.CHAT_LOG, party=Party.CARD_MEMBER)]
    EvidenceVerifier(backend=_AlwaysContradicts()).verify(ev, ReasonCode.C08)
    assert not ev[0].metadata.get("self_defeating")


def test_verify_case_returns_the_case():
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[_ev(EvidenceType.CARRIER_TRACKING, Party.MERCHANT, content="Delivered")],
    )
    assert verify_case(case) is case
    assert "entail" in case.evidence[0].metadata


def test_summarise_counts_verdicts():
    ev = [
        _ev(EvidenceType.RECEIPT, Party.MERCHANT, content="Delivered and signed"),
        _ev(EvidenceType.CHAT_LOG, Party.CARD_MEMBER, content="Never arrived"),
    ]
    EvidenceVerifier().verify(ev, ReasonCode.C08)
    s = summarise(ev)
    assert s["scored"] == 2
    assert sum(s["by_verdict"].values()) == 2


# --------------------------------------------------------------------------------------
# NLI -> ledger integration
#
# The key design decision of this stage: NLI *modulates* the type prior rather than
# replacing it. Letting the raw entail/contradict ratio stand as lambda measured
# 88.8% -> 77.8% on 800 cases with a real DeBERTa-MNLI model, because 2376 of 2888
# exhibits scored neutral and neutral still yields a confident-looking ratio.
# --------------------------------------------------------------------------------------


def test_neutral_nli_stays_near_the_type_prior():
    """The regression this design prevents: neutral must not overwrite the prior."""
    base = compute_lambda(
        Evidence(etype=EvidenceType.DELIVERY_CONFIRMATION, party=Party.MERCHANT)
    )
    neutral = compute_lambda(
        Evidence(
            etype=EvidenceType.DELIVERY_CONFIRMATION,
            party=Party.MERCHANT,
            metadata={"entail": 0.10, "neutral": 0.85, "contradict": 0.05},
        )
    )
    assert math.copysign(1, neutral) == math.copysign(1, base)
    assert abs(neutral - base) < abs(base) * 0.5


def test_entailment_amplifies_the_type_prior():
    base = compute_lambda(
        Evidence(etype=EvidenceType.DELIVERY_CONFIRMATION, party=Party.MERCHANT)
    )
    strong = compute_lambda(
        Evidence(
            etype=EvidenceType.DELIVERY_CONFIRMATION,
            party=Party.MERCHANT,
            metadata={"entail": 0.93, "neutral": 0.05, "contradict": 0.02},
        )
    )
    assert abs(strong) > abs(base)


def test_contradiction_flips_the_sign():
    """A delivery confirmation naming the wrong address should help the Card Member."""
    lam = compute_lambda(
        Evidence(
            etype=EvidenceType.DELIVERY_CONFIRMATION,
            party=Party.MERCHANT,
            metadata={"entail": 0.05, "neutral": 0.15, "contradict": 0.80},
        )
    )
    assert lam > 0  # positive favours the Card Member


def test_weak_contradiction_does_not_flip():
    """Flipping an exhibit against its filer is a strong claim; require confidence."""
    lam = compute_lambda(
        Evidence(
            etype=EvidenceType.DELIVERY_CONFIRMATION,
            party=Party.MERCHANT,
            metadata={
                "entail": 0.30,
                "neutral": 0.40,
                "contradict": NLI_CONTRADICTION_FLOOR - 0.15,
            },
        )
    )
    assert lam < 0  # still favours the merchant


def test_explicit_lambda_still_wins_over_nli():
    lam = compute_lambda(
        Evidence(
            etype=EvidenceType.VISUAL_SIMILARITY,
            party=Party.NETWORK,
            metadata={"lambda_lr": 1.5, "entail": 0.01, "contradict": 0.98},
        )
    )
    assert lam == 1.5


def test_verified_corpus_still_adjudicates():
    """Verification must not break the ledger, including its audit identity.

    The identity only holds when the statute did not clamp — a dispositive rule
    deliberately bypasses the evidence sum.
    """
    v = EvidenceVerifier()
    for gen in generate_corpus(120, CorpusConfig(seed=21)):
        v.verify(gen.case.evidence, gen.reason_code)
        adj = adjudicate(gen.case)
        assert -20 <= adj.posterior_logodds <= 20
        if not adj.decided_by_statute:
            assert adj.audit_sum() == pytest.approx(adj.posterior_logodds)


# --------------------------------------------------------------------------------------
# reason-code classifier
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My laptop stand never arrived and tracking has not moved.", ReasonCode.C08),
        ("They promised me a refund and the credit never appeared.", ReasonCode.C02),
        ("I sent the phone case back with tracking and they still charged me.", ReasonCode.C04),
        ("This order was cancelled but the charge went through.", ReasonCode.C05),
        ("I cancelled my subscription and they keep billing me.", ReasonCode.C28),
        ("The item is not what was advertised, looks nothing like the photos.", ReasonCode.C31),
        ("The blender arrived broken and does not work.", ReasonCode.C32),
        ("I have been charged twice for the same order.", ReasonCode.P08),
        ("They billed me more than the confirmation email said.", ReasonCode.P05),
        ("I did not make this transaction and never authorised it.", ReasonCode.F29),
        ("I asked the merchant for documentation and got no response at all.", ReasonCode.R13),
    ],
)
def test_classifier_maps_narratives_to_codes(text, expected):
    assert classify(text).code is expected


def test_classifier_returns_a_distribution():
    pred = classify("My order never arrived.")
    assert pred.distribution
    assert sum(pred.distribution.values()) == pytest.approx(1.0)


def test_classifier_reports_matched_cues():
    pred = classify("I have been charged twice for the same item.")
    assert pred.matched_cues
    assert "P08" in pred.explain()


def test_vague_text_is_flagged_ambiguous():
    """Intake should ask rather than assume when nothing distinctive fires."""
    pred = classify("Please look at this charge, something is wrong.")
    assert pred.is_ambiguous


def test_top_k_ordering():
    pred = classify("My order never arrived.")
    top = pred.top(3)
    assert len(top) == 3
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_ambiguity_threshold_is_respected():
    pred = CodePrediction(code=ReasonCode.C08, confidence=AMBIGUITY_THRESHOLD - 0.01)
    assert pred.is_ambiguous


def test_classifier_accuracy_on_corpus():
    """Regression gate. Measured 92.8% top-1 at the time of writing."""
    samples = [
        (g.cm_narrative, g.reason_code)
        for g in generate_corpus(600, CorpusConfig(seed=42))
    ]
    res = evaluate_classifier(samples, KeywordClassifier())
    assert res["top1"] >= 0.85
    assert res["top3"] >= 0.92


def test_c08_cues_do_not_swallow_other_codes():
    """The bug this guards: bare "never"/"no" cues pulled 60-70% of C04, C05 and
    R13 into C08, because "I heard nothing since" and "got no response" matched."""
    assert classify("The blender went back to them on day 6 and I heard nothing since.").code is ReasonCode.C04
    assert classify("I asked for documentation and got no response.").code is ReasonCode.R13
    assert classify("This order was cancelled but the charge went through.").code is ReasonCode.C05


def test_refund_promise_outranks_return_mention():
    """"I returned it and was told a credit was coming" is both C04 and C02;
    the broken promise is the gravamen."""
    text = "I returned the winter coat and was told a credit was coming. It never came."
    assert classify(text).code is ReasonCode.C02


# --------------------------------------------------------------------------------------
# transformer backends — construction only, never loaded
# --------------------------------------------------------------------------------------


def test_transformer_nli_construction_is_offline():
    """Constructing must not touch the network; loading is lazy."""
    backend = TransformerNLIBackend()
    assert backend._model is None
    assert "transformer" in backend.name


def test_transformer_classifier_falls_back_when_unavailable():
    clf = TransformerClassifier(model_path="definitely/not-a-real-model")
    pred = clf.predict("My order never arrived and tracking has not moved.")
    assert pred.code is ReasonCode.C08
    assert pred.backend == "keyword"

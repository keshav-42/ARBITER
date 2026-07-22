"""Stage 5 — temperature scaling, conformal abstention, and routing."""

from __future__ import annotations

import math
import random

import pytest

from arbiter.calibration.conformal import (
    DEFAULT_ALPHA,
    ConformalCalibrator,
    PredictionSet,
    Route,
    calibrate,
    evaluate_coverage,
)
from arbiter.calibration.temperature import (
    TemperatureScaler,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)
from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.ledger import DisputeCase, Verdict, adjudicate
from arbiter.core.reason_codes import ReasonCode
from arbiter.data.generator import CorpusConfig, generate_corpus
from arbiter.data.scenarios import GroundTruth


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@pytest.fixture(scope="module")
def labelled():
    """Held-out (logodds, card_member_was_right) pairs, statute cases excluded.

    A dispositive rule is a procedural determination, not a probabilistic one, so
    including its pinned confidence would corrupt the conformal quantile.
    """
    pool = []
    for g in generate_corpus(4000, CorpusConfig(seed=303)):
        adj = adjudicate(g.case)
        if adj.decided_by_statute or g.truth is GroundTruth.GENUINELY_AMBIGUOUS:
            continue
        pool.append(
            (adj.posterior_logodds, g.truth is GroundTruth.CARD_MEMBER_RIGHT)
        )
    return pool


@pytest.fixture(scope="module")
def split(labelled):
    half = len(labelled) // 2
    return labelled[:half], labelled[half:]


# --------------------------------------------------------------------------------------
# temperature scaling
# --------------------------------------------------------------------------------------


def test_default_temperature_is_a_noop():
    s = TemperatureScaler()
    assert s.probability(1.3) == pytest.approx(_sigmoid(1.3))


def test_fit_recovers_a_known_temperature():
    """Generate data that is overconfident by 2x and check the fit finds it."""
    rng = random.Random(0)
    true_t = 2.0
    logodds, labels = [], []
    for _ in range(4000):
        z = rng.uniform(-6, 6)
        logodds.append(z)
        labels.append(rng.random() < _sigmoid(z / true_t))
    s = TemperatureScaler().fit(logodds, labels)
    assert s.fitted
    assert s.temperature == pytest.approx(true_t, abs=0.25)


def test_scaling_never_changes_a_verdict():
    """Monotonicity is why it is safe to apply before conformal calibration."""
    s = TemperatureScaler(temperature=3.4, fitted=True)
    for z in (-5.0, -0.7, -0.01, 0.01, 0.7, 5.0):
        assert (s.probability(z) >= 0.5) == (z >= 0)


def test_higher_temperature_softens_confidence():
    cold = TemperatureScaler(temperature=1.0)
    warm = TemperatureScaler(temperature=3.0)
    assert warm.confidence(2.0) < cold.confidence(2.0)


def test_confidence_is_in_range():
    s = TemperatureScaler(temperature=1.6)
    for z in (-9.0, 0.0, 9.0):
        assert 0.5 <= s.confidence(z) <= 1.0


def test_overconfidence_flag():
    assert TemperatureScaler(temperature=1.8).is_overconfident
    assert not TemperatureScaler(temperature=0.9).is_overconfident


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        TemperatureScaler().fit([0.1, 0.2], [True])


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        TemperatureScaler().fit([], [])


def test_ledger_is_overconfident_in_practice(split):
    """Measured T ~ 1.8: the hand-built posterior really is too sharp."""
    cal, _ = split
    s = TemperatureScaler().fit([z for z, _ in cal], [y for _, y in cal])
    assert s.temperature > 1.2


# --------------------------------------------------------------------------------------
# calibration metrics
# --------------------------------------------------------------------------------------


def test_perfect_calibration_has_zero_ece():
    conf = [0.9] * 90 + [0.9] * 10
    correct = [True] * 90 + [False] * 10
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_overconfidence_shows_up_as_ece():
    conf = [0.99] * 100
    correct = [True] * 60 + [False] * 40
    assert expected_calibration_error(conf, correct) > 0.3


def test_reliability_bins_partition_the_sample():
    conf = [0.5, 0.6, 0.75, 0.9, 1.0]
    correct = [True, False, True, True, True]
    bins = reliability_bins(conf, correct, n_bins=5)
    assert sum(b.count for b in bins) == len(conf)


def test_final_bin_includes_probability_one():
    bins = reliability_bins([1.0], [True], n_bins=10)
    assert sum(b.count for b in bins) == 1


def test_mce_catches_a_single_bad_bin():
    conf = [0.55] * 100 + [0.99] * 10
    correct = [True] * 55 + [False] * 45 + [False] * 10
    assert maximum_calibration_error(conf, correct) > expected_calibration_error(conf, correct)


def test_ece_on_empty_input():
    assert expected_calibration_error([], []) == 0.0


def test_reliability_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        reliability_bins([0.6], [True, False])


# --------------------------------------------------------------------------------------
# conformal mechanics
# --------------------------------------------------------------------------------------


def test_fit_sets_threshold_and_records_size(split):
    cal, _ = split
    c = ConformalCalibrator(alpha=0.1).fit([z for z, _ in cal], [y for _, y in cal])
    assert c.fitted
    assert 0.0 < c.threshold < 1.0
    assert c.n_calibration == len(cal)


def test_lower_alpha_lowers_the_threshold(split):
    """Demanding more coverage means admitting labels more readily."""
    cal, _ = split
    z, y = [a for a, _ in cal], [b for _, b in cal]
    loose = ConformalCalibrator(alpha=0.20).fit(z, y)
    tight = ConformalCalibrator(alpha=0.05).fit(z, y)
    assert tight.threshold < loose.threshold


def test_confident_posterior_gives_a_singleton(split):
    cal, _ = split
    c = ConformalCalibrator(alpha=0.1).fit([z for z, _ in cal], [y for _, y in cal])
    assert c.predict_set(6.0).is_singleton
    assert c.predict_set(6.0).verdict is Verdict.CARD_MEMBER
    assert c.predict_set(-6.0).verdict is Verdict.MERCHANT


def test_balanced_posterior_gives_a_doubleton(split):
    cal, _ = split
    c = ConformalCalibrator(alpha=0.05).fit([z for z, _ in cal], [y for _, y in cal])
    pset = c.predict_set(0.0)
    assert pset.size == 2
    assert pset.verdict is Verdict.CONTESTED


def test_prediction_set_membership():
    pset = PredictionSet(
        labels=frozenset({Verdict.CARD_MEMBER}),
        p_card_member=0.8,
        threshold=0.4,
        alpha=0.1,
    )
    assert pset.contains(Verdict.CARD_MEMBER)
    assert not pset.contains(Verdict.MERCHANT)
    assert not pset.is_empty


def test_empty_set_is_flagged():
    """An empty set means the case is outside calibrated experience."""
    pset = PredictionSet(
        labels=frozenset(), p_card_member=0.5, threshold=0.9, alpha=0.1
    )
    assert pset.is_empty
    assert pset.verdict is Verdict.CONTESTED


def test_fit_rejects_empty_calibration_set():
    with pytest.raises(ValueError):
        ConformalCalibrator().fit([], [])


def test_conformal_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ConformalCalibrator().fit([0.1], [True, False])


def test_calibrate_helper():
    c = calibrate([1.0, -1.0, 2.0, -2.0], [True, False, True, False])
    assert c.fitted
    assert c.alpha == DEFAULT_ALPHA


# --------------------------------------------------------------------------------------
# THE GUARANTEE
#
# Validated on a test split disjoint from calibration. Measured at the time of
# writing: 81.6 / 85.8 / 90.6 / 95.1 against targets of 80 / 85 / 90 / 95.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.20, 0.15, 0.10, 0.05])
def test_coverage_guarantee_holds_on_held_out_data(split, alpha):
    cal, test = split
    c = ConformalCalibrator(alpha=alpha).fit([z for z, _ in cal], [y for _, y in cal])
    rep = evaluate_coverage(c, [z for z, _ in test], [y for _, y in test])
    assert rep.guarantee_holds, rep.summary()
    assert rep.coverage >= (1 - alpha) - 0.02


def test_tighter_alpha_trades_resolution_for_coverage(split):
    """The core tradeoff: more coverage means fewer auto-resolutions."""
    cal, test = split
    zc, yc = [z for z, _ in cal], [y for _, y in cal]
    zt, yt = [z for z, _ in test], [y for _, y in test]

    loose = evaluate_coverage(ConformalCalibrator(alpha=0.20).fit(zc, yc), zt, yt)
    tight = evaluate_coverage(ConformalCalibrator(alpha=0.05).fit(zc, yc), zt, yt)

    assert tight.coverage > loose.coverage
    assert tight.auto_resolve_rate < loose.auto_resolve_rate
    assert tight.mean_set_size > loose.mean_set_size


def test_accuracy_on_auto_resolved_is_high(split):
    """What the guarantee actually bounds: error on cases the system acts on."""
    cal, test = split
    c = ConformalCalibrator(alpha=0.10).fit([z for z, _ in cal], [y for _, y in cal])
    rep = evaluate_coverage(c, [z for z, _ in test], [y for _, y in test])
    assert rep.accuracy_on_resolved >= 0.85


def test_calibration_reduces_ece(split):
    cal, test = split
    zc, yc = [z for z, _ in cal], [y for _, y in cal]
    zt, yt = [z for z, _ in test], [y for _, y in test]

    fitted = ConformalCalibrator(alpha=0.1).fit(zc, yc)
    raw = ConformalCalibrator(alpha=0.1).fit(zc, yc, fit_temperature=False)

    assert evaluate_coverage(fitted, zt, yt).ece <= evaluate_coverage(raw, zt, yt).ece


def test_abstention_concentrates_on_ambiguous_cases(split):
    """Abstention must target genuine ambiguity, not fire at random."""
    cal, test = split
    c = ConformalCalibrator(alpha=0.10).fit([z for z, _ in cal], [y for _, y in cal])

    ambiguous = [
        adjudicate(g.case).posterior_logodds
        for g in generate_corpus(1200, CorpusConfig(seed=404))
        if g.truth is GroundTruth.GENUINELY_AMBIGUOUS
        and not adjudicate(g.case).decided_by_statute
    ]

    def abstain_rate(zs):
        return sum(1 for z in zs if not c.predict_set(z).is_singleton) / len(zs)

    assert abstain_rate(ambiguous) > abstain_rate([z for z, _ in test])


def test_evaluate_rejects_empty():
    c = calibrate([1.0, -1.0], [True, False])
    with pytest.raises(ValueError):
        evaluate_coverage(c, [], [])


# --------------------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(split):
    cal, _ = split
    return ConformalCalibrator(alpha=0.10).fit(
        [z for z, _ in cal], [y for _, y in cal]
    )


def test_statute_cases_route_to_statute(fitted):
    """A dispositive rule is procedural; conformal does not apply to it."""
    case = DisputeCase(
        reason_code=ReasonCode.C04, return_shipped_day=19, return_window_days=14
    )
    rv = fitted.route(adjudicate(case))
    assert rv.route is Route.STATUTE
    assert rv.is_auto_resolved
    assert rv.verdict is Verdict.MERCHANT
    assert "procedural" in rv.explain_route().lower()


def test_confident_case_auto_resolves(fitted):
    case = DisputeCase(
        reason_code=ReasonCode.C08,
        evidence=[
            Evidence(
                etype=EvidenceType.DELIVERY_CONFIRMATION,
                party=Party.MERCHANT,
                verified=True,
                quality=0.96,
                metadata={"lambda_lr": -2.5, "address_match": True},
            ),
            Evidence(
                etype=EvidenceType.USAGE_LOG,
                party=Party.MERCHANT,
                verified=True,
                quality=0.95,
                metadata={"lambda_lr": -2.0},
            ),
        ],
    )
    rv = fitted.route(adjudicate(case))
    assert rv.route is Route.AUTO_RESOLVE
    assert rv.verdict is Verdict.MERCHANT


def test_balanced_settlement_capable_case_routes_to_settlement(fitted):
    case = DisputeCase(
        reason_code=ReasonCode.C31,
        evidence=[
            Evidence(
                etype=EvidenceType.PHOTO_OF_ITEM,
                party=Party.CARD_MEMBER,
                metadata={"lambda_lr": 0.15},
            )
        ],
    )
    rv = fitted.route(adjudicate(case))
    if not rv.prediction_set.is_singleton:
        assert rv.route is Route.SETTLEMENT
        assert not rv.is_auto_resolved


def test_all_or_nothing_code_routes_to_human(fitted):
    """A duplicate charge cannot be half refunded, so ambiguity needs a person."""
    rv = fitted.route(
        adjudicate(DisputeCase(reason_code=ReasonCode.P08)),
        supports_settlement=False,
    )
    if not rv.prediction_set.is_singleton:
        assert rv.route is Route.HUMAN_REVIEW


def test_route_explanations_are_populated(fitted):
    for g in generate_corpus(40, CorpusConfig(seed=505)):
        rv = fitted.route(adjudicate(g.case))
        assert rv.explain_route().strip()
        assert 0.5 <= rv.calibrated_confidence <= 1.0


def test_routed_verdict_respects_statute_over_conformal(fitted):
    """Even if the prediction set were ambiguous, a clamped case keeps its verdict."""
    case = DisputeCase(
        reason_code=ReasonCode.P08, duplicate_confirmed=True
    )
    rv = fitted.route(adjudicate(case))
    assert rv.route is Route.STATUTE
    assert rv.verdict is Verdict.CARD_MEMBER


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------


def test_round_trips_through_dict(fitted):
    restored = ConformalCalibrator.from_dict(fitted.to_dict())
    assert restored.threshold == pytest.approx(fitted.threshold)
    assert restored.scaler.temperature == pytest.approx(fitted.scaler.temperature)
    assert restored.predict_set(1.1).labels == fitted.predict_set(1.1).labels


def test_round_trips_through_disk(fitted, tmp_path):
    path = tmp_path / "calib.json"
    fitted.save(path)
    restored = ConformalCalibrator.load(path)
    assert restored.alpha == fitted.alpha
    assert restored.threshold == pytest.approx(fitted.threshold)

"""Split-conformal abstention — knowing when not to rule.

The claim this module makes possible:

    "We auto-resolve X% of disputes with a mathematically guaranteed 1-alpha
     coverage, and escalate the rest by design."

That is a materially stronger statement than "our model is 91% accurate", and it is
the one a bank actually needs, because it bounds the error rate on the cases the
system *acts* on rather than averaging over cases it should never have touched.

**Method.** Split conformal prediction. On a held-out calibration set, score each case
by nonconformity — how poorly the true label conformed to the prediction:

    s_j = 1 - p_hat_j(y_j)

Take q_hat as the ceil((n+1)(1-alpha))/n empirical quantile. At inference, emit the
set of labels whose predicted probability clears the threshold:

    C(x) = { y : p_hat(y|x) >= 1 - q_hat }

    |C(x)| = 1  ->  auto-resolve. This is the "weeks to minutes".
    |C(x)| = 2  ->  abstain: route to settlement or a human adjudicator.
    |C(x)| = 0  ->  the prediction is worse than the calibration floor; treat as
                    abstention, never as a confident answer.

**The guarantee.** Under exchangeability, P(y in C(x)) >= 1 - alpha, distribution-free
and finite-sample. No assumption about the ledger being well-specified — which matters
here, because the ledger is a hand-built Bayesian model, not a fitted classifier.

**Why this is not just a tuned threshold.** The band sweep in Stage 3 showed the
posterior already ranks uncertainty correctly. Conformal does not improve the ranking;
it converts the ranking into a *promise*. The threshold is derived from the calibration
quantile at a chosen alpha, never hand-picked against the evaluation corpus — that
distinction is the whole point, and it is why CONTESTED_BAND was left untuned.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from arbiter.calibration.temperature import (
    TemperatureScaler,
    expected_calibration_error,
    maximum_calibration_error,
)
from arbiter.core.ledger import Adjudication, Verdict

#: Default miscoverage rate. 0.10 targets 90% coverage, a reasonable operating point
#: for a system where abstention is cheap (route to settlement) and error is not.
DEFAULT_ALPHA: float = 0.10


class Route(str, Enum):
    """What should happen to a dispute after adjudication."""

    AUTO_RESOLVE = "auto_resolve"
    """Singleton prediction set — decide it now."""

    SETTLEMENT = "settlement"
    """Ambiguous, but the reason code supports a partial split."""

    HUMAN_REVIEW = "human_review"
    """Ambiguous and all-or-nothing, or the model fell below the calibration floor."""

    STATUTE = "statute"
    """A dispositive rule decided it; conformal does not apply."""


@dataclass(frozen=True, slots=True)
class PredictionSet:
    """The conformal prediction set for one dispute."""

    labels: frozenset[Verdict]
    p_card_member: float
    threshold: float
    alpha: float

    @property
    def size(self) -> int:
        return len(self.labels)

    @property
    def is_singleton(self) -> bool:
        return self.size == 1

    @property
    def is_empty(self) -> bool:
        """No label cleared the threshold — the case is outside calibrated experience."""
        return self.size == 0

    @property
    def verdict(self) -> Verdict:
        """The decision this set implies."""
        if self.size == 1:
            return next(iter(self.labels))
        return Verdict.CONTESTED

    def contains(self, verdict: Verdict) -> bool:
        return verdict in self.labels


@dataclass(frozen=True, slots=True)
class RoutedVerdict:
    """An adjudication after conformal calibration and routing."""

    adjudication: Adjudication
    prediction_set: PredictionSet
    route: Route
    calibrated_confidence: float

    @property
    def verdict(self) -> Verdict:
        """Post-calibration verdict, which may be CONTESTED where the raw one was not."""
        if self.adjudication.decided_by_statute:
            return self.adjudication.verdict
        return self.prediction_set.verdict

    @property
    def is_auto_resolved(self) -> bool:
        return self.route in (Route.AUTO_RESOLVE, Route.STATUTE)

    def explain_route(self) -> str:
        """One line for the verdict card explaining why this path was taken."""
        if self.route is Route.STATUTE:
            return (
                "Decided by a dispositive rule in the AMEX Chargeback Code Guide. "
                "Statistical calibration does not apply to a procedural determination."
            )
        pct = int(round((1 - self.prediction_set.alpha) * 100))
        if self.route is Route.AUTO_RESOLVE:
            return (
                f"Resolved automatically. The evidence supports a single outcome at "
                f"the calibrated {pct}% coverage level."
            )
        if self.route is Route.SETTLEMENT:
            return (
                f"The evidence supports both outcomes at the {pct}% coverage level, so "
                "no confident ruling is available. Routed to settlement."
            )
        return (
            f"The evidence supports both outcomes at the {pct}% coverage level, and "
            "this reason code does not admit a partial remedy. Routed to human review."
        )


@dataclass(slots=True)
class CoverageReport:
    """Empirical validation of the conformal guarantee."""

    alpha: float
    n: int
    coverage: float
    """Fraction of cases whose true label was in the prediction set."""
    auto_resolve_rate: float
    accuracy_on_resolved: float
    """Error rate here is what the guarantee actually bounds."""
    mean_set_size: float
    abstain_rate: float
    empty_set_rate: float
    ece: float
    mce: float
    temperature: float

    @property
    def guarantee_holds(self) -> bool:
        """Coverage should meet 1-alpha, with slack for finite-sample noise."""
        return self.coverage >= (1 - self.alpha) - 0.02

    def summary(self) -> str:
        target = 1 - self.alpha
        status = "HOLDS" if self.guarantee_holds else "VIOLATED"
        return (
            f"alpha={self.alpha:.2f}  target coverage {target:.0%}  "
            f"empirical {self.coverage:.1%}  [{status}]\n"
            f"  auto-resolved {self.auto_resolve_rate:.1%} of disputes, "
            f"accuracy on those {self.accuracy_on_resolved:.1%}\n"
            f"  abstained {self.abstain_rate:.1%}  mean set size {self.mean_set_size:.2f}  "
            f"ECE {self.ece:.3f}  T={self.temperature:.2f}"
        )


def _quantile_index(n: int, alpha: float) -> int:
    """The conformal order statistic: ceil((n+1)(1-alpha)), clipped to the sample."""
    k = math.ceil((n + 1) * (1 - alpha))
    return min(max(k, 1), n) - 1


@dataclass(slots=True)
class ConformalCalibrator:
    """Split-conformal calibration over binary dispute outcomes.

    Attributes:
        alpha: target miscoverage. Coverage is 1 - alpha.
        threshold: 1 - q_hat, the probability a label must clear to enter the set.
        scaler: temperature calibration applied before thresholding.
        n_calibration: size of the calibration set, recorded for the audit trail.
    """

    alpha: float = DEFAULT_ALPHA
    threshold: float = 0.5
    scaler: TemperatureScaler = field(default_factory=TemperatureScaler)
    n_calibration: int = 0
    fitted: bool = False

    # -- fitting ---------------------------------------------------------------

    def fit(
        self,
        logodds: Sequence[float],
        labels: Sequence[bool],
        *,
        fit_temperature: bool = True,
    ) -> "ConformalCalibrator":
        """Calibrate on held-out adjudications.

        Args:
            logodds: raw posterior log-odds, positive favouring the Card Member.
            labels: True when the Card Member was in fact right.
            fit_temperature: fit temperature scaling first. Since scaling is monotone
                it cannot change a verdict, only the confidence, so it is safe to
                apply before deriving the conformal quantile.

        Statute-clamped cases must be excluded by the caller: a dispositive rule is a
        procedural determination, not a probabilistic one, and including its infinite
        confidence would corrupt the quantile.
        """
        if len(logodds) != len(labels):
            raise ValueError("logodds and labels must be the same length")
        if not logodds:
            raise ValueError("cannot calibrate on an empty set")

        if fit_temperature:
            self.scaler.fit(logodds, labels)

        # Nonconformity: how much probability the model withheld from the truth.
        scores = sorted(
            1.0 - self._p_true(z, y)
            for z, y in zip(logodds, labels, strict=True)
        )
        q_hat = scores[_quantile_index(len(scores), self.alpha)]

        self.threshold = 1.0 - q_hat
        self.n_calibration = len(scores)
        self.fitted = True
        return self

    def _p_true(self, logodds: float, label: bool) -> float:
        p_cm = self.scaler.probability(logodds)
        return p_cm if label else 1.0 - p_cm

    # -- inference -------------------------------------------------------------

    def predict_set(self, logodds: float) -> PredictionSet:
        """Conformal prediction set for one posterior."""
        p_cm = self.scaler.probability(logodds)
        labels: set[Verdict] = set()
        if p_cm >= self.threshold:
            labels.add(Verdict.CARD_MEMBER)
        if (1.0 - p_cm) >= self.threshold:
            labels.add(Verdict.MERCHANT)
        return PredictionSet(
            labels=frozenset(labels),
            p_card_member=p_cm,
            threshold=self.threshold,
            alpha=self.alpha,
        )

    def route(
        self, adj: Adjudication, *, supports_settlement: bool | None = None
    ) -> RoutedVerdict:
        """Calibrate an adjudication and decide what happens to it next.

        Args:
            adj: the raw adjudication.
            supports_settlement: whether the reason code admits a partial remedy.
                Defaults to the spec's own flag. A duplicate charge cannot be half
                refunded, so those abstentions go to a human instead.
        """
        from arbiter.core.reason_codes import get_spec

        if supports_settlement is None:
            supports_settlement = get_spec(adj.reason_code).supports_settlement

        pset = self.predict_set(adj.posterior_logodds)
        confidence = self.scaler.confidence(adj.posterior_logodds)

        if adj.decided_by_statute:
            route = Route.STATUTE
        elif pset.is_singleton:
            route = Route.AUTO_RESOLVE
        elif supports_settlement and not pset.is_empty:
            route = Route.SETTLEMENT
        else:
            # An empty set means the case sits outside calibrated experience. That is
            # a human's problem, never an automatic decision.
            route = Route.HUMAN_REVIEW

        return RoutedVerdict(
            adjudication=adj,
            prediction_set=pset,
            route=route,
            calibrated_confidence=confidence,
        )

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "temperature": self.scaler.temperature,
            "n_calibration": self.n_calibration,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConformalCalibrator":
        scaler = TemperatureScaler(
            temperature=float(data["temperature"]), fitted=True
        )
        return cls(
            alpha=float(data["alpha"]),
            threshold=float(data["threshold"]),
            scaler=scaler,
            n_calibration=int(data.get("n_calibration", 0)),
            fitted=bool(data.get("fitted", True)),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate(
    logodds: Sequence[float],
    labels: Sequence[bool],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> ConformalCalibrator:
    """Fit a calibrator in one call."""
    return ConformalCalibrator(alpha=alpha).fit(logodds, labels)


def evaluate_coverage(
    calibrator: ConformalCalibrator,
    logodds: Sequence[float],
    labels: Sequence[bool],
) -> CoverageReport:
    """Validate the guarantee on a test split disjoint from calibration.

    Coverage measured on the calibration set itself would be optimistic; the whole
    point of split conformal is that the promise holds on unseen data.
    """
    if len(logodds) != len(labels):
        raise ValueError("logodds and labels must be the same length")
    n = len(logodds)
    if n == 0:
        raise ValueError("cannot evaluate on an empty set")

    covered = singletons = correct_singletons = empty = 0
    total_size = 0
    confidences: list[float] = []
    correct: list[bool] = []

    for z, y in zip(logodds, labels, strict=True):
        pset = calibrator.predict_set(z)
        truth = Verdict.CARD_MEMBER if y else Verdict.MERCHANT

        covered += pset.contains(truth)
        total_size += pset.size
        empty += pset.is_empty

        if pset.is_singleton:
            singletons += 1
            correct_singletons += pset.verdict is truth

        confidences.append(calibrator.scaler.confidence(z))
        predicted_cm = calibrator.scaler.probability(z) >= 0.5
        correct.append(predicted_cm == y)

    return CoverageReport(
        alpha=calibrator.alpha,
        n=n,
        coverage=covered / n,
        auto_resolve_rate=singletons / n,
        accuracy_on_resolved=(correct_singletons / singletons) if singletons else 0.0,
        mean_set_size=total_size / n,
        abstain_rate=(n - singletons) / n,
        empty_set_rate=empty / n,
        ece=expected_calibration_error(confidences, correct),
        mce=maximum_calibration_error(confidences, correct),
        temperature=calibrator.scaler.temperature,
    )

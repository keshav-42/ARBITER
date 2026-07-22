"""Temperature scaling and calibration metrics.

The ledger's posterior is a sum of hand-set weights, statute shifts, and model
outputs. Nothing forces the resulting probability to be *calibrated* — when it says
80%, it should be right 80% of the time. It usually is not, and the direction of the
error matters: an overconfident dispute system either auto-resolves cases it should
have escalated, or escalates cases it could have decided.

Temperature scaling fixes this with a single parameter:

    p_calibrated = sigmoid( Lambda / T )

T > 1 softens an overconfident model, T < 1 sharpens an underconfident one. It is
fitted by minimising negative log-likelihood on held-out data, and because it is
monotone it **cannot change any verdict** — only the confidence attached to it. That
property is why it is safe to apply before conformal calibration.

Reported alongside is **ECE** (expected calibration error): bin predictions by
confidence, compare mean confidence to observed accuracy in each bin, average the gaps
weighted by bin size. It is the number to put on a slide, because almost nobody does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _sigmoid(z: float) -> float:
    # Numerically stable both directions.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed miscalibration. Positive means overconfident."""
        return self.mean_confidence - self.accuracy


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin predictions by confidence for a reliability diagram.

    Confidences are expected in [0.5, 1.0] — the probability assigned to whichever
    side won — so the bins span that range rather than [0, 1].
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")

    edges = [0.5 + 0.5 * i / n_bins for i in range(n_bins + 1)]
    bins: list[ReliabilityBin] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include the right edge in the final bin so p=1.0 is counted.
        idx = [
            j
            for j, c in enumerate(confidences)
            if (lo <= c < hi) or (i == n_bins - 1 and c == hi)
        ]
        if not idx:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        mc = sum(confidences[j] for j in idx) / len(idx)
        acc = sum(1 for j in idx if correct[j]) / len(idx)
        bins.append(ReliabilityBin(lo, hi, len(idx), mc, acc))

    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> float:
    """ECE — mean |confidence − accuracy|, weighted by bin population.

    0 is perfect. Above ~0.10 the confidence numbers should not be shown to users as
    probabilities.
    """
    if not confidences:
        return 0.0
    bins = reliability_bins(confidences, correct, n_bins=n_bins)
    n = len(confidences)
    return sum(b.count / n * abs(b.gap) for b in bins if b.count)


def maximum_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> float:
    """The worst single-bin gap. Catches a badly-calibrated region ECE averages away."""
    if not confidences:
        return 0.0
    bins = reliability_bins(confidences, correct, n_bins=n_bins)
    return max((abs(b.gap) for b in bins if b.count), default=0.0)


@dataclass(slots=True)
class TemperatureScaler:
    """Single-parameter calibration of the ledger posterior.

    Attributes:
        temperature: divisor applied to the log-odds. 1.0 is a no-op.
        fitted: whether `fit` has been run.
    """

    temperature: float = 1.0
    fitted: bool = False

    def fit(
        self,
        logodds: Sequence[float],
        labels: Sequence[bool],
        *,
        lo: float = 0.25,
        hi: float = 8.0,
        iterations: int = 60,
    ) -> "TemperatureScaler":
        """Fit temperature by minimising NLL on held-out data.

        Uses golden-section search rather than gradient descent: the objective is
        one-dimensional, smooth, and unimodal in T, so a derivative-free search is
        both simpler and more robust than tuning a learning rate.

        Args:
            logodds: raw posterior log-odds, positive favouring the Card Member.
            labels: True when the Card Member was in fact right.
        """
        if len(logodds) != len(labels):
            raise ValueError("logodds and labels must be the same length")
        if not logodds:
            raise ValueError("cannot fit on an empty calibration set")

        def nll(t: float) -> float:
            total = 0.0
            for z, y in zip(logodds, labels, strict=True):
                p = _sigmoid(z / t)
                p = min(max(p, 1e-9), 1 - 1e-9)
                total -= math.log(p) if y else math.log(1 - p)
            return total / len(logodds)

        invphi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = b - invphi * (b - a), a + invphi * (b - a)
        fc, fd = nll(c), nll(d)

        for _ in range(iterations):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - invphi * (b - a)
                fc = nll(c)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = nll(d)
            if abs(b - a) < 1e-4:
                break

        self.temperature = (a + b) / 2.0
        self.fitted = True
        return self

    def calibrate_logodds(self, logodds: float) -> float:
        """Temperature-scaled log-odds."""
        return logodds / self.temperature

    def probability(self, logodds: float) -> float:
        """Calibrated P(Card Member prevails)."""
        return _sigmoid(self.calibrate_logodds(logodds))

    def confidence(self, logodds: float) -> float:
        """Calibrated probability assigned to the winning side, in [0.5, 1]."""
        p = self.probability(logodds)
        return max(p, 1.0 - p)

    @property
    def is_overconfident(self) -> bool:
        """True when the fit had to soften the raw posterior."""
        return self.temperature > 1.0

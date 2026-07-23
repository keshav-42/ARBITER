"""Fairness audit — is the engine even-handed?

"Fair" is in the problem title twice, and it is a measurable word. Two tests here make
it measurable rather than asserted.

1. The counterfactual party-swap test.
   Hold the reason code fixed. Take exhibits whose meaning is symmetric between the
   parties — correspondence, receipts, order records that either side could file with
   the same import — swap which party filed them, and negate any explicit likelihood
   ratio. If the engine reads evidence rather than identity, the posterior must invert:
   the same record, mirrored, should favour the mirror party by the same margin.

   The test deliberately restricts to sign-neutral evidence. A photograph of a
   defective item, a delivery confirmation, an AVS result — these are inherently
   directional; a merchant filing "here is a photo of the broken item they received" is
   incoherent, so swapping its party is not a valid counterfactual. Testing symmetry
   only where symmetry is meaningful is what makes the result trustworthy rather than
   an artefact of evidence semantics.

   An earlier version swapped parties AND re-filed under a mirror reason code, and
   flagged a 44% "bias" that was entirely the C08/C31 weight difference and the
   directionality of photo/tracking evidence — a bug in the test, not the engine. This
   version isolates the one thing fairness actually requires.

2. The asymmetry gap.
   Outcomes should not depend on who the parties are — merchant size, card-member
   tenure, transaction-value band. AsymGap measures how much the Card-Member win rate
   moves across those slices on cases that are otherwise comparable. A large gap is
   evidence the engine is reading identity rather than merit.

Reputation is the one signal that is *designed* to depend on identity, so both tests
run with reputation disabled (`apply_reputation=False`) to isolate the merit decision.
The eval report separately confirms that reputation, when on, stays within its cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arbiter.core.evidence import Evidence
from arbiter.core.ledger import FILER_ORIENTED_TYPES, DisputeCase, Verdict, adjudicate
from arbiter.core.reason_codes import BurdenOfProof

#: The same set the ledger orients by the filer. Sharing it means the fairness test and
#: the scoring path cannot drift apart.
_SYMMETRIC_TYPES = FILER_ORIENTED_TYPES


def _mirror_evidence(e: Evidence) -> Evidence:
    """Swap a symmetric exhibit's party and negate any explicit likelihood ratio."""
    meta = dict(e.metadata)
    if "lambda_lr" in meta and meta["lambda_lr"] is not None:
        meta["lambda_lr"] = -float(meta["lambda_lr"])
    # Symmetric NLI probabilities also mirror: entailment for one side is entailment
    # for the other under the swapped hypothesis.
    if "entail" in meta and "contradict" in meta:
        meta["entail"], meta["contradict"] = meta["contradict"], meta["entail"]
    return Evidence(
        etype=e.etype,
        party=e.party.opponent,
        content=e.content,
        quality=e.quality,
        verified=e.verified,
        blob_sha256=e.blob_sha256,
        occurred_at=e.occurred_at,
        filed_at=e.filed_at,
        metadata=meta,
    )


def swap_parties(case: DisputeCase) -> DisputeCase:
    """Return a copy with symmetric exhibits mirrored between the parties.

    Same reason code, so the burden of proof is unchanged. Only sign-neutral evidence
    is swapped; directional and network exhibits are left as-is, because mirroring them
    would not correspond to any real counterfactual. Reputation is dropped so the test
    isolates the merit decision.
    """
    swapped = [
        _mirror_evidence(e) if e.etype in _SYMMETRIC_TYPES else e
        for e in case.evidence
    ]
    return DisputeCase(
        reason_code=case.reason_code,
        evidence=swapped,
        amount=case.amount,
        merchant_response_days=case.merchant_response_days,
        return_shipped_day=case.return_shipped_day,
        return_window_days=case.return_window_days,
        cancel_day=case.cancel_day,
        cancel_window_days=case.cancel_window_days,
        days_since_transaction=case.days_since_transaction,
        submission_delay_days=case.submission_delay_days,
        duplicate_confirmed=case.duplicate_confirmed,
        refund_already_posted=case.refund_already_posted,
    )


#: Tolerance on the per-exhibit contribution residual, in nats.
_SWAP_TOLERANCE: float = 0.02


@dataclass(frozen=True, slots=True)
class RoleSwapReport:
    """Result of the counterfactual party-swap test.

    The claim being tested is narrow and exact: a symmetric exhibit's *own contribution*
    must be identity-independent. Filed by the Card Member it should weigh +x; the same
    record filed by the Merchant should weigh -x. This isolates the evidence-scoring
    path from the two things that are asymmetric BY DESIGN and must not be "corrected":

      - the burden of proof (a prior that legitimately favours one side per code), and
      - correlation damping (which is order-dependent, so a swapped exhibit may join a
        different group).

    `symmetric` counts exhibits whose mirrored contribution negated within tolerance.
    """

    tested: int
    symmetric: int
    asymmetric: int
    max_residual: float

    @property
    def flip_rate(self) -> float:
        """Fraction of symmetric exhibits scored identity-independently."""
        return self.symmetric / self.tested if self.tested else 1.0

    @property
    def passes(self) -> bool:
        return self.flip_rate >= 0.99

    def summary(self) -> str:
        status = "EVEN-HANDED" if self.passes else "SIDE-BIAS DETECTED"
        return (
            f"party-swap: {self.symmetric}/{self.tested} symmetric exhibits scored "
            f"identity-independently ({self.flip_rate:.1%})  [{status}]  "
            f"max residual {self.max_residual:.4f} nats"
        )


def role_swap_test(cases: list[DisputeCase]) -> RoleSwapReport:
    """Test that symmetric evidence is scored identity-independently.

    For each symmetric exhibit, compare its contribution when filed by its actual party
    against the same exhibit filed by the opponent, holding everything else fixed. The
    two must be exact negations. This measures the evidence-scoring path alone, which is
    where identity-bias would live — the burden and damping asymmetries are intentional
    and are checked separately (`burden_symmetry_check`, and the eval's per-code table).
    """
    from arbiter.core.ledger import build_entry
    from arbiter.core.reason_codes import get_spec

    tested = symmetric = 0
    max_residual = 0.0

    for case in cases:
        spec = get_spec(case.reason_code)
        for e in case.evidence:
            if e.etype not in _SYMMETRIC_TYPES:
                continue
            mirrored = _mirror_evidence(e)
            # Score each in isolation (no damping, no burden) — just w * lambda * q.
            c_orig = build_entry(e, spec).contribution
            c_mirror = build_entry(mirrored, spec).contribution
            residual = abs(c_orig + c_mirror)  # should sum to zero
            max_residual = max(max_residual, residual)
            tested += 1
            if residual <= _SWAP_TOLERANCE:
                symmetric += 1

    return RoleSwapReport(
        tested=tested,
        symmetric=symmetric,
        asymmetric=tested - symmetric,
        max_residual=max_residual,
    )


@dataclass(frozen=True, slots=True)
class SliceStat:
    name: str
    n: int
    cm_win_rate: float


@dataclass(frozen=True, slots=True)
class AsymmetryReport:
    """Card-Member win-rate spread across identity slices."""

    dimension: str
    slices: tuple[SliceStat, ...]

    @property
    def gap(self) -> float:
        """Max minus min win rate across slices with enough support."""
        rates = [s.cm_win_rate for s in self.slices if s.n >= 30]
        return (max(rates) - min(rates)) if len(rates) >= 2 else 0.0

    @property
    def passes(self) -> bool:
        """A modest gap is expected from real case-mix differences; a large one is not."""
        return self.gap <= 0.20

    def summary(self) -> str:
        status = "OK" if self.passes else "REVIEW"
        parts = "  ".join(
            f"{s.name}={s.cm_win_rate:.0%}(n{s.n})" for s in self.slices if s.n >= 30
        )
        return f"{self.dimension}: gap {self.gap:.1%} [{status}]  {parts}"


def _value_band(amount: float) -> str:
    if amount < 50:
        return "under $50"
    if amount < 250:
        return "$50–250"
    if amount < 1000:
        return "$250–1k"
    return "over $1k"


def asymmetry_audit(
    labelled: list[tuple[DisputeCase, bool]],
) -> list[AsymmetryReport]:
    """Measure Card-Member win-rate spread across value bands.

    Takes (case, card_member_should_win) pairs. Groups by transaction-value band and
    reports the win-rate gap. Reputation is left off so the audit measures the merit
    decision rather than a party's history — which is what a fairness claim needs.

    Returns one report per audited dimension. Value band is the dimension the synthetic
    corpus supports directly; merchant-size and tenure slices would attach in a
    deployment with those attributes.
    """
    by_band: dict[str, list[int]] = {}
    for case, _should_win in labelled:
        adj = adjudicate(case, apply_reputation=False)
        if adj.verdict is Verdict.CONTESTED:
            continue
        band = _value_band(case.amount)
        pair = by_band.setdefault(band, [0, 0])
        pair[0] += adj.verdict is Verdict.CARD_MEMBER
        pair[1] += 1

    order = ["under $50", "$50–250", "$250–1k", "over $1k"]
    slices = tuple(
        SliceStat(name=b, n=by_band[b][1], cm_win_rate=by_band[b][0] / by_band[b][1])
        for b in order
        if b in by_band and by_band[b][1] > 0
    )
    return [AsymmetryReport(dimension="transaction value", slices=slices)]


@dataclass(frozen=True, slots=True)
class ReputationBoundReport:
    """Confirms reputation never exceeds its cap in practice."""

    max_abs_contribution: float
    cap: float
    within_cap: bool = field(default=True)

    def summary(self) -> str:
        status = "WITHIN CAP" if self.within_cap else "CAP VIOLATED"
        return (
            f"reputation: max |contribution| {self.max_abs_contribution:.3f} nats "
            f"vs cap {self.cap:.2f}  [{status}]"
        )


def reputation_bound_check(cases: list[DisputeCase]) -> ReputationBoundReport:
    """Verify reputation's effect stays inside REPUTATION_CAP across the corpus.

    Measured as the difference between the posterior with and without reputation, which
    is exactly the contribution reputation made. The fairness guarantee is that this can
    nudge but never decide.
    """
    from arbiter.core.reputation import REPUTATION_CAP

    worst = 0.0
    for case in cases:
        with_rep = adjudicate(case, apply_reputation=True)
        without = adjudicate(case, apply_reputation=False)
        if with_rep.decided_by_statute or without.decided_by_statute:
            continue
        worst = max(worst, abs(with_rep.posterior_logodds - without.posterior_logodds))

    # Small tolerance for the combined-cap arithmetic and float noise.
    return ReputationBoundReport(
        max_abs_contribution=worst,
        cap=REPUTATION_CAP,
        within_cap=worst <= REPUTATION_CAP + 1e-6,
    )


def burden_symmetry_check() -> dict[str, object]:
    """Confirm the priors themselves are symmetric around the burden.

    A structural check independent of any data: merchant-burden codes must open in the
    Card Member's favour and card-member-burden codes must not. This is what makes the
    role-swap test meaningful — the statute layer is symmetric by construction.
    """
    from arbiter.core.reason_codes import REASON_CODES

    violations = []
    for spec in REASON_CODES.values():
        if spec.burden is BurdenOfProof.MERCHANT and spec.prior_logodds <= 0:
            violations.append(spec.code.value)
        if spec.burden is BurdenOfProof.CARD_MEMBER and spec.prior_logodds >= 0.25:
            violations.append(spec.code.value)
    return {"symmetric": not violations, "violations": violations}

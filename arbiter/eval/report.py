"""End-to-end evaluation report — the numbers behind the pitch.

One function produces every figure a judge asks for:

    agreement with ground truth, overall and by reason code and difficulty
    calibration: ECE, MCE, and the reliability curve, before and after temperature
    conformal coverage across alpha, validated on a held-out split
    fairness: the counterfactual role-swap flip rate and the asymmetry gap
    reputation bound: proof the identity signal stays capped
    latency: p50 / p95 / p99 per dispute, and throughput

Results serialise to JSON for the record and render to a self-contained HTML page the
deck can screenshot. Nothing here needs a GPU or a network: the whole report runs on
the synthetic corpus and the offline engine.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from arbiter.calibration.conformal import ConformalCalibrator, evaluate_coverage
from arbiter.calibration.temperature import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)
from arbiter.core.ledger import DisputeCase, Verdict, adjudicate
from arbiter.data.generator import CorpusConfig, generate_corpus
from arbiter.data.scenarios import GroundTruth
from arbiter.eval.fairness import (
    asymmetry_audit,
    burden_symmetry_check,
    reputation_bound_check,
    role_swap_test,
)

_TRUTH_TO_VERDICT = {
    GroundTruth.CARD_MEMBER_RIGHT: Verdict.CARD_MEMBER,
    GroundTruth.MERCHANT_RIGHT: Verdict.MERCHANT,
}


@dataclass
class EvalReport:
    """Everything measured, in one serialisable object."""

    n_cases: int
    agreement: float
    by_code: dict[str, float]
    by_difficulty: dict[str, float]
    abstain_rate: float
    statute_rate: float
    ambiguous_abstain_rate: float

    ece_raw: float
    ece_calibrated: float
    mce_calibrated: float
    reliability: list[dict[str, float]]

    coverage: list[dict[str, float]]

    role_swap_flip_rate: float
    role_swap_passes: bool
    asymmetry_gap: float
    asymmetry_passes: bool
    reputation_max: float
    reputation_within_cap: bool
    burden_symmetric: bool

    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_per_s: float

    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _labelled(n: int, seed: int) -> tuple[list, list[tuple[DisputeCase, bool]]]:
    """Generate a corpus, returning the raw cases and (case, cm_should_win) pairs."""
    gens = list(generate_corpus(n, CorpusConfig(seed=seed)))
    labelled: list[tuple[DisputeCase, bool]] = []
    for g in gens:
        if g.truth is GroundTruth.GENUINELY_AMBIGUOUS:
            continue
        labelled.append((g.case, g.truth is GroundTruth.CARD_MEMBER_RIGHT))
    return gens, labelled


def run_evaluation(n: int = 4000, seed: int = 202) -> EvalReport:
    """Run the full evaluation over a fresh synthetic corpus."""
    gens, labelled = _labelled(n, seed)

    # -- agreement + routing mix + latency (single pass, timed) --
    decided = correct = contested = statute = 0
    amb_total = amb_abstain = 0
    by_code: dict[str, list[int]] = {}
    by_diff: dict[str, list[int]] = {}
    logodds: list[float] = []
    labels: list[bool] = []
    latencies: list[float] = []

    for g in gens:
        t0 = time.perf_counter()
        adj = adjudicate(g.case)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if adj.decided_by_statute:
            statute += 1

        if g.truth is GroundTruth.GENUINELY_AMBIGUOUS:
            amb_total += 1
            if adj.verdict is Verdict.CONTESTED:
                amb_abstain += 1
            continue

        if not adj.decided_by_statute:
            logodds.append(adj.posterior_logodds)
            labels.append(g.truth is GroundTruth.CARD_MEMBER_RIGHT)

        if adj.verdict is Verdict.CONTESTED:
            contested += 1
            continue

        decided += 1
        hit = adj.verdict is _TRUTH_TO_VERDICT[g.truth]
        correct += hit
        by_code.setdefault(g.reason_code.value, [0, 0])
        by_code[g.reason_code.value][0] += hit
        by_code[g.reason_code.value][1] += 1
        bucket = "easy" if g.difficulty < 0.35 else "medium" if g.difficulty < 0.7 else "hard"
        by_diff.setdefault(bucket, [0, 0])
        by_diff[bucket][0] += hit
        by_diff[bucket][1] += 1

    # -- calibration: shuffle, then split. --
    # Conformal coverage requires exchangeability between the calibration and test
    # sets. The corpus is generated in scenario-weighted order, NOT shuffled, so a
    # naive first-half/second-half split can leave the two halves distributionally
    # different — measured as a systematic 3% undercoverage on some seeds. Shuffling
    # under a fixed seed restores exchangeability while keeping the report reproducible.
    import random as _random

    paired = list(zip(logodds, labels))
    _random.Random(seed).shuffle(paired)
    logodds = [z for z, _ in paired]
    labels = [y for _, y in paired]

    half = len(logodds) // 2
    calib = ConformalCalibrator(alpha=0.10).fit(logodds[:half], labels[:half])

    raw_conf = [1 / (1 + pow(2.718281828, -abs(z))) for z in logodds[half:]]
    raw_correct = [(z > 0) == y for z, y in zip(logodds[half:], labels[half:])]
    ece_raw = expected_calibration_error(raw_conf, raw_correct)

    cal_conf = [calib.scaler.confidence(z) for z in logodds[half:]]
    cal_correct = [
        (calib.scaler.probability(z) >= 0.5) == y for z, y in zip(logodds[half:], labels[half:])
    ]
    ece_cal = expected_calibration_error(cal_conf, cal_correct)
    mce_cal = maximum_calibration_error(cal_conf, cal_correct)
    bins = reliability_bins(cal_conf, cal_correct)

    # -- coverage across alpha --
    coverage = []
    for alpha in (0.20, 0.15, 0.10, 0.05):
        c = ConformalCalibrator(alpha=alpha).fit(logodds[:half], labels[:half])
        rep = evaluate_coverage(c, logodds[half:], labels[half:])
        coverage.append(
            {
                "alpha": alpha,
                "target": round(1 - alpha, 3),
                "empirical": round(rep.coverage, 4),
                "auto_resolve_rate": round(rep.auto_resolve_rate, 4),
                "accuracy_on_resolved": round(rep.accuracy_on_resolved, 4),
                "holds": rep.guarantee_holds,
            }
        )

    # -- fairness --
    swap = role_swap_test([g.case for g in gens])
    asym = asymmetry_audit(labelled)[0]
    rep_bound = reputation_bound_check([g.case for g in gens[: min(len(gens), 1500)]])
    burden = burden_symmetry_check()

    lat = sorted(latencies)

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 4)

    return EvalReport(
        n_cases=len(gens),
        agreement=round(correct / decided, 4) if decided else 0.0,
        by_code={k: round(v[0] / v[1], 4) for k, v in sorted(by_code.items()) if v[1]},
        by_difficulty={
            k: round(by_diff[k][0] / by_diff[k][1], 4)
            for k in ("easy", "medium", "hard")
            if k in by_diff and by_diff[k][1]
        },
        abstain_rate=round(contested / max(1, decided + contested), 4),
        statute_rate=round(statute / len(gens), 4),
        ambiguous_abstain_rate=round(amb_abstain / amb_total, 4) if amb_total else 0.0,
        ece_raw=round(ece_raw, 4),
        ece_calibrated=round(ece_cal, 4),
        mce_calibrated=round(mce_cal, 4),
        reliability=[
            {
                "lower": round(b.lower, 3),
                "upper": round(b.upper, 3),
                "count": b.count,
                "confidence": round(b.mean_confidence, 4),
                "accuracy": round(b.accuracy, 4),
            }
            for b in bins
        ],
        coverage=coverage,
        role_swap_flip_rate=round(swap.flip_rate, 4),
        role_swap_passes=swap.passes,
        asymmetry_gap=round(asym.gap, 4),
        asymmetry_passes=asym.passes,
        reputation_max=round(rep_bound.max_abs_contribution, 4),
        reputation_within_cap=rep_bound.within_cap,
        burden_symmetric=bool(burden["symmetric"]),
        latency_p50_ms=pct(0.50),
        latency_p95_ms=pct(0.95),
        latency_p99_ms=pct(0.99),
        throughput_per_s=round(1000.0 / (sum(lat) / len(lat)), 1) if lat else 0.0,
        notes={
            "temperature": round(calib.scaler.temperature, 4),
            "role_swap": swap.summary(),
            "asymmetry": asym.summary(),
            "reputation": rep_bound.summary(),
        },
    )


def print_report(report: EvalReport) -> None:
    """Human-readable console summary."""
    r = report
    line = "=" * 66
    print(line)
    print("ARBITER — EVALUATION REPORT")
    print(line)
    print(f"\ncorpus: {r.n_cases} cases")
    print("\nAGREEMENT WITH GROUND TRUTH")
    print(f"  overall (decided)   {r.agreement:.1%}")
    for k in ("easy", "medium", "hard"):
        if k in r.by_difficulty:
            print(f"    {k:<8} {r.by_difficulty[k]:.1%}")
    print(f"  abstained (clear)   {r.abstain_rate:.1%}")
    print(f"  decided by statute  {r.statute_rate:.1%}")
    print(f"  abstained ambiguous {r.ambiguous_abstain_rate:.1%}")

    print("\nCALIBRATION")
    print(f"  ECE raw -> calibrated   {r.ece_raw:.3f} -> {r.ece_calibrated:.3f}")
    print(f"  MCE calibrated          {r.mce_calibrated:.3f}")
    print(f"  temperature             {r.notes['temperature']}")

    print("\nCONFORMAL COVERAGE (held-out)")
    for c in r.coverage:
        flag = "HOLDS" if c["holds"] else "VIOLATED"
        print(
            f"  a={c['alpha']:.2f}  target {c['target']:.0%}  empirical {c['empirical']:.1%}"
            f"  auto-resolve {c['auto_resolve_rate']:.1%}  [{flag}]"
        )

    print("\nFAIRNESS")
    print(f"  {r.notes['role_swap']}")
    print(f"  {r.notes['asymmetry']}")
    print(f"  {r.notes['reputation']}")
    print(f"  burden priors symmetric: {r.burden_symmetric}")

    print("\nPERFORMANCE")
    print(f"  latency p50 / p95 / p99   {r.latency_p50_ms:.2f} / {r.latency_p95_ms:.2f} / {r.latency_p99_ms:.2f} ms")
    print(f"  throughput                {r.throughput_per_s:,.0f} disputes/s (single core)")


def write_html(report: EvalReport, path: str | Path) -> None:
    """Render a self-contained HTML report the deck can screenshot."""
    from arbiter.eval.render_html import render

    Path(path).write_text(render(report), encoding="utf-8")

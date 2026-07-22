"""Fit and persist a conformal calibrator from the synthetic corpus.

    python -m arbiter.calibration.fit_calibrator --alpha 0.10 --out models/calibrator.json

Statute-clamped and genuinely-ambiguous cases are excluded from calibration: the
former are procedural determinations with no probabilistic content, the latter have no
ground-truth winner to calibrate against. The calibrator is fitted on one half of the
remaining pool and validated on the other, so the coverage report reflects held-out
performance rather than the fit itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arbiter.calibration.conformal import ConformalCalibrator, evaluate_coverage
from arbiter.core.ledger import adjudicate
from arbiter.data.generator import CorpusConfig, generate_corpus
from arbiter.data.scenarios import GroundTruth


def build_pool(n: int, seed: int) -> list[tuple[float, bool]]:
    pool: list[tuple[float, bool]] = []
    for gen in generate_corpus(n, CorpusConfig(seed=seed)):
        adj = adjudicate(gen.case)
        if adj.decided_by_statute or gen.truth is GroundTruth.GENUINELY_AMBIGUOUS:
            continue
        pool.append(
            (adj.posterior_logodds, gen.truth is GroundTruth.CARD_MEMBER_RIGHT)
        )
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the ARBITER conformal calibrator")
    ap.add_argument("--alpha", type=float, default=0.10, help="target miscoverage")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", type=Path, default=Path("models/calibrator.json"))
    args = ap.parse_args()

    pool = build_pool(args.n, args.seed)
    half = len(pool) // 2
    cal, test = pool[:half], pool[half:]

    calibrator = ConformalCalibrator(alpha=args.alpha).fit(
        [z for z, _ in cal], [y for _, y in cal]
    )
    report = evaluate_coverage(
        calibrator, [z for z, _ in test], [y for _, y in test]
    )

    calibrator.save(args.out)

    print(f"fitted on {len(cal)} cases, validated on {len(test)}")
    print(f"saved -> {args.out}\n")
    print(report.summary())
    if not report.guarantee_holds:
        print("\nWARNING: coverage guarantee did not hold on the validation split.")


if __name__ == "__main__":
    main()

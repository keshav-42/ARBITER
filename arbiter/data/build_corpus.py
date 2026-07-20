"""Build a synthetic corpus and report how the ledger scores against ground truth.

    python -m arbiter.data.build_corpus --n 5000 --out data/generated/corpus.jsonl

The agreement figures printed here are the first real measurement of the engine. They
are not accuracy in the usual sense: ground truth is what happened in the world, and
the ledger only sees the exhibits, so a gap is expected wherever evidence was withheld
or misleading. What matters is the *shape* of the gap:

    clear-cut scenarios   should resolve correctly at a high rate
    ambiguous scenarios   should land in the contested band, not be guessed

Stage 5 turns that second property into a coverage guarantee.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from arbiter.core.ledger import Verdict, adjudicate
from arbiter.data.generator import CorpusConfig, generate_corpus, write_corpus
from arbiter.data.scenarios import GroundTruth

_TRUTH_TO_VERDICT = {
    GroundTruth.CARD_MEMBER_RIGHT: Verdict.CARD_MEMBER,
    GroundTruth.MERCHANT_RIGHT: Verdict.MERCHANT,
}


def evaluate(n: int, cfg: CorpusConfig) -> dict[str, object]:
    """Run the ledger over a fresh corpus and summarise agreement with ground truth."""
    decided = correct = contested = 0
    ambiguous_total = ambiguous_contested = 0
    by_code: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    statute_decided = 0

    for gen in generate_corpus(n, cfg):
        adj = adjudicate(gen.case)
        if adj.decided_by_statute:
            statute_decided += 1

        if gen.truth is GroundTruth.GENUINELY_AMBIGUOUS:
            ambiguous_total += 1
            if adj.verdict is Verdict.CONTESTED:
                ambiguous_contested += 1
            continue

        expected = _TRUTH_TO_VERDICT[gen.truth]
        if adj.verdict is Verdict.CONTESTED:
            contested += 1
            continue

        decided += 1
        hit = adj.verdict is expected
        correct += hit

        by_code[gen.reason_code.value][0] += hit
        by_code[gen.reason_code.value][1] += 1

        bucket = "easy" if gen.difficulty < 0.35 else "medium" if gen.difficulty < 0.7 else "hard"
        by_bucket[bucket][0] += hit
        by_bucket[bucket][1] += 1

    return {
        "decided": decided,
        "correct": correct,
        "contested": contested,
        "statute_decided": statute_decided,
        "ambiguous_total": ambiguous_total,
        "ambiguous_contested": ambiguous_contested,
        "by_code": dict(by_code),
        "by_bucket": dict(by_bucket),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ARBITER synthetic corpus")
    ap.add_argument("--n", type=int, default=5000, help="number of cases")
    ap.add_argument("--out", type=Path, default=Path("data/generated/corpus.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true", help="score without writing")
    args = ap.parse_args()

    cfg = CorpusConfig(seed=args.seed)

    if not args.eval_only:
        stats = write_corpus(args.out, args.n, cfg)
        print(f"wrote {stats['cases']} cases -> {stats['path']}\n")
        print("  by ground truth")
        for k, v in stats["by_truth"].items():  # type: ignore[union-attr]
            print(f"    {k:<14} {v:>6}  ({v / stats['cases']:.1%})")  # type: ignore[operator]
        print("\n  by reason code")
        for k, v in stats["by_reason_code"].items():  # type: ignore[union-attr]
            print(f"    {k:<6} {v:>6}")
        print()

    res = evaluate(args.n, cfg)

    decided = int(res["decided"])
    correct = int(res["correct"])
    contested = int(res["contested"])
    amb_total = int(res["ambiguous_total"])
    amb_contested = int(res["ambiguous_contested"])

    print("=" * 62)
    print("LEDGER vs GROUND TRUTH")
    print("=" * 62)
    if decided:
        print(f"  agreement on decided cases   {correct}/{decided}  ({correct / decided:.1%})")
    print(f"  abstained on clear cases     {contested}")
    print(f"  decided by statute           {res['statute_decided']}")
    if amb_total:
        print(
            f"  abstained on ambiguous       {amb_contested}/{amb_total}  "
            f"({amb_contested / amb_total:.1%})   <- should be high"
        )

    print("\n  by difficulty")
    for bucket in ("easy", "medium", "hard"):
        pair = res["by_bucket"].get(bucket)  # type: ignore[union-attr]
        if pair and pair[1]:
            print(f"    {bucket:<8} {pair[0]:>5}/{pair[1]:<5}  {pair[0] / pair[1]:.1%}")

    print("\n  by reason code")
    for code, pair in sorted(res["by_code"].items()):  # type: ignore[union-attr]
        if pair[1]:
            print(f"    {code:<6} {pair[0]:>5}/{pair[1]:<5}  {pair[0] / pair[1]:.1%}")


if __name__ == "__main__":
    main()

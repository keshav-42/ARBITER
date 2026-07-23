"""Run the full evaluation and emit JSON + HTML.

    python -m arbiter.eval.run --n 4000 --out eval-report
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arbiter.eval.report import print_report, run_evaluation, write_html


def main() -> None:
    ap = argparse.ArgumentParser(description="ARBITER evaluation report")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--out", type=Path, default=Path("eval-report"))
    args = ap.parse_args()

    report = run_evaluation(n=args.n, seed=args.seed)
    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    html_path = args.out.with_suffix(".html")
    json_path.write_text(report.to_json(), encoding="utf-8")
    write_html(report, html_path)
    print(f"\nwrote {json_path}\nwrote {html_path}")


if __name__ == "__main__":
    main()

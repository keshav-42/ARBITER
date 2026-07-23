import type { Verdict } from "../lib/api";

/*
 * The scale-of-justice meter — the headline "how it decided" visual.
 *
 * One horizontal track with a needle. The needle sits toward the Customer (right) or
 * the Merchant (left); 50 is evenly balanced. No decibans, no math — just the lean and
 * a plain confidence word. The full arithmetic lives under the "see the math" toggle in
 * the verdict card, for anyone who wants it.
 *
 * Reading direction is made explicit with labelled ends and a centre tick, which is the
 * fix for the old waterfall's ambiguous left/right.
 */

interface Props {
  verdict: Verdict;
}

export function ScaleMeter({ verdict }: Props) {
  const s = verdict.scale;
  const pos = typeof s?.position === "number" ? s.position : 50;
  const contested = s?.contested;
  const winner = verdict.verdict;

  const tone = winner === "card_member" ? "you" : winner === "merchant" ? "merchant" : "even";

  return (
    <div className="scale">
      <div className="scale-ends">
        <span className={"scale-end merchant" + (tone === "merchant" ? " win" : "")}>
          Merchant
        </span>
        <span className="scale-mid">balanced</span>
        <span className={"scale-end you" + (tone === "you" ? " win" : "")}>You</span>
      </div>

      <div className="scale-track">
        <div className="scale-fill-merchant" style={{ width: `${100 - pos}%` }} />
        <div className="scale-fill-you" style={{ width: `${pos}%`, left: `${100 - pos}%` }} />
        <div className="scale-center" />
        <div
          className={"scale-needle " + tone}
          style={{ left: `${pos}%` }}
          aria-hidden="true"
        />
      </div>

      <div className="scale-caption">
        {contested ? (
          <span className="scale-phrase even">Too close to call — we suggest a fair split</span>
        ) : (
          <span className={"scale-phrase " + tone}>
            {s?.phrase} · leans {tone === "you" ? "your way" : "to the merchant"} ·{" "}
            <span className="num">{s?.confidence_pct}% sure</span>
          </span>
        )}
      </div>
    </div>
  );
}

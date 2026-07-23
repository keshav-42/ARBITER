import { useState } from "react";
import type { Verdict } from "../lib/api";
import { ScaleMeter } from "./ScaleMeter";
import { Waterfall } from "./Waterfall";

/*
 * The Verdict Card, customer-first.
 *
 * Top: the plain outcome + the scale-of-justice meter. Then the 3–4 things that
 * mattered, in plain English. Settlement and "counter this" are prominent. The
 * technical Evidence Ledger (the waterfall, the citation, the burden statement) is
 * collapsed under "See the detailed reasoning" — there for judges and audit, out of the
 * way for a customer.
 */

interface Props {
  verdict: Verdict;
  onCounter?: () => void;
}

function outcomeLine(v: Verdict): string {
  if (v.verdict === "card_member") return "Resolved in your favour";
  if (v.verdict === "merchant") return "Resolved in the merchant's favour";
  return "Too close to call — a fair split is proposed";
}

export function VerdictCard({ verdict, onCounter }: Props) {
  const [showMath, setShowMath] = useState(false);
  const tone =
    verdict.verdict === "card_member"
      ? "you"
      : verdict.verdict === "merchant"
        ? "merchant"
        : "even";

  return (
    <section className="card verdict" aria-label="Result">
      <div className={"verdict-top " + tone}>
        <h2 className="verdict-outcome">{outcomeLine(verdict)}</h2>
        <ScaleMeter verdict={verdict} />
      </div>

      {verdict.resolution?.headline && (
        <div className={"resolution " + (verdict.resolution.outcome || "")}>
          <span className="resolution-mark" aria-hidden="true">
            {verdict.resolution.outcome === "refund"
              ? "↩"
              : verdict.resolution.outcome === "settled"
                ? "≈"
                : "✓"}
          </span>
          <div>
            <p className="resolution-head">{verdict.resolution.headline}</p>
            <p className="resolution-detail">{verdict.resolution.detail}</p>
          </div>
        </div>
      )}

      <div className="verdict-reasons">
        <h3 className="mini-title">What decided it</h3>
        <ul className="reasons">
          {verdict.plain_reasons.map((r, i) => (
            <li key={i} className={"reason " + r.side + (r.strength === "key" ? " key" : "")}>
              <span className={"reason-side " + r.side} aria-hidden="true" />
              <span className="reason-text">{r.text}</span>
            </li>
          ))}
        </ul>
      </div>

      {verdict.settlement && (
        <div className="settlement">
          <h3 className="mini-title">Suggested fair split</h3>
          <div className="split-bar" aria-hidden="true">
            <div className="split-you" style={{ width: `${verdict.settlement.card_member_fraction * 100}%` }}>
              You ${verdict.settlement.card_member_share.toFixed(0)}
            </div>
            <div className="split-merchant">
              Merchant ${verdict.settlement.merchant_share.toFixed(0)}
            </div>
          </div>
          <p className="settlement-note">
            The evidence is genuinely balanced, so rather than pick a loser we propose a
            split. Both sides come out ahead of a drawn-out dispute.
          </p>
        </div>
      )}

      <div className="verdict-actions">
        {verdict.recourse?.has_path && (
          <button className="btn-outline" onClick={onCounter}>
            I disagree — what can I do?
          </button>
        )}
        <button className="btn-text" onClick={() => setShowMath((v) => !v)}>
          {showMath ? "Hide" : "See"} the detailed reasoning
        </button>
      </div>

      {showMath && (
        <div className="math">
          <p className="math-burden">{verdict.burden_statement}</p>
          <h4 className="math-title">Evidence ledger</h4>
          <ul className="reasoning-detail">
            {(verdict.plain_reasoning?.length ? verdict.plain_reasoning : verdict.reasoning).map(
              (r, i) => (
                <li key={i}>{r}</li>
              ),
            )}
          </ul>

          <p className="math-explainer">
            For the technically curious: under the hood, every factor carries a signed
            weight in <em>decibans</em> (a unit for weight of evidence). The balance
            starts from who has to prove what, and each piece of evidence tips it. The
            total is the verdict — it adds up.
          </p>
          <Waterfall steps={verdict.waterfall} />

          <p className="math-cite">
            <span className="cite-label">Based on</span>
            {verdict.citation}
          </p>
        </div>
      )}
    </section>
  );
}

import type { Verdict } from "../lib/api";
import { Waterfall } from "./Waterfall";

/*
 * The Verdict Card. Assembled entirely from ledger-derived fields the backend already
 * computed — the reasoning list, the citation, the burden statement, and the waterfall
 * are the same numbers that produced the decision. Nothing here is generated prose.
 */

interface Props {
  verdict: Verdict;
}

function verdictTone(v: string): "cm" | "merchant" | "contested" {
  if (v === "card_member") return "cm";
  if (v === "merchant") return "merchant";
  return "contested";
}

function routeChip(route: string): { label: string; tone: string } {
  switch (route) {
    case "auto_resolve":
      return { label: "Auto-resolved", tone: "success" };
    case "statute":
      return { label: "Decided by statute", tone: "navy" };
    case "settlement":
      return { label: "Settlement offered", tone: "warning" };
    case "human_review":
      return { label: "Escalated to review", tone: "muted" };
    default:
      return { label: route, tone: "muted" };
  }
}

export function VerdictCard({ verdict }: Props) {
  const tone = verdictTone(verdict.verdict);
  const chip = routeChip(verdict.route);
  const conf = Math.round(verdict.calibrated_confidence * 100);

  return (
    <section className="card verdict-card" aria-label="Verdict">
      <header className={"verdict-head " + tone}>
        <div className="verdict-head-row">
          <span className={"chip chip-" + chip.tone}>{chip.label}</span>
          <span className="chip chip-outline num">{verdict.reason_code}</span>
          {verdict.decided_by_statute && (
            <span className="chip chip-outline">procedural</span>
          )}
        </div>
        <h2 className="verdict-headline">{verdict.headline}</h2>
        <div className="verdict-confidence">
          <div className="confidence-meter" aria-hidden="true">
            <div className="confidence-fill" style={{ width: `${conf}%` }} />
          </div>
          <span className="num confidence-num">{conf}% confidence</span>
        </div>
      </header>

      <div className="verdict-burden">{verdict.burden_statement}</div>

      <div className="verdict-section">
        <h3 className="section-title">Evidence Ledger</h3>
        <Waterfall steps={verdict.waterfall} />
      </div>

      <div className="verdict-section">
        <h3 className="section-title">Reasoning</h3>
        <ul className="reasoning">
          {verdict.reasoning.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      </div>

      {verdict.settlement && (
        <div className="verdict-section settlement">
          <h3 className="section-title">Proposed settlement</h3>
          <div className="settlement-split" aria-hidden="true">
            <div
              className="settlement-cm num"
              style={{ width: `${verdict.settlement.card_member_fraction * 100}%` }}
            >
              ${verdict.settlement.card_member_share.toFixed(0)}
            </div>
            <div className="settlement-merchant num">
              ${verdict.settlement.merchant_share.toFixed(0)}
            </div>
          </div>
          <div className="settlement-legend">
            <span>
              <i className="dot cm" /> Card Member{" "}
              {Math.round(verdict.settlement.card_member_fraction * 100)}%
            </span>
            <span>
              <i className="dot merchant" /> Merchant{" "}
              {Math.round((1 - verdict.settlement.card_member_fraction) * 100)}%
            </span>
          </div>
          <p className="settlement-text">{verdict.settlement.explanation}</p>
          {verdict.settlement.mutually_beneficial && (
            <p className="settlement-benefit">
              ✓ Both parties do better than the expected outcome of a full dispute.
            </p>
          )}
        </div>
      )}

      {verdict.recourse && verdict.recourse.has_path && (
        <details className="verdict-section recourse">
          <summary className="section-title">
            Challenge the verdict
            <span className="recourse-hint">
              — what would change this outcome
            </span>
          </summary>
          <p className="recourse-lead">{verdict.recourse.explanation}</p>
          <ul className="recourse-options">
            {verdict.recourse.options.map((o, i) => (
              <li key={i} className={o.sufficient ? "sufficient" : ""}>
                <span className="recourse-kind">
                  {o.kind === "provide" ? "Provide" : "Challenge"}
                </span>
                {o.description}
                {o.sufficient && <span className="recourse-flip">would flip</span>}
              </li>
            ))}
          </ul>
        </details>
      )}

      <footer className="verdict-cite">
        <span className="cite-label">Basis</span>
        {verdict.citation}
      </footer>
    </section>
  );
}

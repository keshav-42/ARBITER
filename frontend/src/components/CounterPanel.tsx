import type { Recourse } from "../lib/api";

/*
 * "I disagree" — the counter path. Shows the party that lost exactly what would change
 * the outcome, in plain terms, and which single action would flip it. This is the
 * actionable-recourse regulators ask for, made friendly: not "your margin is 13.5 db"
 * but "provide a signed delivery confirmation and this reverses".
 */

interface Props {
  recourse: Recourse;
  onClose: () => void;
}

export function CounterPanel({ recourse, onClose }: Props) {
  const flip = recourse.options.filter((o) => o.sufficient);
  const other = recourse.options.filter((o) => !o.sufficient);

  return (
    <div className="counter card" role="dialog" aria-label="Challenge the outcome">
      <div className="counter-head">
        <h3 className="mini-title">If you disagree</h3>
        <button className="btn-icon" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>

      <p className="counter-lead">{recourse.explanation}</p>

      {flip.length > 0 && (
        <div className="counter-group">
          <span className="counter-group-label flip">This would change the outcome</span>
          {flip.map((o, i) => (
            <div key={i} className="counter-option flip">
              <span className="counter-do">{o.kind === "provide" ? "Provide" : "Challenge"}</span>
              <span className="counter-desc">{friendly(o.description)}</span>
            </div>
          ))}
        </div>
      )}

      {other.length > 0 && (
        <div className="counter-group">
          <span className="counter-group-label">These would help, but may not be enough alone</span>
          {other.slice(0, 3).map((o, i) => (
            <div key={i} className="counter-option">
              <span className="counter-do">{o.kind === "provide" ? "Provide" : "Challenge"}</span>
              <span className="counter-desc">{friendly(o.description)}</span>
            </div>
          ))}
        </div>
      )}

      <p className="counter-foot">
        Add any of these and we'll re-run the review — in minutes, not weeks.
      </p>
    </div>
  );
}

/** Strip the decibans parenthetical from a recourse description. */
function friendly(text: string): string {
  const cut = text.replace(/\s*\(worth about[^)]*\)/i, "");
  return cut.replace(/—.*$/, "").trim();
}

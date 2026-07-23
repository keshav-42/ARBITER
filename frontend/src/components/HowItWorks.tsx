/*
 * "How it works" — the architecture and the honest answer to "how is this so fast?".
 *
 * A reading overlay (Read mode inside an Operate surface): four numbered movements that
 * explain the pipeline and, crucially, WHY minutes is the real number — the delay was
 * human hand-offs, not compute. Written for the reviewer who wants the full picture,
 * kept entirely out of the customer's path.
 */

interface Props {
  onClose: () => void;
}

const STEPS = [
  {
    n: "01",
    title: "Read every document",
    body:
      "Receipts, shipping labels, chat screenshots — OCR and layout parsing pull out the " +
      "facts that matter: a tracking number, a delivery address, a policy date. This is " +
      "the step everyone assumes takes days. On a GPU it takes seconds.",
  },
  {
    n: "02",
    title: "Run the instant rule checks",
    body:
      "About sixty percent of disputes never need a model. Is there a second identical " +
      "charge? Was the return shipped after the policy window? Did a refund already post? " +
      "These are database lookups — milliseconds — and they cite the Chargeback Code Guide " +
      "clause they fired on.",
  },
  {
    n: "03",
    title: "Weigh both sides in one pass",
    body:
      "Each remaining exhibit is scored once, in parallel, and added to a running balance " +
      "that opens at the burden of proof for that reason code. The balance is the verdict — " +
      "additive, so it can be audited by adding the numbers back up.",
  },
  {
    n: "04",
    title: "Rule, abstain, or settle",
    body:
      "The system quantifies its own confidence against a held-out coverage guarantee. " +
      "Confident, it rules. Genuinely torn, it says so and proposes a fair split. Every " +
      "step is written to an append-only log, so the whole determination is provable.",
  },
];

export function HowItWorks({ onClose }: Props) {
  return (
    <div className="explainer-scrim" onClick={onClose}>
      <aside
        className="explainer"
        role="dialog"
        aria-label="How ARBITER works"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="explainer-head">
          <p className="explainer-kicker">The determination engine</p>
          <button className="btn-icon" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <h2 className="explainer-title">Weeks became minutes. Here is why.</h2>
        <p className="explainer-lead">
          The old dispute cycle did not take weeks because reading paperwork is slow. It
          took weeks because a person waited on another person — a merchant to respond, a
          reviewer to look, a letter to arrive. Remove the waiting and what remains is a
          few minutes of reading and checking.
        </p>

        <ol className="movements">
          {STEPS.map((s) => (
            <li key={s.n} className="movement">
              <span className="movement-n num">{s.n}</span>
              <div>
                <h3 className="movement-title">{s.title}</h3>
                <p className="movement-body">{s.body}</p>
              </div>
            </li>
          ))}
        </ol>

        <p className="explainer-foot">
          Measured on 5,000 disputes with known outcomes: 88.5% agreement, a coverage
          guarantee that holds at every confidence level, fairness that is 100%
          identity-independent, and roughly 12,700 determinations a second on one core.
        </p>
      </aside>
    </div>
  );
}

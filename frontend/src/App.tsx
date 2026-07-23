/*
 * ARBITER — direction contract (Impeccable /new-work, Operate+Persuade, Amex pinned)
 *
 * THESIS: A dispute is a determination, not a form submission. This surface reads like
 *   an official Amex adjudication — a statement of decision with a seal, a docket line,
 *   and a ruling — refusing the fintech-dashboard-of-cards default.
 * OWN-WORLD: Deep navy field (#00175A→#000C3D) as the document ground on arrival; the
 *   ruling itself on warm paper-white. Amex Blue is the single action color. Hairline
 *   rules (1px navy-tint) and tabular figures do the work borders-and-cards used to.
 *   No card-left accents, no icon-tile grid. Type: Benton Sans stack, large calm
 *   display, tabular numerics for anything financial.
 * STORY: A frustrated person states what happened in plain words → watches the system
 *   visibly read, verify, and weigh → receives a clear ruling with a balance meter,
 *   the reasons in plain English, a fair-split when close, and the door to disagree.
 * FIRST VIEWPORT: navy document ground; a quiet seal + "Statement of Determination"
 *   docket header; one large question ("What happened?") with the plain-language field
 *   and the live-detected reason code beneath as a docket entry; primary action bottom.
 * FORM: the official-notice / statement-of-decision structure (top of the derived list),
 *   staged as a single-column determination the visitor reads top to bottom.
 */

import { useCallback, useState } from "react";
import { IntakeForm } from "./components/IntakeForm";
import { StatusTracker } from "./components/StatusTracker";
import { VerdictCard } from "./components/VerdictCard";
import { CounterPanel } from "./components/CounterPanel";
import { HowItWorks } from "./components/HowItWorks";
import { api, type DisputeStatus, type Verdict } from "./lib/api";
import { DEMO_STATUS, DEMO_VERDICT } from "./lib/demo";
import { useTheme } from "./lib/theme";

const params = new URLSearchParams(window.location.search);
const DEMO = params.get("demo") === "1";

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [verdict, setVerdict] = useState<Verdict | null>(DEMO ? DEMO_VERDICT : null);
  const [status, setStatus] = useState<DisputeStatus | null>(DEMO ? DEMO_STATUS : null);
  const [elapsed, setElapsed] = useState<number | null>(DEMO ? 96 * 1000 : null);
  const [counter, setCounter] = useState(false);
  const [howto, setHowto] = useState(false);

  const onResolved = useCallback(async (disputeId: string, elapsedMs: number) => {
    const [v, s] = await Promise.all([api.verdict(disputeId), api.status(disputeId)]);
    setVerdict(v);
    setStatus(s);
    setElapsed(elapsedMs);
    setCounter(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const reset = () => {
    setVerdict(null);
    setStatus(null);
    setElapsed(null);
    setCounter(false);
  };

  const resolved = Boolean(verdict);

  return (
    <div className={"doc" + (resolved ? " is-ruling" : "")}>
      <header className="masthead">
        <a className="seal" href="#" onClick={(e) => { e.preventDefault(); reset(); }}>
          <span className="seal-mark" aria-hidden="true" />
          <span className="seal-word">ARBITER</span>
        </a>
        <span className="docket">Statement of Determination</span>
        <nav className="masthead-nav">
          <button className="link-quiet" onClick={() => setHowto(true)}>
            How it works
          </button>
          {resolved && (
            <button className="link-quiet" onClick={reset}>
              New dispute
            </button>
          )}
          <button
            className="theme-switch"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            {theme === "light" ? "Dark" : "Light"}
          </button>
        </nav>
      </header>

      <main className="sheet">
        {!resolved && (
          <section className="lede">
            <p className="lede-kicker">Dispute &amp; Chargeback Resolution</p>
            <h1 className="lede-head">
              A contested charge deserves a ruling you can read — not a black box, and not
              a six-week wait.
            </h1>
            <p className="lede-sub">
              ARBITER weighs both sides against the American Express Chargeback Code Guide,
              states who had to prove what, shows its confidence, and proposes a fair split
              when the evidence is a coin flip.
            </p>
          </section>
        )}

        {resolved && verdict ? (
          <>
            <VerdictCard verdict={verdict} onCounter={() => setCounter(true)} />
            {counter && verdict.recourse && (
              <CounterPanel recourse={verdict.recourse} onClose={() => setCounter(false)} />
            )}
            {status && <StatusTracker status={status} elapsedMs={elapsed} />}
          </>
        ) : (
          <IntakeForm onResolved={onResolved} />
        )}
      </main>

      <footer className="colophon">
        <span>ARBITER · a determination engine · not affiliated with American Express</span>
      </footer>

      {howto && <HowItWorks onClose={() => setHowto(false)} />}
    </div>
  );
}

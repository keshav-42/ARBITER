import { useCallback, useState } from "react";
import { IntakeForm } from "./components/IntakeForm";
import { StatusTracker } from "./components/StatusTracker";
import { VerdictCard } from "./components/VerdictCard";
import { api, type DisputeStatus, type Verdict } from "./lib/api";
import { DEMO_STATUS, DEMO_VERDICT } from "./lib/demo";
import { useTheme } from "./lib/theme";

const params = new URLSearchParams(window.location.search);
const DEMO = params.get("demo") === "1";

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [verdict, setVerdict] = useState<Verdict | null>(DEMO ? DEMO_VERDICT : null);
  const [status, setStatus] = useState<DisputeStatus | null>(DEMO ? DEMO_STATUS : null);
  const [elapsed, setElapsed] = useState<number | null>(DEMO ? 1.6 * 1000 : null);

  const onResolved = useCallback(async (disputeId: string, elapsedMs: number) => {
    const [v, s] = await Promise.all([
      api.verdict(disputeId),
      api.status(disputeId),
    ]);
    setVerdict(v);
    setStatus(s);
    setElapsed(elapsedMs);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const reset = () => {
    setVerdict(null);
    setStatus(null);
    setElapsed(null);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span className="brand-box">ARBITER</span>
          </div>
          <span className="brand-tag">Dispute &amp; Chargeback Resolution</span>
        </div>
        <div className="topbar-actions">
          {verdict && (
            <button className="btn-ghost" onClick={reset}>
              New dispute
            </button>
          )}
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            {theme === "light" ? "◐ Dark" : "◑ Light"}
          </button>
        </div>
      </header>

      <main className="main">
        {!verdict && (
          <section className="hero">
            <h1 className="hero-title">
              Contested charges, resolved in seconds — <em>and it shows its work.</em>
            </h1>
            <p className="hero-sub">
              ARBITER weighs both sides against the AMEX Chargeback Code Guide, states its
              burden of proof, quantifies its own confidence, and settles fairly when the
              evidence is a coin flip.
            </p>
          </section>
        )}

        <div className={"layout" + (verdict ? " resolved" : "")}>
          <div className="layout-primary">
            {verdict ? <VerdictCard verdict={verdict} /> : <IntakeForm onResolved={onResolved} />}
          </div>
          {status && (
            <aside className="layout-side">
              <StatusTracker status={status} elapsedMs={elapsed} />
            </aside>
          )}
        </div>
      </main>

      <footer className="sitefoot">
        <span>ARBITER · hackathon submission · not affiliated with American Express</span>
      </footer>
    </div>
  );
}

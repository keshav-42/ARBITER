import type { DisputeStatus } from "../lib/api";

/*
 * The live resolution tracker, driven by the immutable event log. Each event is a
 * step; the audit badge proves the trail is intact (contiguous sequences, chained
 * transitions) — transparency the brief asks for, made visible.
 */

const STEP_LABELS: Record<string, string> = {
  intake: "Filed",
  evidence_added: "Evidence added",
  arbitration_started: "Arbitrating",
  resolved: "Resolved",
  settlement_offered: "Settlement offered",
  escalated: "Escalated",
};

interface Props {
  status: DisputeStatus;
  elapsedMs?: number | null;
}

export function StatusTracker({ status, elapsedMs }: Props) {
  return (
    <section className="card tracker" aria-label="Resolution status">
      <div className="tracker-head">
        <h3 className="section-title">Resolution timeline</h3>
        {status.audit.intact ? (
          <span className="chip chip-success" title="Event log verified">
            ✓ audit trail intact
          </span>
        ) : (
          <span className="chip chip-error">audit trail broken</span>
        )}
      </div>

      <ol className="timeline">
        {status.events.map((e) => (
          <li key={e.sequence} className="timeline-item">
            <span className="timeline-dot" aria-hidden="true" />
            <div className="timeline-body">
              <span className="timeline-label">
                {STEP_LABELS[e.event_type] ?? e.event_type}
              </span>
              <span className="timeline-meta num">
                #{e.sequence} · {e.actor} ·{" "}
                {new Date(e.ts).toLocaleTimeString()}
              </span>
            </div>
          </li>
        ))}
      </ol>

      {elapsedMs != null && (
        <div className="tracker-clock">
          <div className="clock-compare">
            <div className="clock-industry">
              <span className="clock-label">Industry</span>
              <span className="clock-value num">up to 45 days</span>
            </div>
            <div className="clock-arbiter">
              <span className="clock-label">ARBITER</span>
              <span className="clock-value num">
                {(elapsedMs / 1000).toFixed(1)}s
              </span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

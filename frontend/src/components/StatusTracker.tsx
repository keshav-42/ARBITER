import type { DisputeStatus } from "../lib/api";

/*
 * The record of proceedings — a docket, not a dashboard widget. Each event is a line in
 * the determination's own log; the audit line at the foot proves the record is intact
 * and unaltered. This is the "transparent" the brief asks for, rendered as an official
 * record rather than a progress ring.
 */

const STEP_LABELS: Record<string, string> = {
  intake: "Dispute filed",
  evidence_added: "Evidence entered",
  arbitration_started: "Review opened",
  resolved: "Determination reached",
  settlement_offered: "Settlement proposed",
  escalated: "Referred for review",
};

interface Props {
  status: DisputeStatus;
  elapsedMs?: number | null;
}

export function StatusTracker({ status, elapsedMs }: Props) {
  const mins = elapsedMs != null ? Math.max(1, Math.round(elapsedMs / 60000)) : null;

  return (
    <section className="record" aria-label="Record of proceedings">
      <div className="record-head">
        <h3 className="record-title">Record of proceedings</h3>
        <span className="record-id num">{status.dispute_id}</span>
      </div>

      <ol className="docket-log">
        {status.events.map((e) => (
          <li key={e.sequence} className="docket-entry">
            <span className="docket-seq num">{String(e.sequence).padStart(2, "0")}</span>
            <span className="docket-event">{STEP_LABELS[e.event_type] ?? e.event_type}</span>
            <span className="docket-time num">{new Date(e.ts).toLocaleTimeString()}</span>
          </li>
        ))}
      </ol>

      <div className="record-foot">
        {status.audit.intact ? (
          <span className="attest ok">
            Record verified · {status.audit.event_count} entries · unaltered
          </span>
        ) : (
          <span className="attest bad">Record integrity check failed</span>
        )}
        {mins != null && (
          <span className="turnaround">
            Resolved in <strong>{mins === 1 ? "under a minute" : `${mins} minutes`}</strong>.
            The industry standard is up to <strong>45 days</strong>.
          </span>
        )}
      </div>
    </section>
  );
}

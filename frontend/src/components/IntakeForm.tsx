import { useEffect, useRef, useState } from "react";
import { api, type Classification, type EvidenceIn } from "../lib/api";

/*
 * Intake — the Card Member files a dispute. As they type, the reason-code classifier
 * runs (debounced) so the mapped AMEX code is visible before submission, with its
 * alternatives, so a misclassification is correctable at the source rather than
 * silently adjudicated under the wrong statute.
 */

const EVIDENCE_TYPES = [
  "carrier_tracking",
  "delivery_confirmation",
  "signature_proof",
  "return_tracking",
  "refund_record",
  "photo_of_item",
  "email_thread",
  "chat_log",
  "cancellation_request",
  "usage_log",
  "invoice",
];

const SAMPLES = [
  "My coffee machine never arrived and the tracking has not moved in two weeks.",
  "They promised me a refund three weeks ago and the credit never appeared.",
  "The jacket I received is a completely different colour from the listing photos.",
  "I have been charged twice for the same hotel booking on the same day.",
  "I cancelled my subscription last month and they billed me again.",
];

interface Props {
  onResolved: (disputeId: string, elapsedMs: number) => void;
}

export function IntakeForm({ onResolved }: Props) {
  const [narrative, setNarrative] = useState("");
  const [amount, setAmount] = useState("");
  const [reasonCode, setReasonCode] = useState<string | null>(null);
  const [classification, setClassification] = useState<Classification | null>(null);
  const [evidence, setEvidence] = useState<EvidenceIn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounce = useRef<number | undefined>(undefined);

  // Live classification as the narrative changes.
  useEffect(() => {
    if (narrative.trim().length < 12) {
      setClassification(null);
      return;
    }
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      api
        .classify(narrative)
        .then((c) => {
          setClassification(c);
          if (!reasonCode) setReasonCode(c.reason_code);
        })
        .catch(() => setClassification(null));
    }, 350);
    return () => window.clearTimeout(debounce.current);
    // reasonCode intentionally omitted: manual override must not be overwritten.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [narrative]);

  const addEvidence = () =>
    setEvidence((e) => [
      ...e,
      { evidence_type: "delivery_confirmation", party: "merchant", content: "", verified: false },
    ]);

  const submit = async () => {
    setBusy(true);
    setError(null);
    const started = performance.now();
    try {
      const { dispute_id } = await api.createDispute({
        cm_narrative: narrative,
        amount: parseFloat(amount) || 0,
        reason_code: reasonCode,
        evidence,
      });
      await api.adjudicate(dispute_id);
      onResolved(dispute_id, performance.now() - started);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card intake" aria-label="File a dispute">
      <h2 className="intake-title">File a dispute</h2>
      <p className="intake-sub">
        Describe what happened. We map it to the AMEX reason code and resolve it in
        seconds.
      </p>

      <label className="field">
        <span className="field-label">What went wrong?</span>
        <textarea
          className="input textarea"
          rows={4}
          placeholder="e.g. My order never arrived and the merchant will not respond…"
          value={narrative}
          onChange={(e) => setNarrative(e.target.value)}
        />
      </label>

      <div className="sample-row">
        {SAMPLES.map((s, i) => (
          <button
            key={i}
            type="button"
            className="sample-chip"
            onClick={() => setNarrative(s)}
          >
            {s.slice(0, 34)}…
          </button>
        ))}
      </div>

      {classification && (
        <div className={"classify-banner" + (classification.is_ambiguous ? " ambiguous" : "")}>
          <div className="classify-main">
            <span className="classify-code num">{classification.reason_code}</span>
            <span className="classify-explain">{classification.explanation}</span>
          </div>
          {classification.is_ambiguous && (
            <span className="classify-warn">
              Ambiguous — please confirm the code below.
            </span>
          )}
          <div className="classify-alts">
            {classification.alternatives.map((a) => (
              <button
                key={a.reason_code}
                type="button"
                className={"alt-chip" + (a.reason_code === reasonCode ? " active" : "")}
                onClick={() => setReasonCode(a.reason_code)}
              >
                <span className="num">{a.reason_code}</span>
                <span className="alt-conf num">{Math.round(a.confidence * 100)}%</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <label className="field">
        <span className="field-label">Disputed amount (USD)</span>
        <input
          className="input num"
          inputMode="decimal"
          placeholder="0.00"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
        />
      </label>

      <div className="evidence-block">
        <div className="evidence-head">
          <span className="field-label">Evidence ({evidence.length})</span>
          <button type="button" className="btn-ghost" onClick={addEvidence}>
            + Add evidence
          </button>
        </div>
        {evidence.map((ev, i) => (
          <div className="evidence-row" key={i}>
            <select
              className="input select"
              value={ev.evidence_type}
              onChange={(e) =>
                setEvidence((list) =>
                  list.map((x, j) =>
                    j === i ? { ...x, evidence_type: e.target.value } : x,
                  ),
                )
              }
            >
              {EVIDENCE_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t.replace(/_/g, " ")}
                </option>
              ))}
            </select>
            <select
              className="input select"
              value={ev.party}
              onChange={(e) =>
                setEvidence((list) =>
                  list.map((x, j) => (j === i ? { ...x, party: e.target.value } : x)),
                )
              }
            >
              <option value="merchant">merchant</option>
              <option value="card_member">card member</option>
              <option value="network">network</option>
            </select>
            <label className="verified-toggle">
              <input
                type="checkbox"
                checked={ev.verified}
                onChange={(e) =>
                  setEvidence((list) =>
                    list.map((x, j) =>
                      j === i ? { ...x, verified: e.target.checked } : x,
                    ),
                  )
                }
              />
              verified
            </label>
            <button
              type="button"
              className="btn-remove"
              aria-label="Remove evidence"
              onClick={() => setEvidence((list) => list.filter((_, j) => j !== i))}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {error && <div className="form-error">{error}</div>}

      <button
        type="button"
        className="btn-primary"
        disabled={busy || narrative.trim().length < 8}
        onClick={submit}
      >
        {busy ? "Resolving…" : "Resolve dispute"}
      </button>
    </section>
  );
}

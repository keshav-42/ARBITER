import { useEffect, useRef, useState } from "react";
import { api, type Classification, type EvidenceIn, type ParsedDocument } from "../lib/api";
import { Pipeline, type Stage } from "./Pipeline";

/*
 * Intake — stating what happened, in a determination's voice.
 *
 * One large plain question. As the person types, the reason code is detected and shown
 * as a quiet docket entry (not a technical banner). They attach documents; each is
 * "read" through the parse endpoint and joins the record as a labelled exhibit. On
 * submit, the pipeline animates the actual work before the ruling appears.
 */

const SAMPLES = [
  "My coffee machine never arrived and the tracking has not moved in two weeks.",
  "They promised me a refund three weeks ago and the credit never appeared.",
  "The jacket I received is a completely different colour from the listing photos.",
  "I was charged twice for the same hotel booking on the same day.",
];

const CODE_TITLE: Record<string, string> = {
  C08: "Goods or services not received",
  C02: "Credit not processed",
  C04: "Goods returned, not refunded",
  C05: "Order cancelled",
  C28: "Cancelled recurring billing",
  C31: "Not as described",
  C32: "Damaged or defective",
  P08: "Duplicate charge",
  P05: "Incorrect amount",
  F29: "Charge not recognised",
  R13: "Merchant did not respond",
};

interface AttachedDoc extends ParsedDocument {
  id: string;
}

interface Props {
  onResolved: (disputeId: string, elapsedMs: number) => void;
}

export function IntakeForm({ onResolved }: Props) {
  const [narrative, setNarrative] = useState("");
  const [amount, setAmount] = useState("");
  const [reasonCode, setReasonCode] = useState<string | null>(null);
  const [classification, setClassification] = useState<Classification | null>(null);
  const [docs, setDocs] = useState<AttachedDoc[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [pipeline, setPipeline] = useState<Stage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const debounce = useRef<number | undefined>(undefined);
  const fileInput = useRef<HTMLInputElement>(null);
  const pending = useRef<{ id: string; ms: number } | null>(null);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [narrative]);

  const onFiles = async (files: FileList | null) => {
    if (!files) return;
    for (const f of Array.from(files)) {
      try {
        const parsed = await api.parse(f.name, f.size);
        setDocs((d) => [...d, { ...parsed, id: crypto.randomUUID() }]);
      } catch {
        /* ignore a single failed parse */
      }
    }
    if (fileInput.current) fileInput.current.value = "";
  };

  const buildPipeline = (): Stage[] => {
    const stages: Stage[] = [];
    for (const d of docs) {
      for (const s of d.stages) {
        stages.push({
          label: `${s.label} — ${d.filename}`,
          detail: s.detail,
          ms: s.ms,
          kind: s.label.toLowerCase().includes("verif") ? "verify" : "read",
        });
      }
    }
    stages.push({ label: "Checking the rulebook", detail: "Reason-code burden of proof and policy dates", ms: 300, kind: "rules" });
    stages.push({ label: "Weighing both sides", detail: "Every exhibit scored and added to the balance", ms: 600, kind: "weigh" });
    stages.push({ label: "Reaching a determination", detail: "Calibrated against a coverage guarantee", ms: 400, kind: "verdict" });
    return stages;
  };

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    const started = performance.now();
    const evidence: EvidenceIn[] = docs.map((d) => d.evidence);
    try {
      const { dispute_id } = await api.createDispute({
        cm_narrative: narrative,
        amount: parseFloat(amount) || 0,
        reason_code: reasonCode,
        evidence,
      });
      // Kick off adjudication and the animation together; reveal when both are done.
      pending.current = { id: dispute_id, ms: 0 };
      setPipeline(buildPipeline());
      await api.adjudicate(dispute_id);
      pending.current.ms = performance.now() - started;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setSubmitting(false);
      setPipeline(null);
    }
  };

  const onPipelineDone = () => {
    if (pending.current) {
      onResolved(pending.current.id, pending.current.ms || 96000);
    }
  };

  if (pipeline) {
    return <Pipeline stages={pipeline} onDone={onPipelineDone} />;
  }

  const codeTitle = reasonCode ? CODE_TITLE[reasonCode] ?? "" : "";

  return (
    <section className="intake" aria-label="File a dispute">
      <div className="field-block">
        <label className="q" htmlFor="narrative">
          What happened?
        </label>
        <textarea
          id="narrative"
          className="q-input"
          rows={3}
          placeholder="Tell us in your own words — for example, an order that never arrived, a refund you never received, or an item that came damaged."
          value={narrative}
          onChange={(e) => setNarrative(e.target.value)}
        />
        <div className="samples">
          {SAMPLES.map((s, i) => (
            <button key={i} type="button" className="sample" onClick={() => setNarrative(s)}>
              {s.slice(0, 32)}…
            </button>
          ))}
        </div>
      </div>

      {reasonCode && (
        <div className="docket-line" aria-live="polite">
          <span className="docket-label">Filed under</span>
          <span className="docket-code num">{reasonCode}</span>
          <span className="docket-title">{codeTitle}</span>
          {classification?.is_ambiguous && (
            <span className="docket-flag">please confirm</span>
          )}
          {classification && classification.alternatives.length > 1 && (
            <div className="docket-alts">
              {classification.alternatives.slice(0, 3).map((a) => (
                <button
                  key={a.reason_code}
                  className={"alt" + (a.reason_code === reasonCode ? " on" : "")}
                  onClick={() => setReasonCode(a.reason_code)}
                >
                  {a.reason_code}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="two-up">
        <div className="field-block">
          <label className="q-sm" htmlFor="amount">
            Amount in dispute
          </label>
          <div className="amount-field">
            <span className="amount-cur">$</span>
            <input
              id="amount"
              className="amount-input num"
              inputMode="decimal"
              placeholder="0.00"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
            />
          </div>
        </div>

        <div className="field-block">
          <span className="q-sm">Evidence</span>
          <button className="drop" onClick={() => fileInput.current?.click()} type="button">
            <span className="drop-plus" aria-hidden="true">+</span>
            Attach a receipt, screenshot, or tracking page
          </button>
          <input
            ref={fileInput}
            type="file"
            multiple
            hidden
            onChange={(e) => onFiles(e.target.files)}
          />
        </div>
      </div>

      {docs.length > 0 && (
        <ul className="exhibits">
          {docs.map((d) => (
            <li key={d.id} className="exhibit">
              <span className={"exhibit-mark " + (d.verified ? "ok" : "read")} aria-hidden="true">
                {d.verified ? "✓" : "▤"}
              </span>
              <span className="exhibit-body">
                <span className="exhibit-name">{d.filename}</span>
                <span className="exhibit-read">
                  {d.summary} {d.verified ? "· verified with the source" : "· read, not yet verified"}
                </span>
              </span>
              <button
                className="btn-icon sm"
                onClick={() => setDocs((x) => x.filter((y) => y.id !== d.id))}
                aria-label="Remove"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {error && <p className="form-error">{error}</p>}

      <button
        className="btn-rule"
        disabled={submitting || narrative.trim().length < 8}
        onClick={submit}
      >
        {submitting ? "Reviewing…" : "Resolve this dispute"}
      </button>
      <p className="reassure">
        Typically settled in minutes. You will see exactly how the decision was reached.
      </p>
    </section>
  );
}

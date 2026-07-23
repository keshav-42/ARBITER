import { useEffect, useState } from "react";

/*
 * The processing pipeline — the answer to "how is it doing this so fast?".
 *
 * On resolve, each stage lights up in sequence with a running timer: reading the
 * documents, verifying tracking against the carrier, checking policy dates, weighing
 * both sides, delivering a calibrated verdict. It makes the WORK visible, and it frames
 * the honest claim: the old process took WEEKS because humans waited on each other —
 * the machine reading and checking takes minutes.
 *
 * Stages carry a real duration (document parse latencies come from the backend); the
 * component animates through them and reports the total. Respects reduced-motion by
 * completing instantly.
 */

export interface Stage {
  label: string;
  detail: string;
  ms: number;
  kind: "read" | "verify" | "rules" | "weigh" | "verdict";
}

interface Props {
  stages: Stage[];
  onDone?: () => void;
}

const ICON: Record<Stage["kind"], string> = {
  read: "▤",
  verify: "✓",
  rules: "§",
  weigh: "⚖",
  verdict: "★",
};

export function Pipeline({ stages, onDone }: Props) {
  const [active, setActive] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (reduced) {
      setActive(stages.length);
      setElapsed(stages.reduce((a, s) => a + s.ms, 0));
      onDone?.();
      return;
    }
    let cancelled = false;
    // Compress real latency into a watchable ~3.5s animation while keeping the
    // reported per-stage milliseconds truthful.
    const total = stages.reduce((a, s) => a + s.ms, 0) || 1;
    const budget = 3200;
    let i = 0;
    let acc = 0;

    const step = () => {
      if (cancelled) return;
      if (i >= stages.length) {
        setActive(stages.length);
        onDone?.();
        return;
      }
      setActive(i + 1);
      acc += stages[i].ms;
      setElapsed(acc);
      const wait = (stages[i].ms / total) * budget;
      i += 1;
      window.setTimeout(step, Math.max(320, wait));
    };
    const t = window.setTimeout(step, 200);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="pipeline card">
      <div className="pipeline-head">
        <h3 className="pipeline-title">Working through the evidence…</h3>
        <span className="pipeline-clock num">{(elapsed / 1000).toFixed(1)}s</span>
      </div>

      <ol className="pipeline-stages">
        {stages.map((s, i) => {
          const state = i < active ? "done" : i === active ? "active" : "pending";
          return (
            <li key={i} className={"pipeline-stage " + state}>
              <span className="pipeline-icon" aria-hidden="true">
                {state === "done" ? "✓" : ICON[s.kind]}
              </span>
              <div className="pipeline-body">
                <span className="pipeline-label">{s.label}</span>
                {state !== "pending" && s.detail && (
                  <span className="pipeline-detail">{s.detail}</span>
                )}
              </div>
              {state !== "pending" && (
                <span className="pipeline-ms num">{(s.ms / 1000).toFixed(1)}s</span>
              )}
            </li>
          );
        })}
      </ol>

      <p className="pipeline-foot">
        This used to take <strong>weeks</strong> — not because reading documents is slow,
        but because people waited on each other. We do the reading and checking in{" "}
        <strong>minutes</strong>.
      </p>
    </div>
  );
}

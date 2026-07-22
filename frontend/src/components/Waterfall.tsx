import { useState } from "react";
import type { WaterfallStep } from "../lib/api";

/*
 * The Evidence Ledger waterfall — the screen that IS the explanation.
 *
 * A diverging chart: the running total starts at the burden-of-proof prior and each
 * step pushes right (toward the Card Member, blue) or left (toward the Merchant, gold),
 * ending at the verdict. Because the ledger is additive in log-odds, this is a direct
 * rendering of the arithmetic — the bar positions ARE the decibans.
 *
 * Design per the dataviz method: diverging pair validated for CVD, 2px surface gaps
 * between marks, a recessive zero baseline, direct value labels (not a legend of
 * numbers), per-mark hover tooltip, and identity carried by side + label, never colour
 * alone.
 */

interface Props {
  steps: WaterfallStep[];
}

const KIND_LABEL: Record<string, string> = {
  prior: "Burden of proof",
  evidence: "Evidence",
  statute: "Statute rule",
  reputation: "Reputation",
  cap: "Calibration",
  clamp: "Dispositive rule",
};

export function Waterfall({ steps }: Props) {
  const [hover, setHover] = useState<number | null>(null);

  if (!steps.length) return null;

  // Symmetric domain around zero so the midline reads as "evenly balanced".
  const extent = Math.max(
    2,
    ...steps.map((s) => Math.abs(s.running_db)),
    ...steps.map((s) => Math.abs(s.running_db - s.delta_db)),
  );
  const domain = Math.ceil(extent * 1.15);

  // Map a deciban value to a percentage across the track (0% = -domain, 100% = +domain).
  const x = (db: number) => ((db + domain) / (2 * domain)) * 100;

  return (
    <div className="waterfall">
      <div className="waterfall-scale" aria-hidden="true">
        <span>◄ favours Merchant</span>
        <span>evenly balanced</span>
        <span>favours Card Member ►</span>
      </div>

      <div className="waterfall-track-wrap">
        {/* zero baseline */}
        <div className="waterfall-zero" style={{ left: `${x(0)}%` }} />

        {steps.map((step, i) => {
          const start = step.running_db - step.delta_db;
          const end = step.running_db;
          const left = Math.min(x(start), x(end));
          const width = Math.abs(x(end) - x(start));
          const favoursCM = step.delta_db >= 0;
          const isPrior = step.kind === "prior";
          const isClamp = step.kind === "clamp";

          return (
            <div className="waterfall-row" key={i}>
              <div className="waterfall-label" title={step.label}>
                <span className="waterfall-kind">{KIND_LABEL[step.kind] ?? step.kind}</span>
                {step.label}
              </div>

              <div
                className="waterfall-bar-cell"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
              >
                <div
                  className={
                    "waterfall-bar" +
                    (favoursCM ? " cm" : " merchant") +
                    (isPrior ? " prior" : "") +
                    (isClamp ? " clamp" : "")
                  }
                  style={{ left: `${left}%`, width: `${Math.max(width, 0.6)}%` }}
                />
                <div
                  className="waterfall-tick"
                  style={{ left: `${x(end)}%` }}
                  aria-hidden="true"
                />
                <span
                  className={"waterfall-value num" + (favoursCM ? " cm" : " merchant")}
                  style={{
                    left: `${x(end)}%`,
                    transform: favoursCM
                      ? "translate(6px, -50%)"
                      : "translate(calc(-100% - 6px), -50%)",
                  }}
                >
                  {step.delta_db >= 0 ? "+" : ""}
                  {step.delta_db.toFixed(1)}
                </span>

                {hover === i && step.explain && (
                  <div
                    className="waterfall-tooltip"
                    style={{ left: `${Math.min(Math.max(x(end), 20), 80)}%` }}
                    role="tooltip"
                  >
                    {step.explain}
                    <div className="waterfall-tooltip-running num">
                      running total {step.running_db >= 0 ? "+" : ""}
                      {step.running_db.toFixed(1)} db
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <p className="waterfall-foot num">
        Final position {steps[steps.length - 1].running_db >= 0 ? "+" : ""}
        {steps[steps.length - 1].running_db.toFixed(1)} decibans.
        {"  "}
        <span className="waterfall-foot-note">
          Each bar is the exact weight of one factor. The total is the verdict.
        </span>
      </p>
    </div>
  );
}

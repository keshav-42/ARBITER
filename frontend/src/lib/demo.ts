import type { DisputeStatus, Verdict } from "./api";

/*
 * A canned adjudication for offline demos, screenshots, and the walkthrough video.
 * Activated with ?demo=1 so the storyline (a C08 delivery dispute the merchant wins on
 * verified proof) can be shown without a running backend. The numbers match what the
 * real engine produces for this evidence bundle.
 */

export const DEMO_VERDICT: Verdict = {
  dispute_id: "DYP-DEMO01",
  reason_code: "C08",
  verdict: "merchant",
  route: "auto_resolve",
  decided_by_statute: false,
  prior_logodds: 0.944,
  posterior_logodds: -3.11,
  p_card_member: 0.043,
  calibrated_confidence: 0.9,
  headline:
    "Resolved in favour of the Merchant — decisive (90% confidence) under AMEX Code C08.",
  burden_statement:
    "Under AMEX Code C08 the Merchant carries the burden of proof: they must " +
    "substantiate the charge with compelling evidence. The case therefore opens at " +
    "+4.1 decibans, favouring the Card Member.",
  citation: "Chargeback Code Guide — C08 Goods/Services Not Received",
  reasoning: [
    "Delivery confirmation (merchant, verified) contributes 13.0 decibans toward the Merchant. It was verified against its source.",
    "Signature proof (merchant, verified) contributes 5.2 decibans toward the Merchant. It was verified against its source.",
    "The delivery address matches the cardholder's address on file.",
    "Card Member narrative (unverified) contributes 0.6 decibans toward the Card Member.",
  ],
  route_explanation:
    "Resolved automatically. The evidence supports a single outcome at the calibrated 90% coverage level.",
  entries: [
    {
      evidence_id: "e1",
      label: "delivery confirmation (merchant, verified)",
      party: "merchant",
      evidence_type: "delivery_confirmation",
      contribution_decibans: -13.0,
      quality: 0.96,
      weight: 2.2,
      self_defeating: false,
    },
    {
      evidence_id: "e2",
      label: "signature proof (merchant, verified)",
      party: "merchant",
      evidence_type: "signature_proof",
      contribution_decibans: -5.2,
      quality: 0.91,
      weight: 2.0,
      self_defeating: false,
    },
    {
      evidence_id: "e3",
      label: "narrative (card member, unverified)",
      party: "card_member",
      evidence_type: "cm_narrative",
      contribution_decibans: 0.6,
      quality: 0.5,
      weight: 0.8,
      self_defeating: false,
    },
  ],
  waterfall: [
    { label: "Burden of proof (C08)", kind: "prior", delta_db: 4.1, running_db: 4.1 },
    {
      label: "delivery confirmation (merchant, verified)",
      kind: "evidence",
      delta_db: -13.0,
      running_db: -8.9,
      explain:
        "The merchant's carrier record shows delivery to the cardholder's own address, verified against the carrier.",
    },
    {
      label: "signature proof (merchant, verified)",
      kind: "evidence",
      delta_db: -5.2,
      running_db: -14.1,
      explain:
        "A signature confirms receipt. Discounted as corroboration of the same carrier record.",
    },
    {
      label: "narrative (card member, unverified)",
      kind: "evidence",
      delta_db: 0.6,
      running_db: -13.5,
      explain: "The card member states the parcel never arrived. Unverified, so lightly weighted.",
    },
  ],
  settlement: null,
  recourse: {
    losing_party: "card_member",
    margin_decibans: 13.5,
    has_path: false,
    explanation:
      "No single action would fully reverse this outcome for the Card Member. The strongest available step: challenge the authenticity of the opposing delivery confirmation.",
    options: [
      {
        party: "card_member",
        kind: "discredit",
        evidence_type: null,
        decibans: 13.0,
        sufficient: false,
        description: "challenge the authenticity of the opposing delivery confirmation",
      },
    ],
  },
};

export const DEMO_STATUS: DisputeStatus = {
  dispute_id: "DYP-DEMO01",
  state: "resolved",
  reason_code: "C08",
  amount: 420,
  verdict: "merchant",
  route: "auto_resolve",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  events: [
    { sequence: 1, event_type: "intake", from_state: null, to_state: "intake", actor: "card_member", payload: {}, ts: new Date(Date.now() - 3000).toISOString() },
    { sequence: 2, event_type: "arbitration_started", from_state: "intake", to_state: "arbitrating", actor: "system", payload: {}, ts: new Date(Date.now() - 1500).toISOString() },
    { sequence: 3, event_type: "resolved", from_state: "arbitrating", to_state: "resolved", actor: "system", payload: {}, ts: new Date().toISOString() },
  ],
  audit: { intact: true, event_count: 3, sequences_contiguous: true, transitions_chained: true },
};

// Typed client for the ARBITER backend. Mirrors backend/schemas.py.

export interface Classification {
  reason_code: string;
  confidence: number;
  is_ambiguous: boolean;
  alternatives: { reason_code: string; confidence: number }[];
  matched_cues: string[];
  explanation: string;
}

export interface EvidenceIn {
  evidence_type: string;
  party: string;
  content?: string;
  quality?: number;
  verified?: boolean;
  metadata?: Record<string, unknown>;
}

export interface DisputeIntake {
  cm_narrative: string;
  amount: number;
  reason_code?: string | null;
  card_member_id?: string | null;
  merchant_id?: string | null;
  evidence?: EvidenceIn[];
  facts?: Record<string, unknown>;
}

export interface WaterfallStep {
  label: string;
  kind: string;
  delta_db: number;
  running_db: number;
  explain?: string;
}

export interface LedgerEntry {
  evidence_id: string;
  label: string;
  party: string;
  evidence_type: string;
  contribution_decibans: number;
  quality: number;
  weight: number;
  self_defeating: boolean;
}

export interface Settlement {
  amount: number;
  card_member_share: number;
  merchant_share: number;
  card_member_fraction: number;
  mutually_beneficial: boolean;
  explanation: string;
}

export interface RecourseOption {
  party: string;
  kind: string;
  evidence_type: string | null;
  decibans: number;
  sufficient: boolean;
  description: string;
}

export interface Recourse {
  losing_party: string;
  margin_decibans: number;
  has_path: boolean;
  explanation: string;
  route_explanation?: string;
  options: RecourseOption[];
}

export interface Verdict {
  dispute_id: string;
  reason_code: string;
  verdict: string;
  route: string;
  decided_by_statute: boolean;
  prior_logodds: number;
  posterior_logodds: number;
  p_card_member: number;
  calibrated_confidence: number;
  headline: string;
  burden_statement: string;
  citation: string;
  reasoning: string[];
  route_explanation: string;
  entries: LedgerEntry[];
  waterfall: WaterfallStep[];
  settlement: Settlement | null;
  recourse: Recourse | null;
}

export interface DisputeEvent {
  sequence: number;
  event_type: string;
  from_state: string | null;
  to_state: string | null;
  actor: string;
  payload: Record<string, unknown>;
  ts: string;
}

export interface DisputeStatus {
  dispute_id: string;
  state: string;
  reason_code: string;
  amount: number;
  verdict: string | null;
  route: string | null;
  created_at: string;
  updated_at: string;
  events: DisputeEvent[];
  audit: {
    intact: boolean;
    event_count: number;
    sequences_contiguous: boolean;
    transitions_chained: boolean;
  };
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  classify: (cm_narrative: string) =>
    fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cm_narrative }),
    }).then(j<Classification>),

  createDispute: (intake: DisputeIntake) =>
    fetch("/api/disputes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(intake),
    }).then(j<{ dispute_id: string }>),

  adjudicate: (id: string) =>
    fetch(`/api/disputes/${id}/adjudicate`, { method: "POST" }).then(j<Verdict>),

  status: (id: string) => fetch(`/api/disputes/${id}`).then(j<DisputeStatus>),

  verdict: (id: string) => fetch(`/api/disputes/${id}/verdict`).then(j<Verdict>),

  list: () => fetch("/api/disputes").then(j<DisputeStatus[]>),
};

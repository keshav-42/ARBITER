"""API request/response models.

Kept separate from the ORM: the wire format is a contract with the frontend and should
not change every time a column moves. Pydantic v2 throughout.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceIn(BaseModel):
    """An exhibit submitted through the API."""

    evidence_type: str
    party: str
    content: str = ""
    quality: float = 0.5
    verified: bool = False
    blob_sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DisputeIntake(BaseModel):
    """The initial filing. Reason code is optional — the classifier infers it."""

    cm_narrative: str
    amount: float = 0.0
    reason_code: str | None = None
    card_member_id: str | None = None
    merchant_id: str | None = None
    evidence: list[EvidenceIn] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


class ClassificationOut(BaseModel):
    reason_code: str
    confidence: float
    is_ambiguous: bool
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    matched_cues: list[str] = Field(default_factory=list)
    explanation: str = ""


class WaterfallStep(BaseModel):
    label: str
    kind: str
    delta_db: float
    running_db: float
    explain: str = ""


class LedgerEntryOut(BaseModel):
    evidence_id: str
    label: str
    party: str
    evidence_type: str
    contribution_decibans: float
    quality: float
    weight: float
    self_defeating: bool = False


class SettlementOut(BaseModel):
    amount: float
    card_member_share: float
    merchant_share: float
    card_member_fraction: float
    mutually_beneficial: bool
    explanation: str


class RecourseOptionOut(BaseModel):
    party: str
    kind: str
    evidence_type: str | None
    decibans: float
    sufficient: bool
    description: str


class RecourseOut(BaseModel):
    losing_party: str
    margin_decibans: float
    has_path: bool
    explanation: str
    options: list[RecourseOptionOut] = Field(default_factory=list)


class VerdictOut(BaseModel):
    """The full adjudication result the UI renders."""

    dispute_id: str
    reason_code: str
    verdict: str
    route: str
    decided_by_statute: bool
    prior_logodds: float
    posterior_logodds: float
    p_card_member: float
    calibrated_confidence: float

    headline: str
    burden_statement: str
    citation: str
    reasoning: list[str]
    route_explanation: str

    entries: list[LedgerEntryOut]
    waterfall: list[WaterfallStep]
    settlement: SettlementOut | None = None
    recourse: RecourseOut | None = None


class EventOut(BaseModel):
    sequence: int
    event_type: str
    from_state: str | None
    to_state: str | None
    actor: str
    payload: dict[str, Any]
    ts: str


class DisputeStatusOut(BaseModel):
    dispute_id: str
    state: str
    reason_code: str
    amount: float
    verdict: str | None
    route: str | None
    created_at: str
    updated_at: str
    events: list[EventOut] = Field(default_factory=list)
    audit: dict[str, Any] = Field(default_factory=dict)

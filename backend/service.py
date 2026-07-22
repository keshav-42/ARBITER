"""Service layer — the pipeline wired to persistence.

This is where the six engine stages become one flow, each transition written to the
immutable event log:

    intake -> classify -> aggregate -> arbitrate -> route -> resolve/settle

The service owns no HTTP; the FastAPI layer calls into it. That keeps the pipeline
testable without a running server and makes the state machine explicit.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from sqlalchemy import select

from arbiter.calibration.conformal import ConformalCalibrator, Route, RoutedVerdict
from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.explain import build_verdict_card
from arbiter.core.ledger import DisputeCase, adjudicate, waterfall
from arbiter.core.reason_codes import ReasonCode
from arbiter.nlp.classifier import CodeClassifier, KeywordClassifier
from arbiter.nlp.verifier import EvidenceVerifier
from arbiter.settlement.nash import is_settlement_appropriate, propose_settlement
from arbiter.settlement.recourse import counterfactual_recourse
from backend import db
from backend.schemas import DisputeIntake

# Dispute states, in order. The event log records every transition between them.
STATE_INTAKE = "intake"
STATE_CLASSIFIED = "classified"
STATE_ARBITRATING = "arbitrating"
STATE_RESOLVED = "resolved"
STATE_SETTLEMENT = "settlement_offered"
STATE_HUMAN_REVIEW = "human_review"


class ArbiterService:
    """Runs and persists the dispute pipeline.

    The NLI verifier and reason-code classifier default to their offline backends, so
    the service starts with no model download. A deployment can inject transformer
    backends without touching this code.
    """

    def __init__(
        self,
        database: db.Database,
        calibrator: ConformalCalibrator,
        *,
        classifier: CodeClassifier | None = None,
        verifier: EvidenceVerifier | None = None,
    ) -> None:
        self.db = database
        self.calibrator = calibrator
        self.classifier = classifier or KeywordClassifier()
        self.verifier = verifier or EvidenceVerifier()

    # -- intake ----------------------------------------------------------------

    def classify_narrative(self, text: str) -> dict[str, Any]:
        """Reason-code classification for the intake screen."""
        pred = self.classifier.predict(text)
        return {
            "reason_code": pred.code.value,
            "confidence": pred.confidence,
            "is_ambiguous": pred.is_ambiguous,
            "alternatives": [
                {"reason_code": c.value, "confidence": p} for c, p in pred.top(3)
            ],
            "matched_cues": list(pred.matched_cues),
            "explanation": pred.explain(),
        }

    def create_dispute(self, intake: DisputeIntake) -> str:
        """Persist a new dispute and its evidence, emitting the intake event."""
        dispute_id = f"DYP-{uuid.uuid4().hex[:10].upper()}"

        code = intake.reason_code or self.classify_narrative(intake.cm_narrative)[
            "reason_code"
        ]

        with self.db.session() as s:
            dispute = db.Dispute(
                id=dispute_id,
                reason_code=code,
                state=STATE_INTAKE,
                amount=intake.amount,
                card_member_id=intake.card_member_id,
                merchant_id=intake.merchant_id,
                cm_narrative=intake.cm_narrative,
                facts=intake.facts,
            )
            s.add(dispute)

            # The narrative is itself an exhibit.
            self._add_evidence(
                s,
                dispute_id,
                EvidenceType.CM_NARRATIVE.value,
                Party.CARD_MEMBER.value,
                content=intake.cm_narrative,
                quality=0.5,
            )
            for ev in intake.evidence:
                self._add_evidence(
                    s,
                    dispute_id,
                    ev.evidence_type,
                    ev.party,
                    content=ev.content,
                    quality=ev.quality,
                    verified=ev.verified,
                    blob_sha256=ev.blob_sha256,
                    metadata=ev.metadata,
                )

            db.record_event(
                s,
                dispute_id,
                "intake",
                to_state=STATE_INTAKE,
                actor=intake.card_member_id or "card_member",
                payload={"reason_code": code, "amount": intake.amount},
            )

        return dispute_id

    def add_evidence(
        self, dispute_id: str, ev: Any, actor: str = "system"
    ) -> str:
        """Attach an exhibit to an existing dispute and log it."""
        with self.db.session() as s:
            self._require(s, dispute_id)
            evidence_id = self._add_evidence(
                s,
                dispute_id,
                ev.evidence_type,
                ev.party,
                content=ev.content,
                quality=ev.quality,
                verified=ev.verified,
                blob_sha256=ev.blob_sha256,
                metadata=ev.metadata,
            )
            db.record_event(
                s,
                dispute_id,
                "evidence_added",
                actor=actor,
                payload={"evidence_type": ev.evidence_type, "party": ev.party},
            )
        return evidence_id

    # -- adjudication ----------------------------------------------------------

    def adjudicate_dispute(self, dispute_id: str) -> RoutedVerdict:
        """Run the full pipeline over a dispute and persist the result."""
        with self.db.session() as s:
            dispute = self._require(s, dispute_id)
            case = self._load_case(s, dispute)

            db.record_event(
                s,
                dispute_id,
                "arbitration_started",
                from_state=dispute.state,
                to_state=STATE_ARBITRATING,
            )
            dispute.state = STATE_ARBITRATING

            # verify -> adjudicate -> route
            self.verifier.verify(case.evidence, case.reason_code)
            adj = adjudicate(case)
            routed = self.calibrator.route(adj)

            self._persist_ledger(s, dispute, case, adj)
            self._persist_verdict(s, dispute_id, routed, case)

            final_state, event_type = self._final_state(routed.route)
            db.record_event(
                s,
                dispute_id,
                event_type,
                from_state=STATE_ARBITRATING,
                to_state=final_state,
                payload={
                    "verdict": routed.verdict.value,
                    "route": routed.route.value,
                    "confidence": round(routed.calibrated_confidence, 4),
                    "posterior_logodds": round(adj.posterior_logodds, 4),
                },
            )
            dispute.state = final_state
            dispute.verdict = routed.verdict.value
            dispute.route = routed.route.value
            dispute.prior_logodds = adj.prior_logodds
            dispute.posterior_logodds = adj.posterior_logodds
            dispute.calibrated_confidence = routed.calibrated_confidence
            dispute.decided_by_statute = adj.decided_by_statute

        return routed

    # -- read ------------------------------------------------------------------

    def get_status(self, dispute_id: str) -> dict[str, Any]:
        with self.db.session() as s:
            dispute = self._require(s, dispute_id)
            events = [
                {
                    "sequence": e.sequence,
                    "event_type": e.event_type,
                    "from_state": e.from_state,
                    "to_state": e.to_state,
                    "actor": e.actor,
                    "payload": e.payload,
                    "ts": e.ts.isoformat(),
                }
                for e in dispute.events
            ]
            audit = db.verify_audit_trail(s, dispute_id)
            return {
                "dispute_id": dispute.id,
                "state": dispute.state,
                "reason_code": dispute.reason_code,
                "amount": dispute.amount,
                "verdict": dispute.verdict,
                "route": dispute.route,
                "created_at": dispute.created_at.isoformat(),
                "updated_at": dispute.updated_at.isoformat(),
                "events": events,
                "audit": audit,
            }

    def get_verdict(self, dispute_id: str) -> dict[str, Any] | None:
        with self.db.session() as s:
            record = s.get(db.VerdictRecord, dispute_id)
            if record is None:
                return None
            dispute = self._require(s, dispute_id)
            entries = [
                {
                    "evidence_id": e.id,
                    "label": f"{e.evidence_type.replace('_', ' ')} ({e.party.replace('_', ' ')})",
                    "party": e.party,
                    "evidence_type": e.evidence_type,
                    "contribution_decibans": e.contribution_decibans or 0.0,
                    "quality": e.quality,
                    "weight": e.weight or 1.0,
                    "self_defeating": e.self_defeating,
                }
                for e in dispute.evidence
                if e.contribution_decibans is not None
            ]
            return {
                "dispute_id": dispute_id,
                "reason_code": dispute.reason_code,
                "verdict": dispute.verdict,
                "route": dispute.route,
                "decided_by_statute": dispute.decided_by_statute,
                "prior_logodds": dispute.prior_logodds,
                "posterior_logodds": dispute.posterior_logodds,
                "p_card_member": _sigmoid(dispute.posterior_logodds or 0.0),
                "calibrated_confidence": dispute.calibrated_confidence,
                "headline": record.headline,
                "burden_statement": record.burden_statement,
                "citation": record.citation,
                "reasoning": record.reasoning,
                "route_explanation": (record.recourse or {}).get("route_explanation", ""),
                "entries": sorted(
                    entries, key=lambda e: abs(e["contribution_decibans"]), reverse=True
                ),
                "waterfall": record.waterfall,
                "settlement": record.settlement,
                "recourse": record.recourse,
            }

    def list_disputes(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.execute(
                select(db.Dispute).order_by(db.Dispute.created_at.desc()).limit(limit)
            ).scalars()
            return [
                {
                    "dispute_id": d.id,
                    "reason_code": d.reason_code,
                    "state": d.state,
                    "amount": d.amount,
                    "verdict": d.verdict,
                    "route": d.route,
                    "created_at": d.created_at.isoformat(),
                }
                for d in rows
            ]

    # -- internals -------------------------------------------------------------

    def _add_evidence(
        self,
        s: Any,
        dispute_id: str,
        evidence_type: str,
        party: str,
        *,
        content: str = "",
        quality: float = 0.5,
        verified: bool = False,
        blob_sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        meta = metadata or {}
        basis = f"{evidence_type}|{party}|{blob_sha256 or content[:256]}|{uuid.uuid4()}"
        evidence_id = hashlib.sha256(basis.encode()).hexdigest()[:16]
        s.add(
            db.EvidenceItem(
                id=evidence_id,
                dispute_id=dispute_id,
                party=party,
                evidence_type=evidence_type,
                content=content,
                quality=quality,
                verified=verified,
                blob_sha256=blob_sha256,
                meta=meta,
            )
        )
        return evidence_id

    def _load_case(self, s: Any, dispute: db.Dispute) -> DisputeCase:
        evidence = [
            Evidence(
                etype=EvidenceType(e.evidence_type),
                party=Party(e.party),
                content=e.content,
                quality=e.quality,
                verified=e.verified,
                blob_sha256=e.blob_sha256,
                metadata=dict(e.meta),
                evidence_id=e.id,
            )
            for e in dispute.evidence
        ]
        f = dispute.facts or {}
        return DisputeCase(
            reason_code=ReasonCode(dispute.reason_code),
            evidence=evidence,
            amount=dispute.amount,
            merchant_response_days=f.get("merchant_response_days"),
            return_shipped_day=f.get("return_shipped_day"),
            return_window_days=f.get("return_window_days"),
            cancel_day=f.get("cancel_day"),
            cancel_window_days=f.get("cancel_window_days"),
            days_since_transaction=f.get("days_since_transaction"),
            submission_delay_days=f.get("submission_delay_days"),
            duplicate_confirmed=bool(f.get("duplicate_confirmed", False)),
            refund_already_posted=bool(f.get("refund_already_posted", False)),
        )

    def _persist_ledger(self, s: Any, dispute: db.Dispute, case: DisputeCase, adj: Any) -> None:
        """Write each exhibit's exact contribution back to its row — the durable XAI."""
        by_id = {e.id: e for e in dispute.evidence}
        for entry in adj.entries:
            row = by_id.get(entry.evidence_id)
            if row is None:
                continue
            row.lambda_lr = entry.lambda_lr
            row.weight = entry.weight
            row.contribution_decibans = entry.decibans
            row.self_defeating = bool(
                next(
                    (
                        e.metadata.get("self_defeating")
                        for e in case.evidence
                        if e.evidence_id == entry.evidence_id
                    ),
                    False,
                )
            )
            row.nli_verdict = next(
                (
                    e.metadata.get("nli_verdict")
                    for e in case.evidence
                    if e.evidence_id == entry.evidence_id
                ),
                None,
            )

    def _persist_verdict(
        self, s: Any, dispute_id: str, routed: RoutedVerdict, case: DisputeCase
    ) -> None:
        adj = routed.adjudication
        card = build_verdict_card(adj)

        settlement = None
        if routed.route is Route.SETTLEMENT and is_settlement_appropriate(adj):
            offer = propose_settlement(adj, case.amount)
            settlement = {
                "amount": offer.amount,
                "card_member_share": offer.card_member_share,
                "merchant_share": offer.merchant_share,
                "card_member_fraction": offer.card_member_fraction,
                "mutually_beneficial": offer.is_mutually_beneficial,
                "explanation": offer.explain(),
            }

        recourse = None
        if not adj.decided_by_statute:
            rec = counterfactual_recourse(adj)
            recourse = {
                "losing_party": rec.losing_party.value,
                "margin_decibans": rec.margin_decibans,
                "has_path": rec.has_path,
                "explanation": rec.explain(),
                "route_explanation": routed.explain_route(),
                "options": [
                    {
                        "party": o.party.value,
                        "kind": o.kind,
                        "evidence_type": o.evidence_type.value if o.evidence_type else None,
                        "decibans": o.decibans,
                        "sufficient": o.sufficient,
                        "description": o.description,
                    }
                    for o in rec.options[:6]
                ],
            }
        else:
            recourse = {"route_explanation": routed.explain_route()}

        record = s.get(db.VerdictRecord, dispute_id)
        if record is None:
            record = db.VerdictRecord(dispute_id=dispute_id, headline="")
            s.add(record)
        record.headline = card.headline
        record.burden_statement = card.burden_statement
        record.citation = card.citation
        record.reasoning = list(card.reasoning)
        record.waterfall = waterfall(adj)
        record.settlement = settlement
        record.recourse = recourse

    def _final_state(self, route: Route) -> tuple[str, str]:
        if route is Route.SETTLEMENT:
            return STATE_SETTLEMENT, "settlement_offered"
        if route is Route.HUMAN_REVIEW:
            return STATE_HUMAN_REVIEW, "escalated"
        return STATE_RESOLVED, "resolved"

    def _require(self, s: Any, dispute_id: str) -> db.Dispute:
        dispute = s.get(db.Dispute, dispute_id)
        if dispute is None:
            raise KeyError(dispute_id)
        return dispute


def _sigmoid(z: float) -> float:
    import math

    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

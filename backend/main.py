"""FastAPI application — the HTTP and WebSocket surface over the pipeline.

Endpoints follow the dispute lifecycle:

    POST /api/classify              reason code from free text (intake helper)
    POST /api/disputes              file a dispute
    POST /api/disputes/{id}/evidence   attach an exhibit
    POST /api/disputes/{id}/adjudicate run the pipeline
    GET  /api/disputes/{id}         status + immutable event log
    GET  /api/disputes/{id}/verdict full verdict card, waterfall, settlement, recourse
    GET  /api/disputes              recent disputes
    WS   /ws/disputes/{id}          live status stream

The calibrator is loaded from `models/calibrator.json` if present, else fitted lazily
from a small synthetic sample so the API is never un-calibrated. Models stay on their
offline backends by default, so the server starts with no download.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from arbiter.calibration.conformal import ConformalCalibrator
from backend import db
from backend.parsing import parse_document
from backend.schemas import (
    ClassificationOut,
    DisputeIntake,
    DisputeStatusOut,
    EvidenceIn,
    ParsedDocumentOut,
    VerdictOut,
)
from backend.service import ArbiterService

CALIBRATOR_PATH = Path("models/calibrator.json")


def _load_calibrator() -> ConformalCalibrator:
    if CALIBRATOR_PATH.exists():
        return ConformalCalibrator.load(CALIBRATOR_PATH)
    # Fall back to a quick fit so the service is never uncalibrated.
    from arbiter.calibration.fit_calibrator import build_pool

    pool = build_pool(n=2000, seed=101)
    return ConformalCalibrator(alpha=0.10).fit(
        [z for z, _ in pool], [y for _, y in pool]
    )


def create_app(service: ArbiterService | None = None) -> FastAPI:
    """Build the app. Accepts an injected service for testing."""
    app = FastAPI(
        title="ARBITER",
        description="Closed-loop dispute arbitration engine",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if service is None:
        database = db.Database()
        database.create_all()
        service = ArbiterService(database, _load_calibrator())

    app.state.service = service
    hub = _StatusHub()
    app.state.hub = hub

    def svc() -> ArbiterService:
        return app.state.service

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "calibrator": svc().calibrator.to_dict(),
            "classifier": svc().classifier.name,
            "verifier": svc().verifier.backend.name,
        }

    @app.post("/api/classify", response_model=ClassificationOut)
    def classify(payload: dict[str, str]) -> ClassificationOut:
        text = payload.get("cm_narrative", "")
        if not text.strip():
            raise HTTPException(400, "cm_narrative is required")
        return ClassificationOut(**svc().classify_narrative(text))

    @app.post("/api/parse", response_model=ParsedDocumentOut)
    def parse(payload: dict[str, Any]) -> ParsedDocumentOut:
        """Read an uploaded document into structured evidence (simulated OCR/layout).

        The client posts the filename, size, and declared party — not the bytes — since
        the parse is simulated. A real deployment streams the file to an OCR service and
        returns the same shape.
        """
        filename = str(payload.get("filename", "")).strip()
        if not filename:
            raise HTTPException(400, "filename is required")
        parsed = parse_document(
            filename,
            size_bytes=int(payload.get("size_bytes", 0) or 0),
            declared_party=payload.get("party"),
        )
        return ParsedDocumentOut(
            filename=parsed.filename,
            evidence_type=parsed.evidence_type,
            party=parsed.party,
            verified=parsed.verified,
            quality=parsed.quality,
            extracted=parsed.extracted,
            stages=[{"label": s.label, "ms": s.ms, "detail": s.detail} for s in parsed.stages],
            total_ms=parsed.total_ms,
            summary=parsed.summary(),
            simulated=parsed.simulated,
            evidence=EvidenceIn(**parsed.to_evidence_in()),
        )

    @app.post("/api/disputes")
    def create(intake: DisputeIntake) -> dict[str, str]:
        dispute_id = svc().create_dispute(intake)
        return {"dispute_id": dispute_id}

    @app.post("/api/disputes/{dispute_id}/evidence")
    def add_evidence(dispute_id: str, ev: EvidenceIn) -> dict[str, str]:
        try:
            evidence_id = svc().add_evidence(dispute_id, ev)
        except KeyError:
            raise HTTPException(404, f"dispute {dispute_id} not found") from None
        return {"evidence_id": evidence_id}

    @app.post("/api/disputes/{dispute_id}/adjudicate", response_model=VerdictOut)
    async def adjudicate_endpoint(dispute_id: str) -> VerdictOut:
        try:
            svc().adjudicate_dispute(dispute_id)
        except KeyError:
            raise HTTPException(404, f"dispute {dispute_id} not found") from None
        verdict = svc().get_verdict(dispute_id)
        if verdict is None:
            raise HTTPException(500, "adjudication produced no verdict")
        await hub.broadcast(dispute_id, svc().get_status(dispute_id))
        return VerdictOut(**verdict)

    @app.get("/api/disputes/{dispute_id}", response_model=DisputeStatusOut)
    def status(dispute_id: str) -> DisputeStatusOut:
        try:
            return DisputeStatusOut(**svc().get_status(dispute_id))
        except KeyError:
            raise HTTPException(404, f"dispute {dispute_id} not found") from None

    @app.get("/api/disputes/{dispute_id}/verdict", response_model=VerdictOut)
    def verdict(dispute_id: str) -> VerdictOut:
        result = svc().get_verdict(dispute_id)
        if result is None:
            raise HTTPException(404, "no verdict yet; adjudicate first")
        return VerdictOut(**result)

    @app.get("/api/disputes")
    def list_disputes(limit: int = 50) -> list[dict[str, Any]]:
        return svc().list_disputes(limit)

    @app.websocket("/ws/disputes/{dispute_id}")
    async def ws(websocket: WebSocket, dispute_id: str) -> None:
        await hub.connect(dispute_id, websocket)
        try:
            # Send current status on connect, then hold open for pushes.
            try:
                await websocket.send_json(svc().get_status(dispute_id))
            except KeyError:
                await websocket.send_json({"error": "dispute not found"})
            while True:
                await websocket.receive_text()  # keepalive / client pings
        except WebSocketDisconnect:
            hub.disconnect(dispute_id, websocket)

    return app


class _StatusHub:
    """Minimal pub/sub for live dispute status over WebSocket."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, dispute_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._subscribers.setdefault(dispute_id, set()).add(ws)

    def disconnect(self, dispute_id: str, ws: WebSocket) -> None:
        subs = self._subscribers.get(dispute_id)
        if subs:
            subs.discard(ws)

    async def broadcast(self, dispute_id: str, message: dict[str, Any]) -> None:
        for ws in list(self._subscribers.get(dispute_id, set())):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(dispute_id, ws)


app = create_app()

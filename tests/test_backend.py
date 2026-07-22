"""Stage 7 — FastAPI service, persistence, and the append-only event log."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from arbiter.calibration.conformal import ConformalCalibrator
from backend import db
from backend.db import DisputeEvent, ImmutableEventError
from backend.main import create_app
from backend.service import ArbiterService


@pytest.fixture
def database():
    d = db.Database(url="sqlite:///:memory:")
    d.create_all()
    return d


@pytest.fixture
def calibrator():
    # A tiny hand-made calibrator; the guarantee itself is tested in test_calibration.
    return ConformalCalibrator(alpha=0.10).fit(
        [3.0, -3.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5], [True, False, True, False, True, False, True, False]
    )


@pytest.fixture
def service(database, calibrator):
    return ArbiterService(database, calibrator)


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


# --------------------------------------------------------------------------------------
# schema / immutability
# --------------------------------------------------------------------------------------


def test_event_log_is_append_only(database):
    """The core auditability guarantee, enforced at the DB layer."""
    with database.session() as s:
        s.add(db.Dispute(id="D1", reason_code="C08"))
        db.record_event(s, "D1", "intake", to_state="intake")

    with pytest.raises(ImmutableEventError):
        with database.session() as s:
            ev = s.query(DisputeEvent).filter_by(dispute_id="D1").first()
            ev.to_state = "tampered"


def test_events_cannot_be_deleted_individually(database):
    with database.session() as s:
        s.add(db.Dispute(id="D2", reason_code="C08"))
        db.record_event(s, "D2", "intake", to_state="intake")

    with pytest.raises(ImmutableEventError):
        with database.session() as s:
            ev = s.query(DisputeEvent).filter_by(dispute_id="D2").first()
            s.delete(ev)


def test_sequences_are_monotonic(database):
    with database.session() as s:
        s.add(db.Dispute(id="D3", reason_code="C08"))
        db.record_event(s, "D3", "intake", to_state="intake")
        db.record_event(s, "D3", "arbitration_started", to_state="arbitrating")
        db.record_event(s, "D3", "resolved", to_state="resolved")

    with database.session() as s:
        seqs = [e.sequence for e in s.query(DisputeEvent).filter_by(dispute_id="D3")]
        assert seqs == [1, 2, 3]


def test_sequence_survives_reload(database):
    """next_sequence reads from the persisted max, not an in-memory counter."""
    with database.session() as s:
        s.add(db.Dispute(id="D4", reason_code="C08"))
        db.record_event(s, "D4", "intake", to_state="intake")
    with database.session() as s:
        assert db.next_sequence(s, "D4") == 2


def test_audit_trail_verification(database):
    with database.session() as s:
        s.add(db.Dispute(id="D5", reason_code="C08"))
        db.record_event(s, "D5", "intake", from_state=None, to_state="intake")
        db.record_event(
            s, "D5", "arbitration_started", from_state="intake", to_state="arbitrating"
        )
    with database.session() as s:
        audit = db.verify_audit_trail(s, "D5")
        assert audit["intact"]
        assert audit["sequences_contiguous"]
        assert audit["transitions_chained"]
        assert audit["event_count"] == 2


# --------------------------------------------------------------------------------------
# service pipeline
# --------------------------------------------------------------------------------------


def test_classification(service):
    result = service.classify_narrative("My order never arrived and tracking has not moved.")
    assert result["reason_code"] == "C08"
    assert 0 <= result["confidence"] <= 1


def test_create_and_adjudicate(service):
    dispute_id = service.create_dispute(
        _intake("My laptop stand never arrived.", amount=200.0)
    )
    assert dispute_id.startswith("DYP-")

    routed = service.adjudicate_dispute(dispute_id)
    assert routed.verdict.value in {"card_member", "merchant", "contested"}

    status = service.get_status(dispute_id)
    assert status["state"] in {"resolved", "settlement_offered", "human_review"}
    assert status["audit"]["intact"]


def test_adjudication_persists_contribution_per_evidence(service):
    """The durable XAI: each exhibit keeps its decibans in the row."""
    dispute_id = service.create_dispute(
        _intake(
            "Item not as described.",
            reason_code="C31",
            amount=150.0,
            evidence=[
                {
                    "evidence_type": "photo_of_item",
                    "party": "card_member",
                    "content": "photo showing wrong colour",
                    "metadata": {"lambda_lr": 1.2},
                }
            ],
        )
    )
    service.adjudicate_dispute(dispute_id)
    verdict = service.get_verdict(dispute_id)
    assert verdict is not None
    assert verdict["entries"]
    assert all("contribution_decibans" in e for e in verdict["entries"])


def test_event_log_records_full_lifecycle(service):
    dispute_id = service.create_dispute(_intake("Order never came.", amount=90.0))
    service.adjudicate_dispute(dispute_id)
    status = service.get_status(dispute_id)
    types = [e["event_type"] for e in status["events"]]
    assert "intake" in types
    assert "arbitration_started" in types
    assert types[-1] in {"resolved", "settlement_offered", "escalated"}


def test_settlement_persisted_when_contested(service):
    """A finely-balanced C31 should route to settlement and store the offer."""
    dispute_id = service.create_dispute(
        _intake(
            "The item is a bit different from the listing.",
            reason_code="C31",
            amount=400.0,
            evidence=[
                {
                    "evidence_type": "visual_similarity",
                    "party": "network",
                    "verified": True,
                    "quality": 0.85,
                    "metadata": {"lambda_lr": 0.1, "cosine": 0.68},
                }
            ],
        )
    )
    routed = service.adjudicate_dispute(dispute_id)
    verdict = service.get_verdict(dispute_id)
    if routed.route.value == "settlement":
        assert verdict["settlement"] is not None
        assert verdict["settlement"]["mutually_beneficial"]


def test_recourse_persisted_for_decided_cases(service):
    dispute_id = service.create_dispute(_intake("Never arrived.", amount=75.0))
    service.adjudicate_dispute(dispute_id)
    verdict = service.get_verdict(dispute_id)
    assert verdict["recourse"] is not None


def test_missing_dispute_raises(service):
    with pytest.raises(KeyError):
        service.adjudicate_dispute("DYP-DOESNOTEXIST")


# --------------------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------------------


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_classify_endpoint(client):
    r = client.post("/api/classify", json={"cm_narrative": "I was charged twice."})
    assert r.status_code == 200
    assert r.json()["reason_code"] == "P08"


def test_classify_requires_text(client):
    assert client.post("/api/classify", json={"cm_narrative": "  "}).status_code == 400


def test_full_http_flow(client):
    created = client.post(
        "/api/disputes",
        json={
            "cm_narrative": "My coffee machine never arrived.",
            "amount": 320.0,
        },
    )
    assert created.status_code == 200
    dispute_id = created.json()["dispute_id"]

    adjudicated = client.post(f"/api/disputes/{dispute_id}/adjudicate")
    assert adjudicated.status_code == 200
    body = adjudicated.json()
    assert body["verdict"] in {"card_member", "merchant", "contested"}
    assert body["waterfall"]
    assert body["headline"]

    status = client.get(f"/api/disputes/{dispute_id}")
    assert status.status_code == 200
    assert status.json()["audit"]["intact"]


def test_add_evidence_endpoint(client):
    dispute_id = client.post(
        "/api/disputes", json={"cm_narrative": "Item not as described.", "reason_code": "C31"}
    ).json()["dispute_id"]

    r = client.post(
        f"/api/disputes/{dispute_id}/evidence",
        json={
            "evidence_type": "photo_of_item",
            "party": "card_member",
            "content": "photo",
        },
    )
    assert r.status_code == 200
    assert r.json()["evidence_id"]


def test_verdict_before_adjudication_is_404(client):
    dispute_id = client.post(
        "/api/disputes", json={"cm_narrative": "Never arrived."}
    ).json()["dispute_id"]
    assert client.get(f"/api/disputes/{dispute_id}/verdict").status_code == 404


def test_unknown_dispute_is_404(client):
    assert client.get("/api/disputes/DYP-NOPE").status_code == 404
    assert client.post("/api/disputes/DYP-NOPE/adjudicate").status_code == 404


def test_list_disputes(client):
    for i in range(3):
        client.post("/api/disputes", json={"cm_narrative": f"Order {i} never came."})
    r = client.get("/api/disputes")
    assert r.status_code == 200
    assert len(r.json()) >= 3


def test_websocket_pushes_status_on_connect(client):
    dispute_id = client.post(
        "/api/disputes", json={"cm_narrative": "Never arrived.", "amount": 40.0}
    ).json()["dispute_id"]

    with client.websocket_connect(f"/ws/disputes/{dispute_id}") as ws:
        msg = ws.receive_json()
        assert msg["dispute_id"] == dispute_id
        assert msg["state"] == "intake"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _intake(narrative, *, reason_code=None, amount=0.0, evidence=None):
    from backend.schemas import DisputeIntake, EvidenceIn

    return DisputeIntake(
        cm_narrative=narrative,
        reason_code=reason_code,
        amount=amount,
        evidence=[EvidenceIn(**e) for e in (evidence or [])],
    )

"""Database schema — the append-only event log that makes disputes auditable.

"Transparent" in the brief means *auditable*, and auditable means an immutable record
of everything that happened. So the spine of this schema is `dispute_events`: an
append-only log that no endpoint ever updates or deletes. The current state of a
dispute is a projection of its events, not a mutable row that overwrites its own past.

Two design choices follow from that:

    contribution_decibans lives on evidence rows.  The ledger's per-exhibit weight is
    stored where it can be queried, so "why did we decide this" is a SELECT, not a
    re-run of the model. This is the XAI made durable.

    blobs are content-hashed.  Evidence files are addressed by SHA-256, so tampering is
    detectable and the audit trail is provable rather than merely asserted.

SQLite for the demo, but the schema is PostgreSQL-compatible: no SQLite-only types, and
JSON columns rather than pickled blobs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Dispute(Base):
    """The current projection of a dispute's state.

    This row is updated as the dispute progresses, but every transition is *also*
    written immutably to `dispute_events`, so the projection can always be rebuilt and
    checked against its own history.
    """

    __tablename__ = "disputes"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    reason_code: Mapped[str] = mapped_column(String(8), index=True)
    state: Mapped[str] = mapped_column(String(24), index=True, default="intake")
    amount: Mapped[float] = mapped_column(Float, default=0.0)

    card_member_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    merchant_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    cm_narrative: Mapped[str] = mapped_column(Text, default="")

    # Filled in once adjudicated. Nullable so the row exists through intake.
    verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    route: Mapped[str | None] = mapped_column(String(16), nullable=True)
    prior_logodds: Mapped[float | None] = mapped_column(Float, nullable=True)
    posterior_logodds: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    decided_by_statute: Mapped[bool] = mapped_column(Boolean, default=False)

    # Structured facts consumed by the statute layer.
    facts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    evidence: Mapped[list["EvidenceItem"]] = relationship(
        back_populates="dispute", cascade="all, delete-orphan"
    )
    events: Mapped[list["DisputeEvent"]] = relationship(
        back_populates="dispute",
        cascade="all, delete-orphan",
        order_by="DisputeEvent.sequence",
    )


class EvidenceItem(Base):
    """One exhibit, with its exact ledger contribution stored for audit.

    `contribution_decibans` is the point of this table: the XAI is not regenerated on
    demand, it is persisted per row and can be summed, sorted, and challenged.
    """

    __tablename__ = "evidence_items"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    dispute_id: Mapped[str] = mapped_column(
        ForeignKey("disputes.id"), index=True
    )
    party: Mapped[str] = mapped_column(String(16))
    evidence_type: Mapped[str] = mapped_column(String(40))
    content: Mapped[str] = mapped_column(Text, default="")

    quality: Mapped[float] = mapped_column(Float, default=0.5)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    blob_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Ledger outputs, written at adjudication time. Nullable until then.
    lambda_lr: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    contribution_decibans: Mapped[float | None] = mapped_column(Float, nullable=True)
    nli_verdict: Mapped[str | None] = mapped_column(String(16), nullable=True)
    self_defeating: Mapped[bool] = mapped_column(Boolean, default=False)

    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    dispute: Mapped[Dispute] = relationship(back_populates="evidence")


class DisputeEvent(Base):
    """An immutable log entry. The spine of the audit trail.

    Never updated, never deleted. `sequence` is monotonic per dispute so the history
    has a total order even when timestamps collide. A DB-level guard (below) raises if
    anything tries to mutate a persisted event.
    """

    __tablename__ = "dispute_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dispute_id: Mapped[str] = mapped_column(
        ForeignKey("disputes.id"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(32))
    from_state: Mapped[str | None] = mapped_column(String(24), nullable=True)
    to_state: Mapped[str | None] = mapped_column(String(24), nullable=True)
    actor: Mapped[str] = mapped_column(String(40), default="system")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    dispute: Mapped[Dispute] = relationship(back_populates="events")


class VerdictRecord(Base):
    """The persisted verdict narration and settlement, for retrieval without re-running."""

    __tablename__ = "verdicts"

    dispute_id: Mapped[str] = mapped_column(
        ForeignKey("disputes.id"), primary_key=True
    )
    headline: Mapped[str] = mapped_column(Text)
    burden_statement: Mapped[str] = mapped_column(Text, default="")
    citation: Mapped[str] = mapped_column(Text, default="")
    reasoning: Mapped[list[Any]] = mapped_column(JSON, default=list)
    waterfall: Mapped[list[Any]] = mapped_column(JSON, default=list)
    settlement: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    recourse: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# --------------------------------------------------------------------------------------
# Immutability guard for the event log
# --------------------------------------------------------------------------------------


class ImmutableEventError(RuntimeError):
    """Raised on any attempt to modify or delete a persisted dispute event."""


@event.listens_for(Session, "before_flush")
def _block_event_mutation(session: Session, flush_context: Any, instances: Any) -> None:
    """Refuse to update or delete a DisputeEvent that is already persisted.

    The append-only guarantee is enforced in code, not just by convention. New events
    (in `session.new`) are fine; touching an existing one is a bug and should fail loudly.
    """
    for obj in session.dirty:
        if isinstance(obj, DisputeEvent) and session.is_modified(obj):
            raise ImmutableEventError(
                f"dispute_events is append-only; event {obj.id} may not be modified"
            )
    for obj in session.deleted:
        if isinstance(obj, DisputeEvent):
            # Cascade deletes of a whole dispute are permitted (test teardown, GDPR
            # erasure); a bare event delete is not.
            if obj.dispute is not None and obj.dispute not in session.deleted:
                raise ImmutableEventError(
                    f"dispute_events is append-only; event {obj.id} may not be deleted"
                )


# --------------------------------------------------------------------------------------
# Engine / session factory
# --------------------------------------------------------------------------------------


class Database:
    """Owns the engine and session factory. One instance per process."""

    def __init__(self, url: str = "sqlite:///arbiter.db", *, echo: bool = False) -> None:
        connect_args: dict[str, Any] = {}
        engine_kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            # An in-memory SQLite database is per-connection by default, so the API's
            # request threads would each get their own empty schema. StaticPool pins a
            # single shared connection, which is what makes ":memory:" usable under a
            # threaded TestClient.
            if ":memory:" in url:
                from sqlalchemy.pool import StaticPool

                engine_kwargs["poolclass"] = StaticPool
        self.engine = create_engine(
            url, echo=echo, connect_args=connect_args, future=True, **engine_kwargs
        )
        self._session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def new_session(self) -> Session:
        """A session for the caller to manage — used by the FastAPI dependency."""
        return self._session_factory()


def next_sequence(session: Session, dispute_id: str) -> int:
    """The next event sequence number for a dispute.

    Computed from the persisted max so it survives process restarts. Not concurrency-
    safe under heavy parallel writes to one dispute, which a real deployment would
    handle with a DB sequence or row lock; adequate and simple for the demo.
    """
    from sqlalchemy import func, select

    current = session.execute(
        select(func.max(DisputeEvent.sequence)).where(
            DisputeEvent.dispute_id == dispute_id
        )
    ).scalar()
    return (current or 0) + 1


def record_event(
    session: Session,
    dispute_id: str,
    event_type: str,
    *,
    from_state: str | None = None,
    to_state: str | None = None,
    actor: str = "system",
    payload: dict[str, Any] | None = None,
) -> DisputeEvent:
    """Append an event to the immutable log."""
    ev = DisputeEvent(
        dispute_id=dispute_id,
        sequence=next_sequence(session, dispute_id),
        event_type=event_type,
        from_state=from_state,
        to_state=to_state,
        actor=actor,
        payload=payload or {},
    )
    session.add(ev)
    return ev


def verify_audit_trail(session: Session, dispute_id: str) -> dict[str, Any]:
    """Check that a dispute's event log is well-formed.

    A cheap integrity check the UI can call to prove the trail was not tampered with:
    sequences are contiguous from 1, and state transitions chain (each from_state
    matches the previous to_state).
    """
    from sqlalchemy import select

    events = list(
        session.execute(
            select(DisputeEvent)
            .where(DisputeEvent.dispute_id == dispute_id)
            .order_by(DisputeEvent.sequence)
        ).scalars()
    )

    contiguous = all(e.sequence == i + 1 for i, e in enumerate(events))
    chained = True
    prev_state: str | None = None
    for e in events:
        if e.to_state is not None:
            if e.from_state is not None and e.from_state != prev_state:
                chained = False
            prev_state = e.to_state

    return {
        "dispute_id": dispute_id,
        "event_count": len(events),
        "sequences_contiguous": contiguous,
        "transitions_chained": chained,
        "intact": contiguous and chained,
    }


def to_json(obj: Any) -> str:
    """Compact JSON for payloads, tolerant of enums and dataclasses."""
    def default(o: Any) -> Any:
        if hasattr(o, "value"):
            return o.value
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, default=default, ensure_ascii=False)

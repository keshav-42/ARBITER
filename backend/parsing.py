"""Document parsing — turning an uploaded file into structured evidence.

In production this is where OCR (LayoutLMv3 / Donut) reads a receipt, a shipping label,
or a chat screenshot and extracts the fields that matter: a tracking number, a delivery
address, a policy window, a refund reference. That extraction is the step everyone
assumes takes days — and it is the step that actually takes *seconds* of compute. The
"weeks" in a real dispute is humans waiting on each other, not machines reading paper.

For the demo this parser is SIMULATED: it infers the document kind from the filename and
declared type and emits the structured fields a real parser would, along with a
plausible per-stage latency so the pipeline view can show the work honestly. It is
clearly labelled as simulated wherever it surfaces. Swapping in a real OCR backend means
replacing `parse_document` alone — the evidence it returns is already in the engine's
shape.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from arbiter.core.evidence import EvidenceType, Party

#: Keyword -> (evidence type, filing party, extracted fields, whether verifiable).
#: The parser matches the filename against these to decide what the document is. Order
#: matters: more specific signatures come first, so "return_tracking" resolves to a
#: return (card member) rather than to generic carrier tracking (merchant).
_SIGNATURES: tuple[tuple[tuple[str, ...], EvidenceType, Party, dict[str, Any]], ...] = (
    (
        ("return", "rma"),
        EvidenceType.RETURN_TRACKING,
        Party.CARD_MEMBER,
        {},
    ),
    (
        ("proof of delivery", "delivery confirmation", "pod", "delivered"),
        EvidenceType.DELIVERY_CONFIRMATION,
        Party.MERCHANT,
        {"address_match": True},
    ),
    (
        ("tracking", "carrier", "ups", "fedex", "usps", "dhl", "shipment"),
        EvidenceType.CARRIER_TRACKING,
        Party.MERCHANT,
        {"last_scan": "delivered"},
    ),
    (
        ("signature", "signed"),
        EvidenceType.SIGNATURE_PROOF,
        Party.MERCHANT,
        {},
    ),
    (
        ("refund", "credit note", "credit memo"),
        EvidenceType.REFUND_RECORD,
        Party.MERCHANT,
        {"posted": True},
    ),
    (
        ("receipt", "invoice", "order confirmation"),
        EvidenceType.RECEIPT,
        Party.CARD_MEMBER,
        {},
    ),
    (
        ("photo", "image", "picture", "damaged", "img", "jpg", "jpeg", "png"),
        EvidenceType.PHOTO_OF_ITEM,
        Party.CARD_MEMBER,
        {},
    ),
    (
        ("chat", "conversation", "screenshot", "whatsapp", "messages"),
        EvidenceType.CHAT_LOG,
        Party.CARD_MEMBER,
        {},
    ),
    (
        ("email", "correspondence", "eml"),
        EvidenceType.EMAIL_THREAD,
        Party.CARD_MEMBER,
        {},
    ),
    (
        ("cancel", "cancellation"),
        EvidenceType.CANCELLATION_REQUEST,
        Party.CARD_MEMBER,
        {},
    ),
    (
        ("policy", "terms", "tos"),
        EvidenceType.REFUND_POLICY,
        Party.MERCHANT,
        {},
    ),
    (
        ("statement", "bank"),
        EvidenceType.BANK_STATEMENT,
        Party.CARD_MEMBER,
        {},
    ),
)

#: Machine-verifiable types earn a genuine verification step (and higher authenticity);
#: everything else is "read but unverified".
_VERIFIABLE = {
    EvidenceType.DELIVERY_CONFIRMATION,
    EvidenceType.CARRIER_TRACKING,
    EvidenceType.SIGNATURE_PROOF,
    EvidenceType.RETURN_TRACKING,
    EvidenceType.REFUND_RECORD,
}


@dataclass
class ParseStage:
    """One step of the parse, with a plausible latency for the pipeline view."""

    label: str
    ms: int
    detail: str = ""


@dataclass
class ParsedDocument:
    """The structured result of reading one document."""

    filename: str
    evidence_type: str
    party: str
    verified: bool
    quality: float
    extracted: dict[str, Any]
    stages: list[ParseStage] = field(default_factory=list)
    blob_sha256: str = ""
    simulated: bool = True

    @property
    def total_ms(self) -> int:
        return sum(s.ms for s in self.stages)

    def summary(self) -> str:
        kind = self.evidence_type.replace("_", " ")
        who = self.party.replace("_", " ")
        return f"Read a {kind} filed by the {who}."

    def to_evidence_in(self) -> dict[str, Any]:
        """Shape the engine's EvidenceIn expects."""
        return {
            "evidence_type": self.evidence_type,
            "party": self.party,
            "content": f"{self.filename} (parsed)",
            "quality": self.quality,
            "verified": self.verified,
            "blob_sha256": self.blob_sha256,
            "metadata": {**self.extracted, "source": "document", "simulated": True},
        }


def _classify_filename(name: str) -> tuple[EvidenceType, Party, dict[str, Any]]:
    # Normalise separators so "proof_of_delivery.pdf" and "proof-of-delivery" both match
    # the "proof of delivery" signature.
    low = name.lower().replace("_", " ").replace("-", " ")
    for keywords, etype, party, fields in _SIGNATURES:
        if any(k in low for k in keywords):
            return etype, party, dict(fields)
    # Unrecognised: treat as generic correspondence from the filer's side.
    return EvidenceType.EMAIL_THREAD, Party.CARD_MEMBER, {}


def parse_document(
    filename: str,
    *,
    size_bytes: int = 0,
    declared_party: str | None = None,
    seed: int | None = None,
) -> ParsedDocument:
    """Simulate reading a document into structured evidence.

    Args:
        filename: the uploaded file's name, used to infer the document kind.
        size_bytes: used only to make the OCR latency look plausible.
        declared_party: overrides the inferred filer when the uploader says who they are.
        seed: makes latencies deterministic for tests.
    """
    rng = random.Random(seed if seed is not None else hash(filename) & 0xFFFFFFFF)
    etype, party, fields = _classify_filename(filename)
    if declared_party in ("card_member", "merchant", "network"):
        party = Party(declared_party)

    verifiable = etype in _VERIFIABLE
    blob = hashlib.sha256(f"{filename}:{size_bytes}".encode()).hexdigest()

    # Populate a few realistic extracted fields per kind.
    extracted = dict(fields)
    if etype in (EvidenceType.CARRIER_TRACKING, EvidenceType.DELIVERY_CONFIRMATION):
        extracted.setdefault("tracking_number", f"1Z{rng.randint(10**9, 10**10 - 1)}")
        extracted.setdefault("delivered_on", "day 4")
    if etype is EvidenceType.RETURN_TRACKING:
        extracted.setdefault("return_tracking", f"9400{rng.randint(10**8, 10**9 - 1)}")
        extracted.setdefault("shipped_on", f"day {rng.randint(3, 12)}")
    if etype is EvidenceType.REFUND_RECORD:
        extracted.setdefault("reference", f"CR-{rng.randint(100000, 999999)}")

    # Build the pipeline stages the UI animates.
    read_ms = 400 + min(2600, size_bytes // 800) + rng.randint(120, 500)
    stages = [
        ParseStage(
            "Reading the document",
            read_ms,
            f"Extracted {len(extracted)} field(s) via OCR + layout parsing",
        )
    ]
    quality = rng.uniform(0.45, 0.6)
    verified = False
    if verifiable:
        v_ms = rng.randint(500, 1400)
        stages.append(
            ParseStage(
                "Verifying against the source",
                v_ms,
                "Checked with the carrier / ledger API",
            )
        )
        verified = True
        quality = rng.uniform(0.85, 0.97)

    return ParsedDocument(
        filename=filename,
        evidence_type=etype.value,
        party=party.value,
        verified=verified,
        quality=round(quality, 3),
        extracted=extracted,
        stages=stages,
        blob_sha256=blob,
        simulated=True,
    )

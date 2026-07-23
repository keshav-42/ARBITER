"""Document parsing and the plain-language verdict layer (frontend-facing backend)."""

from __future__ import annotations

from backend.parsing import parse_document
from backend.plain_language import humanize_reasons, scale_position


def test_return_label_is_the_card_members():
    """The ordering fix: a return tracking label is the customer's, not the merchant's."""
    d = parse_document("return_tracking_label.pdf", size_bytes=1000, seed=1)
    assert d.evidence_type == "return_tracking"
    assert d.party == "card_member"


def test_delivery_proof_is_verified_and_merchant():
    d = parse_document("proof_of_delivery.pdf", size_bytes=1000, seed=1)
    assert d.party == "merchant"
    assert d.verified is True
    assert d.quality >= 0.8
    assert d.evidence_type == "delivery_confirmation"


def test_photo_is_unverified_card_member():
    d = parse_document("broken_item_photo.jpg", size_bytes=1000, seed=1)
    assert d.party == "card_member"
    assert d.verified is False
    assert d.evidence_type == "photo_of_item"


def test_parse_has_stages_and_latency():
    d = parse_document("proof_of_delivery.pdf", size_bytes=200000, seed=1)
    assert d.stages
    assert d.total_ms > 0
    # Verifiable docs get a second (verification) stage.
    assert len(d.stages) == 2

    # An unverifiable doc (a photo) has only the reading stage.
    photo = parse_document("item_photo.jpg", size_bytes=200000, seed=1)
    assert len(photo.stages) == 1


def test_parse_is_deterministic_with_seed():
    a = parse_document("receipt.pdf", size_bytes=5000, seed=7)
    b = parse_document("receipt.pdf", size_bytes=5000, seed=7)
    assert a.total_ms == b.total_ms


def test_declared_party_overrides_inference():
    d = parse_document("email.eml", declared_party="merchant", seed=1)
    assert d.party == "merchant"


def test_to_evidence_in_shape():
    d = parse_document("ups_tracking.pdf", size_bytes=1000, seed=1)
    ev = d.to_evidence_in()
    assert ev["evidence_type"] == d.evidence_type
    assert ev["party"] == d.party
    assert ev["metadata"]["source"] == "document"


# --------------------------------------------------------------------------------------
# plain language
# --------------------------------------------------------------------------------------


def _c08_merchant_verdict() -> dict:
    return {
        "verdict": "merchant",
        "decided_by_statute": False,
        "p_card_member": 0.05,
        "calibrated_confidence": 0.9,
        "reasoning": [
            "Delivery confirmation (merchant, verified) contributes 13.0 decibans toward the Merchant.",
        ],
        "entries": [
            {
                "evidence_type": "delivery_confirmation",
                "contribution_decibans": -13.0,
                "quality": 0.95,
                "self_defeating": False,
            },
            {
                "evidence_type": "cm_narrative",
                "contribution_decibans": 0.6,
                "quality": 0.5,
                "self_defeating": False,
            },
        ],
    }


def test_humanize_reasons_are_plain_and_sided():
    reasons = humanize_reasons(_c08_merchant_verdict())
    assert reasons
    # No decibans leak into customer-facing text.
    assert all("deciban" not in r["text"].lower() for r in reasons)
    assert all(r["side"] in ("you", "merchant") for r in reasons)


def test_statute_reason_leads_when_present():
    v = {
        "verdict": "card_member",
        "decided_by_statute": False,
        "p_card_member": 0.95,
        "calibrated_confidence": 0.95,
        "reasoning": [
            "AMEX Code C08 requires valid carrier delivery confirmation. The merchant produced no proof of delivery.",
        ],
        "entries": [],
    }
    reasons = humanize_reasons(v)
    assert "proof of delivery" in reasons[0]["text"].lower()


def test_scale_position_reflects_winner():
    merchant = scale_position(_c08_merchant_verdict())
    assert merchant["position"] < 50  # needle toward merchant
    assert merchant["winner"] == "merchant"
    assert 0 <= merchant["confidence_pct"] <= 100

    cm = scale_position(
        {"verdict": "card_member", "p_card_member": 0.9, "calibrated_confidence": 0.9}
    )
    assert cm["position"] > 50


def test_scale_phrase_words():
    clear = scale_position({"verdict": "merchant", "p_card_member": 0.02, "calibrated_confidence": 0.95})
    close = scale_position({"verdict": "merchant", "p_card_member": 0.45, "calibrated_confidence": 0.55})
    assert clear["phrase"] == "Clear-cut"
    assert close["phrase"] == "Too close to call"

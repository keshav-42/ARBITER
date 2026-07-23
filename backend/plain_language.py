"""Plain-language translation of a verdict for the customer-facing UI.

The engine speaks in decibans and log-odds because that is what makes it auditable. A
Card Member does not care about decibans. This module turns a ledger result into the
three or four things that actually mattered, in words a person understands, plus a
0–100 lean for the scale-of-justice meter. The technical detail stays available under a
"see the math" toggle, driven by the untouched `waterfall`/`entries` fields — this only
adds a friendlier surface, it does not replace the audit trail.
"""

from __future__ import annotations

from typing import Any

# Map exhibit types to how a person would describe them, oriented by who they help.
_EVIDENCE_PHRASE: dict[str, str] = {
    "delivery_confirmation": "confirmed delivery to the address on file",
    "signature_proof": "a signature collected on delivery",
    "carrier_tracking": "carrier tracking for the shipment",
    "usage_log": "records showing the service was used",
    "return_tracking": "tracking showing the item was shipped back",
    "return_receipt": "a receipt for the return shipment",
    "refund_record": "a refund that was issued",
    "credit_note": "a credit note that was issued",
    "photo_of_item": "a photo of the item received",
    "visual_similarity": "an image comparison with the listing",
    "product_description": "the product listing",
    "cancellation_request": "a cancellation request",
    "email_thread": "an email exchange",
    "chat_log": "a support chat",
    "bank_statement": "a bank statement",
    "cm_narrative": "the customer's description",
    "merchant_rebuttal": "the merchant's response",
    "invoice": "the invoice",
    "receipt": "the receipt",
    "order_confirmation": "the order confirmation",
    "auth_log": "the authorisation record",
    "avs_match": "an address-verification check",
    "three_ds_result": "a card authentication check",
}


def _who(verdict: str) -> str:
    return {"card_member": "you", "merchant": "the merchant"}.get(verdict, "neither side")


def humanize_reasons(verdict: dict[str, Any]) -> list[dict[str, str]]:
    """The 3–4 things that decided it, each as a plain sentence with a side.

    Returns dicts of {text, side, strength} where side is 'you' | 'merchant', and
    strength is 'key' | 'supporting' — no numbers. Built from the stored per-exhibit
    contributions so it stays faithful to the actual decision.
    """
    reasons: list[dict[str, str]] = []

    # A statute finding (dispositive OR a strong presumption) is the headline reason —
    # e.g. "the merchant provided no proof of delivery" or "the return was shipped after
    # the policy window". These live at the front of the reasoning list.
    reasoning = verdict.get("reasoning", [])
    if reasoning and _looks_like_rule(reasoning[0]):
        reasons.append(
            {"text": _soften(reasoning[0]), "side": _side_of(verdict), "strength": "key"}
        )

    entries = sorted(
        verdict.get("entries", []),
        key=lambda e: abs(e.get("contribution_decibans", 0)),
        reverse=True,
    )
    for e in entries:
        db = e.get("contribution_decibans", 0)
        if abs(db) < 0.4:
            continue
        etype = e.get("evidence_type", "")
        phrase = _EVIDENCE_PHRASE.get(etype)
        if not phrase:
            continue
        side = "you" if db > 0 else "merchant"
        verified = e.get("quality", 0) >= 0.85
        lead = "There was " if not verified else "We verified "
        text = f"{lead}{phrase}."
        if e.get("self_defeating"):
            text = f"The merchant's own {phrase} actually pointed the other way."
        reasons.append(
            {
                "text": text,
                "side": side,
                "strength": "key" if abs(db) >= 5 else "supporting",
            }
        )
        if len(reasons) >= 4:
            break

    if not reasons:
        reasons.append(
            {
                "text": "Neither side provided decisive evidence, so the rules of who "
                "must prove what decided it.",
                "side": _side_of(verdict),
                "strength": "key",
            }
        )
    return reasons


def _looks_like_rule(line: str) -> str:
    """Whether a reasoning line is a statute finding rather than an evidence sentence.

    Statute rationales reference proof, policy, windows, or the code itself; evidence
    sentences are phrased as "X contributes N decibans toward…".
    """
    low = line.lower()
    if "deciban" in low:
        return False
    markers = (
        "proof of delivery",
        "return window",
        "representment",
        "cancellation window",
        "no proof",
        "produced no",
        "already posted",
        "duplicate",
        "outside",
        "burden",
        "presumption",
        "required",
    )
    return any(m in low for m in markers)


def _soften(line: str) -> str:
    """Trim a technical reasoning line to a plain sentence."""
    # Drop parenthetical citations like "(Chargeback Code Guide — ...)".
    if "(" in line:
        line = line[: line.index("(")].strip()
    return line


def _side_of(verdict: dict[str, Any]) -> str:
    return "you" if verdict.get("verdict") == "card_member" else "merchant"


def scale_position(verdict: dict[str, Any]) -> dict[str, Any]:
    """The scale-of-justice meter: a 0–100 position and a plain confidence phrase.

    0 = fully merchant, 100 = fully customer, 50 = evenly balanced. Uses the calibrated
    probability, so the needle reflects the confidence the system will actually stand
    behind — not the raw, overconfident posterior.
    """
    p_cm = verdict.get("p_card_member", 0.5)
    conf = verdict.get("calibrated_confidence", max(p_cm, 1 - p_cm))
    # Blend toward the calibrated confidence so a 99% raw case shows ~90% after scaling.
    lean = 50 + (p_cm - 0.5) * 100
    # Pull the needle toward the calibrated certainty rather than the raw one.
    if p_cm >= 0.5:
        lean = 50 + (conf - 0.5) * 100
    else:
        lean = 50 - (conf - 0.5) * 100

    if conf >= 0.9:
        phrase = "Clear-cut"
    elif conf >= 0.75:
        phrase = "Fairly confident"
    elif conf >= 0.6:
        phrase = "Leaning, but close"
    else:
        phrase = "Too close to call"

    winner = verdict.get("verdict")
    return {
        "position": round(max(2, min(98, lean)), 1),
        "confidence_pct": round(conf * 100),
        "phrase": phrase,
        "winner": winner,
        "contested": winner == "contested",
    }

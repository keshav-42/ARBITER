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

# How each exhibit reads as a full plain sentence, written from the point of view of
# the side it helps. Two forms: `v` when the record was verified against its source,
# `u` when it was only submitted. Keeping complete sentences (not fragments) is what
# makes the reasoning read clearly instead of "There was a support chat."
_EVIDENCE_SENTENCE: dict[str, dict[str, str]] = {
    "delivery_confirmation": {
        "v": "The carrier confirmed the parcel was delivered to your address.",
        "u": "The merchant submitted a delivery confirmation, though it wasn't independently verified.",
    },
    "signature_proof": {
        "v": "A signature was collected when the parcel was delivered.",
        "u": "The merchant submitted a delivery signature that wasn't independently verified.",
    },
    "carrier_tracking": {
        "v": "Carrier tracking shows the shipment reached its destination.",
        "u": "The merchant provided tracking that couldn't be fully verified with the carrier.",
    },
    "usage_log": {
        "v": "The account records show the service was actually used.",
        "u": "The merchant pointed to usage records for the service.",
    },
    "return_tracking": {
        "v": "Tracking confirms you shipped the item back.",
        "u": "You provided a return shipment that couldn't be fully verified.",
    },
    "return_receipt": {
        "v": "You have a receipt for the return shipment.",
        "u": "You provided a return receipt.",
    },
    "refund_record": {
        "v": "A refund for this charge was issued and has posted.",
        "u": "The merchant referenced a refund, though it hadn't clearly posted.",
    },
    "credit_note": {
        "v": "A credit note covering the charge was issued.",
        "u": "The merchant referenced a credit note.",
    },
    "photo_of_item": {
        "v": "Your photo of the item was consistent with the problem you described.",
        "u": "You submitted a photo of the item you received.",
    },
    "visual_similarity": {
        "v": "An image check compared what arrived against the listing.",
        "u": "An image comparison weighed in on how closely the item matched the listing.",
    },
    "product_description": {
        "v": "The product listing matched what was sent.",
        "u": "The merchant pointed to the product listing.",
    },
    "cancellation_request": {
        "v": "Your cancellation was on record before the charge.",
        "u": "You provided evidence you had cancelled.",
    },
    "email_thread": {
        "v": "The email exchange supports this account.",
        "u": "An email exchange was submitted in support.",
    },
    "chat_log": {
        "v": "The support chat supports this account.",
        "u": "A support chat was submitted in support.",
    },
    "bank_statement": {
        "v": "A bank statement backs up the payment history.",
        "u": "A bank statement was submitted.",
    },
    "merchant_rebuttal": {
        "v": "The merchant's response addressed the claim.",
        "u": "The merchant responded to the claim.",
    },
    "invoice": {"v": "The invoice matched what was agreed.", "u": "An invoice was submitted."},
    "receipt": {"v": "The receipt matched what was agreed.", "u": "A receipt was submitted."},
    "order_confirmation": {
        "v": "The order confirmation showed the agreed price.",
        "u": "An order confirmation was submitted.",
    },
    "auth_log": {
        "v": "The authorisation record supports how the charge was made.",
        "u": "The authorisation record was reviewed.",
    },
    "avs_match": {
        "v": "The billing address matched the one on your account.",
        "u": "An address-verification check was reviewed.",
    },
    "three_ds_result": {
        "v": "The charge was confirmed with card authentication.",
        "u": "A card-authentication result was reviewed.",
    },
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
        # The plain "reasons" list is for evidence that actually mattered. A bare
        # statement of the complaint (cm_narrative / merchant_rebuttal) carries almost
        # no weight on its own and reads as filler, so it is excluded here — the
        # complaint is already the headline of the whole page.
        etype = e.get("evidence_type", "")
        if etype in ("cm_narrative", "merchant_rebuttal") or abs(db) < 0.5:
            continue
        forms = _EVIDENCE_SENTENCE.get(etype)
        if not forms:
            continue
        side = "you" if db > 0 else "merchant"
        verified = e.get("quality", 0) >= 0.85
        text = forms["v"] if verified else forms["u"]
        if e.get("self_defeating"):
            text = "The merchant's own delivery record actually worked against them."
            side = "you"
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


def resolution_line(verdict: dict[str, Any]) -> dict[str, str]:
    """What actually happens now — the outcome in money terms, not a label.

    A customer's real question is "do I get my money back?". This answers it directly.
    """
    winner = verdict.get("verdict")
    has_settlement = verdict.get("settlement") is not None

    if has_settlement or winner == "contested":
        return {
            "outcome": "settled",
            "headline": "We're proposing a fair split",
            "detail": "The evidence is genuinely balanced, so rather than pick a loser we "
            "suggest dividing the amount. You can accept the split or ask a specialist to review.",
        }
    if winner == "card_member":
        return {
            "outcome": "refund",
            "headline": "Your money is being returned",
            "detail": "The charge is reversed and the amount goes back to your account. "
            "Nothing more is needed from you.",
        }
    return {
        "outcome": "upheld",
        "headline": "The charge stands",
        "detail": "The evidence supported the merchant, so the charge remains. If you have "
        "something new, you can challenge this below and we'll review again.",
    }


# Fragments of the raw engine reasoning we rewrite into plain language for the detail view.
def plain_detailed_reasoning(verdict: dict[str, Any]) -> list[str]:
    """Paraphrase the engine's reasoning lines into sentences a person can read.

    The raw lines say things like "Delivery confirmation (merchant, verified)
    contributes 13.0 decibans toward the Merchant." That is precise but unreadable. Here
    we translate each into plain language; the decibans themselves stay visible only on
    the waterfall chart, which is labelled as the technical view.
    """
    out: list[str] = []
    for line in verdict.get("reasoning", []):
        low = line.lower()
        if "deciban" in low:
            # An evidence-weight line: "<Type> (party, verified) contributes N decibans
            # toward the <Side>." Rebuild it plainly, using our sentence bank so the
            # exhibit is named in human terms rather than as its raw type slug.
            side = "you" if "toward the card member" in low else "the merchant"
            raw_name = line.split("(")[0].strip()
            etype = raw_name.lower().replace(" ", "_")
            verified = "verified" in low
            forms = _EVIDENCE_SENTENCE.get(etype)

            # A bare statement of the complaint or rebuttal is not a decisive factor;
            # describe it honestly as such rather than "weighing" for a side.
            if etype in ("cm_narrative", "merchant_rebuttal"):
                out.append(
                    "The written account of the dispute was noted, but on its own it "
                    "carried little weight."
                )
                continue

            if forms:
                sentence = forms["v"] if verified else forms["u"]
                strength = "strongly" if _num_before(line, "deciban") >= 8 else "moderately"
                out.append(f"{sentence} This weighed {strength} in favour of {side}.")
            else:
                strength = "strongly" if _num_before(line, "deciban") >= 8 else "moderately"
                out.append(
                    f"The {raw_name.lower()} weighed {strength} in favour of {side}."
                )
        else:
            # A statute/rule line — already close to plain; just drop the citation.
            out.append(_soften(line))
    return out


def _num_before(line: str, token: str) -> float:
    """Best-effort extraction of the number just before a token, for strength wording."""
    import re

    m = re.search(r"([\d.]+)\s+" + re.escape(token), line)
    try:
        return float(m.group(1)) if m else 0.0
    except ValueError:
        return 0.0

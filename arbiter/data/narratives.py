"""Narrative rendering — the text a Card Member or Merchant actually writes.

Two requirements pull against each other here:

    Variety      a classifier trained on three templates learns the templates. The
                 renderer composes from openers, bodies, details, and closers with
                 independent sampling, so surface forms rarely repeat.

    Faithfulness the text must reflect the scenario's facts. A narrative for
                 `delivered=False` may not mention receiving the item.

Register is varied deliberately: real dispute text ranges from clipped and angry to
long and apologetic, with typos, ALL CAPS, and missing punctuation. A model that only
ever sees clean prose will not survive contact with production input.
"""

from __future__ import annotations

import random
from typing import Any

from arbiter.core.reason_codes import ReasonCode

# --------------------------------------------------------------------------------------
# Fragment banks
# --------------------------------------------------------------------------------------

_OPENERS: tuple[str, ...] = (
    "I am writing to dispute a charge on my account.",
    "I need help with a transaction I do not agree with.",
    "Please review this charge, something has gone wrong.",
    "I would like to formally dispute this transaction.",
    "There is a charge here I should not have to pay.",
    "I have been trying to sort this out with the merchant for weeks.",
    "Requesting a review of the charge below.",
    "",  # some people just start
)

_CLOSERS: tuple[str, ...] = (
    "Please refund this amount.",
    "I would like this reversed as soon as possible.",
    "I have attached everything I have.",
    "Let me know if you need anything else from me.",
    "I have been a member for many years and would appreciate your help.",
    "Thank you for looking into this.",
    "Please advise.",
    "",
)

_FRUSTRATION: tuple[str, ...] = (
    "This has been incredibly frustrating.",
    "I have called them four times with no result.",
    "Nobody at their support line will help me.",
    "I am very disappointed with how this was handled.",
    "This is the worst customer service I have dealt with.",
    "",
    "",
)

#: Body fragments per reason code, conditioned on the scenario facts.
_CM_BODIES: dict[ReasonCode, tuple[str, ...]] = {
    ReasonCode.C08: (
        "I ordered this {item} on {order_date} and it has never arrived.",
        "The {item} I paid for was never delivered to me.",
        "It has been {days} days and I still have not received the {item}.",
        "I paid for a {item} that never showed up at my address.",
        "My order for the {item} never came, and the tracking has not moved.",
    ),
    ReasonCode.C02: (
        "They agreed to refund me for the {item} and the credit never appeared.",
        "I was promised a refund of {amount} and it has not been applied.",
        "The merchant confirmed a refund in writing but nothing has posted.",
        "I returned the {item} and was told a credit was coming. It never came.",
    ),
    ReasonCode.C04: (
        "I returned the {item} and have not been credited.",
        "The {item} went back to them on day {return_day} and I heard nothing since.",
        "I shipped the {item} back within their return period and got no refund.",
        "I sent the {item} back with tracking and they still charged me.",
    ),
    ReasonCode.C05: (
        "I cancelled this order before it shipped and was charged anyway.",
        "I called to cancel the {item} the same day and they billed me regardless.",
        "This order was cancelled but the charge went through.",
    ),
    ReasonCode.C28: (
        "I cancelled this subscription and they keep billing me.",
        "I ended my membership months ago and there is another charge.",
        "This is a recurring charge for a service I already cancelled.",
    ),
    ReasonCode.C31: (
        "The {item} I received is not what was advertised.",
        "What arrived looks nothing like the listing photos.",
        "The {item} is a completely different model from what I ordered.",
        "This is not the {item} shown on their website.",
    ),
    ReasonCode.C32: (
        "The {item} arrived broken.",
        "My {item} was damaged when it came out of the box.",
        "The {item} was defective on arrival and does not work.",
    ),
    ReasonCode.P08: (
        "I have been charged twice for the same {item}.",
        "There are two identical charges of {amount} on my statement.",
        "This transaction appears twice and I only made one purchase.",
    ),
    ReasonCode.P05: (
        "I was quoted one price and charged another.",
        "The charge of {amount} does not match what I agreed to pay.",
        "They billed me more than the confirmation email said.",
    ),
    ReasonCode.P07: (
        "This charge appeared months after I made the purchase.",
        "I authorised this a long time ago and it has only now been billed.",
    ),
    ReasonCode.F29: (
        "I did not make this transaction.",
        "This charge is not mine and I have no idea who made it.",
        "I never authorised this purchase.",
    ),
    ReasonCode.R13: (
        "I asked the merchant for documentation and got no response at all.",
        "The merchant has ignored every request for paperwork.",
    ),
}

_MERCHANT_BODIES: dict[ReasonCode, tuple[str, ...]] = {
    ReasonCode.C08: (
        "Order {order_id} was dispatched and delivered on {delivery_date}.",
        "Our records show the parcel was delivered and the tracking confirms it.",
        "We shipped this order via {carrier} and it was signed for on arrival.",
        "Delivery was completed to the address on file for this account.",
    ),
    ReasonCode.C02: (
        "The refund for order {order_id} was processed on our side.",
        "A credit was issued and should appear within the normal settlement period.",
        "No refund was agreed for this order under our published policy.",
    ),
    ReasonCode.C04: (
        "The return was received outside our published {return_window}-day window.",
        "We have no record of any return being tendered for order {order_id}.",
        "The item was returned late and therefore falls outside our policy.",
    ),
    ReasonCode.C05: (
        "The cancellation request arrived after the order had already shipped.",
        "This order had entered fulfilment before we received any cancellation.",
    ),
    ReasonCode.C28: (
        "The account shows continued activity after the claimed cancellation date.",
        "We have no cancellation on record for this subscription.",
        "Our system shows the plan as active with regular usage.",
    ),
    ReasonCode.C31: (
        "The item shipped matches the listing description exactly.",
        "The product delivered is the one ordered, as shown in our catalogue.",
        "Our listing accurately describes the item that was sent.",
    ),
    ReasonCode.C32: (
        "The item was inspected before dispatch and left our warehouse intact.",
        "The reported damage is consistent with use rather than transit.",
    ),
    ReasonCode.P08: (
        "These are two separate transactions with distinct authorisation codes.",
        "Our terminal log shows two independent purchases, not a duplicate.",
    ),
    ReasonCode.P05: (
        "The final amount includes the service charge disclosed at booking.",
        "The amount billed matches the signed authorisation on file.",
    ),
    ReasonCode.P07: (
        "The charge was submitted within our normal settlement schedule.",
    ),
    ReasonCode.F29: (
        "The transaction carries a successful authentication and matching AVS.",
        "This purchase was made from a device previously used on this account.",
    ),
    ReasonCode.R13: (
        "We have no record of receiving a documentation request.",
    ),
}

_ITEMS: tuple[str, ...] = (
    "laptop stand", "pair of running shoes", "coffee machine", "office chair",
    "wireless headphones", "winter coat", "dining table", "smart watch",
    "camera lens", "bookshelf", "blender", "phone case", "monitor",
    "backpack", "desk lamp", "mattress", "espresso grinder", "tablet",
)

_CARRIERS: tuple[str, ...] = ("UPS", "FedEx", "USPS", "DHL", "Royal Mail")


def _degrade(text: str, rng: random.Random, severity: float) -> str:
    """Introduce realistic input noise.

    Real dispute text is not clean. Roughly proportional to `severity`, this drops
    capitalisation, removes apostrophes, adds filler, or shouts.
    """
    if severity <= 0 or not text:
        return text

    if rng.random() < severity * 0.25:
        text = text.lower()
    if rng.random() < severity * 0.20:
        text = text.replace("'", "")
    if rng.random() < severity * 0.12:
        text = text.upper()
    if rng.random() < severity * 0.15:
        text = text.replace(".", "")
    if rng.random() < severity * 0.18:
        text = text + " " + rng.choice(("please help", "!!", "...", "thanks", "??"))
    return text


def _fill(template: str, ctx: dict[str, Any], rng: random.Random) -> str:
    """Fill a template, tolerating missing keys."""
    values = {
        "item": ctx.get("item", "item"),
        "amount": f"${ctx.get('amount', 100):.2f}",
        "days": ctx.get("days", rng.randint(10, 45)),
        "order_date": ctx.get("order_date", "the 3rd"),
        "delivery_date": ctx.get("delivery_date", "the 9th"),
        "order_id": ctx.get("order_id", "A-00000"),
        "carrier": ctx.get("carrier", rng.choice(_CARRIERS)),
        "return_day": ctx.get("return_day", 7),
        "return_window": ctx.get("return_window", 14),
    }
    try:
        return template.format(**values)
    except KeyError:
        return template


def render_cm_narrative(
    code: ReasonCode,
    ctx: dict[str, Any],
    rng: random.Random,
    *,
    noise: float = 0.4,
) -> str:
    """Compose a Card Member's account of the dispute.

    Args:
        code: reason code the claim falls under.
        ctx: scenario context supplying item, amounts, dates.
        rng: seeded generator, so a corpus is reproducible.
        noise: 0 = clean prose, 1 = heavily degraded input.
    """
    bodies = _CM_BODIES.get(code) or ("I am disputing this charge.",)

    parts = [
        rng.choice(_OPENERS),
        _fill(rng.choice(bodies), ctx, rng),
    ]

    # Occasionally a second sentence of detail, which is where most of the
    # informative signal for a classifier lives.
    if rng.random() < 0.55:
        parts.append(_fill(rng.choice(bodies), ctx, rng))
    if rng.random() < 0.45:
        parts.append(rng.choice(_FRUSTRATION))
    parts.append(rng.choice(_CLOSERS))

    text = " ".join(p for p in parts if p).strip()
    return _degrade(text, rng, noise)


def render_merchant_rebuttal(
    code: ReasonCode,
    ctx: dict[str, Any],
    rng: random.Random,
    *,
    noise: float = 0.1,
) -> str:
    """Compose the Merchant's response.

    Merchant text is materially cleaner than Card Member text — it is usually
    template-generated by a support desk — so the default noise is much lower.
    """
    bodies = _MERCHANT_BODIES.get(code) or ("We consider this charge valid.",)

    parts = [_fill(rng.choice(bodies), ctx, rng)]
    if rng.random() < 0.5:
        parts.append(_fill(rng.choice(bodies), ctx, rng))
    if rng.random() < 0.35:
        parts.append(
            rng.choice(
                (
                    "We consider this charge valid and request it be upheld.",
                    "Supporting documentation is attached.",
                    "Please see the attached records.",
                    "We are unable to offer a refund in this instance.",
                )
            )
        )

    text = " ".join(p for p in parts if p).strip()
    return _degrade(text, rng, noise)


def random_item(rng: random.Random) -> str:
    return rng.choice(_ITEMS)


def random_carrier(rng: random.Random) -> str:
    return rng.choice(_CARRIERS)

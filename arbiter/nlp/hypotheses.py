"""Hypothesis templates — turning each exhibit into a testable claim.

The keystone of the NLP layer. Rather than training a bespoke classifier per evidence
type, every check is reframed as **natural-language inference**:

    premise    = what the exhibit actually says
    hypothesis = the proposition the exhibit is offered to prove
             ->  P(entail), P(neutral), P(contradict)
             ->  lambda = log( P(entail) / P(contradict) )

That ratio is exactly the log-likelihood ratio the ledger already consumes, so the
mathematics arrives for free. It works zero-shot on day one and improves with
fine-tuning.

Two properties make this better than a per-type classifier:

    Falsifiable   an exhibit can *contradict* its own hypothesis. A delivery
                  confirmation naming the wrong address does not merely fail to help
                  the merchant — it actively helps the Card Member, and the sign of
                  lambda flips accordingly.

    Contextual    the same exhibit proves different things under different codes. A
                  delivery confirmation under C08 must prove receipt; under C31 it
                  proves nothing about whether the item matched its description. The
                  hypothesis is keyed on (reason_code, evidence_type).

Hypotheses are written in the *filer's* favour by convention, so entailment always
supports whoever submitted the exhibit. `ledger.compute_lambda` applies the sign flip
for merchant-filed evidence; do not double-apply it here.
"""

from __future__ import annotations

from dataclasses import dataclass

from arbiter.core.evidence import EvidenceType
from arbiter.core.reason_codes import ReasonCode


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A proposition an exhibit is offered to prove.

    Attributes:
        text: the hypothesis, phrased in the filer's favour.
        rationale: why this is the right question for this exhibit under this code.
            Surfaced in the audit trail so a reviewer can challenge the framing
            itself, not just the model's answer.
        dispositive_if_contradicted: when True, a strong contradiction is a serious
            finding — the exhibit undermines the case it was filed to support.
    """

    text: str
    rationale: str = ""
    dispositive_if_contradicted: bool = False


#: Hypotheses keyed by (reason code, evidence type). Consulted before the
#: type-only fallback, because context changes what an exhibit must prove.
_KEYED: dict[tuple[ReasonCode, EvidenceType], Hypothesis] = {
    # --- C08: goods not received. The merchant must prove receipt. ---
    (ReasonCode.C08, EvidenceType.DELIVERY_CONFIRMATION): Hypothesis(
        text=(
            "The parcel was delivered to the cardholder's own address and the "
            "cardholder received it."
        ),
        rationale=(
            "Under C08 the merchant must prove receipt, not merely dispatch. A "
            "confirmation naming another address contradicts this and helps the "
            "Card Member."
        ),
        dispositive_if_contradicted=True,
    ),
    (ReasonCode.C08, EvidenceType.CARRIER_TRACKING): Hypothesis(
        text="The carrier record shows the shipment completed delivery to the cardholder.",
        rationale=(
            "Tracking that halts in transit is evidence the goods never arrived, so "
            "the same exhibit can support either party."
        ),
    ),
    (ReasonCode.C08, EvidenceType.SIGNATURE_PROOF): Hypothesis(
        text="The cardholder or their authorised agent signed for the delivery.",
        rationale="A signature by an unrelated third party does not establish receipt.",
    ),
    (ReasonCode.C08, EvidenceType.USAGE_LOG): Hypothesis(
        text="The cardholder accessed and used the service that was purchased.",
        rationale=(
            "For digital goods, sustained account usage is the equivalent of physical "
            "delivery."
        ),
    ),
    (ReasonCode.C08, EvidenceType.CM_NARRATIVE): Hypothesis(
        text="The cardholder never received the goods or services that were paid for.",
        rationale="The claim itself, stated as a proposition the record can test.",
    ),
    # --- C02: credit not processed. ---
    (ReasonCode.C02, EvidenceType.REFUND_RECORD): Hypothesis(
        text="A refund for this transaction was issued and has settled to the account.",
        rationale=(
            "A refund that was initiated but never settled does not answer the claim; "
            "the Card Member is still out of pocket."
        ),
        dispositive_if_contradicted=True,
    ),
    (ReasonCode.C02, EvidenceType.CREDIT_NOTE): Hypothesis(
        text="A credit note covering the disputed amount was issued to the cardholder.",
    ),
    (ReasonCode.C02, EvidenceType.EMAIL_THREAD): Hypothesis(
        text="The merchant agreed in writing to refund the cardholder.",
        rationale=(
            "Establishes that a promise existed at all — without one there is nothing "
            "for the merchant to have breached."
        ),
    ),
    (ReasonCode.C02, EvidenceType.CHAT_LOG): Hypothesis(
        text="The merchant agreed to refund the cardholder during this conversation.",
    ),
    # --- C04: returned goods. The Card Member must prove a timely return. ---
    (ReasonCode.C04, EvidenceType.RETURN_TRACKING): Hypothesis(
        text=(
            "The cardholder shipped the merchandise back to the merchant within the "
            "merchant's published return window."
        ),
        rationale=(
            "Both halves matter: that a return happened, and that it was timely. The "
            "statute layer decides the date arithmetic; this establishes the return "
            "itself."
        ),
    ),
    (ReasonCode.C04, EvidenceType.RETURN_RECEIPT): Hypothesis(
        text="The cardholder tendered the merchandise to a carrier for return.",
    ),
    (ReasonCode.C04, EvidenceType.REFUND_POLICY): Hypothesis(
        text="The merchant's published policy permitted the return the cardholder made.",
        rationale=(
            "Filed by the merchant to show the return was out of policy, so entailment "
            "here supports the merchant."
        ),
    ),
    # --- C05 / C28: cancellation. ---
    (ReasonCode.C05, EvidenceType.CANCELLATION_REQUEST): Hypothesis(
        text="The cardholder cancelled the order before the merchant dispatched it.",
    ),
    (ReasonCode.C28, EvidenceType.CANCELLATION_REQUEST): Hypothesis(
        text=(
            "The cardholder cancelled the recurring subscription before the disputed "
            "billing cycle began."
        ),
        rationale=(
            "The cycle boundary is what matters; a cancellation after the renewal date "
            "does not make the renewal charge improper."
        ),
    ),
    (ReasonCode.C28, EvidenceType.USAGE_LOG): Hypothesis(
        text="The cardholder continued using the service after the claimed cancellation date.",
        rationale=(
            "Continued use is the strongest available rebuttal to a cancellation claim."
        ),
    ),
    (ReasonCode.C28, EvidenceType.CRM_LOG): Hypothesis(
        text="The merchant's records show the subscription remained active and uncancelled.",
    ),
    # --- C31 / C32: description and condition. ---
    (ReasonCode.C31, EvidenceType.PHOTO_OF_ITEM): Hypothesis(
        text=(
            "The item the cardholder received differs materially from the item that "
            "was advertised."
        ),
        rationale=(
            "C31 turns on material difference, not on preference. A photograph of an "
            "item that matches the listing contradicts the claim."
        ),
    ),
    (ReasonCode.C31, EvidenceType.PRODUCT_DESCRIPTION): Hypothesis(
        text="The listing accurately describes the item that was shipped to the cardholder.",
    ),
    (ReasonCode.C31, EvidenceType.CATALOG_IMAGE): Hypothesis(
        text="The catalogue image accurately depicts the item that was shipped.",
    ),
    (ReasonCode.C32, EvidenceType.PHOTO_OF_ITEM): Hypothesis(
        text="The item arrived damaged or defective rather than being damaged through use.",
        rationale=(
            "Distinguishing transit damage from wear is the whole question in C32, and "
            "how soon it was reported is the main signal."
        ),
    ),
    # --- P-family: processing errors. ---
    (ReasonCode.P08, EvidenceType.AUTH_LOG): Hypothesis(
        text=(
            "The authorisation log shows two separate purchases rather than one "
            "transaction submitted twice."
        ),
        rationale=(
            "Filed by the merchant to rebut a duplicate claim, so entailment supports "
            "the merchant."
        ),
    ),
    (ReasonCode.P05, EvidenceType.ORDER_CONFIRMATION): Hypothesis(
        text="The amount the cardholder agreed to differs from the amount that was billed.",
    ),
    (ReasonCode.P05, EvidenceType.INVOICE): Hypothesis(
        text="The invoiced amount matches the amount the cardholder authorised.",
    ),
    # --- F-family: authorisation disputes. ---
    (ReasonCode.F29, EvidenceType.THREE_DS_RESULT): Hypothesis(
        text="The cardholder personally authenticated this transaction.",
        rationale=(
            "A successful 3-D Secure challenge is direct evidence the cardholder was "
            "present at authorisation."
        ),
    ),
    (ReasonCode.F29, EvidenceType.AVS_MATCH): Hypothesis(
        text="The billing address supplied at purchase matched the cardholder's address on file.",
    ),
    (ReasonCode.F29, EvidenceType.DEVICE_FINGERPRINT): Hypothesis(
        text="The purchase was made from a device the cardholder has used before.",
    ),
    (ReasonCode.F24, EvidenceType.THREE_DS_RESULT): Hypothesis(
        text="The cardholder personally authenticated this transaction.",
    ),
}


#: Fallbacks used when no (code, type) entry exists. Phrased generically but still in
#: the filer's favour.
_BY_TYPE: dict[EvidenceType, Hypothesis] = {
    EvidenceType.DELIVERY_CONFIRMATION: Hypothesis(
        text="The goods were delivered to the cardholder."
    ),
    EvidenceType.CARRIER_TRACKING: Hypothesis(
        text="The carrier record shows the shipment reached the cardholder."
    ),
    EvidenceType.SIGNATURE_PROOF: Hypothesis(
        text="The cardholder signed for the delivery."
    ),
    EvidenceType.RETURN_TRACKING: Hypothesis(
        text="The cardholder returned the merchandise to the merchant."
    ),
    EvidenceType.RETURN_RECEIPT: Hypothesis(
        text="The cardholder tendered the merchandise for return."
    ),
    EvidenceType.REFUND_RECORD: Hypothesis(
        text="A refund was issued to the cardholder for this transaction."
    ),
    EvidenceType.CREDIT_NOTE: Hypothesis(
        text="A credit was issued to the cardholder."
    ),
    EvidenceType.CANCELLATION_REQUEST: Hypothesis(
        text="The cardholder requested cancellation before the merchant performed."
    ),
    EvidenceType.USAGE_LOG: Hypothesis(
        text="The cardholder used the service that was purchased."
    ),
    EvidenceType.CRM_LOG: Hypothesis(
        text="The merchant's records support the validity of this charge."
    ),
    EvidenceType.PHOTO_OF_ITEM: Hypothesis(
        text="The item received was defective or differed from what was advertised."
    ),
    EvidenceType.PRODUCT_DESCRIPTION: Hypothesis(
        text="The listing accurately describes the item that was shipped."
    ),
    EvidenceType.CATALOG_IMAGE: Hypothesis(
        text="The catalogue image depicts the item that was shipped."
    ),
    EvidenceType.CHAT_LOG: Hypothesis(
        text="This conversation supports the cardholder's account of events."
    ),
    EvidenceType.EMAIL_THREAD: Hypothesis(
        text="This correspondence supports the cardholder's account of events."
    ),
    EvidenceType.CM_NARRATIVE: Hypothesis(
        text="The cardholder's account of this dispute is accurate."
    ),
    EvidenceType.MERCHANT_REBUTTAL: Hypothesis(
        text="The merchant's account of this dispute is accurate."
    ),
    EvidenceType.INVOICE: Hypothesis(
        text="The invoice matches what the cardholder agreed to pay."
    ),
    EvidenceType.RECEIPT: Hypothesis(
        text="The receipt matches what the cardholder agreed to pay."
    ),
    EvidenceType.ORDER_CONFIRMATION: Hypothesis(
        text="The order confirmation matches what the cardholder agreed to pay."
    ),
    EvidenceType.BANK_STATEMENT: Hypothesis(
        text="The statement shows the cardholder paid for this transaction by other means."
    ),
    EvidenceType.AUTH_LOG: Hypothesis(
        text="The authorisation record supports the validity of this charge."
    ),
    EvidenceType.AVS_MATCH: Hypothesis(
        text="The address supplied matched the cardholder's address on file."
    ),
    EvidenceType.CVV_MATCH: Hypothesis(
        text="The security code supplied matched the card on file."
    ),
    EvidenceType.THREE_DS_RESULT: Hypothesis(
        text="The cardholder authenticated this transaction."
    ),
    EvidenceType.DEVICE_FINGERPRINT: Hypothesis(
        text="The purchase came from a device associated with the cardholder."
    ),
    EvidenceType.REFUND_POLICY: Hypothesis(
        text="The merchant's published policy governs this transaction."
    ),
    EvidenceType.CANCELLATION_POLICY: Hypothesis(
        text="The merchant's published cancellation policy governs this transaction."
    ),
    EvidenceType.TERMS_OF_SERVICE: Hypothesis(
        text="The merchant's published terms govern this transaction."
    ),
    EvidenceType.SHIPPING_LABEL: Hypothesis(
        text="The merchant dispatched the merchandise to the cardholder."
    ),
}


def hypothesis_for(
    code: ReasonCode, etype: EvidenceType
) -> Hypothesis | None:
    """The proposition an exhibit is offered to prove under a given reason code.

    Resolution order: the (code, type) entry, then the type-only fallback, then None.

    Returning None is meaningful — it marks exhibit types whose value is computed
    rather than read (visual similarity, duplicate matching, policy-window checks).
    Those already carry an explicit `lambda_lr` and must not be routed through NLI.
    """
    keyed = _KEYED.get((code, etype))
    if keyed is not None:
        return keyed
    return _BY_TYPE.get(etype)


def supported_pairs() -> tuple[tuple[ReasonCode, EvidenceType], ...]:
    """Every (code, type) pair with a purpose-written hypothesis."""
    return tuple(_KEYED)


def has_specific_hypothesis(code: ReasonCode, etype: EvidenceType) -> bool:
    """True when a purpose-written hypothesis exists, rather than the generic fallback."""
    return (code, etype) in _KEYED

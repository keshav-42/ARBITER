"""Corpus generator — scenario facts become noisy, partial evidence bundles.

The pipeline for one case:

    scenario  ->  world facts  ->  exhibits  ->  narratives  ->  DisputeCase
                       |
                       +-------------------->  ground truth label

Evidence is a *lossy observation* of the world, never a transcription of it. Three
noise processes model why real dispute files are hard:

    withholding   a party fails to file evidence that exists and would help them.
                  Small merchants especially do not know what to submit.

    corruption    an exhibit exists but cannot be verified — expired tracking link,
                  unreadable screenshot, a receipt photographed at an angle.

    misleading    an exhibit that points the wrong way. A courier marks a parcel
                  delivered when it was left at a neighbour's; the record is
                  authentic and wrong.

Without these, a model trained on the corpus learns "verified delivery confirmation
implies merchant wins" and collapses the moment it meets a real case. With them, the
ambiguous scenarios stay genuinely ambiguous — which is the point, because that is
what the conformal layer in Stage 5 has to learn to abstain on.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from arbiter.core.evidence import Evidence, EvidenceType, Party
from arbiter.core.ledger import DisputeCase
from arbiter.core.reason_codes import ReasonCode
from arbiter.core.reputation import PRIOR_ALPHA, PRIOR_BETA, ReputationState
from arbiter.data.narratives import (
    random_carrier,
    random_item,
    render_cm_narrative,
    render_merchant_rebuttal,
)
from arbiter.data.scenarios import GroundTruth, Scenario, all_scenarios


@dataclass(slots=True)
class CorpusConfig:
    """Knobs controlling corpus realism.

    Defaults are tuned so that clear-cut scenarios resolve correctly most of the time
    while ambiguous ones stay near the decision boundary — the distribution a
    calibrated system needs in order to learn when to abstain.
    """

    seed: int = 42

    #: Probability a party fails to file an exhibit that exists and helps them.
    #: Scaled up by scenario difficulty.
    withhold_rate: float = 0.18

    #: Probability an exhibit that exists cannot be verified.
    corruption_rate: float = 0.15

    #: Probability a merchant exhibit is authentic but points the wrong way.
    misleading_rate: float = 0.08

    #: Fraction of merchants that simply never respond.
    silent_merchant_rate: float = 0.06

    #: Range of disputed amounts, in dollars.
    amount_min: float = 12.0
    amount_max: float = 2400.0

    #: Fraction of parties carrying a prior dispute history.
    reputation_rate: float = 0.55

    #: Text degradation for Card Member narratives.
    cm_noise: float = 0.4


@dataclass(slots=True)
class GeneratedCase:
    """A synthetic dispute with its causal ground truth.

    `truth` is what actually happened, which is deliberately *not* the ledger's
    prediction. Stage 5 calibrates against the gap between them, and Stage 9 audits it.
    """

    case_id: str
    scenario_id: str
    reason_code: ReasonCode
    truth: GroundTruth
    difficulty: float
    amount: float
    case: DisputeCase
    cm_narrative: str
    merchant_rebuttal: str
    #: Facts the renderer used, retained so failures can be traced back to the world.
    world: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Flat JSON-serialisable record for the corpus file and model training."""
        return {
            "case_id": self.case_id,
            "scenario_id": self.scenario_id,
            "reason_code": self.reason_code.value,
            "truth": self.truth.value,
            "difficulty": self.difficulty,
            "amount": round(self.amount, 2),
            "cm_narrative": self.cm_narrative,
            "merchant_rebuttal": self.merchant_rebuttal,
            "evidence": [
                {
                    "type": e.etype.value,
                    "party": e.party.value,
                    "content": e.content,
                    "quality": round(e.quality, 3),
                    "verified": e.verified,
                    "metadata": e.metadata,
                }
                for e in self.case.evidence
            ],
            "facts": {
                "merchant_response_days": self.case.merchant_response_days,
                "return_shipped_day": self.case.return_shipped_day,
                "return_window_days": self.case.return_window_days,
                "cancel_day": self.case.cancel_day,
                "cancel_window_days": self.case.cancel_window_days,
                "days_since_transaction": self.case.days_since_transaction,
                "submission_delay_days": self.case.submission_delay_days,
                "duplicate_confirmed": self.case.duplicate_confirmed,
                "refund_already_posted": self.case.refund_already_posted,
            },
            "world": self.world,
        }


# --------------------------------------------------------------------------------------
# Evidence construction
# --------------------------------------------------------------------------------------


def _quality(rng: random.Random, cfg: CorpusConfig, *, verifiable: bool) -> tuple[float, bool]:
    """Sample an exhibit's authenticity and whether it verified.

    Verifiable exhibits usually verify; when they do not, quality falls too, because
    an exhibit that fails a provenance check looks worse than one never checked.
    """
    if not verifiable:
        return (rng.uniform(0.35, 0.65), False)
    if rng.random() < cfg.corruption_rate:
        return (rng.uniform(0.25, 0.5), False)
    return (rng.uniform(0.82, 0.98), True)


def _maybe(rng: random.Random, cfg: CorpusConfig, difficulty: float) -> bool:
    """True when a party actually files an exhibit they hold.

    Harder scenarios see more withholding — a proxy for the disorganised records that
    make real disputes hard.
    """
    rate = min(0.6, cfg.withhold_rate * (1.0 + difficulty))
    return rng.random() >= rate


def _build_evidence(
    scenario: Scenario,
    world: dict[str, Any],
    rng: random.Random,
    cfg: CorpusConfig,
) -> list[Evidence]:
    """Derive an evidence bundle from world facts."""
    ev: list[Evidence] = []
    f = scenario.facts
    d = scenario.difficulty
    code = scenario.reason_code

    def add(
        etype: EvidenceType,
        party: Party,
        *,
        verifiable: bool = False,
        content: str = "",
        **meta: Any,
    ) -> None:
        q, verified = _quality(rng, cfg, verifiable=verifiable)
        ev.append(
            Evidence(
                etype=etype,
                party=party,
                content=content,
                quality=q,
                verified=verified,
                metadata=meta,
            )
        )

    # --- delivery / fulfilment ---
    if f.get("tracking_exists") and _maybe(rng, cfg, d):
        add(
            EvidenceType.CARRIER_TRACKING,
            Party.MERCHANT,
            verifiable=True,
            content=f"{world['carrier']} tracking {world['tracking_no']}",
            last_scan=f.get("last_scan", "delivered" if f.get("delivered") else "in_transit"),
        )

    if f.get("delivered") and _maybe(rng, cfg, d):
        # A misleading delivery record: authentic, but the parcel never reached the
        # Card Member. This is the noise that makes porch-theft cases hard.
        misleading = rng.random() < cfg.misleading_rate
        add(
            EvidenceType.DELIVERY_CONFIRMATION,
            Party.MERCHANT,
            verifiable=True,
            content=f"Delivered {world['delivery_date']}",
            address_match=bool(f.get("address_match", True)) and not misleading,
            left_unattended=bool(f.get("left_unattended", False)),
            # A partial shipment is genuinely delivered, so the carrier record is
            # truthful yet does not answer the claim: the balance never arrived.
            # Proof of *a* delivery is weak proof of *complete* delivery.
            lambda_lr=0.4 if f.get("partial") else None,
            partial=bool(f.get("partial", False)),
        )

    if f.get("signature") and _maybe(rng, cfg, d):
        add(
            EvidenceType.SIGNATURE_PROOF,
            Party.MERCHANT,
            verifiable=True,
            content=f"Signed by {world['signer']}",
        )

    if f.get("usage_log") or f.get("usage_after_cancel"):
        if _maybe(rng, cfg, d):
            add(
                EvidenceType.USAGE_LOG,
                Party.MERCHANT,
                verifiable=True,
                content="Account activity log",
                sessions=rng.randint(4, 90),
            )

    # --- returns ---
    if f.get("return_tracked") and _maybe(rng, cfg, d):
        add(
            EvidenceType.RETURN_TRACKING,
            Party.CARD_MEMBER,
            verifiable=True,
            content=f"Return {world['return_tracking_no']}",
            shipped_day=f.get("return_day"),
        )
        if rng.random() < 0.6:
            add(
                EvidenceType.RETURN_RECEIPT,
                Party.CARD_MEMBER,
                content="Post office receipt",
            )

    # --- credits ---
    if f.get("refund_issued") and _maybe(rng, cfg, d):
        add(
            EvidenceType.REFUND_RECORD,
            Party.MERCHANT,
            verifiable=True,
            content=f"Credit {world['refund_ref']}",
            posted=bool(f.get("refund_posted", False)),
        )

    if f.get("written_promise") and _maybe(rng, cfg, d):
        add(
            EvidenceType.EMAIL_THREAD,
            Party.CARD_MEMBER,
            content="Merchant email confirming refund",
        )

    # --- policies ---
    if f.get("return_window") and rng.random() < 0.75:
        add(
            EvidenceType.REFUND_POLICY,
            Party.MERCHANT,
            content=f"{f['return_window']}-day return policy",
            window_days=f["return_window"],
        )
    if f.get("cancel_window") and rng.random() < 0.7:
        add(
            EvidenceType.CANCELLATION_POLICY,
            Party.MERCHANT,
            content=f"{f['cancel_window']}-day cancellation policy",
            window_days=f["cancel_window"],
        )

    # --- cancellations ---
    if f.get("cancelled") and _maybe(rng, cfg, d):
        add(
            EvidenceType.CANCELLATION_REQUEST,
            Party.CARD_MEMBER,
            content=f"Cancellation via {f.get('cancel_channel', 'account portal')}",
            channel=f.get("cancel_channel", "portal"),
        )
    if code is ReasonCode.C28 and rng.random() < 0.6:
        add(EvidenceType.CRM_LOG, Party.MERCHANT, verifiable=True, content="Subscription log")

    # --- item condition ---
    if f.get("photo_evidence") or f.get("damaged") or code in (ReasonCode.C31, ReasonCode.C32):
        if _maybe(rng, cfg, d):
            # Damage reported weeks after delivery is consistent with use rather than
            # transit, so the same photograph carries the opposite implication.
            delay = int(f.get("days_before_report", 0))  # type: ignore[arg-type]
            add(
                EvidenceType.PHOTO_OF_ITEM,
                Party.CARD_MEMBER,
                content="Photograph of item received",
                days_before_report=delay or None,
                lambda_lr=(-0.9 if delay > 21 else 1.1) if f.get("damaged") else None,
            )
    if "visual_similarity" in f:
        sim = float(f["visual_similarity"])  # type: ignore[arg-type]
        sim = max(0.0, min(1.0, sim + rng.gauss(0, 0.06)))
        # Map similarity to a likelihood ratio: identical items favour the merchant,
        # divergent ones the Card Member. Centred at 0.7, the empirical point where
        # "same product, different batch" gives way to "different product".
        add(
            EvidenceType.VISUAL_SIMILARITY,
            Party.NETWORK,
            verifiable=True,
            content=f"Catalogue similarity {sim:.2f}",
            lambda_lr=(0.70 - sim) * 5.0,
            cosine=round(sim, 3),
        )
        if rng.random() < 0.7:
            add(
                EvidenceType.CATALOG_IMAGE,
                Party.MERCHANT,
                content="Listing photograph",
            )
        if rng.random() < 0.75:
            add(
                EvidenceType.PRODUCT_DESCRIPTION,
                Party.MERCHANT,
                content="Listing description",
                matches=bool(f.get("description_match", True)),
            )

    # --- authorisation ---
    if code in (ReasonCode.F24, ReasonCode.F29):
        if f.get("three_ds") and _maybe(rng, cfg, d):
            add(
                EvidenceType.THREE_DS_RESULT,
                Party.NETWORK,
                verifiable=True,
                content="3-D Secure authenticated",
                authenticated=True,
            )
        if rng.random() < 0.8:
            add(
                EvidenceType.AVS_MATCH,
                Party.NETWORK,
                verifiable=True,
                content="AVS result",
                matched=bool(f.get("avs_match", False)),
                lambda_lr=-0.9 if f.get("avs_match") else 0.9,
            )
        if f.get("device_match") and rng.random() < 0.7:
            add(
                EvidenceType.DEVICE_FINGERPRINT,
                Party.NETWORK,
                verifiable=True,
                content="Known device",
                lambda_lr=-1.1,
            )

    # --- processing ---
    if code in (ReasonCode.P08, ReasonCode.P05, ReasonCode.P07, ReasonCode.P03):
        if rng.random() < 0.85:
            add(EvidenceType.AUTH_LOG, Party.NETWORK, verifiable=True, content="Authorisation log")
        if rng.random() < 0.6:
            add(EvidenceType.INVOICE, Party.MERCHANT, content="Merchant invoice")
    if f.get("amount_mismatch") and rng.random() < 0.7:
        add(
            EvidenceType.ORDER_CONFIRMATION,
            Party.CARD_MEMBER,
            content="Order confirmation showing quoted price",
            lambda_lr=0.5 if not f.get("disclosed_surcharge") else 0.15,
        )

    # --- correspondence, always plausible ---
    if rng.random() < 0.45:
        add(EvidenceType.CHAT_LOG, Party.CARD_MEMBER, content="Support chat transcript")

    return ev


def _build_world(scenario: Scenario, rng: random.Random, amount: float) -> dict[str, Any]:
    """Concrete details the narratives and exhibits reference."""
    order_date = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    return {
        "item": random_item(rng),
        "carrier": random_carrier(rng),
        "tracking_no": f"1Z{rng.randint(10**9, 10**10 - 1)}",
        "return_tracking_no": f"9400{rng.randint(10**8, 10**9 - 1)}",
        "refund_ref": f"CR-{rng.randint(100000, 999999)}",
        "order_id": f"A-{rng.randint(10000, 99999)}",
        "signer": rng.choice(("J. MARTIN", "RECEPTION", "K. OSEI", "M. TAN", "resident")),
        "order_date": order_date.strftime("%d %b"),
        "delivery_date": (order_date + timedelta(days=rng.randint(2, 9))).strftime("%d %b"),
        "amount": amount,
        "days": rng.randint(12, 60),
        "return_day": scenario.facts.get("return_day", rng.randint(3, 12)),
        "return_window": scenario.facts.get("return_window", 14),
    }


def _reputation(
    rng: random.Random, cfg: CorpusConfig, party: Party
) -> ReputationState | None:
    """Sample a dispute history for a party, or None for a party with no record."""
    if rng.random() > cfg.reputation_rate:
        return None
    n = rng.randint(3, 120)
    # Most parties are fine; a minority are consistently problematic.
    loss_rate = rng.betavariate(2, 6) if rng.random() < 0.8 else rng.betavariate(6, 2)
    losses = round(n * loss_rate)
    return ReputationState(
        party_id=f"{party.value}-{rng.randint(1000, 9999)}",
        party_type=party,
        alpha=PRIOR_ALPHA + losses,
        beta=PRIOR_BETA + (n - losses),
    )


def generate_case(
    scenario: Scenario,
    rng: random.Random,
    cfg: CorpusConfig,
    *,
    index: int = 0,
) -> GeneratedCase:
    """Generate one dispute from a scenario."""
    amount = round(rng.uniform(cfg.amount_min, cfg.amount_max), 2)
    world = _build_world(scenario, rng, amount)
    f = scenario.facts

    evidence = _build_evidence(scenario, world, rng, cfg)

    # Merchant responsiveness: the scenario may force silence, otherwise sample it.
    responds = bool(f.get("merchant_responds", True))
    if responds and rng.random() < cfg.silent_merchant_rate:
        responds = False

    if responds:
        response_days = rng.randint(2, 18)
        rebuttal = render_merchant_rebuttal(scenario.reason_code, world, rng)
        evidence.append(
            Evidence(
                etype=EvidenceType.MERCHANT_REBUTTAL,
                party=Party.MERCHANT,
                content=rebuttal,
                quality=rng.uniform(0.5, 0.65),
            )
        )
    else:
        # Past the window, which the statute layer treats as a default.
        response_days = rng.randint(25, 60)
        rebuttal = ""
        # A silent merchant files nothing at all.
        evidence = [e for e in evidence if e.party is not Party.MERCHANT]

    narrative = render_cm_narrative(
        scenario.reason_code, world, rng, noise=cfg.cm_noise
    )
    evidence.insert(
        0,
        Evidence(
            etype=EvidenceType.CM_NARRATIVE,
            party=Party.CARD_MEMBER,
            content=narrative,
            quality=rng.uniform(0.4, 0.6),
        ),
    )

    case = DisputeCase(
        reason_code=scenario.reason_code,
        evidence=evidence,
        amount=amount,
        merchant_reputation=_reputation(rng, cfg, Party.MERCHANT),
        card_member_reputation=_reputation(rng, cfg, Party.CARD_MEMBER),
        merchant_response_days=response_days,
        return_shipped_day=f.get("return_day"),  # type: ignore[arg-type]
        return_window_days=f.get("return_window"),  # type: ignore[arg-type]
        cancel_day=f.get("cancel_day"),  # type: ignore[arg-type]
        cancel_window_days=f.get("cancel_window"),  # type: ignore[arg-type]
        # The Card Member's filing delay, which is independent of how late the
        # merchant submitted the charge.
        days_since_transaction=rng.randint(5, 90),
        submission_delay_days=(
            int(f["submission_delay_days"]) if "submission_delay_days" in f else None  # type: ignore[arg-type]
        ),
        duplicate_confirmed=bool(f.get("duplicate_confirmed", False)),
        refund_already_posted=bool(f.get("refund_posted", False)),
    )

    return GeneratedCase(
        case_id=f"{scenario.scenario_id}#{index:05d}",
        scenario_id=scenario.scenario_id,
        reason_code=scenario.reason_code,
        truth=scenario.truth,
        difficulty=scenario.difficulty,
        amount=amount,
        case=case,
        cm_narrative=narrative,
        merchant_rebuttal=rebuttal,
        world=world,
    )


def generate_corpus(
    n: int = 5000,
    cfg: CorpusConfig | None = None,
    scenarios: Sequence[Scenario] | None = None,
) -> Iterator[GeneratedCase]:
    """Generate `n` disputes, sampling scenarios by weight.

    Yields lazily so large corpora stream to disk without being held in memory.
    """
    cfg = cfg or CorpusConfig()
    pool = tuple(scenarios) if scenarios is not None else all_scenarios()
    if not pool:
        raise ValueError("no scenarios available")

    rng = random.Random(cfg.seed)
    weights = [s.weight for s in pool]

    for i in range(n):
        scenario = rng.choices(pool, weights=weights, k=1)[0]
        yield generate_case(scenario, rng, cfg, index=i)


def write_corpus(
    path: str | Path,
    n: int = 5000,
    cfg: CorpusConfig | None = None,
) -> dict[str, Any]:
    """Write a corpus as JSON Lines and return summary statistics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    truths: dict[str, int] = {}
    written = 0

    with path.open("w", encoding="utf-8") as fh:
        for gen in generate_corpus(n, cfg):
            fh.write(json.dumps(gen.to_record(), ensure_ascii=False) + "\n")
            counts[gen.reason_code.value] = counts.get(gen.reason_code.value, 0) + 1
            truths[gen.truth.value] = truths.get(gen.truth.value, 0) + 1
            written += 1

    return {
        "path": str(path),
        "cases": written,
        "by_reason_code": dict(sorted(counts.items())),
        "by_truth": dict(sorted(truths.items())),
    }

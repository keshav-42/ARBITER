"""Evidence verification by natural-language inference.

Runs each exhibit against the proposition it is offered to prove and writes the
resulting entailment probabilities into `Evidence.metadata`, where
`ledger.compute_lambda` already knows how to read them. Nothing in the ledger changes.

Backends are pluggable:

    StubNLIBackend          deterministic, offline, no weights. Used by the test
                            suite and as the fallback when a model is unavailable, so
                            the demo never depends on a download succeeding.

    TransformerNLIBackend   a real MNLI cross-encoder, GPU-batched and cached.

The stub is not a toy: it reads the structured metadata the parsers already extract
(address_match, partial, authenticated, posted) and produces calibrated-looking
probabilities from it. That keeps the 121 existing tests fast and offline while still
exercising the full code path.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from arbiter.core.evidence import Evidence, EvidenceType, Party, evidence_polarity
from arbiter.core.ledger import DisputeCase
from arbiter.core.reason_codes import ReasonCode
from arbiter.nlp.hypotheses import Hypothesis, hypothesis_for

#: Exhibit types whose lambda is computed rather than inferred. Routing these through
#: NLI would overwrite a precise measurement with a language model's guess.
_COMPUTED_TYPES: frozenset[EvidenceType] = frozenset(
    {
        EvidenceType.VISUAL_SIMILARITY,
        EvidenceType.DUPLICATE_TXN_MATCH,
        EvidenceType.POLICY_WINDOW_CHECK,
    }
)

#: Clamp on the derived log-likelihood ratio, in nats. An MNLI head can emit
#: probabilities like 0.9997 vs 0.0001, which is a ratio of ~9 nats — far more
#: confidence than a short documentary snippet can justify. The ledger clamps again
#: downstream; this keeps the stored value honest too.
MAX_NLI_LAMBDA: float = 2.2

#: Floor applied to entail/contradict before taking the ratio.
_EPS: float = 1e-4


@dataclass(frozen=True, slots=True)
class NLIResult:
    """Entailment probabilities for one (premise, hypothesis) pair."""

    entail: float
    neutral: float
    contradict: float
    hypothesis: str = ""
    backend: str = ""

    @property
    def lambda_lr(self) -> float:
        """Log-likelihood ratio in the *filer's* favour, clamped.

        `ledger.compute_lambda` applies the party sign flip, so this stays unsigned
        with respect to who filed the exhibit.
        """
        raw = math.log((self.entail + _EPS) / (self.contradict + _EPS))
        return max(-MAX_NLI_LAMBDA, min(MAX_NLI_LAMBDA, raw))

    @property
    def verdict(self) -> str:
        """Which of the three labels dominates — used in the audit trail."""
        top = max(self.entail, self.neutral, self.contradict)
        if top == self.entail:
            return "entailment"
        if top == self.contradict:
            return "contradiction"
        return "neutral"

    @property
    def is_contradicted(self) -> bool:
        """True when the exhibit actively undermines what it was filed to prove."""
        return self.contradict > self.entail and self.contradict > 0.5


class NLIBackend(Protocol):
    """Anything that can score (premise, hypothesis) pairs."""

    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        """Score a batch. Must return one result per input pair, in order."""
        ...


# --------------------------------------------------------------------------------------
# Stub backend — deterministic, offline
# --------------------------------------------------------------------------------------


def _softmax3(a: float, b: float, c: float) -> tuple[float, float, float]:
    m = max(a, b, c)
    ea, eb, ec = math.exp(a - m), math.exp(b - m), math.exp(c - m)
    total = ea + eb + ec
    return (ea / total, eb / total, ec / total)


class StubNLIBackend:
    """Deterministic stand-in for a real MNLI model.

    Scores from structured metadata and lexical cues rather than semantics. It exists
    so tests stay offline and fast, and so a failed model download degrades the demo
    instead of breaking it.

    Determinism comes from hashing the pair, which also gives stable per-exhibit
    jitter — without it every exhibit of a type would score identically and the
    ledger's correlation damping would never be exercised.
    """

    name = "stub"

    #: Phrases suggesting the premise supports its hypothesis. Kept specific: bare
    #: tokens like "no" or "not" match almost any sentence and turn the stub into a
    #: contradiction generator.
    _POSITIVE = (
        "delivered", "signed by", "was received", "refund issued", "credit issued",
        "confirmed", "authentication succeeded", "address matches", "returned",
        "cancellation", "dispatched", "has settled", "recorded sessions",
    )
    #: Phrases suggesting the premise undermines it.
    _NEGATIVE = (
        "never", "does not match", "did not match", "not completed",
        "still in transit", "left unattended", "only part of",
        "has not yet settled", "undelivered", "counterfeit",
        "damaged", "defective", "days after delivery",
    )

    def __init__(self, *, jitter: float = 0.35) -> None:
        self.jitter = jitter

    def _signal(self, premise: str, hypothesis: str) -> float:
        """Crude lexical alignment in roughly [-1, 1]."""
        p = premise.lower()
        pos = sum(1 for t in self._POSITIVE if t in p)
        neg = sum(1 for t in self._NEGATIVE if t in p)
        if pos == neg == 0:
            return 0.0
        return (pos - neg) / max(1.0, float(pos + neg))

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        out: list[NLIResult] = []
        for premise, hypothesis in pairs:
            signal = self._signal(premise, hypothesis)

            # Stable pseudo-random offset in [-jitter, +jitter].
            digest = hashlib.sha256(f"{premise}|{hypothesis}".encode()).digest()
            noise = (digest[0] / 255.0 * 2.0 - 1.0) * self.jitter

            logit_e = 1.4 * signal + noise
            logit_n = 0.35
            logit_c = -1.4 * signal - noise

            e, n, c = _softmax3(logit_e, logit_n, logit_c)
            out.append(
                NLIResult(
                    entail=e, neutral=n, contradict=c,
                    hypothesis=hypothesis, backend=self.name,
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Transformer backend
# --------------------------------------------------------------------------------------

#: Default cross-encoder. Trained on MNLI + FEVER + ANLI, so it handles the
#: adversarial, partially-contradictory text real dispute records contain.
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class TransformerNLIBackend:
    """A real MNLI cross-encoder.

    Loads lazily so that importing this module never touches the network — merely
    constructing the object is safe in a test collection pass.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.name = f"transformer:{model_name}"
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._label_index: dict[str, int] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        model.to(self._device)
        self._model = model

        # Label order varies between MNLI checkpoints, so read it from the config
        # rather than assuming the conventional [contradiction, neutral, entailment].
        id2label = getattr(model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            self._label_index[str(label).lower()[:4]] = int(idx)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        if not pairs:
            return []

        self._ensure_loaded()
        import torch

        results: list[NLIResult] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self._device)

            with torch.no_grad():
                probs = torch.softmax(self._model(**encoded).logits, dim=-1)

            i_e = self._label_index.get("enta", 2)
            i_n = self._label_index.get("neut", 1)
            i_c = self._label_index.get("cont", 0)

            for row, (_, hyp) in zip(probs.tolist(), chunk, strict=True):
                results.append(
                    NLIResult(
                        entail=row[i_e],
                        neutral=row[i_n],
                        contradict=row[i_c],
                        hypothesis=hyp,
                        backend=self.name,
                    )
                )
        return results


# --------------------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------------------


def _premise(evidence: Evidence) -> str:
    """Render an exhibit as a premise sentence.

    Structured metadata is folded into the text because the parsers already extracted
    facts an NLI model cannot recover from a bare label — whether an address matched,
    whether a shipment was partial, whether authentication succeeded.
    """
    parts: list[str] = []
    label = evidence.etype.value.replace("_", " ")
    parts.append(f"{label.capitalize()}.")

    if evidence.content:
        parts.append(evidence.content.strip())

    meta = evidence.metadata
    if meta.get("address_match") is False:
        parts.append("The delivery address does not match the cardholder's address.")
    elif meta.get("address_match") is True:
        parts.append("The delivery address matches the cardholder's address.")

    if meta.get("partial"):
        parts.append("Only part of the order was included in this shipment.")
    if meta.get("left_unattended"):
        parts.append("The parcel was left unattended without a signature.")
    if meta.get("last_scan") == "in_transit":
        parts.append("The final carrier scan shows the parcel still in transit.")
    if meta.get("authenticated") is True:
        parts.append("Authentication succeeded.")
    elif meta.get("authenticated") is False:
        parts.append("Authentication was not completed.")
    if meta.get("matched") is False:
        parts.append("The supplied details did not match the record.")
    if meta.get("posted") is False:
        parts.append("The credit has not yet settled to the account.")
    if meta.get("sessions"):
        parts.append(f"The account shows {meta['sessions']} recorded sessions.")
    if meta.get("days_before_report"):
        parts.append(
            f"The problem was reported {meta['days_before_report']} days after delivery."
        )

    # Provenance is deliberately NOT written into the premise. It is already applied
    # as `Evidence.effective_quality`, which multiplies the contribution downstream;
    # restating it here would double-count it. It also poisoned the lexical stub,
    # whose negation cues fired on the boilerplate "could not be independently
    # verified" and scored contradiction on 4 exhibits in 5.
    return " ".join(parts)


@dataclass(slots=True)
class EvidenceVerifier:
    """Scores exhibits against their hypotheses and annotates them for the ledger.

    Attributes:
        backend: the NLI implementation. Defaults to the offline stub.
        cache: content-hash keyed memo, so re-adjudicating a case costs nothing.
    """

    backend: NLIBackend = field(default_factory=StubNLIBackend)
    cache: dict[str, NLIResult] = field(default_factory=dict)

    def _cache_key(self, premise: str, hypothesis: str) -> str:
        return hashlib.sha256(f"{premise}||{hypothesis}".encode()).hexdigest()

    def verify(
        self, evidence: Sequence[Evidence], code: ReasonCode
    ) -> list[Evidence]:
        """Annotate exhibits in place with entailment probabilities.

        Exhibits are mutated rather than copied because `Evidence.metadata` is the
        agreed channel to the ledger, and the caller's `DisputeCase` should reflect
        the verification.

        Skipped:
          - types whose lambda is computed (visual similarity, duplicate match)
          - exhibits that already carry an explicit `lambda_lr`
          - types with no hypothesis at all
        """
        pending: list[tuple[int, str, Hypothesis]] = []

        for i, ev in enumerate(evidence):
            if ev.etype in _COMPUTED_TYPES:
                continue
            if ev.metadata.get("lambda_lr") is not None:
                continue
            hyp = hypothesis_for(code, ev.etype)
            if hyp is None:
                continue
            pending.append((i, _premise(ev), hyp))

        if not pending:
            return list(evidence)

        # Only score what is not already cached.
        to_score: list[tuple[str, str]] = []
        positions: list[int] = []
        for slot, (_, premise, hyp) in enumerate(pending):
            key = self._cache_key(premise, hyp.text)
            if key not in self.cache:
                to_score.append((premise, hyp.text))
                positions.append(slot)

        if to_score:
            for slot, result in zip(positions, self.backend.score(to_score), strict=True):
                _, premise, hyp = pending[slot]
                self.cache[self._cache_key(premise, hyp.text)] = result

        for idx, premise, hyp in pending:
            result = self.cache[self._cache_key(premise, hyp.text)]
            ev = evidence[idx]
            ev.metadata["entail"] = result.entail
            ev.metadata["neutral"] = result.neutral
            ev.metadata["contradict"] = result.contradict
            ev.metadata["nli_verdict"] = result.verdict
            ev.metadata["nli_hypothesis"] = hyp.text
            ev.metadata["nli_backend"] = result.backend
            if hyp.rationale:
                ev.metadata["nli_rationale"] = hyp.rationale
            # A contradicted exhibit that was meant to be decisive is worth flagging
            # for the verdict card: the party's own evidence works against them.
            if hyp.dispositive_if_contradicted and result.is_contradicted:
                ev.metadata["self_defeating"] = True

        return list(evidence)


def verify_case(
    case: DisputeCase, verifier: EvidenceVerifier | None = None
) -> DisputeCase:
    """Annotate every exhibit in a case, returning the same case for chaining."""
    v = verifier or EvidenceVerifier()
    v.verify(case.evidence, case.reason_code)
    return case


def summarise(evidence: Sequence[Evidence]) -> dict[str, Any]:
    """Aggregate NLI outcomes for logging and the eval harness."""
    counts: dict[str, int] = {"entailment": 0, "neutral": 0, "contradiction": 0}
    scored = self_defeating = 0
    for ev in evidence:
        verdict = ev.metadata.get("nli_verdict")
        if verdict in counts:
            counts[verdict] += 1
            scored += 1
        if ev.metadata.get("self_defeating"):
            self_defeating += 1
    return {
        "scored": scored,
        "unscored": len(evidence) - scored,
        "by_verdict": counts,
        "self_defeating": self_defeating,
    }

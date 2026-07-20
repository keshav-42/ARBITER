"""Reason-code classification — free-text grievance to AMEX code.

The intake step. A Card Member writes "my sofa never turned up and they won't answer
the phone"; the system must decide that this is C08, because everything downstream —
the burden of proof, the statute rules, the evidence weights — is keyed on the code.

Two backends, same interface:

    KeywordClassifier     weak-supervision labelling functions. Offline, instant,
                          interpretable, and the fallback when no model is present.

    TransformerClassifier a fine-tuned sequence classifier. Stage 4 ships the
                          training entry point; the corpus from Stage 3 supplies
                          labelled narratives.

The keyword classifier is not a placeholder to be embarrassed about. Snorkel-style
labelling functions are how the training labels get bootstrapped in the first place,
and a transparent rule that fires on "never arrived" is easier to defend to a judge
than a 0.83 logit. It also sets the floor the neural model has to beat.

Misclassification is recoverable rather than fatal: the intake step returns a
distribution, and a low-confidence result routes to a disambiguation prompt instead of
silently adjudicating under the wrong statute.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from arbiter.core.reason_codes import REASON_CODES, ReasonCode

#: Below this top-1 probability the intake should ask rather than assume.
AMBIGUITY_THRESHOLD: float = 0.45


@dataclass(frozen=True, slots=True)
class CodePrediction:
    """A reason-code prediction with its full distribution."""

    code: ReasonCode
    confidence: float
    distribution: dict[ReasonCode, float] = field(default_factory=dict)
    matched_cues: tuple[str, ...] = ()
    backend: str = ""

    @property
    def is_ambiguous(self) -> bool:
        """True when the top code is not clearly ahead."""
        return self.confidence < AMBIGUITY_THRESHOLD

    def top(self, k: int = 3) -> tuple[tuple[ReasonCode, float], ...]:
        """The k most likely codes, for a disambiguation prompt."""
        ranked = sorted(self.distribution.items(), key=lambda kv: kv[1], reverse=True)
        return tuple(ranked[:k])

    def explain(self) -> str:
        """Why this code was chosen — shown at intake so the member can correct it."""
        spec = REASON_CODES.get(self.code)
        title = spec.title if spec else self.code.value
        if self.matched_cues:
            cues = ", ".join(f'"{c}"' for c in self.matched_cues[:4])
            return f"Classified as {self.code.value} ({title}) from {cues}."
        return f"Classified as {self.code.value} ({title})."


class CodeClassifier(Protocol):
    """Anything that maps a narrative to a reason-code distribution."""

    name: str

    def predict(self, text: str) -> CodePrediction:
        ...


# --------------------------------------------------------------------------------------
# Keyword / weak-supervision backend
# --------------------------------------------------------------------------------------

#: Labelling functions per reason code: (regex, weight). Weights are log-space votes,
#: so several weak cues can outvote one strong one. Patterns are deliberately written
#: against how people actually type — contractions optional, tense loose.
_CUES: dict[ReasonCode, tuple[tuple[str, float], ...]] = {
    # C08 cues must name a *shipment* arriving, not merely something being absent.
    # An earlier version matched bare "never ...ed" and "no ...", which fired on
    # "I heard nothing since" (C04) and "got no response" (R13) and pulled 60-70%
    # of several other codes into C08.
    ReasonCode.C08: (
        (r"(never|not|n.?t) (arrived|delivered|shipped|dispatched|turned up|showed up)", 3.2),
        (r"never (receiv|got)\w* (it|the|my)", 3.0),
        (r"(has|have) not (arrived|been delivered|received)", 2.8),
        (r"(did ?n.?t|never) (receive|get) (the|my|any)", 2.8),
        (r"still (waiting|has ?n.?t (arrived|come|shown))", 2.4),
        (r"no sign of (the|my|it)", 2.4),
        (r"(missing|lost) (order|parcel|package|shipment|delivery)", 2.4),
        (r"tracking (has ?n.?t|never) (moved|updated)", 2.2),
        (r"only (part|some) of (my|the) order", 2.0),
        (r"nothing (ever )?(came|arrived|showed up)", 2.6),
        (r"(order|parcel|package|item) never (came|arrived|showed)", 3.2),
    ),
    # An explicit refund *promise* is what separates C02 from C04. "I returned it and
    # was told a credit was coming" is both a return and a broken promise; the promise
    # is the gravamen, so these cues are weighted above C04's return cues.
    ReasonCode.C02: (
        (r"refund (was )?(promised|agreed|confirmed)", 3.6),
        (r"(credit|refund) (never|has ?n.?t|not) (appear|post|arriv|came|show)", 3.4),
        (r"promised (me )?a refund", 3.6),
        (r"(told|informed) (me )?(that )?a? ?(credit|refund) (was|is) (coming|due|on its way)", 3.8),
        (r"confirmed a refund", 3.6),
        (r"waiting for (my )?(refund|credit)", 2.4),
        (r"agreed to refund", 3.4),
        (r"credit never (came|appeared|arrived|posted)", 3.6),
        (r"(never|has ?n.?t) been applied", 3.0),
    ),
    ReasonCode.C04: (
        (r"(returned|sent (it )?back|shipped (it )?back|posted (it )?back)", 3.0),
        (r"went back to them", 3.2),
        # Multi-word product names ("phone case", "laptop stand") sit between the
        # verb and "back", so \w+ alone does not span them.
        (r"(sent|shipped|posted|took) the [\w\s]{1,25}back", 3.2),
        (r"return (tracking|receipt|label|period|window)", 2.6),
        (r"(gave|took) it back", 2.0),
        (r"refused (the )?delivery", 2.4),
        (r"back (to them |to the merchant )?on day \d+", 3.4),
    ),
    ReasonCode.C05: (
        (r"(this |the |my )?order was cancel(l)?ed", 3.4),
        (r"cancel(l)?ed (the |my )?(order|booking|reservation)", 3.0),
        (r"cancel(l)?ed before (it )?(ship|dispatch|deliver)", 3.2),
        (r"(called|rang|phoned) to cancel", 2.8),
        (r"charge went through", 2.4),
    ),
    ReasonCode.C28: (
        (r"(cancel(l)?ed|ended|stopped) (my |the )?(subscription|membership|plan)", 3.0),
        (r"(still|keep|continue) (billing|charging) me", 2.8),
        (r"recurring (charge|billing|payment)", 2.4),
        (r"auto.?renew", 2.0),
    ),
    ReasonCode.C31: (
        (r"not (what|as) (was )?(advertised|described|shown|pictured)", 3.2),
        (r"(different|nothing like) (from |to )?(what|the) (i ordered|listing|photo|advert)", 3.0),
        (r"(wrong|different) (model|item|product|colour|color|size)", 2.4),
        (r"(fake|counterfeit|knock.?off|replica)", 2.6),
        (r"looks nothing like", 2.8),
    ),
    ReasonCode.C32: (
        (r"(arrived|came|was) (broken|damaged|cracked|smashed|faulty|defective)", 3.2),
        (r"(does ?n.?t|not) work", 2.2),
        (r"damaged (in transit|on arrival|when)", 2.6),
        (r"(defect|fault)", 1.6),
    ),
    ReasonCode.P08: (
        (r"charged twice", 3.4),
        (r"(two|2|double|duplicate) (identical |same )?(charge|transaction|payment)", 3.0),
        (r"appears twice", 2.8),
        (r"double.?billed", 3.0),
    ),
    ReasonCode.P05: (
        (r"(charged|billed) (me )?(more|extra|too much)", 3.0),
        (r"(amount|price|charge) (is )?(wrong|incorrect|different)", 2.6),
        (r"quoted (one|a different) price", 2.8),
        (r"does ?n.?t match (what|the) (i agreed|quote|confirmation)", 2.4),
    ),
    ReasonCode.P07: (
        (r"(months|weeks) (later|after)", 2.4),
        (r"charge appeared (months|long)", 2.8),
        (r"only now been (billed|charged)", 2.6),
    ),
    ReasonCode.F29: (
        (r"(did ?n.?t|not|never) (make|authorise|authorize|recognise|recognize) (this|that|the)", 3.2),
        (r"(not|isn.?t) (my|mine) (charge|transaction|purchase)", 2.8),
        (r"unauthorised|unauthorized", 3.0),
        (r"no idea (who|what) (made|this)", 2.4),
    ),
    ReasonCode.C14: (
        (r"paid (by|with) (cash|another card|different card|bank transfer)", 3.0),
        (r"already paid", 2.2),
    ),
    ReasonCode.R13: (
        (r"(no|never (got|had) (a )?)(response|reply|answer)", 3.2),
        (r"got no response", 3.4),
        (r"(ignored|has ignored|not respond\w*)", 3.0),
        (r"(asked|requested).{0,30}(documentation|paperwork|records)", 3.4),
        (r"every request", 2.8),
    ),
}

#: Compiled once at import.
_COMPILED: dict[ReasonCode, tuple[tuple[re.Pattern[str], float], ...]] = {
    code: tuple((re.compile(p, re.IGNORECASE), w) for p, w in cues)
    for code, cues in _CUES.items()
}

#: Weak prior over codes reflecting real chargeback volume, used to break ties when
#: no cue fires. Without it, a narrative with no recognisable phrasing would produce
#: a uniform distribution and pick a code arbitrarily.
_VOLUME_PRIOR: dict[ReasonCode, float] = {
    ReasonCode.C08: 0.30,
    ReasonCode.C02: 0.16,
    ReasonCode.C31: 0.12,
    ReasonCode.P08: 0.09,
    ReasonCode.C04: 0.09,
    ReasonCode.F29: 0.07,
    ReasonCode.C28: 0.06,
    ReasonCode.C32: 0.05,
    ReasonCode.C05: 0.03,
    ReasonCode.P05: 0.02,
    ReasonCode.P07: 0.01,
}


class KeywordClassifier:
    """Weak-supervision classifier over labelling functions.

    Cue weights are summed per code in log-space, added to a volume prior, and
    softmaxed. Interpretable end to end: `matched_cues` records exactly which patterns
    fired, so the intake screen can show its reasoning and let the member correct it.
    """

    name = "keyword"

    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def predict(self, text: str) -> CodePrediction:
        scores: dict[ReasonCode, float] = {}
        matched: list[str] = []

        for code in REASON_CODES:
            prior = _VOLUME_PRIOR.get(code, 0.004)
            scores[code] = math.log(prior)

        for code, patterns in _COMPILED.items():
            if code not in scores:
                continue
            for pattern, weight in patterns:
                found = pattern.search(text)
                if found:
                    scores[code] += weight
                    matched.append(found.group(0).strip())

        # Softmax over the votes.
        top_score = max(scores.values())
        exps = {
            c: math.exp((s - top_score) / max(self.temperature, 1e-6))
            for c, s in scores.items()
        }
        total = sum(exps.values())
        dist = {c: v / total for c, v in exps.items()}

        best = max(dist.items(), key=lambda kv: kv[1])
        return CodePrediction(
            code=best[0],
            confidence=best[1],
            distribution=dist,
            matched_cues=tuple(dict.fromkeys(matched)),
            backend=self.name,
        )


# --------------------------------------------------------------------------------------
# Transformer backend
# --------------------------------------------------------------------------------------

#: Legal-BERT is pretrained on regulatory and contractual text, the register of the
#: chargeback guide and merchant terms. DeBERTa-v3 is the stronger general encoder.
#: Either is a reasonable fine-tuning base; the choice is set at construction.
DEFAULT_CLASSIFIER_MODEL = "microsoft/deberta-v3-base"
LEGAL_CLASSIFIER_MODEL = "nlpaueb/legal-bert-base-uncased"


class TransformerClassifier:
    """Fine-tuned sequence classifier over reason codes.

    Loads lazily, so constructing it never touches the network. Falls back to the
    keyword classifier if weights are unavailable, because a demo that cannot classify
    is worse than one that classifies transparently.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_CLASSIFIER_MODEL,
        *,
        device: str | None = None,
        max_length: int = 256,
        fallback: CodeClassifier | None = None,
    ) -> None:
        self.model_path = model_path
        self.name = f"transformer:{model_path}"
        self.max_length = max_length
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._labels: list[ReasonCode] = []
        self._fallback = fallback or KeywordClassifier()
        self._unavailable = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._unavailable:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            model.eval()
            model.to(self._device)
            self._model = model

            id2label = getattr(model.config, "id2label", {}) or {}
            labels: list[ReasonCode] = []
            for idx in sorted(id2label, key=lambda k: int(k)):
                try:
                    labels.append(ReasonCode(str(id2label[idx]).upper()))
                except ValueError:
                    labels = []
                    break
            self._labels = labels
            if not labels:
                # An un-finetuned checkpoint has LABEL_0/LABEL_1 heads and cannot
                # produce reason codes; the keyword path is strictly better.
                self._unavailable = True
        except Exception:
            self._unavailable = True

    def predict(self, text: str) -> CodePrediction:
        self._ensure_loaded()
        if self._unavailable or self._model is None:
            return self._fallback.predict(text)

        import torch

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self._device)

        with torch.no_grad():
            probs = torch.softmax(self._model(**encoded).logits, dim=-1)[0].tolist()

        dist = {code: probs[i] for i, code in enumerate(self._labels)}
        best = max(dist.items(), key=lambda kv: kv[1])
        return CodePrediction(
            code=best[0],
            confidence=best[1],
            distribution=dist,
            backend=self.name,
        )


def classify(text: str, classifier: CodeClassifier | None = None) -> CodePrediction:
    """Classify a narrative, defaulting to the offline keyword backend."""
    return (classifier or KeywordClassifier()).predict(text)


def evaluate_classifier(
    samples: Sequence[tuple[str, ReasonCode]],
    classifier: CodeClassifier | None = None,
) -> dict[str, Any]:
    """Score a classifier over labelled narratives.

    Reports top-1, top-3, and the ambiguity rate — how often intake should ask rather
    than assume. A high ambiguity rate is not a failure; routing an unclear complaint
    to a clarifying question is the correct behaviour.
    """
    clf = classifier or KeywordClassifier()
    top1 = top3 = ambiguous = 0
    per_code: dict[str, list[int]] = {}

    for text, expected in samples:
        pred = clf.predict(text)
        hit = pred.code is expected
        top1 += hit
        top3 += expected in {c for c, _ in pred.top(3)}
        ambiguous += pred.is_ambiguous

        pair = per_code.setdefault(expected.value, [0, 0])
        pair[0] += hit
        pair[1] += 1

    n = max(1, len(samples))
    return {
        "n": len(samples),
        "top1": top1 / n,
        "top3": top3 / n,
        "ambiguous_rate": ambiguous / n,
        "per_code": {k: (v[0] / v[1] if v[1] else 0.0, v[1]) for k, v in per_code.items()},
        "backend": clf.name,
    }

"""NLP layer: entailment-based evidence verification and reason-code classification."""

from arbiter.nlp.hypotheses import (
    Hypothesis,
    hypothesis_for,
    supported_pairs,
)
from arbiter.nlp.verifier import (
    EvidenceVerifier,
    NLIResult,
    StubNLIBackend,
    TransformerNLIBackend,
    verify_case,
)

__all__ = [
    "EvidenceVerifier",
    "Hypothesis",
    "NLIResult",
    "StubNLIBackend",
    "TransformerNLIBackend",
    "hypothesis_for",
    "supported_pairs",
    "verify_case",
]

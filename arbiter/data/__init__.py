"""Synthetic dispute corpus generation."""

from arbiter.data.generator import (
    CorpusConfig,
    GeneratedCase,
    generate_case,
    generate_corpus,
)
from arbiter.data.narratives import render_cm_narrative, render_merchant_rebuttal
from arbiter.data.scenarios import GroundTruth, Scenario, scenarios_for

__all__ = [
    "CorpusConfig",
    "GeneratedCase",
    "GroundTruth",
    "Scenario",
    "generate_case",
    "generate_corpus",
    "render_cm_narrative",
    "render_merchant_rebuttal",
    "scenarios_for",
]

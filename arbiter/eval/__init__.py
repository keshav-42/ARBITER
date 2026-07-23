"""Evaluation: fairness audit, calibration metrics, and the end-to-end report."""

from arbiter.eval.fairness import (
    AsymmetryReport,
    RoleSwapReport,
    asymmetry_audit,
    role_swap_test,
    swap_parties,
)
from arbiter.eval.report import EvalReport, run_evaluation

__all__ = [
    "AsymmetryReport",
    "EvalReport",
    "RoleSwapReport",
    "asymmetry_audit",
    "role_swap_test",
    "run_evaluation",
    "swap_parties",
]

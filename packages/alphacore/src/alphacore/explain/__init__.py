"""Model explanation (§9)."""

from __future__ import annotations

from alphacore.explain.shap_runner import (
    ADDITIVITY_TOLERANCE,
    DEFAULT_SAMPLE,
    ShapResult,
    explain_prediction,
    shap_summary,
    verify_additivity,
)

__all__ = [
    "ADDITIVITY_TOLERANCE",
    "DEFAULT_SAMPLE",
    "ShapResult",
    "explain_prediction",
    "shap_summary",
    "verify_additivity",
]

"""Compatibility wrapper exposing modular reflection layer objects."""

from __future__ import annotations

from workspace.bitcore_research.reflection import (
    DEFAULT_WEIGHTS,
    QUALITY_THRESHOLDS,
    ReflectionEngine,
    QualityDimension,
    calculate_overall_score,
    determine_verdict,
    evaluate_quality_scores,
    generate_recommendations,
)

__all__ = [
    "ReflectionEngine",
    "QualityDimension",
    "QUALITY_THRESHOLDS",
    "DEFAULT_WEIGHTS",
    "evaluate_quality_scores",
    "calculate_overall_score",
    "determine_verdict",
    "generate_recommendations",
]


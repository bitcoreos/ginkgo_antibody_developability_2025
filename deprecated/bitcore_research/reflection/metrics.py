"""Quality scoring utilities for the reflection layer."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Mapping


class QualityDimension(str, Enum):
    """Dimension labels used for quality evaluation."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    NOVELTY = "novelty"
    CITATIONS = "citations"


QUALITY_THRESHOLDS: Dict[QualityDimension, float] = {
    QualityDimension.COMPLETENESS: 0.7,
    QualityDimension.ACCURACY: 0.8,
    QualityDimension.RELEVANCE: 0.75,
    QualityDimension.COHERENCE: 0.8,
    QualityDimension.NOVELTY: 0.6,
    QualityDimension.CITATIONS: 0.65,
}

DEFAULT_WEIGHTS: Dict[QualityDimension, float] = {
    QualityDimension.COMPLETENESS: 0.2,
    QualityDimension.ACCURACY: 0.25,
    QualityDimension.RELEVANCE: 0.15,
    QualityDimension.COHERENCE: 0.15,
    QualityDimension.NOVELTY: 0.15,
    QualityDimension.CITATIONS: 0.1,
}


def evaluate_quality_scores(research_data: Mapping) -> Dict[QualityDimension, float]:
    """Compute per-dimension quality scores for the supplied research data."""

    scores = {
        QualityDimension.COMPLETENESS: _evaluate_completeness(research_data),
        QualityDimension.ACCURACY: _evaluate_accuracy(research_data),
        QualityDimension.RELEVANCE: _evaluate_relevance(research_data),
        QualityDimension.COHERENCE: _evaluate_coherence(research_data),
        QualityDimension.NOVELTY: _evaluate_novelty(research_data),
        QualityDimension.CITATIONS: _evaluate_citations(research_data),
    }
    return scores


def calculate_overall_score(
    quality_scores: Mapping[QualityDimension, float],
    weights: Mapping[QualityDimension, float] | None = None,
) -> float:
    """Aggregate per-dimension scores into a single weighted score."""

    weights = weights or DEFAULT_WEIGHTS
    overall = sum(quality_scores[dim] * weights[dim] for dim in weights)
    return round(overall, 3)


def determine_verdict(overall_score: float) -> str:
    """Convert an overall score into a human-readable verdict."""

    if overall_score >= 0.8:
        return "excellent"
    if overall_score >= 0.7:
        return "good"
    if overall_score >= 0.6:
        return "acceptable"
    return "needs_improvement"


def generate_recommendations(
    quality_scores: Mapping[QualityDimension, float], overall_score: float
) -> List[str]:
    """Suggest follow-up actions given quality scores and the overall verdict."""

    recommendations: List[str] = []

    for dimension, score in quality_scores.items():
        threshold = QUALITY_THRESHOLDS[dimension]
        if score >= threshold:
            continue

        if dimension is QualityDimension.COMPLETENESS:
            recommendations.append("Expand research to cover more aspects of the topic")
        elif dimension is QualityDimension.ACCURACY:
            recommendations.append("Verify facts with additional trusted sources")
        elif dimension is QualityDimension.RELEVANCE:
            recommendations.append("Focus on the most relevant aspects of the query")
        elif dimension is QualityDimension.COHERENCE:
            recommendations.append("Improve organization and logical flow of the findings")
        elif dimension is QualityDimension.NOVELTY:
            recommendations.append("Incorporate more recent and cutting-edge sources")
        elif dimension is QualityDimension.CITATIONS:
            recommendations.append("Include additional academic references")

    if overall_score < 0.7:
        recommendations.append(
            "Consider dispatching to a more specialized agent for this research type"
        )
        recommendations.append("Increase the number of sources and total collected results")

    return recommendations


def _evaluate_completeness(research_data: Mapping) -> float:
    sources = research_data.get("sources", {})
    sources_count = len(sources)
    total_results = sum(src.get("count", 0) for src in sources.values())
    completeness = min(1.0, sources_count * 0.3 + total_results * 0.01)
    return round(completeness, 3)


def _evaluate_accuracy(research_data: Mapping) -> float:
    academic_sources: Iterable[str] = {"core_api", "arxiv", "pubmed", "google_scholar"}
    sources = research_data.get("sources", {})
    academic_count = sum(
        1 for source, payload in sources.items() if source in academic_sources and payload.get("count", 0) > 0
    )
    accuracy = min(1.0, academic_count * 0.25)
    return round(accuracy, 3)


def _evaluate_relevance(research_data: Mapping) -> float:
    total_results = sum(src.get("count", 0) for src in research_data.get("sources", {}).values())
    relevance = min(1.0, total_results * 0.05)
    return round(relevance, 3)


def _evaluate_coherence(research_data: Mapping) -> float:
    sources_count = len(research_data.get("sources", {}))
    coherence = min(1.0, sources_count * 0.2)
    return round(coherence, 3)


def _evaluate_novelty(research_data: Mapping) -> float:
    novelty_sources: Iterable[str] = {"arxiv", "preprints", "conference_proceedings"}
    sources = research_data.get("sources", {})
    novelty_count = sum(
        1 for source, payload in sources.items() if source in novelty_sources and payload.get("count", 0) > 0
    )
    novelty = min(1.0, novelty_count * 0.3)
    return round(novelty, 3)


def _evaluate_citations(research_data: Mapping) -> float:
    citation_sources: Iterable[str] = {"core_api", "arxiv", "pubmed", "google_scholar"}
    sources = research_data.get("sources", {})
    citation_count = sum(
        payload.get("count", 0) for source, payload in sources.items() if source in citation_sources
    )
    citations = min(1.0, citation_count * 0.05)
    return round(citations, 3)

"""Primary entrypoint for the reflection layer."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict

from workspace.config import cfg, get_logger

from .artifacts import save_evaluation
from .metrics import (
    DEFAULT_WEIGHTS,
    QUALITY_THRESHOLDS,
    QualityDimension,
    calculate_overall_score,
    determine_verdict,
    evaluate_quality_scores,
    generate_recommendations,
)
from .storage import append_successful_pattern, ensure_directories, load_knowledge_base, save_knowledge_base

logger = get_logger(__name__)


class ReflectionEngine:
    """Evaluates research quality results and updates the knowledge base."""

    def __init__(self) -> None:
        kb_default = os.path.join(cfg.TMP_DIR, "knowledge_base.json")
        reports_default = os.path.join(cfg.TMP_DIR, "assurance_reports")

        self.knowledge_base_path = cfg.KNOWLEDGE_BASE_PATH or kb_default
        self.assurance_reports_path = cfg.ASSURANCE_REPORTS_DIR or reports_default

        ensure_directories(
            {
                os.path.dirname(self.knowledge_base_path),
                self.assurance_reports_path,
                cfg.TMP_DIR,
            },
            logger,
        )

        self.knowledge_base = load_knowledge_base(self.knowledge_base_path, logger)
        logger.info("REFLECTION LAYER: ReflectionEngine initialized")

    def evaluate_quality(self, research_data: Dict, agent_type: str) -> Dict:
        logger.info(
            "Evaluating research quality for query: '%s'",
            research_data.get("query", ""),
        )

        quality_scores = evaluate_quality_scores(research_data)
        overall_score = calculate_overall_score(quality_scores, DEFAULT_WEIGHTS)
        verdict = determine_verdict(overall_score)
        recommendations = generate_recommendations(quality_scores, overall_score)

        serialisable_scores = {dimension.value: score for dimension, score in quality_scores.items()}

        evaluation = {
            "query": research_data.get("query", ""),
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
            "quality_scores": serialisable_scores,
            "dimension_details": {},
            "overall_score": overall_score,
            "verdict": verdict,
            "recommendations": recommendations,
        }

        metrics = self.knowledge_base.setdefault("metrics", {})
        metrics.setdefault("total_evaluations", 0)
        metrics["total_evaluations"] += 1

        save_evaluation(self.assurance_reports_path, evaluation, logger)
        save_knowledge_base(self.knowledge_base_path, self.knowledge_base, logger)

        logger.info(
            "Quality evaluation completed: %s (%s)",
            evaluation["overall_score"],
            evaluation["verdict"],
        )
        return evaluation

    def update_knowledge_base(self, research_data: Dict, evaluation: Dict, agent_type: str) -> None:
        if evaluation.get("overall_score", 0) < 0.7:
            return

        pattern = {
            "query_template": self._extract_query_template(research_data.get("query", "")),
            "agent_type": agent_type,
            "sources_used": list(research_data.get("sources", {}).keys()),
            "results_count": evaluation.get("quality_scores", {}).get(QualityDimension.RELEVANCE.value, 0),
            "quality_score": evaluation.get("overall_score", 0.0),
            "timestamp": datetime.now().isoformat(),
        }

        append_successful_pattern(self.knowledge_base, pattern, logger)
        save_knowledge_base(self.knowledge_base_path, self.knowledge_base, logger)
        logger.info(
            "Updated knowledge base with successful pattern (total patterns: %s)",
            len(self.knowledge_base.get("patterns", [])),
        )

    def _extract_query_template(self, query: str) -> str:
        template = query.lower()
        template = re.sub(r"\b(19|20)\d{2}\b", "{year}", template)
        template = re.sub(r"\b\d+\b", "{number}", template)

        for term in [
            "ai",
            "machine learning",
            "blockchain",
            "quantum",
            "neural",
            "algorithm",
        ]:
            template = template.replace(term, "{domain}")

        return template


__all__ = [
    "ReflectionEngine",
    "QualityDimension",
    "QUALITY_THRESHOLDS",
    "DEFAULT_WEIGHTS",
]

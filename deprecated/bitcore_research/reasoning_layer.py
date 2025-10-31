"""Reasoning layer analyzes query complexity and selects agent routing."""

from datetime import datetime
from enum import Enum

from workspace.config import cfg, get_logger

logger = get_logger(__name__)


MORPHEUS_API_URL = cfg.MORPHEUS_API_URL
MORPHEUS_HEADERS = {
    "Authorization": f"Bearer {cfg.MORPHEUS_API_KEY}" if cfg.MORPHEUS_API_KEY else "",
    "Content-Type": "application/json",
}


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class MorpheusAgentType(Enum):
    GENERAL = "general"
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    EXPERT = "expert"


ROUTING_RULES = {
    ComplexityLevel.SIMPLE: MorpheusAgentType.GENERAL,
    ComplexityLevel.MODERATE: MorpheusAgentType.SCIENTIFIC,
    ComplexityLevel.COMPLEX: MorpheusAgentType.TECHNICAL,
    ComplexityLevel.EXPERT: MorpheusAgentType.EXPERT,
}


class ReasoningEngine:
    """Analyzes incoming queries and selects the appropriate research agent."""

    def __init__(self) -> None:
        self.complexity_thresholds = {
            "simple": 3,
            "moderate": 6,
            "complex": 9,
            "expert": 10,
        }
        logger.info("REASONING LAYER: ReasoningEngine initialized")

    def analyze_complexity(self, query: str) -> ComplexityLevel:
        score = 0

        words = len(query.split())
        if words < 5:
            score += 1
        elif words > 15:
            score += 4

        technical_terms = [
            "quantum",
            "neural",
            "algorithm",
            "framework",
            "protocol",
            "cryptography",
            "blockchain",
            "ai",
            "machine learning",
            "deep learning",
            "transformer",
            "llm",
            "autonomous",
            "cybersecurity",
            "penetration",
            "exploit",
            "vulnerability",
        ]
        q_lower = query.lower()
        score += sum(1 for term in technical_terms if term in q_lower) * 2

        domains = [
            "computer science",
            "artificial intelligence",
            "mathematics",
            "physics",
            "biology",
            "chemistry",
            "engineering",
            "economics",
        ]
        domain_count = sum(1 for domain in domains if domain in q_lower)
        if domain_count >= 2:
            score += 3

        if score <= 3:
            complexity = ComplexityLevel.SIMPLE
        elif score <= 6:
            complexity = ComplexityLevel.MODERATE
        elif score <= 9:
            complexity = ComplexityLevel.COMPLEX
        else:
            complexity = ComplexityLevel.EXPERT

        logger.info(
            "Query complexity analysis: '%s' -> %s (score=%s)",
            query,
            complexity.value,
            score,
        )
        return complexity

    def route_to_agent(self, complexity: ComplexityLevel) -> MorpheusAgentType:
        agent_type = ROUTING_RULES.get(complexity, MorpheusAgentType.GENERAL)
        logger.info(
            "Routing complexity '%s' to agent type '%s'",
            complexity.value,
            agent_type.value,
        )
        return agent_type

    def get_agent_prompt(self, agent_type: MorpheusAgentType) -> str:
        prompts = {
            MorpheusAgentType.GENERAL: "You are a general research assistant. Provide a comprehensive overview of the topic in clear, accessible language.",
            MorpheusAgentType.SCIENTIFIC: "You are a scientific research assistant. Provide detailed information with references to scientific literature and methodologies.",
            MorpheusAgentType.TECHNICAL: "You are a technical research assistant. Provide in-depth technical analysis with specifications, architectures, and implementation details.",
            MorpheusAgentType.EXPERT: "You are an expert research assistant. Provide cutting-edge insights with analysis of recent developments, research gaps, and future directions.",
        }
        return prompts.get(agent_type, prompts[MorpheusAgentType.GENERAL])

    def analyze_and_route(self, query: str) -> dict:
        try:
            complexity = self.analyze_complexity(query)
            agent_type = self.route_to_agent(complexity)
            prompt = self.get_agent_prompt(agent_type)

            result = {
                "query": query,
                "complexity": complexity.value,
                "agent_type": agent_type.value,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
            }
            logger.info("REASONING LAYER: Analysis complete for '%s'", query)
            return result
        except Exception as exc:
            logger.error("Error in analyze_and_route: %s", exc)
            raise


logger.info("REASONING LAYER initialized")

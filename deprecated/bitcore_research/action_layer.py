"""Action layer coordinates data gathering for research tasks."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List

import requests

from workspace.config import cfg, get_logger, is_dry_run

logger = get_logger(__name__)

CORE_API_URL = cfg.CORE_API_URL.rstrip("/") + "/"
CORE_HEADERS = {
    "X-API-KEY": cfg.CORE_API_KEY or "",
    "Content-Type": "application/json",
}

MORPHEUS_API_URL = cfg.MORPHEUS_API_URL
MORPHEUS_HEADERS = {
    "Authorization": f"Bearer {cfg.MORPHEUS_API_KEY}" if cfg.MORPHEUS_API_KEY else "",
    "Content-Type": "application/json",
}

DATA_SOURCE_PRIORITIES = {
    "general": ["wikipedia", "news", "general_web"],
    "scientific": ["core_api", "arxiv", "pubmed", "google_scholar"],
    "technical": ["core_api", "arxiv", "patents", "technical_docs", "github"],
    "expert": [
        "core_api",
        "arxiv",
        "preprints",
        "conference_proceedings",
        "expert_interviews",
    ],
}


class ActionEngine:
    """Coordinates data gathering from available sources."""

    def __init__(self) -> None:
        self.data_sources = {
            "core_api": self._search_core_api,
            "arxiv": self._placeholder_source("arXiv"),
            "wikipedia": self._placeholder_source("Wikipedia"),
            "news": self._placeholder_source("News"),
            "google_scholar": self._placeholder_source("Google Scholar"),
            "pubmed": self._placeholder_source("PubMed"),
            "patents": self._placeholder_source("Patent database"),
            "technical_docs": self._placeholder_source("Technical documentation"),
            "github": self._placeholder_source("GitHub"),
            "conference_proceedings": self._placeholder_source("Conference proceedings"),
            "expert_interviews": self._placeholder_source("Expert interviews"),
            "general_web": self._placeholder_source("General web"),
        }
        logger.info("ACTION LAYER: ActionEngine initialized")

    def gather_data(
        self, query: str, agent_type: str, max_results_per_source: int = 5
    ) -> Dict:
        sources = DATA_SOURCE_PRIORITIES.get(agent_type, ["core_api", "general_web"])
        results: Dict[str, List[Dict]] = {}
        total_items = 0

        for source in sources:
            handler = self.data_sources.get(source)
            if not handler:
                logger.debug("No handler registered for source '%s'", source)
                continue
            try:
                items = handler(query, max_results=max_results_per_source)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Error collecting data from %s: %s", source, exc)
                items = []
            results[source] = items
            total_items += len(items)

        payload = {
            "query": query,
            "agent_type": agent_type,
            "collected_at": datetime.now().isoformat(),
            "sources": results,
            "summary": {
                "total_sources": len(results),
                "total_items": total_items,
            },
        }

        self._persist_collection(payload)
        return payload

    def _persist_collection(self, payload: Dict) -> None:
        try:
            os.makedirs(cfg.TMP_DIR, exist_ok=True)
            filename = f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            target = os.path.join(cfg.TMP_DIR, filename)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            logger.info("Saved data collection artifact to %s", target)
        except Exception as exc:  # pragma: no cover - persistence is best effort
            logger.error("Failed to persist data collection: %s", exc)

    def _search_core_api(self, query: str, max_results: int = 10) -> List[Dict]:
        if is_dry_run() or not cfg.CORE_API_KEY:
            logger.info("CORE API dry-run for '%s'", query)
            return [
                {
                    "title": f"Simulated CORE result for '{query}'",
                    "source": "CORE API",
                    "relevance_score": 0.0,
                    "year": datetime.now().year,
                }
            ]

        try:
            response = requests.get(
                f"{CORE_API_URL}v3/search",
                headers=CORE_HEADERS,
                params={"q": query, "limit": max_results, "metadata": True},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            results: List[Dict] = []
            for item in data.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "authors": ", ".join(
                            author.get("name", "") for author in item.get("authors", [])
                        ),
                        "abstract": item.get("abstract", ""),
                        "year": item.get("year"),
                        "doi": item.get("doi"),
                        "url": item.get("downloadUrl"),
                        "source": "CORE API",
                        "relevance_score": item.get("score", 0.0),
                    }
                )
            return results
        except Exception as exc:
            logger.error("CORE API request failed: %s", exc)
            return []

    def _placeholder_source(self, source_name: str):
        def _handler(query: str, max_results: int = 5) -> List[Dict]:
            logger.info("%s dry-run for '%s'", source_name, query)
            return [
                {
                    "title": f"Simulated {source_name} result",
                    "source": source_name,
                    "query": query,
                }
            ]

        return _handler


logger.info("ACTION LAYER initialized")

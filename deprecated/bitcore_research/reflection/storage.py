"""Persistence helpers for the reflection engine."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable, Mapping


def ensure_directories(directories: Iterable[str], logger) -> None:
    """Ensure directories exist, logging failures but not raising."""

    for directory in directories:
        if not directory:
            continue
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to ensure directory %s: %s", directory, exc)


def load_knowledge_base(path: str, logger) -> Dict:
    """Load a knowledge base document from disk, returning a default if absent."""

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                kb = json.load(handle)
            logger.info("Loaded knowledge base with %s patterns", len(kb.get("patterns", [])))
            return kb
        except Exception as exc:  # pragma: no cover - should not happen in normal flow
            logger.error("Error loading knowledge base: %s", exc)

    kb = {
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "version": "1.0",
        "patterns": [],
        "metrics": {
            "total_evaluations": 0,
            "successful_patterns": 0,
            "optimizations_applied": 0,
        },
    }
    save_knowledge_base(path, kb, logger)
    logger.info("Initialized new knowledge base")
    return kb


def save_knowledge_base(path: str, kb: Dict, logger) -> None:
    """Persist the knowledge base to disk, updating timestamps."""

    try:
        kb["last_updated"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(kb, handle, indent=2)
        logger.info("Knowledge base saved successfully")
    except Exception as exc:  # pragma: no cover - persistence errors are logged
        logger.error("Error saving knowledge base: %s", exc)


def append_successful_pattern(kb: Dict, pattern: Dict, logger, limit: int = 100) -> None:
    """Append a successful pattern while keeping the collection bounded."""

    kb.setdefault("patterns", []).append(pattern)
    kb.setdefault("metrics", {}).setdefault("successful_patterns", 0)
    kb["metrics"]["successful_patterns"] += 1

    if len(kb["patterns"]) > limit:
        kb["patterns"] = kb["patterns"][-limit:]
        logger.debug("Trimmed knowledge base patterns to last %s entries", limit)

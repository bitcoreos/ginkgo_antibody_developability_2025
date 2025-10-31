"""Utility helpers for persisting quality assurance artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict


def save_evaluation(assurance_dir: str, evaluation: Dict, logger) -> None:
    """Persist a quality assurance evaluation to disk."""

    try:
        os.makedirs(assurance_dir, exist_ok=True)
        filename = f"quality_assurance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        target = os.path.join(assurance_dir, filename)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2)
        logger.info("Quality assurance report saved: %s", target)
    except Exception as exc:  # pragma: no cover - persistence reliability best-effort
        logger.error("Error saving quality assurance report: %s", exc)

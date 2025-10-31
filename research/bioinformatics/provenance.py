#!/usr/bin/env python3
"""Utilities for emitting provenance logs for bioinformatics feature scripts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = WORKSPACE_ROOT / "logs" / "pipeline" / "provenance"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_workspace(path: Path) -> Path:
    try:
        return path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return path


def _describe_path(path: Path) -> dict:
    absolute = path.resolve()
    entry = {
        "path": str(_relative_to_workspace(absolute)),
        "exists": absolute.exists(),
    }
    if absolute.is_file():
        entry["sha256"] = _sha256(absolute)
        entry["size_bytes"] = absolute.stat().st_size
    return entry


def record_provenance_event(
    event_name: str,
    inputs: Sequence[Path | str],
    outputs: Sequence[Path | str],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Record provenance metadata for a feature generation event."""

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "event": event_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [_describe_path(Path(p)) for p in inputs],
        "outputs": [_describe_path(Path(p)) for p in outputs],
    }

    if metadata:
        payload["metadata"] = dict(metadata)

    log_path = PROVENANCE_DIR / f"{event_name}_{timestamp}.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return log_path

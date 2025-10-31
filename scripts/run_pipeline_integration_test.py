#!/usr/bin/env python3
"""Execute the pipeline CLI and capture an integration-test log."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = WORKSPACE_ROOT.parent
PIPELINE_SCRIPT = WORKSPACE_ROOT / "scripts" / "run_pipeline.py"
LOG_DIR = WORKSPACE_ROOT / "logs" / "pipeline"


def run_command(args: Sequence[str]) -> Dict[str, Any]:
    start = time.monotonic()
    result = subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.monotonic() - start
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_seconds": duration,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline integration test and log results.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip executing the full pipeline; only record the plan output.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        help="Directory where the integration test log will be written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_integration_{timestamp}.json"

    if not PIPELINE_SCRIPT.is_file():
        raise SystemExit(f"Pipeline script not found: {PIPELINE_SCRIPT}")

    plan_entry = run_command(["python", str(PIPELINE_SCRIPT), "--plan"])
    try:
        plan_payload = json.loads(plan_entry["stdout"]) if plan_entry["stdout"] else None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse pipeline plan output: {exc}\n{plan_entry['stdout']}") from exc

    entries = [plan_entry]
    execution_entry: Dict[str, Any] | None = None

    if not args.dry_run:
        execution_entry = run_command(["python", str(PIPELINE_SCRIPT)])
        entries.append(execution_entry)
        if execution_entry["returncode"] != 0:
            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "plan": plan_payload,
                "steps": entries,
            }
            with log_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(json.dumps({"log_path": str(log_path), "status": "failure"}))
            raise SystemExit(execution_entry["returncode"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan_payload,
        "steps": entries,
    }

    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps({"log_path": str(log_path), "status": "success"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

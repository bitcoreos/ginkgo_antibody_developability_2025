#!/usr/bin/env python3
"""CLI to orchestrate the Track B bioinformatics pipeline with DRY_RUN support."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PIPELINE_ROOT.parent


@dataclass
class PipelineStep:
    name: str
    command: Sequence[str]
    working_dir: Path | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.working_dir is not None:
            data["working_dir"] = str(self.working_dir)
        return data


def build_steps() -> list[PipelineStep]:
    """Define the deterministic step order for the pipeline."""
    return [
        PipelineStep(
            name="Audit Targets",
            command=["python", "scripts/audit_targets.py"],
            working_dir=WORKSPACE_ROOT,
        ),
        PipelineStep(
            name="Competition Pipeline (Modified)",
            command=["python", "scripts/run_competition_pipeline_modified.py"],
            working_dir=WORKSPACE_ROOT,
        ),
    ]


def run_step(step: PipelineStep, dry_run: bool, verbose: bool) -> subprocess.CompletedProcess | None:
    """Execute a single pipeline step unless DRY_RUN is active."""
    if dry_run:
        return None

    result = subprocess.run(
        step.command,
        cwd=step.working_dir,
        check=False,
        text=True,
        capture_output=not verbose,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "no stderr"
        raise RuntimeError(
            f"Step '{step.name}' failed with exit code {result.returncode}: {stderr}"
        )
    if verbose and result.stdout:
        print(result.stdout)
    return result


def render_plan(steps: Iterable[PipelineStep]) -> dict:
    """Materialize the plan metadata for logging/preview purposes."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps": [step.to_dict() for step in steps],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Track B modeling pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running commands.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Output the execution plan JSON and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream command stdout while executing steps.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    steps = build_steps()
    plan = render_plan(steps)

    if args.plan or args.dry_run:
        print(json.dumps(plan, indent=2))
        if args.plan:
            return 0

    for step in steps:
        print(f"→ {step.name}")
        run_step(step, dry_run=args.dry_run, verbose=args.verbose)

    if not args.plan and not args.dry_run:
        print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

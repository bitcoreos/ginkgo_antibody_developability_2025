"""Command line interface for the asynchronous research engine."""

from __future__ import annotations

import argparse
import sys
sys.path.append("/a0/bitcore")  # Add parent directory to path for workspace.config import
import json
import os
from typing import Iterable, List

from workspace.config import cfg
from workspace.research_engine.engine import (
    NeuronEndpoint,
    dispatch_to_neurons,
    load_default_neurons,
    save_results,
)


def _parse_neuron_overrides(paths: Iterable[str]) -> List[NeuronEndpoint]:
    neurons: List[NeuronEndpoint] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and {"name", "url"} <= data.keys():
            neurons.append(NeuronEndpoint(data["name"], data["url"]))
            continue
        if isinstance(data, list):
            for entry in data:
                if {"name", "url"} <= entry.keys():
                    neurons.append(NeuronEndpoint(entry["name"], entry["url"]))
    return neurons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch research topic to neuron webhooks")
    parser.add_argument("--topic", required=False, default="general biology concepts", help="Research topic to dispatch")
    parser.add_argument("--system-prompt", required=False, default="You are a specialized biology research agent.", help="System prompt for all neurons")
    parser.add_argument("--output-dir", default=None, help="Directory for saving results (defaults to config RESULTS_RAW_DIR)")
    parser.add_argument("--timeout", type=int, default=3600, help="Request timeout per neuron in seconds")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum concurrent workers (defaults to config SWARM_MAX_WORKERS)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run regardless of environment configuration")
    parser.add_argument("--neurons", nargs="*", default=None, help="JSON file(s) with neuron definitions [{\"name\": str, \"url\": str}]")
    parser.add_argument("--no-save", action="store_true", help="Do not write results to disk")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dry_run:
        cfg.DRY_RUN = True
        os.environ["DRY_RUN"] = "true"

    neurons: List[NeuronEndpoint]
    if args.neurons:
        neurons = _parse_neuron_overrides(args.neurons)
    else:
        neurons = load_default_neurons()

    results = dispatch_to_neurons(
        topic=args.topic,
        system_prompt=args.system_prompt,
        neurons=neurons,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )

    output_path = None
    if not args.no_save:
        output_path = save_results(
            topic=args.topic,
            system_prompt=args.system_prompt,
            results=results,
            output_dir=args.output_dir,
        )

    print(json.dumps({"results": results, "saved_to": output_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

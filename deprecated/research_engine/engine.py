"""Utilities for dispatching research requests to neuron webhooks."""

from __future__ import annotations
import time
import random

import concurrent.futures
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import requests

from workspace.config import cfg, get_logger, is_dry_run

logger = get_logger(__name__)


@dataclass(frozen=True)
class NeuronEndpoint:
    """Represents a single neuron webhook endpoint."""

    name: str
    url: str


_DEFAULT_NEURONS: Sequence[NeuronEndpoint] = (
    NeuronEndpoint("llama-3.2-3b", "https://n8n.bitwiki.org/webhook/429d66fc-10f5-4f43-8c61-9f0af1fdde0f"),
    NeuronEndpoint("qwen3-4b", "https://n8n.bitwiki.org/webhook/f29c348a-d66d-41b6-9571-7bdcf61bc134"),
    NeuronEndpoint("llama-3.3-70b", "https://n8n.bitwiki.org/webhook/2208dcde-7716-4bd5-8910-bae78edc1931"),
    NeuronEndpoint("mistral-31-24b", "https://n8n.bitwiki.org/webhook/304264b7-4128-470e-b7c5-e549aa2b62b5"),
    NeuronEndpoint("venice-uncensored", "https://n8n.bitwiki.org/webhook/05720333-8ec6-408d-9531-d5cb5eb87993"),
    NeuronEndpoint("qwen3-235b", "https://n8n.bitwiki.org/webhook/0b10e6c5-51a5-4d61-99bc-26409bc2c7a8"),
)


def _env_override(name: str) -> Optional[str]:
    env_name = f"NEURON_{name.replace('-', '_').replace('.', '_').upper()}_URL"
    return os.getenv(env_name)


def load_default_neurons() -> List[NeuronEndpoint]:
    neurons: List[NeuronEndpoint] = []
    for neuron in _DEFAULT_NEURONS:
        override = _env_override(neuron.name)
        neurons.append(NeuronEndpoint(neuron.name, override or neuron.url))
    return neurons


def build_payload(topic: str, system_prompt: str) -> Dict[str, str]:
    return {
        "systemprompt": system_prompt,
        "researchTopic": topic,
    }


def _dispatch_single(session: requests.Session, neuron: NeuronEndpoint, payload: Dict[str, str], dry_run: bool, timeout: Optional[int] = None) -> Dict:
    if dry_run:
        logger.info("DRY_RUN: would dispatch to %s", neuron.name)
        return {
            "status": "success",
            "neuron": neuron.name,
            "response": {"simulated": True},
            "timestamp": datetime.now().isoformat(),
        }

    response = session.post(neuron.url, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Dispatch to %s failed: %s", neuron.name, exc)
        return {
            "status": "error",
            "neuron": neuron.name,
            "status_code": response.status_code,
            "message": response.text,
            "timestamp": datetime.now().isoformat(),
        }

    data: Dict = {}
    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text}

    return {
        "status": "success",
        "neuron": neuron.name,
        "response": data,
        "timestamp": datetime.now().isoformat(),
    }


def dispatch_to_neurons(
    topic: str,
    system_prompt: str,
    neurons: Optional[Iterable[NeuronEndpoint]] = None,
    max_workers: Optional[int] = None,
    timeout: Optional[int] = None
) -> Dict[str, Dict]:
    if neurons is None:
        neurons = load_default_neurons()

    neuron_list = list(neurons)
    if not neuron_list:
        raise ValueError("No neuron endpoints provided")

    payload = build_payload(topic, system_prompt)
    dry_run = is_dry_run()
    worker_count = max_workers or cfg.SWARM_MAX_WORKERS

    results: Dict[str, Dict] = {}
    with requests.Session() as session:
        # Process neurons sequentially to prevent n8n server overload
        neuron_list = list(neuron_list)  # Ensure we have a list
        for i, neuron in enumerate(neuron_list):
            try:
                # Dispatch to single neuron
                result = _dispatch_single(session, neuron, payload, dry_run, timeout=timeout)
                results[neuron.name] = result

                # Add delay between requests except after the last one
                if i < len(neuron_list) - 1:
                    delay = random.uniform(5, 10)
                    logger.info(f"Waiting {{delay:.1f}} seconds before next neuron request")
                    time.sleep(delay)
            except Exception as exc:  # pragma: no cover
                logger.error("Dispatch to %s raised unexpected error: %s", neuron.name, exc)
                results[neuron.name] = {
                    "status": "error",
                    "neuron": neuron.name,
                    "message": str(exc),
                    "timestamp": datetime.now().isoformat(),
                }
    return results


def save_results(
    topic: str,
    system_prompt: str,
    results: Dict[str, Dict],
    output_dir: Optional[str] = None,
    prefix: str = "research_engine_results",
) -> str:
    directory = output_dir or cfg.RESULTS_RAW_DIR
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(directory, f"{prefix}_{timestamp}.json")

    payload = {
        "topic": topic,
        "system_prompt": system_prompt,
        "results": results,
        "dry_run": is_dry_run(),
        "saved_at": datetime.now().isoformat(),
    }

    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved research engine results to %s", target)
    return target


__all__ = [
    "NeuronEndpoint",
    "build_payload",
    "dispatch_to_neurons",
    "load_default_neurons",
    "save_results",
]

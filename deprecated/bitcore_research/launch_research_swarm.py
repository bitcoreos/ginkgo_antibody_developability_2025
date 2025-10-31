import concurrent.futures
import json
import os
from datetime import datetime

import requests

from workspace.config import cfg, get_logger, is_dry_run

logger = get_logger(__name__)


def send_research_request(payload, model_config):
    """Send a research request to the Morpheus API with the specified model"""
    url = cfg.MORPHEUS_API_URL

    # Extract model name and construct the full model string
    model_name = model_config["model"]
    full_model = f"{model_name}:web"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {cfg.MORPHEUS_API_KEY}" if cfg.MORPHEUS_API_KEY else "",
        "Content-Type": "application/json"
    }

    # Create the request body
    data = {
        "model": full_model,
        "messages": [
            {
                "role": "system",
                "content": model_config["system_prompt"]
            },
            {
                "role": "user",
                "content": f"Research the following topic in depth: {payload['researchTopic']}"
            }
        ],
        "stream": False
    }

    if is_dry_run():
        logger.info("DRY_RUN: simulated request for model %s", model_name)
        return {
            "status": "success",
            "model": model_name,
            "response": {"simulated": True},
            "timestamp": datetime.now().isoformat()
        }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=300)  # 5 minute timeout

        if response.status_code == 200:
            return {
                "status": "success",
                "model": model_name,
                "response": response.json(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "model": model_name,
                "status_code": response.status_code,
                "message": response.text,
                "timestamp": datetime.now().isoformat()
            }

    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "model": model_name,
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


def launch_research_swarm(payload=None, models=None, max_workers=None):
    """Launch the research swarm. Returns results dict."""
    if payload is None:
        payload = {
            "researchTopic": "Hidden features in antibody sequences that influence developability using Hidden Markov Models",
            "timestamp": datetime.now().isoformat()
        }

    if models is None:
        models = [
            {"model": "llama-3.2-3b", "system_prompt": "You are a biochemical analysis specialist. Focus on molecular features that determine antibody developability."},
            {"model": "qwen3-4b", "system_prompt": "You are a structural biology specialist. Focus on 3D conformations and their influence on function."},
            {"model": "llama-3.3-70b", "system_prompt": "You are an evolutionary biology specialist. Focus on patterns from natural selection."},
            {"model": "mistral-31-24b", "system_prompt": "You are a clinical research specialist. Focus on real-world outcomes and applications."},
            {"model": "venice-uncensored", "system_prompt": "You are a computational biology specialist. Focus on model relationships and prediction methods."},
            {"model": "qwen-3-235b", "system_prompt": "You are a thermodynamics specialist. Focus on energy landscapes governing folding."}
        ]

    if max_workers is None:
        max_workers = cfg.SWARM_MAX_WORKERS

    logger.info("Launching research swarm at %s", datetime.now().isoformat())
    logger.info("Research topic: %s", payload["researchTopic"])

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {executor.submit(send_research_request, payload, model): model["model"] for model in models}
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results[model_name] = result
                logger.info("Completed research for %s", model_name)
            except Exception as exc:
                results[model_name] = {
                    "status": "error",
                    "model": model_name,
                    "message": str(exc),
                    "timestamp": datetime.now().isoformat(),
                }
                logger.error("Error in research for %s: %s", model_name, exc)

    # Save results
    base_dir = cfg.RESULTS_BASE_DIR
    results_dir = cfg.RESULTS_RAW_DIR
    try:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"research_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({"payload": payload, "models": [m["model"] for m in models], "results": results, "completion_time": datetime.now().isoformat()}, f, indent=2)
        logger.info("Research swarm completed. Results saved to %s", results_file)
    except Exception as e:
        logger.error("Error saving swarm results: %s", e)
        try:
            import tempfile
            tmpdir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fb = os.path.join(tmpdir, f"research_results_{timestamp}.json")
            with open(fb, 'w') as f:
                json.dump({"payload": payload, "models": [m["model"] for m in models], "results": results, "completion_time": datetime.now().isoformat()}, f, indent=2)
            logger.info("Research swarm completed. Results saved to fallback %s", fb)
        except Exception as e2:
            logger.error("Final fallback saving failed: %s", e2)

    return results

if __name__ == "__main__":
    launch_research_swarm()

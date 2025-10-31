
"""
PERCEPTION LAYER for Autonomous Research Coordinator

Based on documented evidence in memory, the 'agent-zero-final' webhook is functional
and has been confirmed with 200 responses from test POST requests.

This module implements the entry point for the four-layer cognition engine.
"""

import json
import os
import tempfile
from datetime import datetime

import requests

from workspace.config import cfg, get_logger, is_dry_run

logger = get_logger(__name__)

# Webhook configuration from config
WEBHOOK_URL = cfg.PERCEPTION_WEBHOOK_URL
HEADERS = {
    "X-N8N-API-KEY": cfg.N8N_API_KEY or "",
    "Content-Type": "application/json"
}

# Function to send research request to the perception layer
def send_research_request(query="latest AI developments", additional_data=None):
    """
    Send a research request to the n8n webhook endpoint.

    Args:
        query (str): The research query
        additional_data (dict): Additional data to include in the request

    Returns:
        dict: Response from the webhook
    """

    # Prepare the payload
    payload = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "source": "AgentZero research coordinator"
    }

    # Add any additional data if provided
    if additional_data:
        payload.update(additional_data)

    # Respect DRY_RUN: do not make network calls if dry-run
    if is_dry_run():
        logger.info("DRY_RUN enabled - skipping webhook POST")
        # Return a simulated successful response structure
        simulated = {
            "success": True,
            "status_code": 200,
            "result": {"message": "dry-run: simulated response"}
        }
        # Save simulated result to tmp for debugging
        _save_result(simulated)
        return simulated

    try:
        # Send the POST request to the webhook
        response = requests.post(
            WEBHOOK_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )

        # Log the response
        logger.info("Webhook response status: %s", response.status_code)

        if response.status_code == 200:
            try:
                result = response.json()
                logger.info("Research request successful")

                # Save the result
                _save_result(result)

                return {
                    "success": True,
                    "status_code": response.status_code,
                    "result": result
                }
            except json.JSONDecodeError:
                logger.warning("Webhook returned 200 but response is not JSON")
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "result": response.text
                }
        else:
            logger.error("Webhook request failed: %s", response.text)
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text
            }

    except Exception as e:
        logger.error("Exception sending research request: %s", e)
        return {
            "success": False,
            "error": str(e)
        }


def _save_result(result: dict) -> None:
    filename = "research_result.json"
    candidates = [cfg.TMP_DIR, tempfile.gettempdir(), os.getcwd()]
    for base in candidates:
        try:
            if not base:
                continue
            os.makedirs(base, exist_ok=True)
            target = os.path.join(base, filename)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            logger.info("Saved perception result to %s", target)
            return
        except Exception:
            continue
    logger.warning("Unable to persist perception result to any temp directory")

# Test function
def test_connection():
    """Test the connection to the webhook endpoint."""
    logger.info("Testing connection to perception layer webhook")
    result = send_research_request("test query for connection validation")

    if result["success"]:
        logger.info("PERCEPTION LAYER: Connection test successful")
    else:
        logger.error("PERCEPTION LAYER: Connection test failed")

    return result

# Initialize
logger.info("PERCEPTION LAYER initialized")

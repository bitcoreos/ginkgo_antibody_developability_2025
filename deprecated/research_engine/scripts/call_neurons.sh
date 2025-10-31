#!/bin/bash
set -euo pipefail

RESEARCH_TOPIC=${RESEARCH_TOPIC:-"general biology concepts"}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a specialized biology research agent. Your role is to analyze biological concepts, identify key principles, and provide comprehensive explanations."}

exec python -m workspace.research_engine.cli \
    --topic "${RESEARCH_TOPIC}" \
    --system-prompt "${SYSTEM_PROMPT}" \
    "$@"


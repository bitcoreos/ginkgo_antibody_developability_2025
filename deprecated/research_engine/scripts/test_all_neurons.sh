#!/bin/bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-"data/biology/antibodies/websearch"}
mkdir -p "${OUTPUT_DIR}"

python -m workspace.research_engine.cli \
    --topic "antibodies" \
    --system-prompt "You are a biology websearcher." \
    --output-dir "${OUTPUT_DIR}" \
    --dry-run \
    "$@"

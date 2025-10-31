# Research Engine

For standardized terminology definitions, see [Terminology Glossary](../../docs/terminology_glossary.md).

# Research Engine

> **Status (2025-10-14): Deprecated.**
> All recent runs returned hallucinated, non-biological outputs due to upstream API instability. Keep the codebase for reference, but do not trigger new jobs until a trusted replacement service is vetted.

## Purpose

The research_engine is a script-based orchestration system that manages asynchronous calls to the neural cluster. It enables parallel processing of research tasks across multiple AI models via webhook endpoints.

## Architecture

- **Input**: System prompt and research topic
- **Processing**: Asynchronous dispatch to 6 neuron webhooks
- **Output**: Distributed research results in designated output directories
- **Flow**: `User → research_engine → 6 Neurons → research_outputs/`

## API Specification

| Parameter | Type | Required | Description |
|---------|------|---------|-------------|
| `systemprompt` | string | Yes | Role definition for AI model |
| `researchTopic` | string | Yes | Research topic or query |

## Neuron Endpoints

| Model | Webhook URL | Purpose |
|------|-----------|---------|
| llama-3.2-3b | `https://n8n.bitwiki.org/webhook/429d66fc-10f5-4f43-8c61-9f0af1fdde0f` | General biology research |
| qwen3-4b | `https://n8n.bitwiki.org/webhook/f29c348a-d66d-41b6-9571-7bdcf61bc134` | Specialized analysis |
| llama-3.3-70b | `https://n8n.bitwiki.org/webhook/2208dcde-7716-4bd5-8910-bae78edc1931` | Deep reasoning |
| mistral-31-24b | `https://n8n.bitwiki.org/webhook/304264b7-4128-470e-b7c5-e549aa2b62b5` | Pattern recognition |
| venice-uncensored | `https://n8n.bitwiki.org/webhook/05720333-8ec6-408d-9531-d5cb5eb87993` | Unrestricted analysis |
| qwen3-235b | `https://n8n.bitwiki.org/webhook/0b10e6c5-51a5-4d61-99bc-26409bc2c7a8` | Advanced reasoning |

## Usage Examples

```bash
# Run with custom parameters
RESEARCH_TOPIC="antibody stability" SYSTEM_PROMPT="You are a protein stability expert..." ./scripts/call_neurons.sh
```

## Directory Structure

```
research_engine/
├── scripts/          # Orchestration scripts
├── config/           # Neuron configurations
├── data/             # Research parameters and search terms
├── docs/             # Architecture and CI/CD documentation
└── README.md         # This document
```

## Maintenance

- Update neuron configurations in config/ when webhook URLs change
- Add new search terms to scholarly_search_terms.txt as needed
- Update CI/CD pipeline in docs/semantic_mesh_ci.yml for new requirements

# Neural Cluster System Architecture

## Overview

A distributed asynchronous neural cluster system with six specialized AI models processing research tasks in parallel.

## Neuron Configuration

| Neuron | Model | Webhook URL | Status |
|--------|-------|-------------|--------|
| Neuron_llama-3.2-3b | llama-3.2-3b | https://n8n.bitwiki.org/webhook/429d66fc-10f5-4f43-8c61-9f0af1fdde0f | Active |
| Neuron_qwen3-4b | qwen3-4b | https://n8n.bitwiki.org/webhook/f29c348a-d66d-41b6-9571-7bdcf61bc134 | Active |
| Neuron_llama-3.3-70b | llama-3.3-70b | https://n8n.bitwiki.org/webhook/2208dcde-7716-4bd5-8910-bae78edc1931 | Error-prone |
| Neuron_mistral-31-24b | mistral-31-24b | https://n8n.bitwiki.org/webhook/304264b7-4128-470e-b7c5-e549aa2b62b5 | Active |
| Neuron_venice-uncensored | venice-uncensored | https://n8n.bitwiki.org/webhook/05720333-8ec6-408d-9531-d5cb5eb87993 | Active |
| Neuron_qwen3-235b | qwen3-235b | https://n8n.bitwiki.org/webhook/0b10e6c5-51a5-4d61-99bc-26409bc2c7a8 | Active |

## System Design Principles

1. **Redundancy**: Multiple neurons ensure results even if some fail
2. **Asynchronous Processing**: Workflows run independently without blocking
3. **Scope Sensitivity**: Research topics must be narrowly focused
4. **Error Tolerance**: Some API failures are expected in test environment
5. **Fast Data Retrieval**: Prompts should enable quick information access

## Testing Protocol

1. Use simple, focused research questions
2. Dispatch requests asynchronously
3. Do not wait for responses
4. Collect results later
5. Compare outputs across neurons

## Known Issues

- Neuron_llama-3.3-70b frequently returns errors
- JSON path references need correction: change `$json.researchTopic` to `$json["Initialize Parameters"].json.researchTopic`
- Gateway timeouts occur with broad research scopes
- Some nodes in workflows may not function correctly

## Best Practices

- Use `researchterm` instead of `researchTopic` in payloads
- Keep system prompts simple: "You are a biology websearcher"
- Ask for specific, concise answers
- Allow sufficient processing time (10+ minutes)
- Expect some failures in test environment

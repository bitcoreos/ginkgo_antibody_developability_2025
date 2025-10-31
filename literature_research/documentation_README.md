# Documentation Hub

## Purpose
- Centralize ownership for modular READMEs without collapsing context.
- Expose a live manifest pointing to the definitive documentation for each subsystem.
- Surface hygiene signals so duplication and drift are corrected before harming execution.

## Directory Manifest

| Path | Primary README | Scope | Steward | Validation Hooks |
|------|----------------|-------|---------|------------------|
| workspace/README.md | workspace/README.md | Root layout and directory duties | CTO (agent) | `tree` diff vs `workspace/sonar_map.txt` |
| workspace/plans | workspace/plans/README.md | Strategy, tactics, and operational plans | Planning steward | Weekly timestamp audit of `plans/` artifacts |
| workspace/semantic_mesh | workspace/semantic_mesh/README.md | Knowledge lattice, ontology assets, validation logs | Mesh steward | `semantic_mesh/validation_report_*.md` review |
| workspace/bitcore_research | workspace/bitcore_research/README.md | Cognition layers (perception → reflection) | Research steward | Layer unit tests under `tests/` |
| workspace/research_engine | workspace/research_engine/README.md | Swarm orchestration runtime | Runtime steward | `tests/test_research_engine.py` |
| workspace/data | workspace/data/README.md | Datasets, provenance, manifests | Data steward | `data/MANIFEST.yaml` checksum scan |
| workspace/ai_finetuning | workspace/ai_finetuning/README.md | Embedding + fine-tuning tracks | Model steward | README task board burn-down |
| workspace/bioinformatics | workspace/bioinformatics/README.md | Feature extraction modules + notebooks | Bioinformatics steward | `bioinformatics/tests/` suite |
| deprecated/research_engine/raw_outputs | deprecated/research_engine/raw_outputs/README.md | Model/week-scoped research output staging | Runtime steward | Promotion ledger → `semantic_mesh/` |
| workspace/responses | workspace/responses/README.md | Raw model payload responses | Runtime steward | Payload checksum spot checks |

> Maintain this ledger as directories evolve; it is the canonical ownership map.

## Hygiene Protocol
- Keep every directory README self-contained: scope, structure, steward, validation entry points.
- Avoid copy/paste between READMEs; consolidate shared context here instead.
- Onboarding a new module requires adding its row, steward, and validation hooks.
- Run a README hash audit (planned automation) each Friday to flag clones for rewrite.
- Mirror high-impact updates to `init.md` and `bitcore.framework.md` so global context stays aligned.

## Active Remediation Tasks
1. Automate manifest generation for this table and `data/MANIFEST.yaml` (hash + steward sync).
2. Relocate historical README variants (README_*.md) into an `archive/` folder with provenance notes.
3. Extend CI to include documentation hash drift checks alongside semantic mesh validators.

## Notes
- Directory READMEs remain the context injection surface for agents; do not collapse them into pointers.
- Use this hub before refactors, during audits, and when onboarding new contributors.# DOCUMENTATION

## Consolidated README Content


### README_2.md
# Neuron Cluster n8n System

## Project Overview

A distributed asynchronous neural cluster system that leverages six specialized AI models to process research tasks in parallel. The system transforms individual AI models into specialized neurons, enabling enhanced research capabilities through redundancy and parallel processing.

## Architecture

- **Neurons**: 6 specialized AI models (llama-3.2-3b, qwen3-4b, llama-3.3-70b, mistral-31-24b, venice-uncensored, qwen3-235b)
- **Processing**: Asynchronous, non-blocking execution
- **Redundancy**: Multiple neurons ensure results even if some fail
- **Orchestration**: n8n workflows with webhook triggers

## Directory Structure

```
neuron_cluster/
├── README.md            # Project overview
├── ARCHITECTURE.md      # System architecture details
├── docs/               # Detailed documentation
├── workflows/          # n8n workflow exports and descriptions
├── tests/              # Test scripts and results
├── data/               # Research data and outputs
└── scripts/            # Utility scripts for system management
```

## Status

- All six neuron workflows are active
- Testing in progress with antibody research topic
- Results will be collected asynchronously


### README_3.md
# Antibody Competition Research Library

A comprehensive research system for antibody competition analysis using the Morpheus AI swarm and n8n workflows.

Created: 2025-10-06 06:56:00

## Project Overview

This research library is designed to systematically investigate hidden features in antibody sequences using Hidden Markov Models and a distributed AI swarm architecture. The system leverages multiple specialized models working in parallel through n8n workflows.

## Directory Structure

- **workflows**: Contains all n8n workflow definitions
- **neural_clusters**: Contains neural cluster configurations and models
- **documentation**: Research documentation, plans, and reports
- **research_outputs**: Raw and processed research results (now located in deprecated/research_engine/raw_outputs/)
- **scripts**: Automation scripts for research workflows

## Research Objectives

1. Identify hidden features in antibody sequences that influence developability
2. Analyze competition dynamics between different antibody variants
3. Develop predictive models for antibody performance

## Execution Strategy

- Phased research execution to prevent system overload
- Staggered API calls with 5-10 minute intervals between batches
- Continuous monitoring of system performance

## Status

- Directory structure created
- Documentation initialized
- Endpoint validation in progress

## Next Steps

1. Restore neural cluster workflow files from backup or recreate
2. Activate workflows on n8n server
3. Begin phased research execution

## Contact

Agent Zero - Primary Research Orchestrator


### README_4.md
# Antibody Research n8n Workflows

Created: 2025-10-06 06:57:24

## Project Overview

This repository contains n8n workflows for the antibody competition research system. The workflows orchestrate a distributed AI swarm using multiple specialized models to analyze hidden features in antibody sequences.

## Directory Structure

- **mvp**: Minimum Viable Product workflows - currently active research workflows
- **experiments**: Experimental workflows for testing new research approaches
- **templates**: Template workflows for creating new research workflows
- **documentation**: Detailed documentation for workflows and research methodology
- **archived**: Archived workflows no longer in active use

## Workflow Files

The following neural cluster workflows are available:

- neuron_llama-3.3-70b.json: Research using Llama 3.3 70B model
- neuron_qwen3-4b.json: Research using Qwen3 4B model
- neuron_llama-3.2-3b.json: Research using Llama 3.2 3B model
- neuron_mistral-31-24b.json: Research using Mistral 31 24B model
- neuron_venice-uncensored.json: Research using Venice uncensored model
- neuron_qwen3-235b.json: Research using Qwen3 235B model

## Setup Instructions

1. Ensure you have access to the external n8n instance at https://n8n.bitwiki.org
2. Never install or run n8n locally - always use the external instance
3. Import workflows from the mvp directory into the n8n instance
4. Activate each workflow to enable its webhook endpoint
5. Test each endpoint before conducting research

## Research Execution

To conduct research:

1. Send a POST request to the appropriate webhook endpoint
2. Include a JSON payload with a "researchTopic" field
3. Wait for the response containing the research results
4. Process and analyze the results

Example:
```bash\ncurl -X POST https://n8n.bitwiki.org/webhook/neuron/llama-3.3-70b \\n  -H "Content-Type: application/json" \\n  -d '{"researchTopic": "hidden features in antibody sequences"}'
```

## Troubleshooting

### Webhook Returns 404
- Ensure the workflow is imported and activated in the n8n instance
- Verify the webhook path matches the workflow configuration
- Check that the n8n instance is running and accessible

### Workflow Fails to Process
- Verify the model service is available
- Check the payload format - must include "researchTopic" field
- Review n8n logs for error details

### Slow Response
- The research process may take several minutes depending on complexity
- Implement staggered calls to avoid overwhelming the system
- Monitor system performance during research execution

## Maintenance

- Regularly backup workflow files
- Document any changes to workflows
- Test workflows after any modifications
- Keep the archived directory organized

## Contact

Agent Zero - Research Orchestrator


### README_5.md
# Antibody Research n8n Workflows

Created: 2025-10-06 06:57:24

## Project Overview

This repository contains n8n workflows for the antibody competition research system. The workflows orchestrate a distributed AI swarm using multiple specialized models to analyze hidden features in antibody sequences.

## Directory Structure

- **mvp**: Minimum Viable Product workflows - currently active research workflows
- **experiments**: Experimental workflows for testing new research approaches
- **templates**: Template workflows for creating new research workflows
- **documentation**: Detailed documentation for workflows and research methodology
- **archived**: Archived workflows no longer in active use

## Workflow Files

The following neural cluster workflows are available:

- neuron_llama-3.3-70b.json: Research using Llama 3.3 70B model
- neuron_qwen3-4b.json: Research using Qwen3 4B model
- neuron_llama-3.2-3b.json: Research using Llama 3.2 3B model
- neuron_mistral-31-24b.json: Research using Mistral 31 24B model
- neuron_venice-uncensored.json: Research using Venice uncensored model
- neuron_qwen3-235b.json: Research using Qwen3 235B model

## Setup Instructions

1. Ensure you have access to the external n8n instance at https://n8n.bitwiki.org
2. Never install or run n8n locally - always use the external instance
3. Import workflows from the mvp directory into the n8n instance
4. Activate each workflow to enable its webhook endpoint
5. Test each endpoint before conducting research

## Research Execution

To conduct research:

1. Send a POST request to the appropriate webhook endpoint
2. Include a JSON payload with a "researchTopic" field
3. Wait for the response containing the research results
4. Process and analyze the results

Example:
```bash\ncurl -X POST https://n8n.bitwiki.org/webhook/neuron/llama-3.3-70b \\n  -H "Content-Type: application/json" \\n  -d '{"researchTopic": "hidden features in antibody sequences"}'
```

## Troubleshooting

### Webhook Returns 404
- Ensure the workflow is imported and activated in the n8n instance
- Verify the webhook path matches the workflow configuration
- Check that the n8n instance is running and accessible

### Workflow Fails to Process
- Verify the model service is available
- Check the payload format - must include "researchTopic" field
- Review n8n logs for error details

### Slow Response
- The research process may take several minutes depending on complexity
- Implement staggered calls to avoid overwhelming the system
- Monitor system performance during research execution

## Maintenance

- Regularly backup workflow files
- Document any changes to workflows
- Test workflows after any modifications
- Keep the archived directory organized

## Contact

Agent Zero - Research Orchestrator


### README_6.md
# Antibody Research Scripts

Automation scripts for data processing, API calls, and report generation

Created: 2025-10-06 06:56:00

## Contents

This directory contains files and subdirectories related to automation scripts for data processing, api calls, and report generation.

## Usage

- Place all scripts files in this directory
- Follow the naming convention: lowercase with hyphens, no spaces
- Document any changes in the README

## Dependencies

- None specific to this directory

## Troubleshooting

- If files are missing, check the backup directory
- Verify file permissions if access is denied


### README_7.md
# Antibody Research Research Outputs

Raw and processed research results, summaries, and reports

Created: 2025-10-06 06:56:00

## Contents

This directory contains files and subdirectories related to raw and processed research results, summaries, and reports.

## Usage

- Place all research outputs files in this directory
- Follow the naming convention: lowercase with hyphens, no spaces
- Document any changes in the README

## Dependencies

- None specific to this directory

## Troubleshooting

- If files are missing, check the backup directory
- Verify file permissions if access is denied


### README_8.md
# Antibody Research Documentation

Research documentation, plans, reports, and meeting notes

Created: 2025-10-06 06:56:00

## Contents

This directory contains files and subdirectories related to research documentation, plans, reports, and meeting notes.

## Usage

- Place all documentation files in this directory
- Follow the naming convention: lowercase with hyphens, no spaces
- Document any changes in the README

## Dependencies

- None specific to this directory

## Troubleshooting

- If files are missing, check the backup directory
- Verify file permissions if access is denied


### readme.md
# refactor when needed to enhance redability.

this folder contains important information about the competion. however it is framented into various runs.

need to reorganize to better query and fit our needs. ask the ceo dev pi for information .


agent-zero is the coy of the repo in bitcoreos



---


after refactoring the structure, consolidating and optimizing logic.

-  view all uique items and everything is placed and talk to human.


### WORKSPACE_LAYOUT
- **BITCORE**: /bitcore
- **WORKSPACE**: /bitcore/workspace
- **SANDBOX**: /bitcore/sandbox
- **LOCAL_GITHUB_FOR_COMPETITION_FROM_BITCOREOS**: /bitcore/competition/agent-zero # this repo is at https://github.com/bitcoreos/agent-zero
- **MEMORY_STORE**: /a0/tmp/memories
- **LOGS**: /a0/logs
- **KNOWLEDGE_BASE**: /a0/knowledge


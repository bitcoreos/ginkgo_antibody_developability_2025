# Research Outputs

> **Status (2025-10-14): Raw responses are quarantined.**
> Keep the structure for provenance, but do not consume `raw_responses/` artifacts—they are confirmed hallucinations from failed API calls.

## Purpose

The research_outputs directory contains all research findings from the neural cluster. This is the raw output of our research swarm, organized for easy comparison and analysis.

## Directory Structure

```
research_outputs/
├── model/
│   ├── llama-3.2-3b/
│   │   └── week_40/
│   ├── llama-3.3-70b/
│   │   └── week_40/
│   ├── mistral-31-24b/
│   │   └── week_40/
│   ├── qwen3-4b/
│   │   └── week_40/
│   ├── qwen3-235b/
│   │   └── week_40/
│   └── venice-uncensored/
│       └── week_40/
├── metadata.json
└── README.md
```

## Organization Logic

1. **Model-First Hierarchy**:
   - Primary organization by AI model
   - Enables direct comparison of model performance
   - Matches research_engine configuration structure

2. **Weekly Buckets**:
   - Uses ISO week numbers (current: week_40)
   - Aligns with sprint cycles
   - Simplifies time-based analysis

## Quality Levels

| Directory | Quality Level | Preservation |
|---------|--------------|-------------|
| model/llama-3.3-70b/week_40/ | Preliminary | Temporary |
| model/other_models/week_40/ | Failed | Temporary |

## Current Status

- **Active Week**: 2025, week 40
- **Successful Models**: llama-3.3-70b (1/6)
- **Failed Models**: qwen3-4b, venice-uncensored, mistral-3124b, llama-3.2-3b, qwen-3-235b
- **Failure Reason**: API connectivity issues and database constraints

## Maintenance

1. **New Research Runs**:
   - Create new week directory (e.g., week_41) for each research cycle
   - Populate with output from all models

2. **Validation Process**:
   - Preliminary findings must be re-run with fixed infrastructure
   - Only validated findings are integrated into the semantic mesh source of truth

3. **Archiving**:
   - After final validation, move successful findings to semantic mesh source of truth
   - Archive research_outputs by week after competition

This structure ensures clear organization of research outputs while maintaining separation from the semantic mesh source of truth.
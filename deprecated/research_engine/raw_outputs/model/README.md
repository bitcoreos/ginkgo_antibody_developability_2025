# Model Research Outputs

## Purpose

This directory contains research outputs organized by AI model and week. Each model has its own directory to enable direct comparison of performance across different models.

## Directory Structure

```
model/
├── llama-3.2-3b/
│   └── week_40/
├── llama-3.3-70b/
│   └── week_40/
│       └── semantic_mesh_preliminary_20251006_051208.json
├── mistral-31-24b/
│   └── week_40/
├── qwen3-235b/
│   └── week_40/
├── qwen3-4b/
│   └── week_40/
└── venice-uncensored/
    └── week_40/
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

## Current Status

- **Active Week**: 2025, week 40
- **Successful Models**: llama-3.3-70b (1/6)
- **Failed Models**: qwen3-4b, venice-uncensored, mistral-3124b, llama-3.2-3b, qwen-3-235b
- **Failure Reason**: API connectivity issues and database constraints

This structure ensures clear organization of research outputs while maintaining separation from the semantic mesh source of truth.
# AGENT ZERO COMPETITION PLAN

## PROJECT PURPOSE
Win the Ginkgo Bio 2025 AbDev competition by creating an antibody model using the BITCORE framework based on fractal mathematics and hyperbolic geometry.


## ARCHITECTURAL UPDATE - 2025-10-08

### Current Flaw
- Morpheus AI agents (https://api.mor.org/api/v1/chat/completions) cannot access CORE API data
- Reasoning and reflection layers send queries to external agents without access to primary research data
- Creates broken validation loop - agents reason about data they cannot see

### Proposed Fix
1. a0 conducts research using CORE API with full data access
2. a0 generates validated findings and structured prompts
3. a0 sends prompts to Morpheus agents for corroboration
4. Unidirectional flow ensures data security while enabling external validation

### Structural Changes
- Renamed `/a0/bitcore/workspace/scripts/` to `/a0/bitcore/workspace/bitcore_research/`
- New directory contains Four-Layer Cognition Engine (perception, reasoning, action, reflection)
- Accurate naming reflects system's function as core research orchestrator
```

## CURRENT STATE
- All neural cluster files consolidated into /a0/workspace/neural_cluster/
- Neural cluster system with 6 neurons: llama-3.2-3b, qwen3-4b, llama-3.3-70b, mistral-31-24b, venice-uncensored, qwen3-235b
- System requires asynchronous handling, monitoring scripts, and result aggregation
- Previous attempts failed due to webhook configuration issues and payload structure errors
- Semantic mesh exists in /a0/bitcore/agent-zero/semantic_mesh/
- Competition deadline: November 1, 2025

## KEY CHALLENGES
1. Webhook configuration - all webhooks must accept POST requests
2. Payload structure - must use 'researchTopic' field name, not 'researchterm'
3. Workspace organization - must use /a0/workspace, not ad-hoc directories
4. Proper planning - must notarize plans before execution

## NEXT STEPS
1. Verify all neuron webhook URLs and configurations
2. Fix payload structure to use correct field names
3. Implement staggered request timing to prevent server overload
4. Create validation scripts to check response quality
5. Build comprehensive semantic mesh from converged results
6. Develop predictive models for antibody developability metrics

## PRINCIPLES
- Always notarize plans before execution
- Use proper workspace layout (/a0/workspace, /a0/sandbox)
- Validate all assumptions before acting
- Document all decisions and changes
- Prioritize reliability over speed

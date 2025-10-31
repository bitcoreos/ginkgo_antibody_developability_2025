# Feature Evidence Map (2025-10-14)

Purpose: Tie priority developability feature workstreams to the strongest vetted citations so engineering, modeling, and QA stay aligned on evidence-backed deliverables.

## Quick Legend
- **Feature Track**: Planned workstream or module in feature roadmap.
- **Primary Citations**: Key references from `bitcore_research/citation/` supporting the approach.
- **Evidence Highlights**: What the papers actually deliver that we rely on (metrics, methods, datasets).
- **Usage Notes**: How we plan to apply the evidence or any cautions.

## Feature-to-Citation Matrix

| Feature Track | Primary Citations | Evidence Highlights | Usage Notes |
| --- | --- | --- | --- |
| **CDR & Sequence-Derived Features** | `citation_cdr_analysis_1760067088.md`, `citation_cdr_analysis_1760106901.md`, `citation_vhh_research_1760017144.md` | Benchmarks for inverse folding, CDR-H3 flexibility motifs, nanobody CDR3 length trends; curated VHH sequence databases | Drive CDR feature engineering sprint; borrow test splits/metrics for regression baselines; ensure nanobody edge cases stay in scope |
| **Information-Theoretic / Mutual Information Signals** | `citation_information_theoretic_approach_1760204876.md`, `citation_dynamic_fitness_landscape_1760199292.md` | Formal MI framework, epistasis analysis, channel capacity framing for FLAb datasets | Reuse MI estimation recipes; constrain feature selection to high-MI residue pairs; document theoretical rationale in model cards |
| **Aggregation Propensity & Hydrophobic Risk** | `citation_aggregation_propensity_1760318967.md`, `citation_ai_ml_prediction_1759987335.md` | Demonstrated ML pipelines for aggregation with r=0.91 using surface curvature + electrostatics; surveys of developability ML benchmarks | Anchor surface-charge feature set; cross-check evaluation metrics before reproducing; highlight curvature descriptors as differentiator |
| **Thermostability & Tm2 Prediction** | `citation_thermal_stability_1760149915.md`, `citation_multidimensional_modeling_1760364146.md` | Sequence+structure ML achieving Spearman 0.4-0.52; MD-informed AbMelt models with R² 0.57-0.60; embedding-based nanobody Tm estimator | Justify hybrid sequence/structure head; borrow AbMelt MD-derived descriptors; extend to VHH via TEMPRO references |
| **Temporal Dynamics / Neural-ODE Channel** | `citation_antibody_temporal_dynamics_1760365867.md`, `citation_adcnet_kinetics_1760227735.md` | AbODE conjoined ODE design paradigm; ADCnet Neural-ODE kinetics for payload release with experimental validation | Build roadmap for dynamic developability predictor; cite when scoping temporal feature backlog; flag need for time-series labels |
| **Uncertainty-Aware Structural Features** | `citation_uncertainty_aware_architecture_1760366138.md`, `citation_multidimensional_modeling_1760364894.md` | ImmuneBuilder ensemble variance for confidence estimates; multimodal fusion recipes incorporating uncertainty weighting | Integrate ABodyBuilder2 uncertainty into feature pipeline; document calibration targets; share with validation for risk-aware thresholds |
| **High-Throughput Assay Alignment** | `citation_high_throughput_platforms_1760017345.md`, `citation_validation_methods_1760365565.md` | Enumerates assay stack (HIC, AC-SINS, DLS) and validation pipelines; provides wet-lab KPIs and cross-validation procedures | Map competition metrics to wet-lab corollaries; build QA checklist referencing assay confidence intervals |
| **Productization & Competitive Context** | `citation_abdev_platform_1759985414.md`, `citation_prophet_ab_platform_1759987142.md`, `citation_revoab_product_1759987090.md` | Market positioning for competitor platforms; technical deep dives into PROPHET-Ab and RevoAb capabilities | Use in feature priority discussions; reference when writing differentiators and customer-facing collateral |

## Outstanding Evidence Gaps
- `citation_information_theoretic_approach_1760204859.md` remains a stub—either populate from the richer MI packs or retire to avoid dangling references.
- Need empirical sources for codon-usage and expression yield features (no clean citations yet).
- Lack wet-lab confirmation for uncertainty calibration thresholds; flag for future partnership outreach.

## Next Actions
1. Sync this matrix with `operational_BACKLOG.md` so each critical-path task cites supporting literature.
2. When drafting model specs or experiment briefs, reference the "Primary Citations" column verbatim to keep traceability.
3. Update the matrix as new citation packs land or when evidence is invalidated.

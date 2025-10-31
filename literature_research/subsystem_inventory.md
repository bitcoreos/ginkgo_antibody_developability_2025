# Workspace Inventory Log

updated_utc: 2025-10-14T00:00:00Z (initial draft)
owner: modeling_lead
purpose: Track which artefacts inside high-noise directories have been read and what they contain.

## Disposition Summary
- **Keep & use immediately**: `bioinformatics/`, `bitcore_research/citation/` (authoritative code/tests and vetted literature).
- **Keep as structural references**: `research_engine/output/` (schema examples only).
- **Quarantine / archive-ready**: `research_engine/raw_results/`, `deprecated/research_engine/raw_outputs/raw_responses/` (corrupted hallucinations—retain solely for QA notes, no model reuse).

## bioinformatics/
| File | Status | Notes |
| --- | --- | --- |
| README.md | Reviewed | Large status log: outlines three strategic advantages, exhaustive pipeline todo list, and data provenance notes for GDPa1 preprocessing. |
| COMPETITION_ROADMAP.md | Reviewed | Phase roadmap for AbDev competition covering CV/test prediction pipelines, upcoming information-theoretic and LM work. |
| __init__.py | Reviewed | Package marker. |
| debug_scoring_analysis.py | Reviewed | Debug script printing isotype prediction scores for test sequences (newline escaping is broken). |
| debug_sequence_analysis.py | Reviewed | Debug script dumping sequence features/motifs for sample IgG/IgA/... sequences (newline escaping is broken). |
| processing_log.json | Reviewed | JSON audit trail of preprocessing steps, hashes, and outputs for GDPa1 baseline pipeline. |
| modules/__init__.py | Reviewed | Package marker. |
| modules/cdr_extraction.py | Reviewed | Implements AHO-numbered CDR extraction with validation and logging. |
| modules/isotype_modeling.py | Reviewed | Scores sequences against isotype-specific motifs; returns prediction dataclass. |
| modules/mutual_information.py | Reviewed | Mutual information utilities (positional, joint, whole-matrix) with pseudo-count support. |
| tests/__init__.py | Reviewed | Package marker. |
| tests/test_cdr_extraction.py | Reviewed | Unit tests covering happy path, validation, logging for CDR extraction. |
| tests/test_isotype_modeling.py | Reviewed | Unit tests for isotype scoring/prediction including batch helper. |
| tests/test_mutual_information.py | Reviewed | Unit tests for MI helpers (correlated vs uncorrelated sequences, error cases). |

## bitcore_research/citation/
| File | Status | Notes |
| --- | --- | --- |
| .completed_1760199292 | Reviewed | Value: marks dynamic fitness landscape citation pack finished; Implementation: use as checkpoint before opening new swarm tasks. |
| .completed_1760365565 | Reviewed | Value: confirms validation-methods batch shipped; Implementation: reference to avoid duplicate citation requests. |
| .completed_1760365699 | Reviewed | Value: duplicate completion flag; Implementation: retain for provenance but no action. |
| citation_abdev_platform_1759985414.md | Reviewed | Value: curates PROPHET-Ab positioning quotes; Implementation: feed into partner briefings and marketing collateral. |
| citation_ac_sins_assay_1759986712.md | Reviewed | Value: AC-SINS assay exemplars; Implementation: justify including AC-SINS in lab validation menu. |
| citation_adc_pharmacokinetics_1760317829.md | Reviewed | Value: succinct ADC PK review; Implementation: ground pharmacokinetic Priors for Neural-ODE channel. |
| citation_adcnet_kinetics_1760227735.md | Reviewed | Value: documents ADCnet Neural-ODE framework; Implementation: adapt architecture baselines for temporal module. |
| citation_aggregation_propensity_1760318967.md | Reviewed | Value: aggregation ML survey; Implementation: extract feature sets for hydrophobicity risk head. |
| citation_ai_ml_prediction_1759987335.md | Reviewed | Value: overview of ML developability frameworks (PROPERMAB, SOLart); Implementation: benchmark against existing AUROC figures. |
| citation_antibody_research_1759984612.md | Reviewed | Value: broad AI+wet lab integration references; Implementation: cite in exec summary to position BITCORE tech stack. |
| citation_antibody_research_1760016409.md | Reviewed | Value: 2025 antibody landscape intel; Implementation: mine for competitive differentiators in pitch decks. |
| citation_antibody_research_1760227630.md | Reviewed | Value: AlphaFold3 + ADC structural modeling coverage; Implementation: justify structural inference roadmap. |
| citation_antibody_research_1760247201.md | Reviewed | Value: ImmuScope T-cell epitope ML method; Implementation: borrow epitope scoring features for liability checks. |
| citation_antibody_temporal_dynamics_1760365867.md | Reviewed | Value: AbODE + ADCnet temporal dynamics; Implementation: replicate Neural-ODE training recipe. |
| citation_cdr_analysis_1760067088.md | Reviewed | Value: CDR inverse folding benchmarks; Implementation: select baseline models for CDR redesign loop. |
| citation_cdr_analysis_1760067123.md | Reviewed | Value: expanded docking/design corpus; Implementation: enrich retrieval set for design agent. |
| citation_cdr_analysis_1760067158.md | Reviewed | Value: zero-shot CDR design exemplars; Implementation: adapt Chai-2 prompt templates. |
| citation_cdr_analysis_1760106901.md | Reviewed | Value: CDR-H3 flexibility motifs; Implementation: derive structural penalties for loop overextension. |
| citation_consortium_1759985442.md | Reviewed | Value: Ginkgo+Apheris consortium overview; Implementation: context for federated pitch messaging. |
| citation_consortium_competition_1759986740.md | Reviewed | Value: ties consortium to AbDev competition; Implementation: inform competition strategy narrative. |
| citation_consortium_enrollment_1759986072.md | Reviewed | Value: enrollment workflows; Implementation: model partnership onboarding steps. |
| citation_consortium_enrollment_1759986850.md | Reviewed | Value: federated learning enrollment detail; Implementation: reuse governance language in templates. |
| citation_consortium_latest_1759986128.md | Reviewed | Value: consortium + RevoAb messaging (dup); Implementation: tag as duplicate in knowledge graph. |
| citation_consortium_latest_1759986184.md | Reviewed | Value: similar consortium narrative; Implementation: same as above. |
| citation_consortium_latest_1759986208.md | Reviewed | Value: consortium AI tooling references; Implementation: mine tooling list for integration targets. |
| citation_consortium_latest_1759986230.md | Reviewed | Value: duplicate consortium AI tooling; Implementation: maintain for time-stamped provenance. |
| citation_dynamic_fitness_landscape_1760199292.md | Reviewed | Value: FLAb benchmark + epistasis papers; Implementation: use dataset to stress-test fitness channel. |
| citation_federated_infrastructure_1759986264.md | Reviewed | Value: Apheris infra write-up; Implementation: align security posture statements. |
| citation_gdp1_dataset_1759986294.md | Reviewed | Value: GDPa1 launch summary; Implementation: cite when framing dataset ingestion. |
| citation_gdp1_dataset_1759986423.md | Reviewed | Value: duplicate GDPa1 context; Implementation: mark redundant but keep for traceability. |
| citation_gdp1_dataset_1759986680.md | Reviewed | Value: GDPa1 specifics (246 antibodies, 9 properties); Implementation: schedule Hugging Face pull into feature store. |
| citation_high_throughput_platforms_1760017345.md | Reviewed | Value: high-throughput assay stack references; Implementation: adapt platform KPIs for roadmap. |
| citation_index.md | Reviewed | Value: compact map of recent citation batches; Implementation: drive DOI lookup automation. |
| citation_index.md.backup | Reviewed | Value: legacy index snapshot; Implementation: diff vs current to catch dropped topics. |
| citation_information_theoretic_approach_1760204859.md | Reviewed | Value: draft stub with no sources yet; Implementation: flag for completion in next swarm. |
| citation_information_theoretic_approach_1760204876.md | Reviewed | Value: core info-theory references; Implementation: translate into mutual-information feature tests. |
| citation_information_theoretic_approach_1760208099.md | Reviewed | Value: alternate info-theory bibliography; Implementation: compare to 4876 for coverage gaps. |
| citation_information_theoretic_approach_1760284661.md | Reviewed | Value: extended info-theory/biology set; Implementation: underpin communication-channel modeling spec. |
| citation_multidimensional_modeling_1760364146.md | Reviewed | Value: AlphaFold3 + Neural-ODE anchor citations; Implementation: cite in multidimensional design doc. |
| citation_multidimensional_modeling_1760364894.md | Reviewed | Value: multi-modal fusion refs incl. FLAb; Implementation: blueprint for sequence-structure-temporal fusion. |
| citation_nextgen_ai_frameworks_1760009783.md | Reviewed | Value: next-gen AI platform survey; Implementation: prioritize IgBert/IgT5 fine-tuning tasks. |
| citation_nextgen_ai_frameworks_1760009850.md | Reviewed | Value: duplicate of 9783 for consistency; Implementation: keep until dedupe pass. |
| citation_product_offering_1759986322.md | Reviewed | Value: Ginkgo commercial data offering; Implementation: align sales collateral with features. |
| citation_prophet_ab_platform_1759987142.md | Reviewed | Value: PROPHET-Ab technical deep dive; Implementation: mine methods for assay emulation. |
| citation_revoab_product_1759987090.md | Reviewed | Value: RevoAb launch narrative; Implementation: craft customer-facing FAQ. |
| citation_revoab_product_1759987164.md | Reviewed | Value: duplicate RevoAb narrative; Implementation: keep for metadata parity. |
| citation_thermal_stability_1760149915.md | Reviewed | Value: thermostability modeling papers; Implementation: port features into Tm prediction head. |
| citation_uncertainty_aware_architecture_1760366138.md | Reviewed | Value: uncertainty-aware hybrid architecture case; Implementation: integrate ABodyBuilder2 confidence weights. |
| citation_validation_methods_1760365565.md | Reviewed | Value: multidimensional validation pipeline; Implementation: copy assay roster into test plan. |
| citation_validation_methods_1760365699.md | Reviewed | Value: complementary validation stack; Implementation: reference when planning wet-lab confirmation. |
| citation_vhh_research_1760017144.md | Reviewed | Value: nanobody developability insights; Implementation: expand VHH-specific feature library. |
| current_citations.txt | Reviewed | Value: latest generated citation list; Implementation: monitor for missing ingest. |
| research_log.md | Reviewed | Value: task chronology + next research leads; Implementation: queue ST-GNN scouting task. |

## research_engine/output/
| File | Status | Notes |
| --- | --- | --- |
| research_engine_results_20251011_165747.json | Reviewed | Value: dry-run metadata for hydrophobicity HIC pass; Implementation: use as template for configuring neuron roster in future live runs. |
| research_engine_results_20251011_170255.json | Reviewed | Value: dry-run polyreactivity PSP scaffolding; Implementation: reuse topic definitions when re-running with real inference. |
| research_engine_results_20251011_170847.json | Reviewed | Value: dry-run literature review shell for salt-retention topic; Implementation: adjust prompts before enabling non-simulated responses. |
| research_engine_results_20251011_172533.json | Reviewed | Value: full multi-neuron output (concept mapper, investigator, evaluator, etc.) for salt retention study; Implementation: harvest agent role outputs to build downstream summarizers/checkers. |

## research_engine/raw_results/
| File | Status | Notes |
| --- | --- | --- |
| Expression_Yield_Manufacturability/research_engine_results_20251011_022129.json | Reviewed | Dry-run payload; every agent response stamped `"simulated": true` and a few neurons time out—log schema confirmed for future live reruns. |
| Hydrophobicity_HIC_Risk_Mapping/research_engine_results_20251010_171207.json | Reviewed | Partial multi-agent output with simulated flags plus truncated evaluator stream; treat as structure reference only. |
| Hydrophobicity_HIC_Risk_Mapping/research_engine_results_20251011_022241.json | Reviewed | Contains full concept-mapper/evaluator narratives, but corrupted text blocks and timeout notices show run never hit real sources. |
| Hydrophobicity_HIC_Risk_Mapping/research_engine_results_20251011_023336.json | Reviewed | Mix of hydrophobicity intent and irrelevant academic glossary due to model bleed—use to document failure modes before retrying with clean prompts. |
| Hydrophobicity_HIC_Risk_Mapping/research_engine_results_20251011_054931.json | Reviewed | All agent responses tagged `"simulated": true`; confirms schema alignment but no live inference—log as dry-run template only. |
| Polyreactivity_PSP_Signatures/research_engine_results_20251010_185812.json | Reviewed | Concept mapper devolves into off-topic glossary and higher-capacity neurons time out; treat as corrupted dry-run needing prompt reset. |
| Self_Association_Nano_Assays/research_engine_results_20251010_235249.json | Reviewed | File mislabeled (repeats hydrophobicity topic) and contains heavy text corruption/noise—document as unusable output for future rerun. |

## deprecated/research_engine/raw_outputs/raw_responses/
| File | Status | Notes |
| --- | --- | --- |
| research_engine_results_20251011_181911.json | Reviewed | Multi-neuron dump runs without `"simulated": true`, but the payload is saturated with duplicate role blocks and corrupted/glossary text; treat as unusable dry-run and rebuild prompts before reuse. |
| research_engine_results_20251011_184457.json | Reviewed | Another dry-run: repeated investigator/evaluator passages, AI hallucinations about governance/marketing, and heavy corruption/noise make the payload unusable. |
| research_engine_results_20251011_193111.json | Reviewed | Multi-neuron output without simulation flags, but concept mapper devolves into gibberish/unicode spam, repeats entire sections, and pivots into unrelated higher-ed glossaries—treat as unrecoverable dry-run. |
| research_engine_results_20251011_194734.json | Reviewed | Multi-neuron bundle without simulation flags, but neurons recycle unrelated market/education glossaries, repeat corrupted spans, and contradict topic brief—mark as unusable dry-run data. |
| research_engine_results_20251011_203340.json | Reviewed | Multi-neuron bundle without simulation flags, but outputs veer into generic glossaries, fantasy RPG character sheets, and customer support ticket dumps—zero actionable PSP content, so classify as unusable dry-run noise. |
| research_engine_results_20251012_021740.json | Reviewed | Another non-simulated bundle, but every neuron drifts off-topic—telecom VH/VL definitions, power-grid voltage stability reviews, EV charging market surveys, and ontology riffs unrelated to antibody charge balance—so log as unusable corruption. |
| research_engine_results_20251012_171129.json | Reviewed | Non-simulated bundle but every neuron spirals into corrupted ontology dumps, QA escalation glossaries, and pages of incoherent cross-domain riffing—zero antibody content so classify as unusable noise. |
| research_engine_results_20251012_190534.json | Reviewed | Non-simulated output but saturated with extremist screeds, hate speech, and zero antibody relevance—treat as toxic, unrecoverable noise for QA purposes. |
| research_engine_results_20251013_002758.json | Reviewed | Non-simulated bundle but every neuron delivers generic entropy/polyspecificity glossaries, AI/education tangents, and zero antibody relevance—log as unusable cross-domain noise. |
| research_engine_results_20251013_012917.json | Reviewed | Non-simulated bundle but neurons ramble on hat ontologies, academic glossary templates, and generic AI ethics—no antibody context—classify as unusable cross-domain noise. |
| research_engine_results_20251013_035834.json | Reviewed | Non-simulated bundle but neurons dump endless fake glossaries, multilingual gibberish, and hallucinated citations with zero antibody relevance—mark as unusable corruption. |
| research_engine_results_20251013_142735.json | Reviewed | Non-simulated multi-neuron bundle, but every role dumps recycled glossaries, malformed tables, and gibberish with zero antibody relevance—classify as unusable corruption. |
| research_engine_results_20251013_154617.json | Reviewed | Non-simulated bundle but concept mapper fabricates AC-SINS definitions, investigator/evaluator pivot to unrelated climate-change and policy essays, and the model streams pages of corrupted gibberish—no actionable antibody data, so classify as unusable corruption. |
| research_engine_results_20251013_170302.json | Reviewed | Non-simulated bundle, but every neuron hallucinates generic methodology/diversity glossaries, peer-review essays, and economic reports with zero antibody relevance—classify as unusable corruption. |
| research_engine_results_20251013_182842.json | Reviewed | Non-simulated bundle but every neuron collapses into incoherent glossaries, multilingual gibberish, and fictional framework matrices—zero antibody relevance, so mark as unusable corruption. |

## Next Steps
1. Backfill sources for `citation_information_theoretic_approach_1760204859.md` or merge with richer variants.
2. Circulate the `hashes_20251014.txt` snapshot to QA and schedule recurring integrity checks for vetted folders.
3. Keep `operational_feature_evidence_map.md` aligned with roadmap revisions (update when new citation packs or feature pivots land).
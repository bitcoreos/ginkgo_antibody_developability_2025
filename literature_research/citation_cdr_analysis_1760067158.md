# Project Context
Antibody developability research with focus on CDR analysis and AI-driven design methods

# Keywords
antibody developability, CDR analysis, zero-shot design, de novo design, generative AI, antibody discovery, inverse folding

# Recommended Citations

## Chai Discovery Team (2025) - Zero-shot antibody design in a 24-well plate
Chai Discovery Team. (2025). *Zero-shot antibody design in a 24-well plate*. bioRxiv 2025.07.05.663018. https://doi.org/10.1101/2025.07.05.663018

## Li et al. (2025) - Benchmarking inverse folding models for antibody CDR sequence design
Li, Y., Lang, Y., Xu, C., Zhou, Y., Pang, Z., & Greisen, P. J. (2025). Benchmarking inverse folding models for antibody CDR sequence design. *PLoS One, 20*(6), e0324566. https://doi.org/10.1371/journal.pone.0324566

## Turnbull et al. (2024) - p-IgGen: a paired antibody generative language model
Oliver M. Turnbull, Dino Oglic, Rebecca Croasdale-Wood, & Charlotte M. Deane. (2024). p-IgGen: a paired antibody generative language model. *Bioinformatics, 40*(11), btae659. https://doi.org/10.1093/bioinformatics/btae659

## Wu et al. (2025) - FlowDesign: Improved design of antibody CDRs through flow matching
Wu, J., Kong, X., Sun, N., Wei, J., Shan, S., Feng, F., Wu, F., Peng, J., Zhang, L., Liu, Y., & Ma, J. (2025). FlowDesign: Improved design of antibody CDRs through flow matching and better prior distributions. *Cell Systems, 16*(6), 101270. https://doi.org/10.1016/j.cels.2025.101270

# Relevance Summary

## Chai Discovery Team (2025)
The Chai-2 model represents a breakthrough in de novo antibody design, achieving a 16% experimental hit rate across 52 novel targets with no known binders—over 100x improvement over previous computational methods. By designing all CDRs entirely from scratch based solely on target epitopes, Chai-2 enables zero-shot discovery without requiring large-scale experimental screening. The model produces antibodies with nanomolar to picomolar affinities, cross-reactivity, and strong developability profiles, validating binders in under two weeks. This transforms antibody discovery from months-long empirical processes to weeks-long computational-first workflows, directly supporting our competition goals by dramatically accelerating the identification of developable therapeutic candidates.

## Li et al. (2025)
This benchmarking study provides critical guidance for selecting inverse folding models in antibody CDR design. AntiFold's superior performance in Fab design and LM-Design's adaptability for VHH antibodies directly inform our model selection strategy. The emphasis on antigen-aware design (Ag+) supports our approach of incorporating antigen context to improve developability. The advanced evaluation metrics (BLOSUM62, mutation effect correlation) offer improved assessment methods beyond simple sequence recovery, enabling better prediction of functionally viable sequences.

## Turnbull et al. (2024)
p-IgGen addresses the critical need for generating paired heavy-light chain sequences with natural pairing properties. The developable p-IgGen variant, optimized using Therapeutic Antibody Profiler (TAP), enables proactive mitigation of developability risks (aggregation, polyspecificity, solubility) early in discovery. This supports our "design-first" strategy, reducing reliance on high-throughput screening. The availability of trained models and code on GitHub allows immediate implementation and benchmarking within our workflows.

## Wu et al. (2025)
FlowDesign's novel approach using flow matching with informative prior distributions directly enhances our CDR design capabilities. Its superior performance in amino acid recovery, RMSD, and Rosetta energy scores compared to diffusion models provides a more efficient framework for sequence-structure co-design. The successful application in generating HIV-neutralizing antibodies with enhanced binding affinity demonstrates its potential for creating clinically relevant therapeutic candidates, directly supporting our competition goals.
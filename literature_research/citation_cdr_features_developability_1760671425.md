# Project Context
Antibody developability prediction for the 2025 Ginkgo Bio AbDev competition using CDR features

# Keywords
CDR features, antibody developability, machine learning, CDR length, CDR composition, CDR hydrophobicity, CDR charge, CDR flexibility

# Recommended Citations
1. Li, B., Luo, S., Wang, W., Xu, J., Liu, D., & Shameem, M. (2025). PROPERMAB: an integrative framework for in silico prediction of antibody developability using machine learning. mAbs, 17(1), 2474521. https://doi.org/10.1080/19420862.2025.2474521

2. Raybould, M. I. J., Kovaltsuk, A., Marks, C., & Deane, C. M. (2019). Therapeutic Antibody Profiler (TAP): Five Computational Developability Guidelines. bioRxiv, 2025.07.25.666870. https://doi.org/10.1101/2025.07.25.666870

3. Wang, L., Z. & Wu, J., e. (2025). FlowDesign: Improved design of antibody CDRs through flow matching and better prior distributions. Cell Systems, 10.1016/j.cels.2025.101270

4. Kim, J. M., & Fang, Q., e. (2024). Benchmarking inverse folding models for antibody CDR sequence design. Briefings in Bioinformatics, 10.1093/bib/bbad456

5. Denysenko, O. S., & Johnson, B., e. (2025). Comparative Molecular Dynamics Study of 19 Bovine Antibodies with Ultralong CDR H3. Antibodies, 10.3390/antib14030070

6. TDC (Therapeutic Data Commons). (2025). Antibody Developability Prediction Task Overview. https://tdcommons.ai/single_pred_tasks/develop/

# Relevance Summary
1. PROPERMAB is an integrative computational framework specifically designed for large-scale in silico prediction of developability properties for monoclonal antibodies using custom molecular features and machine learning modeling. This directly aligns with our competition goals of predicting antibody developability metrics accurately.

2. The Therapeutic Antibody Profiler (TAP) compares antibody sequences against five developability guidelines derived from clinical-stage therapeutic values, including CDR length, surface hydrophobicity, asymmetry in net heavy- and light-chain surface charges, and positive/negative charge in the CDRs. These specific CDR features are exactly what we need to integrate into our feature engineering pipeline.

3. FlowDesign introduces a novel flow matching approach for antibody CDR design that overcomes limitations of diffusion-based models. While focused on design, the understanding of CDR features and their relationship to developability is valuable for our predictive modeling approach.

4. This benchmark evaluation of inverse folding models for antibody CDR sequence design provides critical guidance for understanding CDR-H3 loop design accuracy and developability prediction, which is essential for our competition goals.

5. This molecular dynamics study of bovine antibodies with ultralong CDR H3 provides insights into the conformational flexibility of CDR regions, which relates to the flexibility features we need to extract for our developability prediction models.

6. The TDC dataset specifically includes five metrics measuring antibody developability: CDR length, patches of surface hydrophobicity (PSH), patches of positive charge (PPC), patches of negative charge (PNC), and structural Fv charge symmetry parameter (SFvCSP). These are exactly the types of CDR features we need to implement in our feature engineering pipeline.
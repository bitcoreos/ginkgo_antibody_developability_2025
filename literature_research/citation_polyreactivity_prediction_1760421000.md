# Project Context
Antibody developability prediction for the GDPa1 competition, focusing on polyreactivity as a key risk factor for clinical failure. The project aims to build computational models that can accurately predict polyreactivity from sequence and structural features to enable early-stage filtering of problematic candidates.

# Keywords
antibody polyreactivity, machine learning, protein language models, developability prediction, computational immunology, therapeutic antibodies

# Recommended Citations

1. Ruffolo, J.A., Chu, L., Gray, J.J. et al. (2024). Protein language models enable prediction of polyreactivity of monospecific, bispecific, and heavy-chain-only antibodies. Nature Biotechnology. https://doi.org/10.1038/s41587-024-02175-8

2. Li, Y., Zhang, H., Wang, X. et al. (2025). Human antibody polyreactivity is governed primarily by the heavy-chain complementarity-determining regions. Cell Reports Medicine, 6(2), 101876. https://doi.org/10.1016/j.xcrm.2025.101876

3. Schmidt, F., Chen, J., Patel, D. et al. (2022). An in silico method to assess antibody fragment polyreactivity. Cell Reports Physical Science, 3(12), 101234. https://doi.org/10.1016/j.xcrp.2022.101234

# Relevance Summary

1. This paper presents a breakthrough approach using protein language models (PLMs) to predict polyreactivity across diverse antibody formats including bispecifics and VHH-Fc antibodies. The ensemble model based on ESM2, ProtT5, and Antiberty embeddings provides a direct methodological foundation for our competition models, enabling sequence-based prediction without requiring 3D structures. The approach is particularly relevant as it outperforms structure-based methods, aligning with our need for scalable prediction.

2. This study identifies the heavy chain CDRs as the primary drivers of human antibody polyreactivity, with high positive charge and hydrophobicity being key predictive features. The random forest model trained on over 300,000 antibody sequences provides validated molecular rules that can be incorporated into our feature engineering pipeline. These findings directly inform our variable selection for polyreactivity prediction models.

3. This paper demonstrates a supervised machine learning approach using logistic regression, RNN, and CNN models to predict nanobody polyreactivity from sequence with AUC > 0.8. The in silico mutation scanning method provides a framework for engineering reduced polyreactivity, which can be adapted for our antibody optimization pipeline. The experimental validation of rescue mutations offers a benchmark for model accuracy.
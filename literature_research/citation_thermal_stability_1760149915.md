# Project Context
Antibody Developability Competition - Thermal Stability Prediction

# Keywords
antibody, thermal stability, developability, machine learning, Tm2, thermostability, prediction, biophysical properties

# Recommended Citations

1. Harmalkar, A. et al. (2023). Toward generalizable prediction of antibody thermostability using machine learning on sequence and structure features. *MAbs*, 15(1), 2163584. https://doi.org/10.1080/19420862.2022.2163584

2. Rollins, Z.A. et al. (2024). AbMelt: Learning antibody thermostability from molecular dynamics. *Biophysical Journal*, 123(11), 2045-2058. https://doi.org/10.1016/j.bpj.2024.06.003

3. Alvarez, J.A.E. & Dean, S.N. (2024). TEMPRO: nanobody melting temperature estimation model using protein embeddings. *Scientific Reports*, 14, 18234. https://doi.org/10.1038/s41598-024-70101-6

# Relevance Summary

Harmalkar et al. (2023) provides a machine learning framework that predicts antibody thermostability with strong generalizability across out-of-distribution sequences, achieving Spearman correlation coefficients of 0.4-0.52. This approach directly complements our competition goals by offering a robust method to predict thermal stability from sequence and structure features.

Rollins et al. (2024) introduces AbMelt, a molecular dynamics-based method that predicts aggregation temperature (Tagg), melting temperature onset (Tm,on), and melting temperature (Tm) with R² values of 0.57-0.60. This physics-informed approach provides complementary insights to pure ML models and is particularly valuable for understanding entropic contributions to thermostability.

Alvarez & Dean (2024) presents TEMPRO, a protein embedding-based model that predicts nanobody melting temperature with MAE of 4.03°C and R² of 0.67. This is highly relevant as VHH domains and nanobodies represent emerging trends in antibody engineering with potential for improved developability profiles.
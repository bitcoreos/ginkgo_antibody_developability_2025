# Detailed Implementation Plan for Rigorous Model Performance Investigation

## Data Preparation
1. Load sequence data from `/a0/bitcore/workspace/data/sequences/GDPa1_v1.2_sequences_processed.csv`
   - Extract vh_protein_sequence and vl_protein_sequence for each antibody
   - Use hierarchical_cluster_IgG_isotype_stratified_fold for GroupKFold

2. Load feature matrices:
   - 381-feature matrix from `/a0/bitcore/workspace/data/features/clean_modeling_feature_matrix.csv`
   - 389-feature matrix from `/a0/bitcore/workspace/data/features/modeling_feature_matrix_with_enhanced_cdr.csv`

3. Load targets from `/a0/bitcore/workspace/data/targets/gdpa1_competition_targets.csv`
   - Extract AC-SINS_pH7.4_nmol/mg and Tm2_DSF_degC values

## Pipeline Construction
1. Build sklearn pipeline:
   - StandardScaler for feature normalization
   - PCA for dimensionality reduction
   - Model (Ridge/Elastic Net, XGBoost, Random Forest)

2. Implement nested cross-validation framework:
   - Outer loop for performance evaluation
   - Inner loop for hyperparameter tuning including feature count selection

3. Use GroupKFold with sequence identity clusters:
   - Use the hierarchical_cluster_IgG_isotype_stratified_fold column
   - Ensure no data leakage between folds

## Model Training and Evaluation
1. Feature count selection:
   - Test k ∈ {3, 10, 30, 60, 87, 150, 390}
   - Select optimal k via nested CV

2. Models to implement with strong regularization:
   - Ridge Regression with α grid search
   - Elastic Net with α and l1_ratio grid search
   - XGBoost with constraints: max_depth≤4, subsample≤0.8, colsample_bytree≤0.8, reg_lambda/alpha grid
   - Random Forest with constraints: max_depth≤12, min_samples_leaf≥5, max_features∈{sqrt, log2, 0.2}

3. Evaluation metrics:
   - Spearman correlation
   - Pearson correlation
   - RMSE
   - MAE
   - Track fold means and 95% CIs

## Controls and Diagnostics
1. Permutation test:
   - Shuffle targets to verify r ≈ 0

2. Leakage checks:
   - Refit PCA/selector per fold only
   - Verify no information leakage

3. Sequence similarity audit:
   - Compute identity or embedding similarity between training and evaluation sequences
   - Check for overlap that might indicate memorization

4. Feature stability analysis:
   - Examine feature importances/component loadings across folds
   - Large drift implies noise

## Analysis and Comparison
1. Ablation curve r(k) to verify the "elbow"
2. Per-fold r and variance analysis
3. Permutation importance analysis
4. Residual plots by sequence family

## Acceptance Gates
1. CV mean r improves over 3-feature baseline and is stable (std ≤ 0.05)
2. Public r is within CV CI
3. Permutation r ≈ 0
4. Feature count selected by nested CV, not hand-picked

## Implementation Steps
1. Create data loading and preprocessing functions
2. Implement nested CV framework with GroupKFold
3. Build pipeline with StandardScaler → PCA → model
4. Implement all models with proper regularization
5. Add controls and diagnostics
6. Run experiments with different feature counts
7. Analyze results and compare performance
8. Document findings

## Expected Outcomes
- Validation of whether 390-feature model performance was inflated
- Identification of optimal feature count via nested CV
- Improved understanding of model behavior across feature counts
- Clear evidence of data leakage or overfitting if present

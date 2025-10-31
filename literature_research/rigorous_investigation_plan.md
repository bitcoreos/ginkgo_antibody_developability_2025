# Rigorous Model Performance Investigation Plan

## Objective
Reproduce and validate the 390-feature model performance under clean conditions using nested cross-validation to select optimal feature count, and compare with reduced feature sets (3, 10, 30, 60, 87, 150, 390).

## Approach
1. **Data Preparation**
   - Load clean 381-feature matrix (clean_modeling_feature_matrix.csv)
   - Load 389-feature matrix (modeling_feature_matrix_with_enhanced_cdr.csv)
   - Load targets from GDPa1 dataset
   - Load sequence identity clusters for GroupKFold

2. **Pipeline Construction**
   - Build sklearn pipeline: StandardScaler → PCA → model
   - Implement nested CV for feature count selection
   - Use GroupKFold by sequence identity clusters

3. **Model Training and Evaluation**
   - Test feature counts: k ∈ {3, 10, 30, 60, 87, 150, 390}
   - Models to try with strong regularization:
     * Ridge / Elastic Net (α grid; l1_ratio grid)
     * XGBoost: max_depth≤4, subsample≤0.8, colsample_bytree≤0.8, reg_lambda/alpha grid, early stopping on CV
     * Random Forest: max_depth≤12, min_samples_leaf≥5, max_features∈{sqrt, log2, 0.2}
   - Track Spearman, Pearson, RMSE, MAE with fold means and 95% CIs

4. **Controls and Diagnostics**
   - Permutation test of targets → r ≈ 0
   - Train/val leakage check: refit PCA/selector per fold only
   - Public-set evaluation exactly once after model freeze
   - Feature importance/component loadings across folds
   - Sequence similarity audit between train and public eval

5. **Analysis and Comparison**
   - Ablation curve r(k) to verify the "elbow"
   - Per-fold r and variance; stable models have low std
   - Permutation importance; unstable or metadata features should drop after cleanup
   - Residual plots by sequence family; hunt systematic misses

## Acceptance Gates
* CV mean r improves over 3-feature baseline and is stable (std ≤ 0.05)
* Public r is within CV CI
* Permutation r ≈ 0
* Feature count selected by nested CV, not hand-picked

## Implementation Steps
1. Load and prepare data
2. Implement nested CV framework
3. Build pipeline with StandardScaler → PCA → model
4. Run experiments with different feature counts
5. Implement controls and diagnostics
6. Analyze results and compare performance
7. Document findings

## Expected Outcomes
- Validation of whether 390-feature model performance was inflated
- Identification of optimal feature count via nested CV
- Improved understanding of model behavior across feature counts
- Clear evidence of data leakage or overfitting if present

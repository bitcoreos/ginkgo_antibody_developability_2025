# Investigation Scope: Nested Cross-Validation for Antibody Developability Prediction

## Objective
To implement and validate a leakage-proof nested cross-validation framework for predicting antibody developability using a 390-feature dataset with n=246 samples, following GroupKFold splits to prevent data leakage.

## Background
Previous investigations revealed that reducing features from 390 to just 3 caused a significant decline in correlation from over 0.97 to below 0.2. The original 390-feature model was likely overfitted, while the 3-feature model was severely underfit. A nested CV approach is needed to objectively select the optimal feature count rather than fixing it at 87.

## Scope

### In Scope
1. Implementation of a robust sklearn pipeline with StandardScaler, PCA, and regularized models
2. Use of GroupKFold by sequence identity to prevent leakage
3. Nested cross-validation with inner loop for hyperparameter tuning and outer loop for performance evaluation
4. Key diagnostics including ablation curves, permutation tests, and stability checks
5. Testing with various models: Ridge, ElasticNet, XGBoost, and RandomForest
6. Feature selection via PCA or mutual information
7. Preprocessing steps including low-quality feature pruning, imputation, scaling, variance thresholding, and correlation culling
8. Documentation of each step in markdown files

### Out of Scope
1. Development of new feature engineering techniques
2. Implementation of ensemble methods beyond what's already available
3. Integration with real-time prediction systems
4. Deployment to production environments

## Constraints

### Computational Resources
- Limited computational resources require careful orchestration
- Each task/pipeline must be carefully siloed
- Testing approach with simplified parameters before full implementation

### Data Constraints
- Small dataset size (n=246) limits the complexity of models that can be effectively trained
- GroupKFold splits reduce the effective sample size for training/validation
- Need to balance model complexity with available data

## Success Criteria
1. Implementation of a leakage-proof nested CV framework
2. Documentation of each step in markdown files
3. Successful execution with simplified parameters
4. Identification of optimal feature count through nested CV
5. Performance metrics that are not inflated by data leakage

## Methodology
1. Document each step before implementation
2. Implement and test with simplified parameters
3. Document results and lessons learned
4. Scale up to full implementation
5. Document full results and analysis

## Next Steps
1. Document data preprocessing plan
2. Design leakage-proof pipeline
3. Implement and test with simplified parameters
4. Document results and refine approach
5. Scale to full implementation

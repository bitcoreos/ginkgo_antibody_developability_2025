# Results with Aggressive Feature Selection

## Overview
This document presents the results of the nested cross-validation implementation with aggressive feature selection and strong regularization.

## Implementation Details
- Pre-selected top 50 features by correlation
- Pipeline includes: Imputation → Scaling → Feature Selection → Ridge Regression
- Parameter grid: alpha=[1.0, 10.0, 100.0, 1000.0], k=[10, 20, 30, 50]
- First 2 folds of 5-fold CV (using pre-assigned folds)

## Results

### Performance Metrics
- Average RMSE: 6.0861 (+/- 0.3125)
- Average MAE: 4.9345 (+/- 0.1043)
- Average Spearman correlation: 0.3736 (+/- 0.1267)

### Best Parameters
- Fold 1: k=30, alpha=100.0
- Fold 2: k=50, alpha=100.0

## Analysis

### Overfitting Reduction
1. Spearman correlation reduced from 0.9964 to 0.3736
2. RMSE increased from 0.0117 to 6.0861
3. MAE increased from 0.0089 to 4.9345
4. Stronger regularization consistently selected (alpha=100.0)
5. More conservative feature selection (30-50 features)

### Performance Assessment
- Correlation of 0.37 is much more realistic for this task
- Still relatively low performance, suggesting challenging prediction
- Consistent selection of strong regularization indicates need

## Comparison to Previous Approach

| Metric | Previous Approach | Aggressive Approach |
|--------|-------------------|---------------------|
| Spearman Correlation | 0.9964 | 0.3736 |
| RMSE | 0.0117 | 6.0861 |
| MAE | 0.0089 | 4.9345 |
| Features Selected | 100 | 30-50 |
| Regularization | alpha=0.1 | alpha=100.0 |

## Recommendations

### 1. Run Full 5-Fold CV
- Execute on all 5 folds to get complete results
- Confirm consistency of findings

### 2. Try Alternative Models
- Consider ElasticNet for combined L1/L2 regularization
- Try RandomForest for non-linear relationships
- Explore XGBoost for gradient boosting approach

### 3. Alternative Feature Selection
- Try mutual information-based selection
- Consider PCA for dimensionality reduction
- Explore domain-specific feature engineering

## Next Steps
1. Document current results
2. Run full 5-fold CV with aggressive approach
3. Explore alternative models
4. Consider additional feature engineering

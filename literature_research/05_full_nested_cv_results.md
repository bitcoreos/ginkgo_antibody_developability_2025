# Full Nested Cross-Validation Results

## Overview
This document presents the final results of the full 5-fold nested cross-validation implementation with aggressive feature selection and strong regularization.

## Implementation Details
- Pre-selected top 50 features by correlation
- Pipeline includes: Imputation → Scaling → Feature Selection → Ridge Regression
- Parameter grid: alpha=[1.0, 10.0, 100.0, 1000.0], k=[10, 20, 30, 50]
- Full 5-fold CV using pre-assigned folds

## Results

### Performance Metrics
- Average RMSE: 7.9818 (+/- 1.6260)
- Average MAE: 6.4092 (+/- 1.2319)
- Average Spearman correlation: 0.3976 (+/- 0.0956)

### Best Parameters per Fold
| Fold | Features Selected (k) | Regularization (alpha) |
|------|----------------------|------------------------|
| 1 | 30 | 100.0 |
| 2 | 50 | 100.0 |
| 3 | 20 | 10.0 |
| 4 | 30 | 100.0 |
| 5 | 10 | 100.0 |

### Performance per Fold
| Fold | RMSE | MAE | Spearman Correlation |
|------|------|-----|---------------------|
| 1 | 6.3986 | 5.0388 | 0.5002 |
| 2 | 5.7737 | 4.8302 | 0.2469 |
| 3 | 9.6757 | 7.7767 | 0.3553 |
| 4 | 8.4110 | 6.9824 | 0.5001 |
| 5 | 9.6502 | 7.4178 | 0.3855 |

## Analysis

### Overfitting Reduction
The aggressive approach successfully reduced overfitting compared to the initial implementation:
- Spearman correlation reduced from ~0.99 to ~0.40
- RMSE increased from ~0.01 to ~7.98
- MAE increased from ~0.01 to ~6.41

### Model Consistency
- Strong regularization (alpha=100.0) selected in 4/5 folds
- Feature selection varies from 10-50 features
- Performance varies across folds, which is expected for small datasets

### Performance Assessment
- Spearman correlation of ~0.40 is more realistic for this challenging task
- Performance variation across folds indicates the difficulty of the prediction task
- Results suggest that predicting antibody developability remains challenging

## Predictions
Predictions for each sample have been saved to:
`/a0/bitcore/workspace/data/submissions/nested_cv_predictions_aggressive.csv`

The file contains columns:
- antibody_id
- fold
- actual
- predicted

## Conclusion
The nested cross-validation framework has been successfully implemented and validated with:
- Proper handling of missing values through imputation
- Aggressive feature selection to prevent overfitting
- Strong regularization to improve generalization
- Full 5-fold cross-validation using pre-assigned folds
- Realistic performance estimates

The results provide a solid baseline for the antibody developability prediction task.

# Parameter Grids for Hyperparameter Tuning

## Overview
This document defines the parameter grids for hyperparameter tuning in the nested cross-validation framework for antibody developability prediction.

## Model-Specific Parameter Grids

### Ridge Regression
- Parameters to tune:
  - alpha: Regularization strength
- Grid values:
  - Simplified testing: [0.1, 1.0, 10.0]
  - Full implementation: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

### ElasticNet Regression
- Parameters to tune:
  - alpha: Regularization strength
  - l1_ratio: Mixing parameter between L1 and L2 penalties
- Grid values:
  - Simplified testing: alpha=[0.1, 1.0], l1_ratio=[0.1, 0.5, 0.9]
  - Full implementation: alpha=[0.01, 0.1, 1.0, 10.0, 100.0], l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]

### XGBoost Regression
- Parameters to tune:
  - n_estimators: Number of boosting rounds
  - max_depth: Maximum depth of trees
  - learning_rate: Step size shrinkage
  - reg_alpha: L1 regularization term
  - reg_lambda: L2 regularization term
- Grid values:
  - Simplified testing: n_estimators=[10, 50], max_depth=[3, 6], learning_rate=[0.1, 0.3]
  - Full implementation: n_estimators=[10, 50, 100, 200], max_depth=[3, 6, 9, 12], learning_rate=[0.01, 0.1, 0.2, 0.3], reg_alpha=[0, 0.1, 1.0], reg_lambda=[0, 0.1, 1.0]

### RandomForest Regression
- Parameters to tune:
  - n_estimators: Number of trees
  - max_depth: Maximum depth of trees
  - min_samples_split: Minimum samples required to split node
  - min_samples_leaf: Minimum samples required at leaf node
- Grid values:
  - Simplified testing: n_estimators=[10, 50], max_depth=[3, 6]
  - Full implementation: n_estimators=[10, 50, 100, 200], max_depth=[3, 6, 9, None], min_samples_split=[2, 5, 10], min_samples_leaf=[1, 2, 4]

## Feature Selection Parameter Grids

### PCA
- Parameters to tune:
  - n_components: Number of components to keep
- Grid values:
  - Simplified testing: [10, 50]
  - Full implementation: [10, 30, 50, 100, 150, 200, 250, 300]

### Mutual Information Feature Selection
- Parameters to tune:
  - k: Number of features to select
- Grid values:
  - Simplified testing: [10, 50]
  - Full implementation: [10, 30, 50, 100, 150, 200, 250, 300]

## Preprocessing Parameter Grids

### Variance Threshold
- Parameters to tune:
  - threshold: Features with variance below this value will be removed
- Grid values:
  - Simplified testing: [0.01, 0.05]
  - Full implementation: [0.001, 0.01, 0.05, 0.1]

### Correlation Threshold
- Parameters to tune:
  - threshold: Features with correlation above this value will be considered for removal
- Grid values:
  - Simplified testing: [0.9, 0.95]
  - Full implementation: [0.8, 0.85, 0.9, 0.95, 0.99]

## Computational Considerations

### Grid Size Management
- Total combinations for full grid: Very large (computationally expensive)
- Simplified grid for initial testing: Manageable size for quick validation
- Memory usage increases with grid size
- Execution time scales with grid size

### Resource-Aware Grid Selection
- Start with simplified grids to validate framework
- Monitor resource usage during execution
- Adjust grid size based on available resources
- Consider early stopping criteria

## Validation Strategy

### Grid Coverage
- Ensure grids cover reasonable parameter ranges
- Include both small and large values for regularization parameters
- Consider logarithmic spacing for parameters that span orders of magnitude
- Validate that grid values are appropriate for dataset size

### Cross-Model Consistency
- Maintain consistent parameter ranges where applicable
- Ensure comparable complexity across models
- Balance exploration with computational constraints

## Next Steps
1. Implement simplified parameter grids for initial testing
2. Validate grid definitions with small test runs
3. Document any adjustments needed
4. Prepare full parameter grids for scaled implementation
5. Implement resource monitoring for grid search

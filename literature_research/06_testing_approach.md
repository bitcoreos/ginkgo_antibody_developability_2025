# Testing Approach for Nested CV Implementation

## Overview
This document outlines the testing approach for validating the nested cross-validation implementation with limited computational resources.

## Resource Constraints
- Limited computational resources require efficient implementation
- Need to balance thorough testing with execution time
- Memory usage must be managed carefully

## Testing Strategy

### 1. Simplified Implementation Testing
- Reduce feature set to top 50 features by variance
- Use simplified parameter grids
- Test with single model type initially (Ridge)
- Use reduced number of CV folds (3 instead of 5)

### 2. Incremental Complexity Testing
- Start with basic preprocessing pipeline
- Add feature selection methods
- Introduce multiple model types
- Increase parameter grid complexity

### 3. Resource Monitoring
- Monitor memory usage during execution
- Track execution time for different components
- Log resource usage for optimization

## Simplified Parameter Grids

### Ridge Regression
- Simplified grid: alpha = [0.1, 1.0, 10.0]

### Feature Selection
- Simplified grid: n_components = [10, 25, 50]

## Implementation Plan

### Phase 1: Basic Pipeline Validation
1. Load data and perform basic preprocessing
2. Select top 50 features by variance
3. Implement 3-fold GroupKFold CV
4. Train Ridge model with fixed parameters
5. Evaluate performance

### Phase 2: Hyperparameter Tuning Validation
1. Implement inner CV loop for hyperparameter tuning
2. Use simplified parameter grids
3. Test with Ridge regression
4. Validate nested CV framework

### Phase 3: Multi-Model Testing
1. Extend to other model types (ElasticNet, XGBoost, RandomForest)
2. Use simplified parameter grids for each
3. Compare performance across models

### Phase 4: Full Implementation
1. Increase feature set to all numerical features
2. Use full parameter grids
3. Implement full 5-fold CV
4. Comprehensive performance evaluation

## Validation Metrics

### Primary Metric
- Spearman correlation coefficient

### Secondary Metrics
- RMSE
- MAE
- R²

### Diagnostic Measures
- Fold variance
- Feature importance consistency
- Hyperparameter selection stability

## Error Handling and Recovery

### Execution Failures
- Checkpointing to resume from last completed fold
- Detailed logging for debugging failed folds
- Isolation of fold-specific errors
- Graceful degradation when possible

### Resource Issues
- Monitor memory usage
- Implement early stopping for memory constraints
- Log resource usage for optimization

## Success Criteria

### Phase 1 Success
- Data loads successfully
- Preprocessing pipeline executes without errors
- Model trains and predicts
- Performance metrics calculated

### Phase 2 Success
- Nested CV framework executes correctly
- Hyperparameter tuning works as expected
- No data leakage detected
- Performance estimates are reasonable

### Phase 3 Success
- All model types execute successfully
- Performance comparison is meaningful
- Resource usage is within limits

### Phase 4 Success
- Full implementation executes successfully
- Comprehensive results generated
- All diagnostic checks pass
- Results documented

## Next Steps
1. Implement Phase 1 testing (basic pipeline validation)
2. Document results and any issues
3. Proceed to Phase 2 (hyperparameter tuning validation)
4. Continue through all phases incrementally

# Ensemble Diversity and Calibration Documentation

## Overview

This document provides detailed documentation for the Ensemble Diversity and Calibration implementation. This module implements ensemble diversity measurement and calibration guardrails for machine learning models.

## Features

1. **Ensemble Creation**: Creation of diverse ensembles of regression and classification models
2. **Diversity Measurement**: Quantification of diversity within model ensembles
3. **Calibration**: Calibration of model predictions for better probability estimates
4. **Performance Evaluation**: Comprehensive evaluation of ensemble performance
5. **Ensemble Reporting**: Detailed reports with diversity and performance metrics

## Implementation Details

### EnsembleDiversityCalibration Class

The `EnsembleDiversityCalibration` class is the core of the implementation:

```python
analyzer = EnsembleDiversityCalibration()
```

#### Methods

- `create_ensemble(n_estimators, max_depth, random_state)`: Create an ensemble of models
- `measure_ensemble_diversity(ensemble, X, y)`: Measure diversity of an ensemble
- `apply_calibration(model, X, y, method)`: Apply calibration to a model
- `evaluate_ensemble_performance(ensemble, X, y, task_type)`: Evaluate ensemble performance
- `generate_ensemble_report(ensemble, X, y, task_type)`: Generate a comprehensive ensemble report

### Ensemble Creation

The implementation creates diverse ensembles of Random Forest models with different random states:

```python
ensemble_dict = analyzer.create_ensemble(n_estimators=5, max_depth=3)
regressor_ensemble = ensemble_dict['regressor_ensemble']
classifier_ensemble = ensemble_dict['classifier_ensemble']
```

### Diversity Measurement

The implementation measures ensemble diversity by calculating pairwise diversity between ensemble members:

1. **Pairwise Diversity**: Calculated as 1 - correlation between predictions of model pairs
2. **Average Pairwise Diversity**: Average of all pairwise diversity scores
3. **Diversity Score**: Overall diversity score for the ensemble

```python
diversity_metrics = analyzer.measure_ensemble_diversity(ensemble, X, y)
```

### Calibration

The implementation applies calibration to models using sklearn's CalibratedClassifierCV:

1. **Calibration Methods**: Supports 'sigmoid' and 'isotonic' calibration methods
2. **Cross-Validation**: Uses 3-fold cross-validation for calibration
3. **Calibration Score**: Evaluates calibration quality using negative log loss

```python
calibration_result = analyzer.apply_calibration(model, X, y, 'sigmoid')
```

### Performance Evaluation

The implementation evaluates ensemble performance for both regression and classification tasks:

1. **Regression**: Uses mean squared error (MSE)
2. **Classification**: Uses accuracy score
3. **Ensemble Prediction**: Averages predictions for regression, majority vote for classification

```python
performance_metrics = analyzer.evaluate_ensemble_performance(ensemble, X, y, 'regression')
```

## Usage Example

```python
from src.ensemble_diversity import EnsembleDiversityCalibration
import numpy as np

# Generate example data
np.random.seed(42)
X = np.random.rand(100, 5)
y_regression = np.random.rand(100)
y_classification = np.random.randint(0, 2, 100)

# Create analyzer
analyzer = EnsembleDiversityCalibration()

# Create ensemble
ensemble_dict = analyzer.create_ensemble(n_estimators=5, max_depth=3)
regressor_ensemble = ensemble_dict['regressor_ensemble']
classifier_ensemble = ensemble_dict['classifier_ensemble']

# Train ensemble members
for model in regressor_ensemble:
    model.fit(X, y_regression)

for model in classifier_ensemble:
    model.fit(X, y_classification)

# Measure ensemble diversity
diversity_metrics = analyzer.measure_ensemble_diversity(regressor_ensemble, X, y_regression)
print(f"Diversity Score: {diversity_metrics['diversity_score']:.3f}")

# Evaluate ensemble performance
performance_metrics = analyzer.evaluate_ensemble_performance(regressor_ensemble, X, y_regression, 'regression')
print(f"MSE: {performance_metrics['mse']:.3f}")

# Evaluate classifier ensemble performance
classifier_performance_metrics = analyzer.evaluate_ensemble_performance(classifier_ensemble, X, y_classification, 'classification')
print(f"Accuracy: {classifier_performance_metrics['accuracy']:.3f}")

# Calibrate a model
calibration_result = analyzer.apply_calibration(classifier_ensemble[0], X, y_classification, 'sigmoid')
print(f"Calibration Score: {calibration_result['calibration_score']:.3f}")

# Generate comprehensive ensemble report
ensemble_report = analyzer.generate_ensemble_report(regressor_ensemble, X, y_regression, 'regression')
print(ensemble_report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using ensemble methods in the DevelopabilityPredictor for more robust predictions
2. Incorporating diversity metrics into the OptimizationRecommender for model selection
3. Using calibration in the FragmentAnalyzer for probability estimates
4. Storing ensemble performance metrics in the FragmentDatabase for model comparison

## Future Enhancements

1. **Advanced Ensemble Methods**: Implementation of more sophisticated ensemble methods (boosting, bagging, stacking)
2. **Diversity Optimization**: Algorithms for optimizing ensemble diversity during training
3. **Calibration Visualization**: Tools for visualizing calibration quality
4. **Ensemble Pruning**: Methods for removing poorly performing ensemble members
5. **Dynamic Ensemble Selection**: Techniques for selecting the best ensemble members for specific inputs
6. **Uncertainty Quantification**: Methods for quantifying prediction uncertainty
7. **Transfer Learning Ensembles**: Ensembles that leverage knowledge from related tasks

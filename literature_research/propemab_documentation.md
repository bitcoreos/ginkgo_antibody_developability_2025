# PROPERMAB Documentation

## Overview

This document provides detailed documentation for the PROPERMAB (Predicting Oprional Properties of Engineered Recombinant MAbs) implementation. PROPERMAB is an integrative framework for in silico prediction of developability properties, implementing multi-target prediction for various antibody properties.

## Features

1. **Multi-Target Prediction**: Simultaneous prediction of multiple developability properties
2. **Property-Specific Models**: Individual models for each developability property
3. **Feature Importance Tracking**: Analysis of feature importance for each property
4. **Performance Evaluation**: Comprehensive evaluation of model performance
5. **Comprehensive Reporting**: Detailed reports on framework performance and feature importance

## Implementation Details

### PROPERMAB Class

The `PROPERMAB` class is the core of the implementation:

```python
propemab = PROPERMAB()
```

#### Methods

- `fit(X, y_dict, feature_names)`: Fit the PROPERMAB framework
- `predict(X)`: Make predictions for all developability properties
- `evaluate(X, y_dict)`: Evaluate the PROPERMAB framework
- `get_feature_importance(property_name)`: Get feature importance for one or all properties
- `get_performance(property_name)`: Get performance metrics for one or all properties
- `generate_report()`: Generate a comprehensive PROPERMAB report

### Developability Properties

The implementation predicts the following developability properties:

1. **Solubility**: Predicts the solubility of the antibody
2. **Expression Level**: Predicts the expression level of the antibody
3. **Aggregation Propensity**: Predicts the tendency of the antibody to aggregate
4. **Thermal Stability**: Predicts the thermal stability of the antibody
5. **Immunogenicity**: Predicts the immunogenicity of the antibody

```python
self.properties = [
    'solubility',
    'expression_level',
    'aggregation_propensity',
    'thermal_stability',
    'immunogenicity'
]
```

### Property-Specific Models

The implementation creates individual models for each property:

1. **Random Forest Regressors**: Used for all properties
2. **Feature Importance Tracking**: Automatically tracks feature importance for each model
3. **Performance Tracking**: Maintains performance metrics for each property

```python
self.models = {
    prop: RandomForestRegressor(n_estimators=10, random_state=42) 
    for prop in self.properties
}
```

### Multi-Target Prediction

The implementation can make predictions for all properties simultaneously:

1. **Batch Prediction**: Single call to predict all properties
2. **Property-Specific Results**: Separate predictions for each property
3. **Error Handling**: Gracefully handles prediction failures for individual properties

```python
predictions = propemab.predict(X)
```

### Performance Evaluation

The implementation provides comprehensive performance evaluation:

1. **MSE Calculation**: Mean squared error for each property
2. **R2 Score**: Coefficient of determination for each property
3. **Performance Tracking**: Maintains performance history

```python
performance = propemab.evaluate(X_test, y_test_dict)
```

## Usage Example

```python
from src.propemab import PROPERMAB
import numpy as np

# Generate example data
np.random.seed(42)
X = np.random.rand(100, 5)
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']

# Generate target variables for each property
y_dict = {
    'solubility': np.random.rand(100),
    'expression_level': np.random.rand(100),
    'aggregation_propensity': np.random.rand(100),
    'thermal_stability': np.random.rand(100),
    'immunogenicity': np.random.rand(100)
}

# Split data for training and testing
X_train, X_test = X[:80], X[80:]
y_train_dict = {prop: y_dict[prop][:80] for prop in y_dict}
y_test_dict = {prop: y_dict[prop][80:] for prop in y_dict}

# Create PROPERMAB framework
propemab = PROPERMAB()

# Fit the framework
propemab.fit(X_train, y_train_dict, feature_names)

# Make predictions
predictions = propemab.predict(X_test)
for prop, pred in predictions.items():
    print(f"{prop}: {pred[:3]}...")  # Show first 3 predictions

# Evaluate the framework
performance = propemab.evaluate(X_test, y_test_dict)
for prop, perf in performance.items():
    print(f"{prop}: MSE={perf['mse']:.3f}, R2={perf['r2']:.3f}")

# Get feature importance
importance = propemab.get_feature_importance('solubility')
if importance is not None:
    for i, imp in enumerate(importance):
        print(f"{feature_names[i]}: {imp:.3f}")

# Generate comprehensive report
report = propemab.generate_report()
print(report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using PROPERMAB in the DevelopabilityPredictor for comprehensive property prediction
2. Incorporating feature importance into the OptimizationRecommender for targeted optimization
3. Using performance metrics in the FragmentAnalyzer for model monitoring
4. Generating PROPERMAB reports as part of the analysis pipeline

## Future Enhancements

1. **Advanced Models**: Integration with deep learning models for each property
2. **Multi-Task Learning**: Implementation of true multi-task learning approaches
3. **Cross-Property Dependencies**: Modeling dependencies between different properties
4. **Uncertainty Quantification**: Adding uncertainty estimates to predictions
5. **Transfer Learning**: Leveraging pre-trained models for related properties
6. **Active Learning**: Implementing active learning strategies for data-efficient training
7. **Explainable AI**: Adding explainability features to understand model decisions

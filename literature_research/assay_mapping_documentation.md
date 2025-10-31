# PSR/PSP Assay Mapping Documentation

## Overview

This document provides detailed documentation for the PSR/PSP Assay Mapping implementation. This module maps polyreactivity features to PSR/PSP assay data and creates decision rules for antibody developability assessment.

## Features

1. **Polyreactivity Feature Mapping**: Mapping of polyreactivity features to assay data
2. **PSR/PSP Score Calculation**: Calculation of Polyreactivity Specificity Ratio (PSR) and Polyreactivity Specificity Potential (PSP) scores
3. **Decision Rule Application**: Application of decision rules for polyreactivity assessment, developability risk, recommended actions, and priority ranking
4. **Comprehensive Assay Reporting**: Generation of detailed assay reports

## Implementation Details

### AssayMapper Class

The `AssayMapper` class is the core of the implementation:

```python
mapper = AssayMapper()
```

#### Methods

- `map_polyreactivity_features(sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score)`: Map polyreactivity features to assay data and create decision rules
- `generate_assay_report(sequence, charge_imbalance_score, clustering_risk_score, binding_potential, dynamics_risk_score)`: Generate a comprehensive assay report

### Scoring System

#### Polyreactivity Specificity Ratio (PSR)

The PSR is a weighted combination of polyreactivity features (0-1, lower is better):

```
psr = (
    0.3 * charge_imbalance +
    0.3 * clustering_risk +
    0.2 * binding_potential +
    0.2 * dynamics_risk
)
```

#### Polyreactivity Specificity Potential (PSP)

The PSP is a modified version of PSR that considers potential for improvement (0-1, lower is better):

```
# Calculate base PSR
psr = calculate_psr(assay_features)

# Calculate "improvability factor" (0-1, higher means easier to improve)
improvability = 1.0 - psr  # Inverse relationship

# PSP is PSR adjusted by improvability
psp = psr * (1.0 + 0.5 * (1.0 - improvability))
```

### Decision Rules

The implementation applies several decision rules:

#### 1. Polyreactivity Assessment

- PSR < 0.2: Low polyreactivity - favorable for specificity
- PSR 0.2-0.4: Moderate polyreactivity - generally acceptable
- PSR 0.4-0.6: High polyreactivity - may affect specificity
- PSR > 0.6: Very high polyreactivity - likely to cause specificity issues

#### 2. Developability Risk Assessment

- PSP < 0.3: Low developability risk
- PSP 0.3-0.5: Moderate developability risk
- PSP 0.5-0.7: High developability risk
- PSP > 0.7: Very high developability risk

#### 3. Recommended Actions

Based on individual feature scores:

- Charge imbalance > 0.3: Consider charge balancing modifications
- Clustering risk > 0.3: Consider breaking up residue clusters
- Binding potential > 0.4: Consider reducing hydrophobic patches
- Dynamics risk > 0.4: Consider stabilizing paratope dynamics

#### 4. Priority Ranking

- PSR > 0.6 or PSP > 0.7: High - immediate attention required
- PSR > 0.4 or PSP > 0.5: Medium - should be addressed
- Otherwise: Low - monitor but no immediate action needed

## Usage Example

```python
from src.assay_mapping import AssayMapper

# Example sequence
sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"

# Example feature scores (from previous analyses)
charge_imbalance_score = 0.013
clustering_risk_score = 0.760
binding_potential = 0.311
dynamics_risk_score = 0.361

# Create mapper
mapper = AssayMapper()

# Generate comprehensive assay report
assay_report = mapper.generate_assay_report(
    sequence, charge_imbalance_score, clustering_risk_score,
    binding_potential, dynamics_risk_score
)

print(assay_report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Adding assay mapping features to the FragmentAnalyzer output
2. Using PSR/PSP scores as inputs to the DevelopabilityPredictor
3. Incorporating assay mapping results into the OptimizationRecommender

## Future Enhancements

1. **Experimental Data Integration**: Integration with actual PSR/PSP assay data
2. **Machine Learning Models**: Training models to predict assay outcomes
3. **Advanced Decision Rules**: More sophisticated decision rules based on experimental data
4. **Validation Studies**: Experimental validation of the assay mapping approach

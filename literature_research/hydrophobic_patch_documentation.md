# Hydrophobic Patch Analysis Documentation

## Overview

This document provides detailed documentation for the Hydrophobic Patch Analysis implementation. This module identifies and analyzes hydrophobic patches in antibody variable domains for surface binding prediction.

## Features

1. **Hydrophobic Patch Identification**: Identification of hydrophobic patches using a sliding window approach
2. **Patch Metrics Calculation**: Calculation of various metrics including patch counts, densities, and hydrophobic fractions
3. **Binding Potential Scoring**: Scoring method that predicts the likelihood of non-specific binding based on hydrophobic patches

## Implementation Details

### HydrophobicPatchAnalyzer Class

The `HydrophobicPatchAnalyzer` class is the core of the implementation:

```python
analyzer = HydrophobicPatchAnalyzer()
```

#### Methods

- `analyze_hydrophobic_patches(sequence, window_size)`: Analyze hydrophobic patches in a sequence
- `calculate_binding_potential(sequence, window_size)`: Calculate hydrophobic patch binding potential

### Hydrophobic Amino Acids

The implementation uses the following hydrophobic amino acids:

- A (Alanine)
- I (Isoleucine)
- L (Leucine)
- M (Methionine)
- F (Phenylalanine)
- P (Proline)
- W (Tryptophan)
- V (Valine)

### Functionality

#### Hydrophobic Patch Identification

The implementation uses a sliding window approach to identify hydrophobic patches:

1. A window of specified size (default 5) slides across the sequence
2. For each window position, it counts the number of hydrophobic residues
3. If more than half of the residues in the window are hydrophobic, it's considered a patch

#### Patch Metrics Calculation

For the identified patches, the implementation calculates:

- Patch count
- Patch density (patch count / possible window positions)
- Average hydrophobic fraction (average of hydrophobic fractions in all patches)
- Maximum hydrophobic fraction (highest hydrophobic fraction in any patch)
- Patch score (patch count normalized by maximum possible patches)
- Hydrophobicity score (total hydrophobic residues / sequence length)

#### Binding Potential Scoring

The implementation calculates a binding potential score using a weighted approach:

```
binding_potential = (
    0.5 * patch_score +
    0.3 * hydrophobicity_score +
    0.2 * avg_hydrophobic_fraction
)
```

The score is interpreted as follows:

- < 0.2: Low binding potential - favorable for solubility and specificity
- 0.2-0.4: Moderate binding potential - generally acceptable
- 0.4-0.6: High binding potential - may affect solubility and specificity
- > 0.6: Very high binding potential - likely to cause non-specific binding

## Usage Example

```python
from src.hydrophobic_patch import HydrophobicPatchAnalyzer

# Example sequence
sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"

# Create analyzer
analyzer = HydrophobicPatchAnalyzer()

# Calculate binding potential
binding_potential = analyzer.calculate_binding_potential(sequence)

print(f"Binding potential: {binding_potential['binding_potential']:.3f}")
print(f"Interpretation: {binding_potential['interpretation']}")
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Adding hydrophobic patch features to the FragmentAnalyzer output
2. Using binding potential scores as inputs to the DevelopabilityPredictor
3. Incorporating hydrophobic patch analysis into the OptimizationRecommender

## Future Enhancements

1. **3D Structure Integration**: Incorporating structural information for more accurate patch analysis
2. **Surface Exposure Prediction**: Predicting which patches are surface-exposed
3. **Experimental Validation**: Comparing predictions with experimental binding data
4. **Machine Learning Models**: Training models to predict binding-related developability issues

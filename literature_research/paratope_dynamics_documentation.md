# Paratope Dynamics Proxies Documentation

## Overview

This document provides detailed documentation for the Paratope Dynamics Proxies implementation. This module computes proxies for paratope dynamics (entropy of predicted paratope states) in antibody variable domains.

## Features

1. **Paratope Region Analysis**: Analysis of Complementarity Determining Regions (CDRs) for dynamics properties
2. **Dynamics Proxy Calculation**: Calculation of flexibility, rigidity, paratope, and entropy proxies for each CDR region
3. **Dynamics Risk Scoring**: Scoring method that predicts the likelihood of binding instability based on paratope dynamics

## Implementation Details

### ParatopeDynamicsAnalyzer Class

The `ParatopeDynamicsAnalyzer` class is the core of the implementation:

```python
analyzer = ParatopeDynamicsAnalyzer()
```

#### Methods

- `analyze_paratope_dynamics(sequence, cdr_regions)`: Analyze paratope dynamics proxies in a sequence
- `calculate_dynamics_risk_score(sequence, cdr_regions)`: Calculate a comprehensive dynamics risk score

### Amino Acid Property Groups

The implementation uses the following amino acid property groups relevant to paratope dynamics:

- **Paratope Amino Acids**: Y (Tyrosine), S (Serine), G (Glycine), N (Asparagine), F (Phenylalanine), D (Aspartic acid), H (Histidine), W (Tryptophan)
- **Flexible Amino Acids**: G (Glycine), S (Serine)
- **Rigid Amino Acids**: P (Proline), W (Tryptophan)

### Functionality

#### Paratope Region Analysis

The implementation analyzes the Complementarity Determining Regions (CDRs) of antibody variable domains:

1. If CDR regions are not provided, it uses default approximate CDR regions (CDR1: ~24-34, CDR2: ~50-56, CDR3: ~95-106)
2. It extracts the CDR sequences from the variable domain sequence
3. It analyzes each CDR region for dynamics properties

#### Dynamics Proxy Calculation

For each CDR region, the implementation calculates:

- Flexible residue count and flexibility score (flexible residues / region length)
- Rigid residue count and rigidity score (rigid residues / region length)
- Paratope residue count and paratope score (paratope residues / region length)
- Entropy proxy (simplified entropy calculation based on residue composition diversity)

#### Dynamics Risk Scoring

The implementation calculates an overall dynamics score using a weighted approach:

```
dynamics_score = (
    0.4 * avg_flexibility_score +
    0.2 * avg_rigidity_score +
    0.2 * avg_paratope_score +
    0.2 * min(1.0, avg_entropy_proxy / 5.0)  # Normalize entropy
)
```

The score is interpreted as follows:

- < 0.2: Low dynamics - may limit binding flexibility
- 0.2-0.4: Moderate dynamics - generally favorable for binding
- 0.4-0.6: High dynamics - may affect binding specificity
- > 0.6: Very high dynamics - likely to cause binding instability

## Usage Example

```python
from src.paratope_dynamics import ParatopeDynamicsAnalyzer

# Example sequence
sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"

# Define CDR regions (approximate)
cdr_regions = [(23, 33), (49, 55), (94, 105)]

# Create analyzer
analyzer = ParatopeDynamicsAnalyzer()

# Calculate dynamics risk score
dynamics_risk = analyzer.calculate_dynamics_risk_score(sequence, cdr_regions)

print(f"Dynamics risk score: {dynamics_risk['dynamics_risk_score']:.3f}")
print(f"Interpretation: {dynamics_risk['interpretation']}")
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Adding paratope dynamics features to the FragmentAnalyzer output
2. Using dynamics risk scores as inputs to the DevelopabilityPredictor
3. Incorporating paratope dynamics analysis into the OptimizationRecommender

## Future Enhancements

1. **3D Structure Integration**: Incorporating structural information for more accurate dynamics analysis
2. **Experimental Validation**: Comparing predictions with experimental dynamics data
3. **Machine Learning Models**: Training models to predict dynamics-related developability issues
4. **Advanced Entropy Calculation**: Implementing more sophisticated entropy calculations based on structural ensembles

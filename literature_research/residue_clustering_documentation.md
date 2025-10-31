# Residue Clustering Pattern Analysis Documentation

## Overview

This document provides detailed documentation for the Residue Clustering Pattern Analysis implementation. This module implements advanced residue clustering pattern analysis for antibody variable domains, going beyond simple density calculations to identify and analyze clusters of charged, hydrophobic, and aromatic residues.

## Features

1. **Residue Cluster Identification**: Identification of clusters of 2 or more consecutive residues of the same type
2. **Multi-type Clustering Analysis**: Analysis of charged, hydrophobic, and aromatic residue clustering
3. **Comprehensive Cluster Metrics**: Calculation of various metrics including cluster counts, densities, average sizes, and maximum sizes
4. **Clustering Risk Scoring**: Weighted scoring method that considers multiple aspects of residue clustering

## Implementation Details

### ResidueClusteringAnalyzer Class

The `ResidueClusteringAnalyzer` class is the core of the implementation:

```python
analyzer = ResidueClusteringAnalyzer()
```

#### Methods

- `analyze_residue_clustering(sequence)`: Analyze residue clustering patterns in a sequence
- `calculate_clustering_risk_score(sequence)`: Calculate a comprehensive clustering risk score

### Amino Acid Property Groups

The implementation uses the following amino acid property groups:

- **Charged Amino Acids**: K (Lysine), R (Arginine), H (Histidine), D (Aspartic acid), E (Glutamic acid)
- **Hydrophobic Amino Acids**: A (Alanine), I (Isoleucine), L (Leucine), M (Methionine), F (Phenylalanine), P (Proline), W (Tryptophan), V (Valine)
- **Aromatic Amino Acids**: F (Phenylalanine), W (Tryptophan), Y (Tyrosine)

### Functionality

#### Residue Cluster Identification

The implementation identifies clusters of 2 or more consecutive residues of the same type by scanning through the sequence and identifying continuous stretches of residues belonging to the same property group.

#### Cluster Metrics Calculation

For each residue type, the implementation calculates:

- Cluster count
- Cluster density (cluster count / sequence length)
- Average cluster size
- Maximum cluster size

#### Clustering Scoring

The implementation calculates clustering scores for each residue type (0-1, higher means more clustering):

```
clustering_score = min(1.0, (avg_cluster_size * cluster_count) / 10)
```

It then calculates an overall clustering score using a weighted approach:

```
overall_clustering_score = (
    0.4 * charged_clustering_score +
    0.4 * hydrophobic_clustering_score +
    0.2 * aromatic_clustering_score
)
```

The score is interpreted as follows:

- < 0.1: Low clustering - favorable for solubility and stability
- 0.1-0.2: Moderate clustering - generally acceptable
- 0.2-0.3: High clustering - may affect solubility and stability
- > 0.3: Very high clustering - likely to cause developability issues

## Usage Example

```python
from src.residue_clustering import ResidueClusteringAnalyzer

# Example sequence
sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"

# Create analyzer
analyzer = ResidueClusteringAnalyzer()

# Calculate clustering risk score
risk_score = analyzer.calculate_clustering_risk_score(sequence)

print(f"Clustering risk score: {risk_score['clustering_risk_score']:.3f}")
print(f"Interpretation: {risk_score['interpretation']}")
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Adding clustering features to the FragmentAnalyzer output
2. Using clustering risk scores as inputs to the DevelopabilityPredictor
3. Incorporating clustering analysis into the OptimizationRecommender

## Future Enhancements

1. **3D Structure Integration**: Incorporating structural information for more accurate clustering analysis
2. **Experimental Validation**: Comparing predictions with experimental clustering data
3. **Machine Learning Models**: Training models to predict clustering-related developability issues
4. **Dynamic Clustering Analysis**: Modeling how clustering changes under different conditions

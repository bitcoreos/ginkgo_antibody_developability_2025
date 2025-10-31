# Motif-Based Risk Scoring Documentation

## Overview

This document provides detailed documentation for the Motif-Based Risk Scoring implementation. This module identifies and scores problematic motifs in antibody sequences that may affect developability, including aggregation-prone regions, stability issues, cleavage sites, deamidation sites, and isomerization sites.

## Features

1. **Motif Database Management**: Comprehensive database of known problematic motifs
2. **Category-Based Scoring**: Risk scoring organized by motif categories
3. **Advanced Risk Scoring Algorithms**: Diminishing returns model for multiple motif occurrences
4. **Comprehensive Reporting**: Detailed motif scoring reports
5. **Extensible Design**: Ability to add/remove motifs from the database

## Implementation Details

### MotifScorer Class

The `MotifScorer` class is the core of the implementation:

```python
scorer = MotifScorer()
```

#### Methods

- `score_motifs(sequence)`: Score motifs in an antibody sequence
- `generate_motif_report(sequence)`: Generate a comprehensive motif scoring report
- `add_motif(category, motif, risk_score, description)`: Add a new motif to the database
- `remove_motif(category, motif)`: Remove a motif from the database

### Motif Database

The implementation includes a comprehensive database of predefined problematic motifs organized by category:

#### 1. Aggregation-Prone Motifs

- "GGG": Glycine-rich regions associated with aggregation (risk score: 0.8)
- "WWW": Tryptophan-rich regions associated with aggregation (risk score: 0.8)
- "FFFF": Phenylalanine-rich regions associated with aggregation (risk score: 0.8)
- "YYYY": Tyrosine-rich regions associated with aggregation (risk score: 0.7)
- "MMM": Methionine-rich regions susceptible to oxidation (risk score: 0.7)
- "AAAA": Alanine-rich regions associated with aggregation (risk score: 0.6)
- "VVVV": Valine-rich regions associated with aggregation (risk score: 0.6)
- "IIII": Isoleucine-rich regions associated with aggregation (risk score: 0.6)
- "LLLL": Leucine-rich regions associated with aggregation (risk score: 0.6)

#### 2. Stability Issues

- "CC": Cysteine pairs that may form incorrect disulfide bonds (risk score: 0.7)
- "DD": Aspartic acid pairs that may cause isomerization (risk score: 0.6)
- "NN": Asparagine pairs that may cause deamidation (risk score: 0.6)
- "PP": Proline pairs that may affect folding (risk score: 0.5)
- "SS": Serine pairs that may cause unwanted interactions (risk score: 0.4)
- "TT": Threonine pairs that may cause unwanted interactions (risk score: 0.4)

#### 3. Cleavage Sites

- "FR": Phe-Arg motifs associated with proteolytic cleavage (risk score: 0.6)
- "KR": Lys-Arg motifs associated with proteolytic cleavage (risk score: 0.5)
- "RR": Arg-Arg motifs associated with proteolytic cleavage (risk score: 0.5)

#### 4. Deamidation Sites

- "NG": Asn-Gly motifs associated with deamidation (risk score: 0.6)
- "NS": Asn-Ser motifs associated with deamidation (risk score: 0.5)
- "NT": Asn-Thr motifs associated with deamidation (risk score: 0.5)

#### 5. Isomerization Sites

- "DG": Asp-Gly motifs associated with isomerization (risk score: 0.6)
- "DS": Asp-Ser motifs associated with isomerization (risk score: 0.5)
- "DT": Asp-Thr motifs associated with isomerization (risk score: 0.5)

### Risk Scoring System

The implementation uses an advanced risk scoring system:

1. **Base Risk Scores**: Each motif has a predefined risk score between 0 and 1
2. **Adjusted Risk Scores**: For motifs that occur multiple times, a diminishing returns model is used:
   - Adjusted Risk Score = 1 - (1 - Base Risk Score)^Count
3. **Category Scores**: For each category, the maximum adjusted risk score among all motifs in that category is used
4. **Total Risk Score**: The weighted average of all category scores

### Risk Assessment Categories

- Total Risk Score < 0.2: Low risk - few or no problematic motifs identified
- Total Risk Score 0.2-0.4: Moderate risk - some potentially problematic motifs identified
- Total Risk Score 0.4-0.6: High risk - several problematic motifs identified
- Total Risk Score > 0.6: Very high risk - many problematic motifs identified

## Usage Example

```python
from src.motif_scoring import MotifScorer

# Example sequence
sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"

# Create scorer
scorer = MotifScorer()

# Score motifs
motif_results = scorer.score_motifs(sequence)

print(f"Total Risk Score: {motif_results['total_risk_score']:.3f}")

# Generate comprehensive motif report
motif_report = scorer.generate_motif_report(sequence)

print(motif_report['summary'])

# Add a new motif
scorer.add_motif("custom_category", "QQQ", 0.4, "Glutamine-rich regions")
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Adding motif scoring features to the FragmentAnalyzer output
2. Using motif risk scores as inputs to the DevelopabilityPredictor
3. Incorporating motif scoring results into the OptimizationRecommender

## Future Enhancements

1. **Machine Learning Models**: Training models to predict motif risk scores
2. **Structural Data Integration**: Integration with structural data for more accurate motif identification
3. **Experimental Validation**: Experimental validation of identified motifs
4. **Motif Database Expansion**: Expansion of the motif database with new findings
5. **Context-Aware Scoring**: Adjusting motif risk scores based on sequence context

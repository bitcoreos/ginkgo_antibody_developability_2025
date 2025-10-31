# VH/VL Charge Imbalance Analysis Documentation

## Overview

This document provides detailed documentation for the VH/VL Charge Imbalance Analysis implementation. This module implements advanced charge imbalance analysis for antibody variable domains, going beyond basic net charge calculations to provide more sophisticated predictions of antibody behavior.

## Features

1. **Detailed Charge Distribution Analysis**: Analysis of charge distribution within individual VH and VL domains
2. **Charge Pairing Analysis**: Analysis of charge pairing between VH and VL domains
3. **Comprehensive Charge Imbalance Scoring**: Weighted scoring method that considers multiple aspects of charge imbalance

## Implementation Details

### ChargeImbalanceAnalyzer Class

The `ChargeImbalanceAnalyzer` class is the core of the implementation:

```python
analyzer = ChargeImbalanceAnalyzer()
```

#### Methods

- `analyze_charge_distribution(sequence)`: Analyze detailed charge distribution in a sequence
- `analyze_charge_pairing(vh_sequence, vl_sequence)`: Analyze charge pairing between VH and VL domains
- `calculate_charge_imbalance_score(vh_sequence, vl_sequence)`: Calculate a comprehensive charge imbalance score

### Amino Acid Charge Properties

The implementation uses the following amino acid charge properties:

- **Positive Charge Amino Acids**: K (Lysine), R (Arginine), H (Histidine)
- **Negative Charge Amino Acids**: D (Aspartic acid), E (Glutamic acid)

### Functionality

#### Charge Distribution Analysis

For a given sequence, the implementation calculates:

- Positive residue count
- Negative residue count
- Net charge (positive count - negative count)
- Charge density ((positive count + negative count) / sequence length)
- Positive density (positive count / sequence length)
- Negative density (negative count / sequence length)
- Charge balance (abs(net charge) / sequence length)

#### Charge Pairing Analysis

For VH/VL domain pairs, the implementation calculates:

- Individual charge analysis for each domain
- Pairing imbalance (absolute difference in net charge)
- Pairing compatibility (absolute sum of net charges)
- Pairing score (pairing imbalance / total length)

#### Charge Imbalance Scoring

The implementation calculates a comprehensive charge imbalance score using a weighted approach:

```
imbalance_score = (
    0.5 * pairing_score +
    0.25 * vh_charge_balance +
    0.25 * vl_charge_balance
)
```

The score is interpreted as follows:

- < 0.1: Low charge imbalance - favorable for stability and solubility
- 0.1-0.2: Moderate charge imbalance - generally acceptable
- 0.2-0.3: High charge imbalance - may affect stability and solubility
- > 0.3: Very high charge imbalance - likely to cause developability issues

## Usage Example

```python
from src.charge_imbalance import ChargeImbalanceAnalyzer

# Example VH and VL sequences
vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYGSSPYTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

# Create analyzer
analyzer = ChargeImbalanceAnalyzer()

# Calculate comprehensive charge imbalance score
imbalance_score = analyzer.calculate_charge_imbalance_score(vh_sequence, vl_sequence)

print(f"Imbalance score: {imbalance_score['imbalance_score']:.3f}")
print(f"Interpretation: {imbalance_score['interpretation']}")
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Adding charge imbalance features to the FragmentAnalyzer output
2. Using charge imbalance scores as inputs to the DevelopabilityPredictor
3. Incorporating charge imbalance analysis into the OptimizationRecommender

## Future Enhancements

1. **pH-dependent Charge Analysis**: Modeling how charge distribution changes with pH
2. **Structural Context**: Incorporating structural information for more accurate charge analysis
3. **Experimental Validation**: Comparing predictions with experimental charge measurement data
4. **Machine Learning Models**: Training models to predict charge-related developability issues

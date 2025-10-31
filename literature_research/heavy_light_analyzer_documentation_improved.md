# Heavy-Light Coupling Analysis Documentation

## Overview

This document provides detailed documentation for the Heavy-Light Coupling Analysis implementation. This module implements heavy-light chain coupling analysis for antibody developability, including VH-VL pairing compatibility and isotype-specific feature engineering.

## Features

1. **VH-VL Pairing Analysis**: Compatibility analysis based on charge, hydrophobicity, and length
2. **Isotype-Specific Feature Engineering**: Analysis of isotype-related properties
3. **Gene Family Prediction**: Prediction of VH and VL gene families based on sequence properties
4. **Comprehensive Coupling Reports**: Detailed reports with pairing and isotype analysis

## Implementation Details

### HeavyLightAnalyzer Class

The `HeavyLightAnalyzer` class is the core of the implementation:

```python
analyzer = HeavyLightAnalyzer()
```

#### Methods

- `analyze_vh_vl_pairing(vh_sequence, vl_sequence, vh_gene, vl_gene)`: Analyze VH-VL pairing compatibility
- `analyze_isotype_features(heavy_chain_sequence, light_chain_sequence, isotype)`: Analyze isotype-specific features
- `generate_coupling_report(vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, isotype)`: Generate a comprehensive heavy-light coupling report

### VH-VL Pairing Analysis

The VH-VL pairing analysis evaluates compatibility based on three factors:

1. **Charge Compatibility**: Compatibility based on charge pairing rules between VH and VL gene families
2. **Hydrophobicity Compatibility**: Compatibility based on hydrophobicity pairing rules between VH and VL gene families
3. **Length Compatibility**: Compatibility based on the length difference between VH and VL domains, with automatic extraction of VL domains from full light chain sequences

The overall pairing score is calculated as a weighted combination of these factors:
- Charge compatibility: 40%
- Hydrophobicity compatibility: 40%
- Length compatibility: 20%

Note: When full light chain sequences are provided, the VL domain is automatically extracted for length compatibility calculation.

```python
pairing_analysis = analyzer.analyze_vh_vl_pairing(vh_sequence, vl_sequence)
```

### Isotype-Specific Feature Analysis

The isotype-specific feature analysis evaluates properties related to the antibody isotype:

1. **Flexibility Score**: Based on the isotype's inherent flexibility and proline content in the heavy chain
2. **Effector Function Score**: Based on the isotype's effector function capacity and glycine content in the heavy chain

The overall isotype score is calculated as a weighted combination of these factors:
- Flexibility score: 50%
- Effector function score: 50%

```python
isotype_analysis = analyzer.analyze_isotype_features(heavy_chain_sequence, light_chain_sequence, 'IgG1')
```

### Gene Family Prediction

The implementation includes a simple gene family prediction method based on sequence properties:

- For VH chains: Prediction based on sequence length
- For VL chains: Prediction based on kappa/lambda content in the CDR3 region

### Supported Isotypes

The implementation includes information about various antibody isotypes:

- IgG1, IgG2, IgG3, IgG4
- IgA1, IgA2
- IgM, IgE, IgD

Each isotype has associated properties such as flexibility and effector function capacity.

## Usage Example

```python
from src.heavy_light_analyzer import HeavyLightAnalyzer

# Example sequences
vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
heavy_chain_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
light_chain_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"

# Create analyzer
analyzer = HeavyLightAnalyzer()

# Analyze VH-VL pairing
pairing_analysis = analyzer.analyze_vh_vl_pairing(vh_sequence, vl_sequence)
print(f"Pairing Score: {pairing_analysis['pairing_score']:.3f}")

# Analyze isotype features
isotype_analysis = analyzer.analyze_isotype_features(heavy_chain_sequence, light_chain_sequence, 'IgG1')
print(f"Isotype Score: {isotype_analysis['isotype_score']:.3f}")

# Generate comprehensive coupling report
coupling_report = analyzer.generate_coupling_report(vh_sequence, vl_sequence, heavy_chain_sequence, light_chain_sequence, 'IgG1')
print(coupling_report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using pairing analysis in the DevelopabilityPredictor for more accurate predictions
2. Incorporating isotype-specific features into the FragmentAnalyzer
3. Using coupling reports in the OptimizationRecommender for pairing optimization suggestions
4. Storing coupling analysis results in the FragmentDatabase for rapid comparison

## Future Enhancements

1. **Advanced Pairing Models**: More sophisticated models for predicting VH-VL compatibility
2. **Structural Data Integration**: Integration with structural data for more accurate pairing analysis
3. **Experimental Validation**: Experimental validation of pairing predictions
4. **Isotype Engineering**: Advanced isotype engineering strategies
5. **Context-Aware Analysis**: Pairing analysis that considers the antigen context
6. **Machine Learning Models**: Training models to predict pairing compatibility
7. **Dynamic Isotype Switching**: Analysis of isotype switching effects

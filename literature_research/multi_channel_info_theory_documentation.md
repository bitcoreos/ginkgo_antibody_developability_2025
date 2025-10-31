# Multi-Channel Information Theory Framework Documentation

## Overview

This document provides detailed documentation for the Multi-Channel Information Theory Framework implementation. This framework integrates sequence, structure, and temporal dynamics channels to provide a comprehensive information-theoretic analysis of antibody developability.

## Features

1. **Multi-Channel Integration**: Integrates information from sequence, structure, and temporal channels
2. **Information Metrics**: Computes entropy, variance, and mean for each channel
3. **Cross-Channel Analysis**: Analyzes relationships between different channels
4. **Mutual Information**: Computes mutual information between channels
5. **Cross-Correlation**: Computes cross-correlation between channels
6. **Cosine Similarity**: Computes cosine similarity between channels
7. **Comprehensive Reporting**: Generates detailed reports on information-theoretic analysis

## Implementation Details

### MultiChannelInfoTheory Class

The `MultiChannelInfoTheory` class is the core of the implementation:

```python
info_theory = MultiChannelInfoTheory()
```

#### Methods

- `add_channel_data(channel, data)`: Add data for a specific channel
- `compute_entropy(data)`: Compute entropy of data
- `compute_mutual_information(x, y)`: Compute mutual information between two variables
- `compute_channel_information(channel)`: Compute information metrics for a specific channel
- `compute_cross_channel_information(channel1, channel2)`: Compute cross-channel information metrics
- `analyze_all_channels()`: Analyze information across all channels
- `get_channel_data(channel)`: Get data for a specific channel
- `get_information_metrics(channel)`: Get information metrics for one or all channels
- `get_cross_channel_metrics(channel1, channel2)`: Get cross-channel metrics for a specific pair or all pairs
- `generate_report()`: Generate a comprehensive Multi-Channel Information Theory report

### Channels

The framework supports three channels:

1. **Sequence Channel**: Information about amino acid sequences
2. **Structure Channel**: Information about structural properties
3. **Temporal Channel**: Information about temporal dynamics

```python
self.channels = ['sequence', 'structure', 'temporal']
```

### Information Metrics

The framework computes several information metrics for each channel:

1. **Entropy**: Measures the uncertainty or randomness in the data
2. **Variance**: Measures the spread of the data
3. **Mean**: Measures the central tendency of the data

```python
# Entropy
metrics['entropy'] = self.compute_entropy(data.flatten())

# Variance
metrics['variance'] = np.var(data)

# Mean
metrics['mean'] = np.mean(data)
```

### Cross-Channel Analysis

The framework analyzes relationships between different channels:

1. **Mutual Information**: Measures the mutual dependence between channels
2. **Cross-Correlation**: Measures the similarity between channels as a function of displacement
3. **Cosine Similarity**: Measures the cosine of the angle between channel vectors

```python
# Mutual information
metrics['mutual_information'] = self.compute_mutual_information(data1, data2)

# Cross-correlation
cross_corr = np.correlate(data1.flatten(), data2.flatten(), mode='valid')
metrics['cross_correlation'] = np.mean(cross_corr) if len(cross_corr) > 0 else 0.0

# Cosine similarity
cos_sim = cosine_similarity(data1.reshape(1, -1), data2.reshape(1, -1))[0, 0]
metrics['cosine_similarity'] = cos_sim
```

## Usage Example

```python
from src.multi_channel_info_theory import MultiChannelInfoTheory
import numpy as np

# Generate example data for different channels
np.random.seed(42)
seq_len = 50

# Sequence channel data (e.g., amino acid frequencies)
sequence_data = np.random.rand(seq_len)

# Structure channel data (e.g., secondary structure propensities)
structure_data = np.random.rand(seq_len)

# Temporal channel data (e.g., stability over time)
temporal_data = np.random.rand(seq_len)

# Create Multi-Channel Information Theory Framework
info_theory = MultiChannelInfoTheory()

# Add data for each channel
info_theory.add_channel_data('sequence', sequence_data)
info_theory.add_channel_data('structure', structure_data)
info_theory.add_channel_data('temporal', temporal_data)

# Analyze all channels
analysis_results = info_theory.analyze_all_channels()

# Get information metrics for sequence channel
seq_metrics = info_theory.get_information_metrics('sequence')
for metric, value in seq_metrics.items():
    print(f"{metric}: {value:.4f}")

# Get cross-channel metrics
cross_metrics = info_theory.get_cross_channel_metrics('sequence', 'structure')
for metric, value in cross_metrics.items():
    print(f"{metric}: {value:.4f}")

# Generate comprehensive report
report = info_theory.generate_report()
print(report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using the framework in the DevelopabilityPredictor for information-theoretic feature analysis
2. Incorporating information metrics into the OptimizationRecommender for understanding feature importance
3. Using cross-channel analysis in the FragmentAnalyzer for multi-modal property analysis
4. Generating information theory reports as part of the analysis pipeline

## Future Enhancements

1. **Advanced Information Measures**: Integration with more sophisticated information-theoretic measures
2. **Dynamic Information Theory**: Implementing time-varying information measures
3. **Multivariate Information Theory**: Extending to multivariate information measures
4. **Information Bottleneck**: Implementing information bottleneck methods for feature selection
5. **Transfer Entropy**: Adding transfer entropy for causal analysis between channels
6. **Information Visualization**: Adding tools for visualizing information-theoretic relationships
7. **Channel-Specific Metrics**: Implementing specialized metrics for each channel type

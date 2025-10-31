# Cross-Attention Mechanisms Documentation

## Overview

This document provides detailed documentation for the Cross-Attention Mechanisms implementation. Cross-attention mechanisms are used for fusing structural and sequential representations, implementing attention-based fusion of different modalities in antibody developability prediction.

## Features

1. **Cross-Modal Attention**: Attention mechanisms for fusing different representation types
2. **Multi-Head Attention**: Multiple attention heads for capturing different aspects of relationships
3. **Scaled Dot-Product Attention**: Efficient attention computation with scaling
4. **Bidirectional Fusion**: Fusion in both directions (sequential to structural and vice versa)
5. **Representation Similarity**: Computation of similarity between original and fused representations
6. **Attention Weight Tracking**: Maintains and provides access to attention weights

## Implementation Details

### CrossAttention Class

The `CrossAttention` class is the core of the implementation:

```python
cross_attention = CrossAttention(embedding_dim=64, attention_heads=2)
```

#### Methods

- `scaled_dot_product_attention(queries, keys, values, mask)`: Compute scaled dot-product attention
- `multi_head_attention(query_seq, key_seq, value_seq, mask)`: Compute multi-head attention
- `fuse_representations(sequential_repr, structural_repr)`: Fuse sequential and structural representations using cross-attention
- `get_attention_weights()`: Get attention weights
- `generate_report()`: Generate a comprehensive Cross-Attention report

### Cross-Modal Attention

The implementation computes attention between different modalities:

1. **Sequential to Structural**: Uses sequential representations as queries and structural representations as keys/values
2. **Structural to Sequential**: Uses structural representations as queries and sequential representations as keys/values
3. **Bidirectional Fusion**: Combines both directions of attention

```python
# Cross-attention: sequential queries, structural keys and values
fused_seq_to_struct, attention_weights_seq_to_struct = self.multi_head_attention(
    sequential_repr, structural_repr, structural_repr
)
```

### Multi-Head Attention

The implementation uses multiple attention heads:

1. **Multiple Perspectives**: Each head captures different aspects of relationships
2. **Parallel Processing**: Heads are processed in parallel
3. **Concatenation**: Outputs from all heads are concatenated

```python
# Process each attention head
for head in range(self.attention_heads):
    # Apply linear transformations
    queries = np.dot(query_seq, self.query_weights[head])
    keys = np.dot(key_seq, self.key_weights[head])
    values = np.dot(value_seq, self.value_weights[head])
```

### Scaled Dot-Product Attention

The implementation uses scaled dot-product attention:

1. **Dot Product**: Computes similarity between queries and keys
2. **Scaling**: Scales by square root of embedding dimension for stability
3. **Softmax**: Applies softmax to obtain attention weights
4. **Weighted Sum**: Computes weighted sum of values

```python
# Compute attention scores
scores = np.dot(queries, keys.T)

# Scale by square root of embedding dimension
scores = scores / np.sqrt(self.embedding_dim)

# Apply softmax to get attention weights
attention_weights = softmax(scores)

# Compute attention output
attention_output = np.dot(attention_weights, values)
```

### Representation Fusion

The implementation fuses representations in both directions:

1. **Sequential to Structural Fusion**: Enhances sequential representations with structural information
2. **Structural to Sequential Fusion**: Enhances structural representations with sequential information
3. **Bidirectional Fusion**: Combines both directions for comprehensive fusion

```python
# Combine both directions of fusion
bidirectional_fusion = fused_seq_to_struct + fused_struct_to_seq
```

## Usage Example

```python
from src.cross_attention import CrossAttention
import numpy as np

# Generate example embeddings
np.random.seed(42)
seq_len = 10
embedding_dim = 64

# Sequential representations (e.g., protein sequence embeddings)
sequential_repr = np.random.randn(seq_len, embedding_dim)

# Structural representations (e.g., 3D structure embeddings)
structural_repr = np.random.randn(seq_len, embedding_dim)

# Create Cross-Attention mechanism
cross_attention = CrossAttention(embedding_dim=embedding_dim, attention_heads=2)

# Fuse representations
fusion_results = cross_attention.fuse_representations(sequential_repr, structural_repr)

print(f"Sequential representations shape: {sequential_repr.shape}")
print(f"Structural representations shape: {structural_repr.shape}")
print(f"Fused sequential shape: {fusion_results['fused_sequential'].shape}")
print(f"Fused structural shape: {fusion_results['fused_structural'].shape}")
print(f"Bidirectional fusion shape: {fusion_results['bidirectional_fusion'].shape}")

# Check similarity between original and fused representations
seq_sim = fusion_results['seq_similarity']
struct_sim = fusion_results['struct_similarity']
print(f"Average sequential similarity: {np.mean(seq_sim):.3f}")
print(f"Average structural similarity: {np.mean(struct_sim):.3f}")

# Get attention weights
attention_weights = cross_attention.get_attention_weights()
print(f"Query weights shape: {attention_weights['query_weights'].shape}")

# Generate comprehensive report
report = cross_attention.generate_report()
print(report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using cross-attention in the DevelopabilityPredictor for fusing sequence and structure features
2. Incorporating attention weights into the OptimizationRecommender for understanding feature relationships
3. Using fused representations in the FragmentAnalyzer for enhanced property analysis
4. Generating cross-attention reports as part of the analysis pipeline

## Future Enhancements

1. **Advanced Attention Mechanisms**: Integration with more sophisticated attention mechanisms
2. **Hierarchical Attention**: Implementing hierarchical attention for multi-level fusion
3. **Sparse Attention**: Implementing sparse attention for computational efficiency
4. **Learned Positional Encodings**: Adding learned positional encodings for sequential data
5. **Cross-Modal Contrastive Learning**: Implementing contrastive learning between modalities
6. **Attention Visualization**: Adding tools for visualizing attention patterns
7. **Dynamic Attention**: Implementing dynamic attention that adapts to input characteristics

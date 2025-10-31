# Protein Sequence Embeddings Documentation

## Overview

This document provides detailed documentation for the Protein Sequence Embeddings implementation. This module implements protein sequence embeddings for feature extraction using various approaches including property-based embeddings, one-hot embeddings, k-mer embeddings, and combined embeddings.

## Features

1. **Multiple Embedding Approaches**: Property-based, one-hot, k-mer, and combined embeddings
2. **Embedding Similarity Calculation**: Cosine similarity between embeddings
3. **Comprehensive Embedding Reports**: Detailed reports with embedding information
4. **Extensible Design**: Modular design that can be extended with more advanced embedding methods

## Implementation Details

### ProteinEmbedder Class

The `ProteinEmbedder` class is the core of the implementation:

```python
embedder = ProteinEmbedder()
```

#### Methods

- `generate_property_embedding(sequence)`: Generate embedding based on amino acid properties
- `generate_onehot_embedding(sequence)`: Generate one-hot encoding embedding
- `generate_kmer_embedding(sequence, k)`: Generate k-mer frequency embedding
- `generate_combined_embedding(sequence)`: Generate combined embedding from multiple sources
- `calculate_embedding_similarity(embedding1, embedding2)`: Calculate similarity between two embeddings
- `generate_embedding_report(sequence)`: Generate a comprehensive embedding report

### Embedding Types

#### 1. Property-Based Embeddings

Property-based embeddings are generated using amino acid properties such as hydrophobicity, charge, polarity, and volume. The embedding is a 4-dimensional vector representing the average values of these properties across the sequence.

```python
property_embedding = embedder.generate_property_embedding(sequence)
```

#### 2. One-Hot Embeddings

One-hot embeddings represent the amino acid composition of the sequence. The embedding is a 20-dimensional vector (one for each amino acid) representing the normalized frequency of each amino acid in the sequence.

```python
onehot_embedding = embedder.generate_onehot_embedding(sequence)
```

#### 3. K-mer Embeddings

K-mer embeddings represent the frequency of k-length amino acid subsequences in the sequence. For a k-mer length of 2, the embedding is a 400-dimensional vector (20^2) representing the normalized frequency of each possible dipeptide.

```python
kmer_embedding = embedder.generate_kmer_embedding(sequence, k=2)
```

#### 4. Combined Embeddings

Combined embeddings concatenate property-based, one-hot, and k-mer embeddings into a single high-dimensional vector. For the default configuration (property: 4, one-hot: 20, k-mer k=2: 400), the combined embedding is a 424-dimensional vector.

```python
combined_embedding = embedder.generate_combined_embedding(sequence)
```

### Embedding Similarity

The implementation includes functionality for calculating the cosine similarity between two embeddings:

```python
similarity = embedder.calculate_embedding_similarity(embedding1, embedding2)
```

### Embedding Reports

The implementation can generate comprehensive embedding reports that include all embedding types and summary information:

```python
embedding_report = embedder.generate_embedding_report(sequence)
```

## Usage Example

```python
from src.protein_embeddings import ProteinEmbedder

# Example sequence
sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"

# Create embedder
embedder = ProteinEmbedder()

# Generate property-based embedding
property_embedding = embedder.generate_property_embedding(sequence)
print(f"Property embedding shape: {property_embedding['embedding'].shape}")

# Generate one-hot embedding
onehot_embedding = embedder.generate_onehot_embedding(sequence)
print(f"One-hot embedding shape: {onehot_embedding['embedding'].shape}")

# Generate k-mer embedding
kmer_embedding = embedder.generate_kmer_embedding(sequence, 2)
print(f"K-mer embedding shape: {kmer_embedding['embedding'].shape}")

# Generate combined embedding
combined_embedding = embedder.generate_combined_embedding(sequence)
print(f"Combined embedding shape: {combined_embedding['embedding'].shape}")

# Calculate similarity between embeddings
similarity = embedder.calculate_embedding_similarity(
    property_embedding['embedding'], 
    property_embedding['embedding']
)
print(f"Self-similarity: {similarity:.3f}")

# Generate comprehensive embedding report
embedding_report = embedder.generate_embedding_report(sequence)
print(embedding_report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using embeddings as additional features in the FragmentAnalyzer
2. Incorporating embedding similarity into the OptimizationRecommender for sequence comparison
3. Using embeddings in the DevelopabilityPredictor as additional input features
4. Storing embeddings in the FragmentDatabase for rapid sequence comparison

## Future Enhancements

1. **Advanced Language Models**: Integration with pre-trained protein language models (e.g., ESM, ProtBERT)
2. **Structural Embeddings**: Integration with structural information for more accurate embeddings
3. **Context-Aware Embeddings**: Embeddings that consider the context of each amino acid in the sequence
4. **Embedding Visualization**: Tools for visualizing and interpreting embeddings
5. **Embedding-Based Clustering**: Clustering sequences based on their embeddings
6. **Transfer Learning**: Using embeddings for transfer learning to related tasks
7. **Embedding Refinement**: Refining embeddings based on experimental data

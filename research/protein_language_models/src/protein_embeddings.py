"""
Protein Sequence Embeddings Module

This module implements protein sequence embeddings for feature extraction.
"""

import numpy as np
from typing import Dict, List, Union
from collections import Counter

# Amino acid properties for embedding generation
AMINO_ACID_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'polarity': 0, 'volume': 88.6},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'polarity': 1, 'volume': 173.4},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'volume': 114.1},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'volume': 111.1},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'polarity': 0, 'volume': 108.5},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'volume': 143.9},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'volume': 138.4},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'polarity': 0, 'volume': 60.1},
    'H': {'hydrophobicity': -3.2, 'charge': 1, 'polarity': 1, 'volume': 153.2},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'polarity': 0, 'volume': 166.7},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'polarity': 0, 'volume': 166.7},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'polarity': 1, 'volume': 168.6},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'polarity': 0, 'volume': 162.9},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'polarity': 0, 'volume': 189.9},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'polarity': 0, 'volume': 112.7},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'polarity': 1, 'volume': 89.0},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'polarity': 1, 'volume': 116.1},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'polarity': 0, 'volume': 227.8},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'polarity': 1, 'volume': 193.6},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'polarity': 0, 'volume': 140.0}
}

# Amino acid one-hot encoding
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class ProteinEmbedder:
    """
    Embedder for protein sequences.
    """
    
    def __init__(self):
        """
        Initialize the protein embedder.
        """
        pass
    
    def generate_property_embedding(self, sequence: str) -> Dict[str, Union[np.ndarray, int, bool]]:
        """
        Generate embedding based on amino acid properties.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Property-based embedding
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'embedding': np.zeros(4),
                'embedding_complete': True
            }
        
        # Calculate property averages
        hydrophobicity = 0.0
        charge = 0.0
        polarity = 0.0
        volume = 0.0
        
        valid_count = 0
        for aa in sequence:
            if aa in AMINO_ACID_PROPERTIES:
                props = AMINO_ACID_PROPERTIES[aa]
                hydrophobicity += props['hydrophobicity']
                charge += props['charge']
                polarity += props['polarity']
                volume += props['volume']
                valid_count += 1
        
        if valid_count > 0:
            hydrophobicity /= valid_count
            charge /= valid_count
            polarity /= valid_count
            volume /= valid_count
        
        # Create embedding vector
        embedding = np.array([hydrophobicity, charge, polarity, volume])
        
        return {
            'sequence': sequence,
            'length': length,
            'embedding': embedding,
            'embedding_complete': True
        }
    
    def generate_onehot_embedding(self, sequence: str) -> Dict[str, Union[np.ndarray, int, bool]]:
        """
        Generate one-hot encoding embedding.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: One-hot encoding embedding
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'embedding': np.zeros(len(AMINO_ACIDS)),
                'embedding_complete': True
            }
        
        # Calculate amino acid composition
        aa_counts = Counter(sequence)
        
        # Create one-hot-like embedding (normalized composition)
        embedding = np.zeros(len(AMINO_ACIDS))
        for aa, count in aa_counts.items():
            if aa in AA_TO_INDEX:
                index = AA_TO_INDEX[aa]
                embedding[index] = count / length
        
        return {
            'sequence': sequence,
            'length': length,
            'embedding': embedding,
            'embedding_complete': True
        }
    
    def generate_kmer_embedding(self, sequence: str, k: int = 3) -> Dict[str, Union[np.ndarray, int, bool]]:
        """
        Generate k-mer frequency embedding.
        
        Args:
            sequence (str): Amino acid sequence
            k (int): Length of k-mers
            
        Returns:
            Dict: K-mer frequency embedding
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length < k:
            return {
                'sequence': sequence,
                'length': length,
                'k': k,
                'embedding': np.zeros(len(AMINO_ACIDS) ** k),
                'embedding_complete': True
            }
        
        # Generate all possible k-mers
        possible_kmers = []
        for i in range(len(AMINO_ACIDS) ** k):
            kmer = ''
            temp = i
            for _ in range(k):
                kmer = AMINO_ACIDS[temp % len(AMINO_ACIDS)] + kmer
                temp //= len(AMINO_ACIDS)
            possible_kmers.append(kmer)
        
        # Count k-mers in sequence
        kmer_counts = Counter()
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if all(aa in AMINO_ACIDS for aa in kmer):
                kmer_counts[kmer] += 1
        
        # Create embedding vector
        embedding = np.zeros(len(possible_kmers))
        for i, kmer in enumerate(possible_kmers):
            if kmer in kmer_counts:
                embedding[i] = kmer_counts[kmer] / (length - k + 1)
        
        return {
            'sequence': sequence,
            'length': length,
            'k': k,
            'embedding': embedding,
            'embedding_complete': True
        }
    
    def generate_combined_embedding(self, sequence: str) -> Dict[str, Union[np.ndarray, int, bool]]:
        """
        Generate combined embedding from multiple sources.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Combined embedding
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'embedding': np.zeros(4 + len(AMINO_ACIDS) + len(AMINO_ACIDS) ** 2),
                'embedding_complete': True
            }
        
        # Generate individual embeddings
        property_embedding = self.generate_property_embedding(sequence)['embedding']
        onehot_embedding = self.generate_onehot_embedding(sequence)['embedding']
        kmer_embedding = self.generate_kmer_embedding(sequence, 2)['embedding']
        
        # Combine embeddings
        combined_embedding = np.concatenate([property_embedding, onehot_embedding, kmer_embedding])
        
        return {
            'sequence': sequence,
            'length': length,
            'embedding': combined_embedding,
            'embedding_complete': True
        }
    
    def calculate_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings using cosine similarity.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity between embeddings
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def generate_embedding_report(self, sequence: str) -> Dict[str, Union[str, np.ndarray, float]]:
        """
        Generate a comprehensive embedding report.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Comprehensive embedding report
        """
        # Generate embeddings
        property_embedding = self.generate_property_embedding(sequence)
        onehot_embedding = self.generate_onehot_embedding(sequence)
        kmer_embedding = self.generate_kmer_embedding(sequence, 2)
        combined_embedding = self.generate_combined_embedding(sequence)
        
        # Generate summary
        summary = f"""
Protein Embedding Report
=======================

Sequence Length: {property_embedding['length']}

Embedding Dimensions:
- Property-based: {len(property_embedding['embedding'])}
- One-hot: {len(onehot_embedding['embedding'])}
- K-mer (k=2): {len(kmer_embedding['embedding'])}
- Combined: {len(combined_embedding['embedding'])}
"""
        
        return {
            'sequence': sequence,
            'length': property_embedding['length'],
            'property_embedding': property_embedding['embedding'],
            'onehot_embedding': onehot_embedding['embedding'],
            'kmer_embedding': kmer_embedding['embedding'],
            'combined_embedding': combined_embedding['embedding'],
            'embedding_dimensions': {
                'property': len(property_embedding['embedding']),
                'onehot': len(onehot_embedding['embedding']),
                'kmer': len(kmer_embedding['embedding']),
                'combined': len(combined_embedding['embedding'])
            },
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the protein embedder.
    """
    # Example sequence
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Create embedder
    embedder = ProteinEmbedder()
    
    # Generate property-based embedding
    property_embedding = embedder.generate_property_embedding(sequence)
    print("Property-Based Embedding:")
    print(f"  Sequence Length: {property_embedding['length']}")
    print(f"  Embedding Shape: {property_embedding['embedding'].shape}")
    print(f"  Embedding Values: {property_embedding['embedding']}")
    
    # Generate one-hot embedding
    onehot_embedding = embedder.generate_onehot_embedding(sequence)
    print("\nOne-Hot Embedding:")
    print(f"  Sequence Length: {onehot_embedding['length']}")
    print(f"  Embedding Shape: {onehot_embedding['embedding'].shape}")
    print(f"  Embedding Values (first 10): {onehot_embedding['embedding'][:10]}")
    
    # Generate k-mer embedding
    kmer_embedding = embedder.generate_kmer_embedding(sequence, 2)
    print("\nK-mer Embedding (k=2):")
    print(f"  Sequence Length: {kmer_embedding['length']}")
    print(f"  K-mer Length: {kmer_embedding['k']}")
    print(f"  Embedding Shape: {kmer_embedding['embedding'].shape}")
    print(f"  Embedding Values (first 10): {kmer_embedding['embedding'][:10]}")
    
    # Generate combined embedding
    combined_embedding = embedder.generate_combined_embedding(sequence)
    print("\nCombined Embedding:")
    print(f"  Sequence Length: {combined_embedding['length']}")
    print(f"  Embedding Shape: {combined_embedding['embedding'].shape}")
    print(f"  Embedding Values (first 10): {combined_embedding['embedding'][:10]}")
    
    # Calculate similarity between embeddings
    similarity = embedder.calculate_embedding_similarity(
        property_embedding['embedding'], 
        property_embedding['embedding']
    )
    print(f"\nSelf-similarity of property embedding: {similarity:.3f}")
    
    # Generate comprehensive embedding report
    embedding_report = embedder.generate_embedding_report(sequence)
    print("\nEmbedding Report Summary:")
    print(embedding_report['summary'])


if __name__ == "__main__":
    main()

"""
Protein Language Model & Embedding Strategies Implementation

This module implements protein sequence embeddings, transformer-based representations,
and embedding-based similarity and anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Amino acid alphabet
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


class KmerEmbedding:
    """
    K-mer based protein sequence embedding.
    """
    
    def __init__(self, k: int = 3, normalize: bool = True):
        """
        Initialize the k-mer embedding.
        
        Args:
            k (int): Length of k-mers
            normalize (bool): Whether to normalize the embedding vectors
        """
        self.k = k
        self.normalize = normalize
        self.kmer_vocab = self._generate_kmer_vocabulary()
        self.kmer_to_index = {kmer: i for i, kmer in enumerate(self.kmer_vocab)}
    
    def _generate_kmer_vocabulary(self) -> List[str]:
        """
        Generate vocabulary of all possible k-mers.
        
        Returns:
            List[str]: List of all possible k-mers
        """
        import itertools
        kmers = [''.join(kmer) for kmer in itertools.product(AMINO_ACIDS, repeat=self.k)]
        return sorted(kmers)
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a protein sequence as a k-mer frequency vector.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: K-mer frequency vector
        """
        # Initialize embedding vector
        embedding = np.zeros(len(self.kmer_vocab))
        
        # Count k-mers in sequence
        kmer_counts = defaultdict(int)
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            if all(aa in AMINO_ACIDS for aa in kmer):  # Only count valid k-mers
                kmer_counts[kmer] += 1
        
        # Fill embedding vector
        for kmer, count in kmer_counts.items():
            if kmer in self.kmer_to_index:
                embedding[self.kmer_to_index[kmer]] = count
        
        # Normalize if requested
        if self.normalize and np.sum(embedding) > 0:
            embedding = embedding / np.sum(embedding)
        
        return embedding
    
    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Encode multiple protein sequences.
        
        Args:
            sequences (List[str]): List of protein sequences
            
        Returns:
            np.ndarray: Matrix of k-mer frequency vectors
        """
        embeddings = []
        for sequence in sequences:
            embedding = self.encode_sequence(sequence)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class PositionalEmbedding:
    """
    Positional embedding for protein sequences.
    """
    
    def __init__(self, embedding_dim: int = 20):
        """
        Initialize the positional embedding.
        
        Args:
            embedding_dim (int): Dimension of the embedding
        """
        self.embedding_dim = embedding_dim
        self.aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a protein sequence as a positional embedding.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Positional embedding matrix
        """
        # Initialize embedding matrix
        embedding = np.zeros((len(sequence), self.embedding_dim))
        
        # Fill embedding matrix
        for i, aa in enumerate(sequence):
            if aa in self.aa_to_index:
                # One-hot encoding based on amino acid type
                embedding[i, self.aa_to_index[aa]] = 1.0
                
                # Add positional information
                embedding[i, -1] = i / len(sequence)  # Relative position
                
                # Add some physicochemical properties
                # Hydrophobicity (simplified)
                hydrophobic_aa = set('AILMFWV')
                if aa in hydrophobic_aa:
                    embedding[i, -2] = 1.0
                
                # Charge (simplified)
                charged_aa = set('RKDE')
                if aa in charged_aa:
                    embedding[i, -3] = 1.0
        
        return embedding
    
    def encode_sequences(self, sequences: List[str]) -> List[np.ndarray]:
        """
        Encode multiple protein sequences.
        
        Args:
            sequences (List[str]): List of protein sequences
            
        Returns:
            List[np.ndarray]: List of positional embedding matrices
        """
        embeddings = []
        for sequence in sequences:
            embedding = self.encode_sequence(sequence)
            embeddings.append(embedding)
        
        return embeddings


class TransformerRepresentation:
    """
    Transformer-based representations of antibody sequences.
    ""
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        """
        Initialize the transformer representation.
        
        Args:
            hidden_dim (int): Hidden dimension of the transformer
            num_layers (int): Number of transformer layers
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.aa_to_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        self.index_to_aa = {i: aa for aa, i in self.aa_to_index.items()}
    
    def _attention_weights(self, query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Compute attention weights.
        
        Args:
            query (np.ndarray): Query matrix
            key (np.ndarray): Key matrix
            
        Returns:
            np.ndarray: Attention weights
        """
        # Compute dot product
        scores = np.dot(query, key.T)
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        return attention_weights
    
    def _self_attention(self, sequence_embedding: np.ndarray) -> np.ndarray:
        """
        Apply self-attention to sequence embedding.
        
        Args:
            sequence_embedding (np.ndarray): Sequence embedding
            
        Returns:
            np.ndarray: Sequence embedding with self-attention applied
        """
        # Initialize weight matrices (simplified)
        W_q = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        W_k = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        W_v = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        
        # Project to query, key, value
        query = np.dot(sequence_embedding, W_q)
        key = np.dot(sequence_embedding, W_k)
        value = np.dot(sequence_embedding, W_v)
        
        # Compute attention weights
        attention_weights = self._attention_weights(query, key)
        
        # Apply attention to values
        attended_values = np.dot(attention_weights, value)
        
        return attended_values
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a protein sequence using transformer-based representation.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Transformer-based representation
        """
        # Convert sequence to numerical representation
        sequence_indices = [self.aa_to_index.get(aa, 0) for aa in sequence]
        
        # Initialize embedding matrix
        embedding = np.zeros((len(sequence), self.hidden_dim))
        
        # Fill embedding matrix with learned embeddings (simplified)
        for i, idx in enumerate(sequence_indices):
            # Simple embedding (in practice, this would be learned)
            embedding[i, idx % self.hidden_dim] = 1.0
            
            # Add positional encoding
            embedding[i, -1] = np.sin(i / 10000**(2 * (idx % (self.hidden_dim // 2)) / self.hidden_dim))
        
        # Apply self-attention layers
        for _ in range(self.num_layers):
            embedding = self._self_attention(embedding)
        
        # Global average pooling to get fixed-size representation
        sequence_representation = np.mean(embedding, axis=0)
        
        return sequence_representation
    
    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Encode multiple protein sequences using transformer-based representation.
        
        Args:
            sequences (List[str]): List of protein sequences
            
        Returns:
            np.ndarray: Matrix of transformer-based representations
        """
        embeddings = []
        for sequence in sequences:
            embedding = self.encode_sequence(sequence)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class EmbeddingSimilarity:
    """
    Embedding-based similarity and anomaly detection.
    """
    
    def __init__(self):
        """
        Initialize the embedding similarity analyzer.
        """
        pass
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        
        return float(similarity)
    
    def pairwise_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for embeddings.
        
        Args:
            embeddings (np.ndarray): Matrix of embeddings
            
        Returns:
            np.ndarray: Pairwise similarity matrix
        """
        n_samples = embeddings.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                similarity_matrix[i, j] = self.cosine_similarity(embeddings[i], embeddings[j])
        
        return similarity_matrix
    
    def detect_anomalies(self, embeddings: np.ndarray, method: str = 'isolation_forest', 
                        contamination: float = 0.1) -> Dict[str, Union[List[int], np.ndarray]]:
        """
        Detect anomalies in embeddings.
        
        Args:
            embeddings (np.ndarray): Matrix of embeddings
            method (str): Anomaly detection method ('isolation_forest', 'z_score', 'mahalanobis')
            contamination (float): Expected proportion of anomalies
            
        Returns:
            Dict[str, Union[List[int], np.ndarray]]: Anomaly detection results
        """
        anomalies = []
        anomaly_scores = np.zeros(len(embeddings))
        
        if method == 'z_score':
            # Z-score based anomaly detection
            mean_embedding = np.mean(embeddings, axis=0)
            std_embedding = np.std(embeddings, axis=0)
            
            # Avoid division by zero
            std_embedding = np.where(std_embedding == 0, 1, std_embedding)
            
            # Calculate z-scores for each sample
            z_scores = np.abs((embeddings - mean_embedding) / std_embedding)
            anomaly_scores = np.mean(z_scores, axis=1)
            
            # Determine threshold based on contamination
            threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
            anomalies = np.where(anomaly_scores > threshold)[0].tolist()
            
        elif method == 'mahalanobis':
            # Mahalanobis distance based anomaly detection
            mean_embedding = np.mean(embeddings, axis=0)
            cov_matrix = np.cov(embeddings, rowvar=False)
            
            # Add small value to diagonal to avoid singular matrix
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8
            
            # Calculate inverse covariance matrix
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            
            # Calculate Mahalanobis distances
            for i, embedding in enumerate(embeddings):
                diff = embedding - mean_embedding
                mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff))
                anomaly_scores[i] = mahalanobis_dist
            
            # Determine threshold based on contamination
            threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
            anomalies = np.where(anomaly_scores > threshold)[0].tolist()
            
        else:  # Default to isolation forest approach (simplified)
            # Simplified isolation approach based on distance to centroid
            mean_embedding = np.mean(embeddings, axis=0)
            distances = np.array([np.linalg.norm(embedding - mean_embedding) for embedding in embeddings])
            anomaly_scores = distances
            
            # Determine threshold based on contamination
            threshold = np.percentile(distances, 100 * (1 - contamination))
            anomalies = np.where(distances > threshold)[0].tolist()
        
        return {
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'threshold': float(threshold),
            'method': method
        }


def main():
    """
    Example usage of the protein embeddings implementation.
    """
    # Example antibody sequences
    example_sequences = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGSTYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGTFAYYAMDYWGQGTLVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGMNWVRQAPGQGLEWMGWISAGSGSTYYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARGGTFAYYAMDYWGQGTLVTVSS",
        "DIQMTQSPSSLSASVGDRVTITCRASQSVSSSYLAWYQQKPGKAPKLLIYDASNRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQRSNWPRTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC",
        "DIVMTQSPDSLAVSLGERATINCKSSQSVLYHSNKNYLAWYQQKPGQSPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    ]
    
    # K-mer embedding
    kmer_embedding = KmerEmbedding(k=3, normalize=True)
    kmer_embeddings = kmer_embedding.encode_sequences(example_sequences)
    
    print("K-mer Embedding Results:")
    print(f"  Embedding dimension: {kmer_embeddings.shape[1]}")
    print(f"  Number of sequences: {kmer_embeddings.shape[0]}")
    print(f"  First sequence embedding (first 10 dimensions): {kmer_embeddings[0][:10]}")
    
    # Positional embedding
    positional_embedding = PositionalEmbedding(embedding_dim=25)
    positional_embeddings = positional_embedding.encode_sequences(example_sequences)
    
    print("\nPositional Embedding Results:")
    print(f"  First sequence embedding shape: {positional_embeddings[0].shape}")
    print(f"  First sequence embedding (first 5 positions, first 5 dimensions):\n{positional_embeddings[0][:5, :5]}")
    
    # Transformer representation
    transformer_repr = TransformerRepresentation(hidden_dim=64, num_layers=2)
    transformer_embeddings = transformer_repr.encode_sequences(example_sequences)
    
    print("\nTransformer Representation Results:")
    print(f"  Embedding dimension: {transformer_embeddings.shape[1]}")
    print(f"  Number of sequences: {transformer_embeddings.shape[0]}")
    print(f"  First sequence embedding (first 10 dimensions): {transformer_embeddings[0][:10]}")
    
    # Embedding similarity
    embedding_similarity = EmbeddingSimilarity()
    
    # Compute similarity between first two sequences using k-mer embeddings
    similarity = embedding_similarity.cosine_similarity(kmer_embeddings[0], kmer_embeddings[1])
    
    print("\nEmbedding Similarity Results:")
    print(f"  Cosine similarity between first two sequences (k-mer): {similarity:.4f}")
    
    # Compute pairwise similarity matrix
    similarity_matrix = embedding_similarity.pairwise_similarity_matrix(kmer_embeddings)
    
    print("\nPairwise Similarity Matrix (first 3x3):")
    print(similarity_matrix[:3, :3])
    
    # Anomaly detection
    anomaly_results = embedding_similarity.detect_anomalies(kmer_embeddings, method='z_score', contamination=0.2)
    
    print("\nAnomaly Detection Results:")
    print(f"  Detected anomalies: {anomaly_results['anomalies']}")
    print(f"  Anomaly scores: {anomaly_results['anomaly_scores']}")
    print(f"  Threshold: {anomaly_results['threshold']:.4f}")


if __name__ == "__main__":
    main()

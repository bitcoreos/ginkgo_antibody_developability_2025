"""
Protein Language Model & Embedding Strategies Implementation

This module implements protein sequence embeddings, transformer-based representations,
and embedding-based similarity/anomaly detection for antibody developability prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')


class ProteinEmbeddingExtractor:
    """
    Extractor for protein sequence embeddings using simplified language model approaches.
    """
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize the protein embedding extractor.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        # Simplified amino acid embedding matrix (20 standard amino acids + padding)
        self.aa_embedding_matrix = np.random.randn(21, embedding_dim)
        # Amino acid to index mapping
        self.aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            'X': 20  # Unknown/ambiguous amino acid
        }
        # Index to amino acid mapping
        self.idx_to_aa = {v: k for k, v in self.aa_to_idx.items()}
    
    def sequence_to_indices(self, sequence: str) -> List[int]:
        """
        Convert protein sequence to indices.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            List[int]: List of amino acid indices
        """
        return [self.aa_to_idx.get(aa.upper(), 20) for aa in sequence]
    
    def extract_kmer_embeddings(self, sequence: str, k: int = 3) -> np.ndarray:
        """
        Extract k-mer embeddings from protein sequence.
        
        Args:
            sequence (str): Protein sequence
            k (int): Length of k-mers
            
        Returns:
            np.ndarray: Average embedding vector for the sequence
        """
        # Convert sequence to indices
        indices = self.sequence_to_indices(sequence)
        
        # Extract k-mers
        kmer_embeddings = []
        for i in range(len(indices) - k + 1):
            kmer_indices = indices[i:i+k]
            # Average embeddings of k-mer amino acids
            kmer_embedding = np.mean([self.aa_embedding_matrix[idx] for idx in kmer_indices], axis=0)
            kmer_embeddings.append(kmer_embedding)
        
        # Return average of all k-mer embeddings
        if kmer_embeddings:
            return np.mean(kmer_embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def extract_positional_embeddings(self, sequence: str) -> np.ndarray:
        """
        Extract positional embeddings from protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Embedding vector combining amino acid and positional information
        """
        # Convert sequence to indices
        indices = self.sequence_to_indices(sequence)
        
        # Create positional embeddings (simplified sinusoidal encoding)
        seq_len = len(indices)
        pos_embeddings = []
        
        for pos, idx in enumerate(indices):
            # Simplified positional encoding
            pos_encoding = np.zeros(self.embedding_dim)
            for i in range(0, self.embedding_dim, 2):
                pos_encoding[i] = np.sin(pos / (10000 ** (i / self.embedding_dim)))
                if i + 1 < self.embedding_dim:
                    pos_encoding[i + 1] = np.cos(pos / (10000 ** (i / self.embedding_dim)))
            
            # Combine amino acid embedding with positional encoding
            combined_embedding = self.aa_embedding_matrix[idx] + pos_encoding
            pos_embeddings.append(combined_embedding)
        
        # Return average of all position embeddings
        if pos_embeddings:
            return np.mean(pos_embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def extract_composite_embeddings(self, sequence: str, k: int = 3) -> np.ndarray:
        """
        Extract composite embeddings combining k-mer and positional information.
        
        Args:
            sequence (str): Protein sequence
            k (int): Length of k-mers
            
        Returns:
            np.ndarray: Composite embedding vector
        """
        # Extract k-mer embeddings
        kmer_embedding = self.extract_kmer_embeddings(sequence, k)
        
        # Extract positional embeddings
        pos_embedding = self.extract_positional_embeddings(sequence)
        
        # Combine embeddings
        composite_embedding = np.concatenate([kmer_embedding, pos_embedding])
        
        return composite_embedding


class AntibodyTransformerRepresentation:
    """
    Transformer-based representation of antibody sequences.
    """
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8):
        """
        Initialize the antibody transformer representation.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            num_heads (int): Number of attention heads
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding_extractor = ProteinEmbeddingExtractor(embedding_dim)
    
    def self_attention(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Simplified self-attention mechanism.
        
        Args:
            embeddings (np.ndarray): Embedding matrix of shape (seq_len, embedding_dim)
            
        Returns:
            np.ndarray: Attention-weighted embeddings
        """
        seq_len, embed_dim = embeddings.shape
        
        # Simplified attention computation
        # Compute attention scores
        attention_scores = np.dot(embeddings, embeddings.T) / np.sqrt(embed_dim)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores, axis=1)
        
        # Apply attention weights
        attended_embeddings = np.dot(attention_weights, embeddings)
        
        return attended_embeddings
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Compute softmax values for array.
        
        Args:
            x (np.ndarray): Input array
            axis (int): Axis along which to compute softmax
            
        Returns:
            np.ndarray: Softmax values
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def encode_antibody_sequence(self, vh_sequence: str, vl_sequence: str) -> Dict[str, np.ndarray]:
        """
        Encode antibody sequence using transformer-based approach.
        
        Args:
            vh_sequence (str): Heavy chain sequence
            vl_sequence (str): Light chain sequence
            
        Returns:
            Dict[str, np.ndarray]: Encoded representations
        """
        # Extract embeddings for heavy and light chains
        vh_embeddings = self.embedding_extractor.extract_composite_embeddings(vh_sequence)
        vl_embeddings = self.embedding_extractor.extract_composite_embeddings(vl_sequence)
        
        # Combine embeddings
        combined_embeddings = np.concatenate([vh_embeddings, vl_embeddings])
        
        # Reshape for attention mechanism (simulating sequence of tokens)
        # In a real implementation, this would be done at the residue level
        seq_len = 10  # Simulated sequence length
        reshaped_embeddings = combined_embeddings.reshape(seq_len, -1)
        
        # Apply self-attention
        attended_embeddings = self.self_attention(reshaped_embeddings)
        
        # Global average pooling to get final representation
        final_representation = np.mean(attended_embeddings, axis=0)
        
        return {
            'vh_embeddings': vh_embeddings,
            'vl_embeddings': vl_embeddings,
            'combined_embeddings': combined_embeddings,
            'attended_embeddings': attended_embeddings,
            'final_representation': final_representation
        }


class EmbeddingSimilarityDetector:
    """
    Detector for embedding-based similarity and anomaly detection.
    """
    
    def __init__(self):
        """
        Initialize the embedding similarity detector.
        """
        pass
    
    def calculate_cosine_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix for embeddings.
        
        Args:
            embeddings (np.ndarray): Embedding matrix of shape (n_samples, embedding_dim)
            
        Returns:
            np.ndarray: Cosine similarity matrix of shape (n_samples, n_samples)
        """
        return cosine_similarity(embeddings)
    
    def detect_anomalies_isolation_forest(self, embeddings: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            embeddings (np.ndarray): Embedding matrix of shape (n_samples, embedding_dim)
            contamination (float): Expected proportion of anomalies
            
        Returns:
            np.ndarray: Anomaly labels (-1 for anomaly, 1 for normal)
        """
        detector = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = detector.fit_predict(embeddings)
        return anomaly_labels
    
    def detect_anomalies_local_outlier_factor(self, embeddings: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            embeddings (np.ndarray): Embedding matrix of shape (n_samples, embedding_dim)
            n_neighbors (int): Number of neighbors for LOF
            
        Returns:
            np.ndarray: Anomaly labels (-1 for anomaly, 1 for normal)
        """
        detector = LocalOutlierFactor(n_neighbors=n_neighbors)
        anomaly_labels = detector.fit_predict(embeddings)
        return anomaly_labels
    
    def detect_sequence_anomalies(self, embeddings: np.ndarray, method: str = 'isolation_forest') -> Dict[str, np.ndarray]:
        """
        Detect sequence anomalies using embedding-based methods.
        
        Args:
            embeddings (np.ndarray): Embedding matrix of shape (n_samples, embedding_dim)
            method (str): Anomaly detection method ('isolation_forest' or 'local_outlier_factor')
            
        Returns:
            Dict[str, np.ndarray]: Anomaly detection results
        """
        if method == 'isolation_forest':
            anomaly_labels = self.detect_anomalies_isolation_forest(embeddings)
        elif method == 'local_outlier_factor':
            anomaly_labels = self.detect_anomalies_local_outlier_factor(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate anomaly scores
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=0.1, random_state=42)
            detector.fit(embeddings)
            anomaly_scores = detector.decision_function(embeddings)
        else:  # local_outlier_factor
            detector = LocalOutlierFactor(n_neighbors=20)
            anomaly_scores = detector.fit_predict(embeddings)
            # LOF returns -1 for anomalies, 1 for normal
            # Convert to anomaly scores (lower is more anomalous)
            anomaly_scores = -anomaly_scores
        
        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores
        }


def main():
    """
    Example usage of the protein language model implementation.
    """
    # Example antibody sequences
    vh_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLRTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNYWPLTFGQGTKVEIK"
    
    # Extract protein embeddings
    embedding_extractor = ProteinEmbeddingExtractor()
    vh_embedding = embedding_extractor.extract_composite_embeddings(vh_sequence)
    vl_embedding = embedding_extractor.extract_composite_embeddings(vl_sequence)
    
    print("Protein Embedding Extraction:")
    print(f"VH embedding shape: {vh_embedding.shape}")
    print(f"VL embedding shape: {vl_embedding.shape}")
    
    # Encode antibody sequence using transformer-based approach
    transformer_encoder = AntibodyTransformerRepresentation()
    encoded_antibody = transformer_encoder.encode_antibody_sequence(vh_sequence, vl_sequence)
    
    print("\nTransformer-Based Antibody Encoding:")
    print(f"Final representation shape: {encoded_antibody['final_representation'].shape}")
    
    # Detect embedding similarities and anomalies
    similarity_detector = EmbeddingSimilarityDetector()
    
    # Create example embeddings for multiple antibodies
    np.random.seed(42)
    example_embeddings = np.random.randn(100, 256)  # 100 antibodies, 256-dim embeddings
    
    # Calculate similarity matrix
    similarity_matrix = similarity_detector.calculate_cosine_similarity(example_embeddings[:10])  # Use first 10 for similarity
    
    print("\nEmbedding-Based Similarity Detection:")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Detect anomalies
    anomaly_results = similarity_detector.detect_sequence_anomalies(example_embeddings, method='isolation_forest')
    
    print("\nEmbedding-Based Anomaly Detection:")
    print(f"Number of anomalies detected: {np.sum(anomaly_results['anomaly_labels'] == -1)}")
    print(f"Anomaly scores range: {np.min(anomaly_results['anomaly_scores']):.4f} to {np.max(anomaly_results['anomaly_scores']):.4f}")


if __name__ == "__main__":
    main()

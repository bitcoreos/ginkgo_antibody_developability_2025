"""
Protein Language Model & Embedding Strategies Implementation

This module implements protein sequence embeddings using ESM-2 protein language models.
"""

import torch
import esm
import numpy as np
from typing import List, Dict, Tuple, Union

class ESM2Integrator:
    """
    Integrator for ESM-2 protein language model to extract embeddings from antibody sequences.
    """

    def __init__(self, model_name: str = "esm2_t6_8M_UR50D", device: str = None):
        """
        Initialize the ESM2 integrator.

        Args:
            model_name (str): Name of the ESM-2 model to use
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._load_model()

    def _load_model(self):
        """
        Load the ESM-2 model and alphabet.
        """
        print(f"Loading {self.model_name} model...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Disable dropout for inference
        self.model.to(self.device)  # Move model to device
        print(f"Model loaded successfully on {self.device}")

    def preprocess_sequences(self, sequences: List[str]) -> List[Tuple[str, str]]:
        """
        Preprocess sequences for ESM-2 model input.

        Args:
            sequences (List[str]): List of protein sequences

        Returns:
            List[Tuple[str, str]]: List of tuples (sequence_id, sequence)
        """
        # Create sequence IDs
        sequence_data = []
        for i, seq in enumerate(sequences):
            # Remove any whitespace and convert to uppercase
            seq = seq.strip().upper()
            # Validate sequence (basic check for standard amino acids)
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            if all(aa in valid_aa for aa in seq):
                sequence_data.append((f"seq_{i}", seq))
            else:
                print(f"Warning: Sequence {i} contains non-standard amino acids. Skipping.")

        return sequence_data

    def extract_embeddings(self, sequences: List[str], layer: int = 6) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from protein sequences using ESM-2.

        Args:
            sequences (List[str]): List of protein sequences
            layer (int): Layer to extract embeddings from (-1 for last layer)

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping sequence IDs to embeddings
        """
        # Preprocess sequences
        sequence_data = self.preprocess_sequences(sequences)

        if not sequence_data:
            return {}

        # Convert to batches
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequence_data)
        batch_tokens = batch_tokens.to(self.device)

        # Extract embeddings
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layer], return_contacts=False)

        # Get embeddings
        token_embeddings = results["representations"][layer]

        # Convert to numpy
        embeddings = {}
        for i, (label, seq) in enumerate(sequence_data):
            # Remove start and end tokens
            seq_embedding = token_embeddings[i, 1 : len(seq) + 1].cpu().numpy()
            embeddings[label] = seq_embedding

        return embeddings

    def extract_sequence_embeddings(self, sequences: List[str], layer: int = 6) -> Dict[str, np.ndarray]:
        """
        Extract per-sequence embeddings (pooled) from protein sequences using ESM-2.

        Args:
            sequences (List[str]): List of protein sequences
            layer (int): Layer to extract embeddings from (-1 for last layer)

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping sequence IDs to per-sequence embeddings
        """
        # Extract per-residue embeddings
        residue_embeddings = self.extract_embeddings(sequences, layer)

        # Pool embeddings (mean pooling)
        sequence_embeddings = {}
        for seq_id, embeddings in residue_embeddings.items():
            sequence_embeddings[seq_id] = np.mean(embeddings, axis=0)

        return sequence_embeddings

def main():
    """
    Example usage of the ESM2Integrator.
    """
    # Example antibody sequences
    vh_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLRTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNYWPLTFGQGTKVEIK"

    # Initialize ESM2 integrator
    esm2_integrator = ESM2Integrator()

    # Extract per-residue embeddings
    residue_embeddings = esm2_integrator.extract_embeddings([vh_sequence, vl_sequence])

    print("Per-residue embeddings:")
    for seq_id, embeddings in residue_embeddings.items():
        print(f"{seq_id}: {embeddings.shape}")

    # Extract per-sequence embeddings
    sequence_embeddings = esm2_integrator.extract_sequence_embeddings([vh_sequence, vl_sequence])

    print("\nPer-sequence embeddings:")
    for seq_id, embeddings in sequence_embeddings.items():
        print(f"{seq_id}: {embeddings.shape}")

if __name__ == "__main__":
    main()

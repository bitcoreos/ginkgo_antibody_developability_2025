"""
FLAb Analyzer for Protein Language Models

This module implements a FLAb analyzer for protein language model embeddings.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any

    from esm2_integrator import ESM2Integrator
except ImportError:
    print("Warning: ESM2Integrator not found. Please ensure the module is properly installed.")
    ESM2Integrator = None

class FLAbProteinLanguageModelAnalyzer:
    """
    FLAb analyzer for protein language model embeddings.
    """

    def __init__(self):
        """
        Initialize the FLAb protein language model analyzer.
        """
        self.analyzer_name = "ProteinLanguageModelAnalyzer"
        self.esm2_integrator = None

        # Initialize ESM2 integrator if available
        if ESM2Integrator is not None:
            try:
                self.esm2_integrator = ESM2Integrator()
            except Exception as e:
                print(f"Warning: Failed to initialize ESM2Integrator: {e}")
        else:
            print("Warning: ESM2Integrator not available")

    def analyze(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Any]:
        """
        Analyze antibody sequences using protein language models.

        Args:
            vh_sequence (str): Heavy chain sequence
            vl_sequence (str): Light chain sequence

        Returns:
            Dict[str, Any]: Dictionary of extracted features
        """
        features = {}

        # Check if ESM2 integrator is available
        if self.esm2_integrator is None:
            print("Warning: ESM2Integrator not available, returning empty features")
            return features

        try:
            # Prepare sequences for analysis
            sequences = []
            if vh_sequence and len(vh_sequence.strip()) > 0:
                sequences.append(vh_sequence.strip())
            if vl_sequence and len(vl_sequence.strip()) > 0:
                sequences.append(vl_sequence.strip())

            # If no valid sequences, return empty features
            if not sequences:
                print("Warning: No valid sequences provided, returning empty features")
                return features

            # Extract per-sequence embeddings
            sequence_embeddings = self.esm2_integrator.extract_sequence_embeddings(sequences)

            # Process embeddings for VH and VL
            vh_embedding = np.array([])
            vl_embedding = np.array([])

            # Extract VH embedding if provided
            if vh_sequence and len(vh_sequence.strip()) > 0:
                vh_embedding = sequence_embeddings.get('seq_0', np.array([]))
                # If VH is the second sequence (no VL provided), it would be 'seq_0'
                if len(vh_embedding) == 0 and len(sequences) == 1:
                    vh_embedding = sequence_embeddings.get('seq_0', np.array([]))

            # Extract VL embedding if provided
            if vl_sequence and len(vl_sequence.strip()) > 0:
                # If VH was provided, VL is 'seq_1', otherwise VL is 'seq_0'
                if vh_sequence and len(vh_sequence.strip()) > 0:
                    vl_embedding = sequence_embeddings.get('seq_1', np.array([]))
                else:
                    vl_embedding = sequence_embeddings.get('seq_0', np.array([]))

            # Extract features from embeddings
            if len(vh_embedding) > 0:
                # Statistical features from VH embedding
                features['vh_esm2_mean'] = np.mean(vh_embedding)
                features['vh_esm2_std'] = np.std(vh_embedding)
                features['vh_esm2_min'] = np.min(vh_embedding)
                features['vh_esm2_max'] = np.max(vh_embedding)
                features['vh_esm2_dim'] = len(vh_embedding)

            if len(vl_embedding) > 0:
                # Statistical features from VL embedding
                features['vl_esm2_mean'] = np.mean(vl_embedding)
                features['vl_esm2_std'] = np.std(vl_embedding)
                features['vl_esm2_min'] = np.min(vl_embedding)
                features['vl_esm2_max'] = np.max(vl_embedding)
                features['vl_esm2_dim'] = len(vl_embedding)

            # Combined features
            if len(vh_embedding) > 0 and len(vl_embedding) > 0:
                # Difference features
                features['vh_vl_esm2_mean_diff'] = features['vh_esm2_mean'] - features['vl_esm2_mean']
                features['vh_vl_esm2_std_diff'] = features['vh_esm2_std'] - features['vl_esm2_std']

                # Cosine similarity
                dot_product = np.dot(vh_embedding, vl_embedding)
                norm_vh = np.linalg.norm(vh_embedding)
                norm_vl = np.linalg.norm(vl_embedding)
                if norm_vh != 0 and norm_vl != 0:
                    features['vh_vl_esm2_cosine_similarity'] = dot_product / (norm_vh * norm_vl)
                else:
                    features['vh_vl_esm2_cosine_similarity'] = 0

        except Exception as e:
            print(f"Warning: Error in protein language model analysis: {e}")
            import traceback
            traceback.print_exc()

        return features
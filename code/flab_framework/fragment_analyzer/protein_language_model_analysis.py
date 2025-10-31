"""
Protein Language Model Analysis Module for FLAb Framework

This module provides functionality for analyzing protein sequences using protein language models.
"""

import sys
import os

# Add the path to import FLAbProteinLanguageModelAnalyzer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_algorithms', 'protein_language_models'))

try:
    from flab_protein_language_model_analyzer import FLAbProteinLanguageModelAnalyzer
except ImportError:
    print("Warning: FLAbProteinLanguageModelAnalyzer not found. Please ensure the module is properly installed.")
    FLAbProteinLanguageModelAnalyzer = None

def analyze_protein_language_model_features(vh_sequence, vl_sequence=None):
    """
    Analyze protein sequences using protein language models.

    Args:
        vh_sequence (str): Heavy chain sequence
        vl_sequence (str, optional): Light chain sequence

    Returns:
        dict: Protein language model analysis including:
            - vh_esm2_mean: Mean of VH ESM-2 embeddings
            - vh_esm2_std: Standard deviation of VH ESM-2 embeddings
            - vh_esm2_min: Minimum of VH ESM-2 embeddings
            - vh_esm2_max: Maximum of VH ESM-2 embeddings
            - vh_esm2_dim: Dimension of VH ESM-2 embeddings
            - vl_esm2_mean: Mean of VL ESM-2 embeddings (if VL sequence provided)
            - vl_esm2_std: Standard deviation of VL ESM-2 embeddings (if VL sequence provided)
            - vl_esm2_min: Minimum of VL ESM-2 embeddings (if VL sequence provided)
            - vl_esm2_max: Maximum of VL ESM-2 embeddings (if VL sequence provided)
            - vl_esm2_dim: Dimension of VL ESM-2 embeddings (if VL sequence provided)
            - vh_vl_esm2_mean_diff: Difference in mean embeddings between VH and VL
            - vh_vl_esm2_std_diff: Difference in std embeddings between VH and VL
            - vh_vl_esm2_cosine_similarity: Cosine similarity between VH and VL embeddings
    """
    # Initialize analyzer
    if FLAbProteinLanguageModelAnalyzer is None:
        print("Warning: FLAbProteinLanguageModelAnalyzer not available, returning empty features")
        return {}

    try:
        analyzer = FLAbProteinLanguageModelAnalyzer()

        # If only VH sequence is provided, analyze it alone
        if vl_sequence is None:
            features = analyzer.analyze(vh_sequence, None)
        else:
            # Analyze both VH and VL sequences
            features = analyzer.analyze(vh_sequence, vl_sequence)

        return features
    except Exception as e:
        print(f"Warning: Error in protein language model analysis: {e}")
        return {}

if __name__ == "__main__":
    # Test the protein language model analysis
    vh_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLRTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNYWPLTFGQGTKVEIK"

    print("Testing protein language model analysis:")
    print(f"VH sequence length: {len(vh_sequence)}")
    print(f"VL sequence length: {len(vl_sequence)}")

    plm_features = analyze_protein_language_model_features(vh_sequence, vl_sequence)

    print("Protein language model features:")
    for feature, value in plm_features.items():
        print(f"  {feature}: {value}")

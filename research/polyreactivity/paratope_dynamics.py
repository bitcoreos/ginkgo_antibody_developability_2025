"""
Paratope Dynamics Proxies Implementation

This module implements paratope dynamics analysis for antibody sequences,
focusing on entropy-based proxies for paratope state predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter


def predict_paratope_residues(sequence: str, method: str = "kabat") -> List[int]:
    """
    Predict paratope residues based on canonical CDR definitions.
    
    Args:
        sequence (str): Antibody sequence
        method (str): Method for CDR definition ('kabat', 'chothia', 'contact')
        
    Returns:
        List[int]: List of residue indices predicted to be in the paratope
    """
    # Simplified CDR definitions (Kabat numbering)
    # In practice, this would use a more sophisticated paratope prediction method
    
    # Approximate CDR lengths for a typical antibody
    cdr_lengths = {
        'H1': 10, 'H2': 16, 'H3': 12,
        'L1': 10, 'L2': 16, 'L3': 12
    }
    
    # Approximate positions of CDRs in a typical antibody sequence
    # This is highly simplified and would need to be replaced with actual numbering
    total_length = len(sequence)
    
    if total_length < 100:  # Too short to be a full antibody sequence
        return []
    
    # Approximate CDR positions (highly simplified)
    cdr_positions = []
    
    # Heavy chain CDRs (approximate positions)
    h1_start = total_length // 4
    h1_end = h1_start + cdr_lengths['H1']
    cdr_positions.extend(range(h1_start, min(h1_end, total_length)))
    
    h2_start = total_length // 2
    h2_end = h2_start + cdr_lengths['H2']
    cdr_positions.extend(range(h2_start, min(h2_end, total_length)))
    
    h3_start = int(total_length * 0.75)
    h3_end = h3_start + cdr_lengths['H3']
    cdr_positions.extend(range(h3_start, min(h3_end, total_length)))
    
    # Light chain CDRs (approximate positions)
    l1_start = total_length // 8
    l1_end = l1_start + cdr_lengths['L1']
    cdr_positions.extend(range(l1_start, min(l1_end, total_length)))
    
    l2_start = int(total_length * 0.375)
    l2_end = l2_start + cdr_lengths['L2']
    cdr_positions.extend(range(l2_start, min(l2_end, total_length)))
    
    l3_start = int(total_length * 0.625)
    l3_end = l3_start + cdr_lengths['L3']
    cdr_positions.extend(range(l3_start, min(l3_end, total_length)))
    
    # Remove duplicates and sort
    paratope_residues = sorted(list(set(cdr_positions)))
    
    return paratope_residues


def calculate_sequence_entropy(sequence: str) -> float:
    """
    Calculate the Shannon entropy of a protein sequence.
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Sequence entropy
    """
    if not sequence:
        return 0.0
    
    # Count amino acid frequencies
    aa_counts = Counter(sequence)
    total_aa = len(sequence)
    
    # Calculate entropy
    entropy = 0.0
    for count in aa_counts.values():
        probability = count / total_aa
        if probability > 0:  # Avoid log(0)
            entropy -= probability * np.log2(probability)
    
    return entropy


def calculate_paratope_entropy(sequence: str, paratope_indices: List[int] = None) -> Dict[str, float]:
    """
    Calculate entropy-based metrics for paratope dynamics.
    
    Args:
        sequence (str): Antibody sequence
        paratope_indices (List[int]): Indices of paratope residues (if known)
        
    Returns:
        Dict[str, float]: Paratope entropy metrics
    """
    if not sequence:
        return {
            'paratope_entropy': 0.0,
            'paratope_length': 0,
            'normalized_entropy': 0.0
        }
    
    # Predict paratope residues if not provided
    if paratope_indices is None:
        paratope_indices = predict_paratope_residues(sequence)
    
    # Extract paratope sequence
    paratope_sequence = ''.join([sequence[i] for i in paratope_indices if i < len(sequence)])
    
    # Calculate paratope entropy
    paratope_entropy = calculate_sequence_entropy(paratope_sequence)
    paratope_length = len(paratope_sequence)
    
    # Normalize entropy by sequence length
    if paratope_length > 0:
        normalized_entropy = paratope_entropy / np.log2(paratope_length)
    else:
        normalized_entropy = 0.0
    
    return {
        'paratope_entropy': paratope_entropy,
        'paratope_length': paratope_length,
        'normalized_entropy': normalized_entropy
    }


def calculate_paratope_dynamics_proxy(vh_sequence: str, vl_sequence: str) -> Dict[str, float]:
    """
    Calculate paratope dynamics proxies based on entropy of predicted paratope states.
    
    Args:
        vh_sequence (str): Heavy chain variable domain sequence
        vl_sequence (str): Light chain variable domain sequence
        
    Returns:
        Dict[str, float]: Paratope dynamics proxies
    """
    # Calculate paratope entropy for each chain
    vh_paratope_entropy = calculate_paratope_entropy(vh_sequence)
    vl_paratope_entropy = calculate_paratope_entropy(vl_sequence)
    
    # Calculate combined metrics
    combined_entropy = vh_paratope_entropy['paratope_entropy'] + vl_paratope_entropy['paratope_entropy']
    
    # Entropy difference as a measure of paratope dynamics
    entropy_difference = abs(vh_paratope_entropy['paratope_entropy'] - vl_paratope_entropy['paratope_entropy'])
    
    # Normalized combined entropy
    total_paratope_length = vh_paratope_entropy['paratope_length'] + vl_paratope_entropy['paratope_length']
    if total_paratope_length > 0:
        normalized_combined_entropy = combined_entropy / np.log2(total_paratope_length)
    else:
        normalized_combined_entropy = 0.0
    
    return {
        'vh_paratope_entropy': vh_paratope_entropy['paratope_entropy'],
        'vl_paratope_entropy': vl_paratope_entropy['paratope_entropy'],
        'combined_paratope_entropy': combined_entropy,
        'entropy_difference': entropy_difference,
        'normalized_combined_entropy': normalized_combined_entropy,
        'vh_paratope_length': vh_paratope_entropy['paratope_length'],
        'vl_paratope_length': vl_paratope_entropy['paratope_length'],
        'total_paratope_length': total_paratope_length
    }


def analyze_paratope_composition(sequence: str, paratope_indices: List[int] = None) -> Dict[str, float]:
    """
    Analyze the amino acid composition of the paratope.
    
    Args:
        sequence (str): Antibody sequence
        paratope_indices (List[int]): Indices of paratope residues (if known)
        
    Returns:
        Dict[str, float]: Paratope composition metrics
    """
    if not sequence:
        return {}
    
    # Predict paratope residues if not provided
    if paratope_indices is None:
        paratope_indices = predict_paratope_residues(sequence)
    
    # Extract paratope sequence
    paratope_sequence = ''.join([sequence[i] for i in paratope_indices if i < len(sequence)])
    
    if not paratope_sequence:
        return {}
    
    # Define amino acid property groups
    hydrophobic = {'A', 'F', 'I', 'L', 'M', 'P', 'V', 'W', 'Y'}
    polar = {'C', 'N', 'Q', 'S', 'T'}
    positive = {'H', 'K', 'R'}
    negative = {'D', 'E'}
    
    # Count amino acids in each group
    hydrophobic_count = sum(1 for aa in paratope_sequence if aa in hydrophobic)
    polar_count = sum(1 for aa in paratope_sequence if aa in polar)
    positive_count = sum(1 for aa in paratope_sequence if aa in positive)
    negative_count = sum(1 for aa in paratope_sequence if aa in negative)
    
    total = len(paratope_sequence)
    
    return {
        'hydrophobic_fraction': hydrophobic_count / total if total > 0 else 0.0,
        'polar_fraction': polar_count / total if total > 0 else 0.0,
        'positive_fraction': positive_count / total if total > 0 else 0.0,
        'negative_fraction': negative_count / total if total > 0 else 0.0,
        'aromatic_fraction': sum(1 for aa in paratope_sequence if aa in {'F', 'W', 'Y'}) / total if total > 0 else 0.0
    }


def main():
    """
    Example usage of the paratope dynamics implementation.
    """
    # Example VH and VL sequences (simplified)
    vh_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLRTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNHWPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # Calculate paratope dynamics proxies
    dynamics = calculate_paratope_dynamics_proxy(vh_sequence, vl_sequence)
    
    print("Paratope Dynamics Proxies:")
    print(f"VH Paratope Entropy: {dynamics['vh_paratope_entropy']:.2f}")
    print(f"VL Paratope Entropy: {dynamics['vl_paratope_entropy']:.2f}")
    print(f"Combined Paratope Entropy: {dynamics['combined_paratope_entropy']:.2f}")
    print(f"Entropy Difference: {dynamics['entropy_difference']:.2f}")
    print(f"Normalized Combined Entropy: {dynamics['normalized_combined_entropy']:.4f}")
    
    # Analyze paratope composition
    vh_composition = analyze_paratope_composition(vh_sequence)
    vl_composition = analyze_paratope_composition(vl_sequence)
    
    print("\nParatope Composition Analysis:")
    print(f"VH Hydrophobic Fraction: {vh_composition['hydrophobic_fraction']:.2f}")
    print(f"VL Positive Fraction: {vl_composition['positive_fraction']:.2f}")


if __name__ == "__main__":
    main()

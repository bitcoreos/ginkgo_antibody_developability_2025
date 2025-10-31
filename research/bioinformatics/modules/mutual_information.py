"""
Module: mutual_information.py

This module provides functionality for calculating mutual information between
amino acid positions in antibody sequences. Mutual information is a measure of
the dependence between two positions, indicating how much knowing the amino acid
at one position reduces uncertainty about the amino acid at another position.

Mutual information is widely used in bioinformatics to identify co-evolving
residues and potential structural or functional constraints in protein families.

The calculation follows the standard formula:
MI(X;Y) = Σ Σ p(x,y) * log(p(x,y)/(p(x)*p(y)))

Author: Bioinformatics Pipeline Developer
Date: 2025-10-11
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import math

# Standard amino acid alphabet
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZ*")


def calculate_positional_frequencies(sequences: List[str]) -> Dict[int, Dict[str, float]]:
    """Calculate the frequency of each amino acid at each position in a multiple sequence alignment.

    Args:
        sequences: List of aligned protein sequences

    Returns:
        Dictionary mapping position to amino acid frequency distribution

    Raises:
        ValueError: If sequences are not the same length or contain invalid amino acids
    """
    if not sequences:
        raise ValueError("Empty sequence list")

    # Check that all sequences have the same length
    seq_length = len(sequences[0])
    if not all(len(seq) == seq_length for seq in sequences):
        raise ValueError("All sequences must have the same length")

    # Validate sequences and convert to uppercase
    sequences = [seq.upper() for seq in sequences]

    # Check for invalid amino acids
    for i, seq in enumerate(sequences):
        invalid_aas = set(seq) - AMINO_ACIDS
        if invalid_aas:
            raise ValueError(f"Invalid amino acids in sequence {i}: {invalid_aas}")

    # Calculate frequencies for each position
    frequencies = {}
    n_sequences = len(sequences)

    for pos in range(seq_length):
        aa_counts = Counter(seq[pos] for seq in sequences)
        pos_freqs = {aa: count / n_sequences for aa, count in aa_counts.items()}
        frequencies[pos] = pos_freqs

    return frequencies

def calculate_joint_frequencies(sequences: List[str], pos1: int, pos2: int) -> Dict[Tuple[str, str], float]:
    """Calculate the joint frequency of amino acid pairs at two positions.

    Args:
        sequences: List of aligned protein sequences
        pos1: First position (0-based)
        pos2: Second position (0-based)

    Returns:
        Dictionary mapping amino acid pairs to their joint frequencies

    Raises:
        ValueError: If positions are out of range or sequences are invalid
    """
    if not sequences:
        raise ValueError("Empty sequence list")

    seq_length = len(sequences[0])
    if pos1 < 0 or pos1 >= seq_length:
        raise ValueError(f"Position 1 out of range: {pos1}")

    if pos2 < 0 or pos2 >= seq_length:
        raise ValueError(f"Position 2 out of range: {pos2}")

    # Validate sequences
    sequences = [seq.upper() for seq in sequences]
    for i, seq in enumerate(sequences):
        invalid_aas = set(seq) - AMINO_ACIDS
        if invalid_aas:
            raise ValueError(f"Invalid amino acids in sequence {i}: {invalid_aas}")

    # Calculate joint frequencies
    joint_counts = Counter((seq[pos1].upper(), seq[pos2].upper()) for seq in sequences)
    n_sequences = len(sequences)
    joint_freqs = {pair: count / n_sequences for pair, count in joint_counts.items()}

    return joint_freqs

def calculate_mutual_information(sequences: List[str], pos1: int, pos2: int, 
                                pseudo_count: float = 1e-10) -> float:
    """Calculate mutual information between two positions in a multiple sequence alignment.

    Args:

        sequences: List of aligned protein sequences

        pos1: First position (0-based)

        pos2: Second position (0-based)

        pseudo_count: Small value added to frequencies to avoid log(0) (default: 1e-10)

    Returns:

        Mutual information value in bits

    Raises:

        ValueError: If positions are out of range or sequences are invalid

    """

    if not sequences:

        raise ValueError("Empty sequence list")

        

    seq_length = len(sequences[0])

    if pos1 < 0 or pos1 >= seq_length:

        raise ValueError(f"Position 1 out of range: {pos1}")

        

    if pos2 < 0 or pos2 >= seq_length:

        raise ValueError(f"Position 2 out of range: {pos2}")

        

    # Calculate marginal frequencies for both positions

    all_freqs = calculate_positional_frequencies(sequences)

    freqs1 = all_freqs[pos1]
    freqs2 = all_freqs[pos2]

    # Calculate joint frequencies
    joint_freqs = calculate_joint_frequencies(sequences, pos1, pos2)

    # Calculate mutual information
    mi = 0.0
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            # Get frequencies, add pseudo-count to avoid log(0)
            p_x = freqs1.get(aa1, 0.0) + pseudo_count
            p_y = freqs2.get(aa2, 0.0) + pseudo_count
            p_xy = joint_freqs.get((aa1, aa2), 0.0) + pseudo_count


            # Calculate contribution to MI
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

    return mi

def calculate_mi_matrix(sequences: List[str], pseudo_count: float = 1e-10) -> np.ndarray:
    """Calculate mutual information matrix for all pairs of positions.

    Args:
        sequences: List of aligned protein sequences
        pseudo_count: Small value added to frequencies to avoid log(0)

    Returns:
        2D numpy array with MI values for all position pairs
    """
    if not sequences:
        raise ValueError("Empty sequence list")

    seq_length = len(sequences[0])

    # Initialize MI matrix
    mi_matrix = np.zeros((seq_length, seq_length))

    # Calculate MI for all pairs of positions
    for i in range(seq_length):
        for j in range(i + 1, seq_length):  # Only upper triangle
            mi = calculate_mutual_information(sequences, i, j, pseudo_count)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi  # Symmetric

    return mi_matrix

# Version information
__version__ = "1.0.0"
__author__ = "Bioinformatics Pipeline Developer"
__date__ = "2025-10-11"

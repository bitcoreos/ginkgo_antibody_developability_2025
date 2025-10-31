"""
Test module for mutual_information.py

This module contains unit tests for the mutual information calculation functions.
Tests cover valid inputs, edge cases, and numerical accuracy.
"""

import unittest
import numpy as np
from bioinformatics.modules.mutual_information import (
    calculate_positional_frequencies,
    calculate_joint_frequencies,
    calculate_mutual_information,
    calculate_mi_matrix
)

class TestMutualInformation(unittest.TestCase):
    """Test cases for mutual information calculation functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Simple test sequences with known MI properties
        self.sequences = [
            "ACDE", "ACDF", "ACDG", "ACDH",  # Position 0: A(100%), Position 1: C(100%)
            "BCDE", "BCDF", "BCDG", "BCDH"   # Position 0: B(100%), Position 1: C(100%)
        ]

        # Sequences with perfect correlation
        self.correlated_sequences = [
            "AA", "BB", "CC", "DD", "EE"
        ]

        # Sequences with no correlation
        # Create truly uncorrelated sequences with uniform marginal distributions
        # Each position has independent uniform distribution of A, B, C, D
        aa_list = ['A', 'B', 'C', 'D']
        self.uncorrelated_sequences = []
        for aa1 in aa_list:
            for aa2 in aa_list:
                self.uncorrelated_sequences.append(aa1 + aa2)
        # This creates 16 sequences with perfect independence

    def test_calculate_positional_frequencies(self):
        """Test calculation of positional frequencies."""
        freqs = calculate_positional_frequencies(self.sequences)

        # Test position 0
        self.assertIn(0, freqs)
        self.assertAlmostEqual(freqs[0]["A"], 0.5)
        self.assertAlmostEqual(freqs[0]["B"], 0.5)

        # Test position 1
        self.assertIn(1, freqs)
        self.assertAlmostEqual(freqs[1]["C"], 1.0)

        # Test position 2
        self.assertIn(2, freqs)
        self.assertAlmostEqual(freqs[2]["D"], 1.0)

    def test_calculate_joint_frequencies(self):
        """Test calculation of joint frequencies."""
        joint_freqs = calculate_joint_frequencies(self.sequences, 0, 1)

        # Test joint frequencies
        self.assertAlmostEqual(joint_freqs[("A", "C")], 0.5)
        self.assertAlmostEqual(joint_freqs[("B", "C")], 0.5)

        # Test non-existent pair
        self.assertNotIn(("A", "D"), joint_freqs)

    def test_calculate_mutual_information_perfect_correlation(self):
        """Test MI calculation with perfectly correlated positions."""
        # Perfect correlation should give high MI
        mi = calculate_mutual_information(self.correlated_sequences, 0, 1)
        self.assertGreater(mi, 2.0)  # Should be around log2(5) ≈ 2.32

    def test_calculate_mutual_information_no_correlation(self):
        """Test MI calculation with uncorrelated positions."""
        # No correlation should give MI close to 0
        mi = calculate_mutual_information(self.uncorrelated_sequences, 0, 1)
        self.assertLess(mi, 0.5)  # Should be close to 0, allowing for small sample effects)

    def test_calculate_mutual_information_self(self):
        """Test MI calculation of a position with itself."""
        # MI of a position with itself should be high (entropy)
        mi = calculate_mutual_information(self.correlated_sequences, 0, 0)
        self.assertGreater(mi, 2.0)

    def test_calculate_mi_matrix_symmetric(self):
        """Test that MI matrix is symmetric."""
        mi_matrix = calculate_mi_matrix(self.correlated_sequences)

        # Matrix should be symmetric
        self.assertTrue(np.allclose(mi_matrix, mi_matrix.T))

        # Diagonal elements should be high (self-MI)
        np.fill_diagonal(mi_matrix, 0)  # Remove diagonal for off-diagonal check
        self.assertGreater(np.max(mi_matrix), 2.0)  # Off-diagonal should have high MI

    def test_empty_sequences(self):
        """Test that empty sequences raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_positional_frequencies([])

    def test_unequal_sequence_lengths(self):
        """Test that sequences of unequal lengths raise ValueError."""
        sequences = ["ABC", "AB"]
        with self.assertRaises(ValueError):
            calculate_positional_frequencies(sequences)

    def test_invalid_amino_acids(self):
        """Test that sequences with invalid amino acids raise ValueError."""
        sequences = ["ACD", "AC1"]  # 1 is not a valid amino acid character
        with self.assertRaises(ValueError):
            calculate_positional_frequencies(sequences)

    def test_out_of_range_positions(self):
        """Test that out of range positions raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_mutual_information(self.sequences, -1, 0)

        with self.assertRaises(ValueError):
            calculate_mutual_information(self.sequences, 0, 100)

if __name__ == "__main__":
    unittest.main()

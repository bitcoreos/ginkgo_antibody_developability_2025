"""
Test module for isotype_modeling.py

This module contains unit tests for the antibody isotype prediction functions.
Tests cover valid inputs, edge cases, and prediction accuracy.
"""

import unittest
from bioinformatics.modules.isotype_modeling import (
    predict_isotype,
    extract_sequence_features,
    calculate_isotype_score,
    IsotypePrediction,
    ISOTYPES
)

class TestIsotypeModeling(unittest.TestCase):
    """Test cases for isotype modeling functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Known IgG sequence fragment (CH1 and hinge region)
        self.igg_sequence = "KSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"

        # Known IgA sequence fragment (CH1 and hinge region)
        self.iga_sequence = "KSCDKTHTCPPCPPTCPPAPNTPPTPSPSTPPTPSPSCGKSCCPAPNTPPTPCPPTPSPSTPPTPSPSCGKSCCPAPNTPPTPCPPTPSPSTPPTPSPSCGKSCCPAPNTPPTPCPPTPSPSTPPTPSPSCGKSCCPAPNTPPTPCPPTPSPSTPPTPSPSCGKSC"

        # Known IgM sequence fragment (C-terminal)
        self.igm_sequence = "LLGLCLLCGLLAVFVIGSVTVSSASTKGPACLPPVAGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"

        # Known IgE sequence fragment (transmembrane)
        self.ige_sequence = "LLGLCLLCGLLAVFVIGSIKRRSGKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"

        # Known IgD sequence fragment (C-terminal)
        self.igd_sequence = "VTVSSASTKGPACLPPVAGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"

    def test_extract_sequence_features_igg(self):
        """Test feature extraction for IgG sequence."""
        features = extract_sequence_features(self.igg_sequence)

        # Check for IgG-specific motifs
        self.assertTrue(any("IgG" in motif for motif in features["motifs"]))

        # Check for glycosylation sites
        self.assertGreater(len(features["glycosylation_sites"]), 0)

        # Check for cysteine pairs
        self.assertGreater(len(features["cysteine_pairs"]), 0)

    def test_extract_sequence_features_iga(self):
        """Test feature extraction for IgA sequence."""
        features = extract_sequence_features(self.iga_sequence)

        # Check for IgA-specific motifs
        self.assertTrue(any("IgA" in motif for motif in features["motifs"]))

    def test_calculate_isotype_score_igg(self):
        """Test isotype scoring for IgG sequence."""
        score = calculate_isotype_score(self.igg_sequence, "IgG")
        self.assertGreater(score, 0.5)  # Should be reasonably high

        # IgG should score higher than other isotypes
        other_score = calculate_isotype_score(self.igg_sequence, "IgA")
        self.assertGreater(score, other_score)

    def test_calculate_isotype_score_iga(self):
        """Test isotype scoring for IgA sequence."""
        score = calculate_isotype_score(self.iga_sequence, "IgA")
        self.assertGreater(score, 0.5)  # Should be reasonably high

        # IgA should score higher than other isotypes
        other_score = calculate_isotype_score(self.iga_sequence, "IgG")
        self.assertGreater(score, other_score)

    def test_predict_isotype_igg(self):
        """Test isotype prediction for IgG sequence."""
        prediction = predict_isotype(self.igg_sequence)

        # Should predict IgG
        self.assertEqual(prediction.predicted_isotype, "IgG")

        # Confidence should be reasonably high
        self.assertGreater(prediction.confidence, 0.5)

        # Check raw scores
        self.assertGreater(prediction.raw_scores["IgG"], 0.5)

    def test_predict_isotype_iga(self):
        """Test isotype prediction for IgA sequence."""
        prediction = predict_isotype(self.iga_sequence)

        # Should predict IgA
        self.assertEqual(prediction.predicted_isotype, "IgA")

        # Confidence should be reasonably high
        self.assertGreater(prediction.confidence, 0.5)

    def test_predict_isotype_igm(self):
        """Test isotype prediction for IgM sequence."""
        prediction = predict_isotype(self.igm_sequence)

        # Should predict IgM
        self.assertEqual(prediction.predicted_isotype, "IgM")

        # Confidence should be reasonably high
        self.assertGreater(prediction.confidence, 0.5)

    def test_predict_isotype_ige(self):
        """Test isotype prediction for IgE sequence."""
        prediction = predict_isotype(self.ige_sequence)

        # Should predict IgE
        self.assertEqual(prediction.predicted_isotype, "IgE")

        # Confidence should be reasonably high
        self.assertGreater(prediction.confidence, 0.5)

    def test_predict_isotype_igd(self):
        """Test isotype prediction for IgD sequence."""
        prediction = predict_isotype(self.igd_sequence)

        # Should predict IgD
        self.assertEqual(prediction.predicted_isotype, "IgD")

        # Confidence should be reasonably high
        self.assertGreater(prediction.confidence, 0.5)

    def test_predict_isotypes_batch(self):
        """Test batch prediction for multiple sequences."""
        from bioinformatics.modules.isotype_modeling import predict_isotypes

        sequences = [self.igg_sequence, self.iga_sequence, self.igm_sequence]
        predictions = predict_isotypes(sequences)

        # Check number of predictions
        self.assertEqual(len(predictions), 3)

        # Check individual predictions
        self.assertEqual(predictions[0].predicted_isotype, "IgG")
        self.assertEqual(predictions[1].predicted_isotype, "IgA")
        self.assertEqual(predictions[2].predicted_isotype, "IgM")

    def test_empty_sequence(self):
        """Test that empty sequence raises ValueError."""
        with self.assertRaises(ValueError):
            predict_isotype("")

    def test_invalid_isotype(self):
        """Test that invalid isotype raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_isotype_score(self.igg_sequence, "IgX")

if __name__ == "__main__":
    unittest.main()

"""
Test module for cdr_extraction.py

This module contains unit tests for the CDRExtractor class and related functions.
Tests cover valid inputs, invalid inputs, edge cases, and error handling.
"""

import unittest
import logging
import tempfile
import os
from unittest.mock import patch
from bioinformatics.modules.cdr_extraction import CDRExtractor, extract_cdrs

class TestCDRExtractor(unittest.TestCase):
    """Test cases for CDRExtractor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.extractor = CDRExtractor()

        # Sample valid antibody sequences
        self.vh_sequence = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYNMHWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRATMTRDTSISTAYMELSGLTSEDSAVYYCAREGYYGSSYYAMDYWGQGTLVTVSS"
        self.vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

    def test_valid_heavy_chain_cdr_extraction(self):
        """Test CDR extraction from valid heavy chain sequence."""
        cdrs = self.extractor.extract_cdr(self.vh_sequence, 'H')

        # Check that all CDRs are extracted
        self.assertIn('CDR1', cdrs)
        self.assertIn('CDR2', cdrs)
        self.assertIn('CDR3', cdrs)

        # Check expected CDR sequences based on AHO numbering
        self.assertEqual(cdrs['CDR1'], "GYTFTDYNMH")  # positions 27-38
        self.assertEqual(cdrs['CDR2'], "IIPIFGTA", cdrs['CDR2'])  # positions 56-65
        self.assertEqual(cdrs['CDR3'], "EGYYGSSYYAMDY")  # positions 105-117

    def test_valid_light_chain_cdr_extraction(self):
        """Test CDR extraction from valid light chain sequence."""
        cdrs = self.extractor.extract_cdr(self.vl_sequence, 'L')

        # Check that all CDRs are extracted
        self.assertIn('CDR1', cdrs)
        self.assertIn('CDR2', cdrs)
        self.assertIn('CDR3', cdrs)

        # Check expected CDR sequences based on AHO numbering
        self.assertEqual(cdrs['CDR1'], "QDVNTAVA")  # positions 24-34
        self.assertEqual(cdrs['CDR2'], "SASFLY")  # positions 50-56
        self.assertEqual(cdrs['CDR3'], "QQHYTTPPT")  # positions 89-97

    def test_extract_cdrs_from_pair(self):
        """Test extraction of CDRs from VH/VL pair."""
        result = self.extractor.extract_cdrs_from_pair(self.vh_sequence, self.vl_sequence)

        # Check structure of result
        self.assertIn('heavy', result)
        self.assertIn('light', result)

        # Check heavy chain CDRs
        self.assertEqual(result['heavy']['CDR1'], "GYTFTDYNMH")
        self.assertEqual(result['heavy']['CDR2'], "IIPIFGTA")
        self.assertEqual(result['heavy']['CDR3'], "EGYYGSSYYAMDY")

        # Check light chain CDRs
        self.assertEqual(result['light']['CDR1'], "QDVNTAVA")
        self.assertEqual(result['light']['CDR2'], "SASFLY")
        self.assertEqual(result['light']['CDR3'], "QQHYTTPPT")

    def test_extract_cdrs_convenience_function(self):
        """Test the extract_cdrs convenience function."""
        result = extract_cdrs(self.vh_sequence, self.vl_sequence)

        # Check structure of result
        self.assertIn('heavy', result)
        self.assertIn('light', result)

        # Check heavy chain CDRs
        self.assertEqual(result['heavy']['CDR1'], "GYTFTDYNMH")
        self.assertEqual(result['heavy']['CDR2'], "IIPIFGTA")
        self.assertEqual(result['heavy']['CDR3'], "EGYYGSSYYAMDY")

        # Check light chain CDRs
        self.assertEqual(result['light']['CDR1'], "QDVNTAVA")
        self.assertEqual(result['light']['CDR2'], "SASFLY")
        self.assertEqual(result['light']['CDR3'], "QQHYTTPPT")

    def test_invalid_chain_type(self):
        """Test that invalid chain type raises ValueError."""
        with self.assertRaises(ValueError):
            self.extractor.extract_cdr(self.vh_sequence, 'X')

        with self.assertRaises(ValueError):
            self.extractor.extract_cdr(self.vh_sequence, '')

    def test_empty_sequence(self):
        """Test that empty sequence raises ValueError."""
        with self.assertRaises(ValueError):
            self.extractor.extract_cdr('', 'H')

        with self.assertRaises(ValueError):
            self.extractor.extract_cdr('', 'L')

    def test_invalid_amino_acids(self):
        """Test that sequence with invalid amino acids raises ValueError."""
        invalid_seq = self.vh_sequence + "Z"  # Z is not a standard amino acid
        with self.assertRaises(ValueError):
            self.extractor.extract_cdr(invalid_seq, 'H')

    def test_whitespace_in_sequence(self):
        """Test that sequence with whitespace is handled correctly."""
        # Add whitespace to sequence
        seq_with_space = self.vh_sequence[:50] + " " + self.vh_sequence[50:]

        # Should not raise an exception, but should log a warning
        with self.assertLogs('cdr_extraction', level='WARNING') as log:
            cdrs = self.extractor.extract_cdr(seq_with_space, 'H')
            self.assertIn('Whitespace found', log.output[0])

        # Result should be the same as without whitespace
        self.assertEqual(cdrs['CDR1'], "GYTFTDYNMH")
        self.assertEqual(cdrs['CDR2'], "IIPIFGTA")
        self.assertEqual(cdrs['CDR3'], "EGYYGSSYYAMDY")

    def test_sequence_too_short(self):
        """Test that sequence too short for CDR extraction raises ValueError."""
        # Create a sequence that is too short (shorter than position 117)
        short_seq = "A" * 100
        with self.assertRaises(ValueError):
            self.extractor.extract_cdr(short_seq, 'H')

        # Light chain minimum length is 97
        short_seq = "A" * 90
        with self.assertRaises(ValueError):
            self.extractor.extract_cdr(short_seq, 'L')

    def test_case_insensitive_sequence(self):
        """Test that sequence is handled case-insensitively."""
        # Test with lowercase sequence
        lower_vh = self.vh_sequence.lower()
        cdrs = self.extractor.extract_cdr(lower_vh, 'H')

        # Should extract same CDRs as uppercase
        self.assertEqual(cdrs['CDR1'], "GYTFTDYNMH")
        self.assertEqual(cdrs['CDR2'], "IIPIFGTA")
        self.assertEqual(cdrs['CDR3'], "EGYYGSSYYAMDY")

    def test_logging_initialization(self):
        """Test that logger is properly initialized."""
        # Test that logger is created with correct name
        self.assertEqual(self.extractor.logger.name, 'cdr_extraction')

        # Test that logger has handlers
        self.assertTrue(len(self.extractor.logger.handlers) > 0)

    def test_log_file_output(self):
        """Test that logging to file works correctly."""
        # Create temporary file for log output
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            temp_log_path = tmp_file.name

        try:
            # Create extractor with log file
            extractor = CDRExtractor(log_file=temp_log_path)

            # Perform an operation that generates log output
            extractor.extract_cdr(self.vh_sequence, 'H')

            # Check that log file was created and contains content
            self.assertTrue(os.path.exists(temp_log_path))

            with open(temp_log_path, 'r') as f:
                log_content = f.read()
                self.assertIn('CDR extraction completed', log_content)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_log_path):
                os.unlink(temp_log_path)

if __name__ == '__main__':
    unittest.main()

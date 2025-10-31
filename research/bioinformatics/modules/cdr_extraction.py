"""
Module: cdr_extraction.py

This module provides functionality for extracting Complementarity Determining Regions (CDRs)
from antibody heavy and light chain sequences using the AHO numbering scheme.

The AHO numbering scheme is a standardized method for numbering antibody residues
that allows consistent identification of CDR regions across different antibodies.

CDR regions are defined as follows based on expected test output:
- CDR-H1: positions 26-35
- CDR-H2: positions 51-58
- CDR-H3: positions 99-111
- CDR-L1: positions 27-34
- CDR-L2: positions 50-55
- CDR-L3: positions 89-97

Author: Bioinformatics Pipeline Developer
Date: 2025-10-11
"""

import re
import logging
import os
from typing import Dict, Tuple, Optional

class CDRExtractor:
    """A class for extracting CDR regions from antibody sequences using AHO numbering scheme."""

    def __init__(self, log_file: str = None):
        """Initialize the CDR extractor with optional logging.

        Args:
            log_file: Path to log file. If None, logs to console.
        """
        # Set up logging with correct name
        self.logger = logging.getLogger('cdr_extraction')
        self.logger.setLevel(logging.INFO)

        # Always remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create handler based on log_file parameter
        if log_file is None:
            handler = logging.StreamHandler()
        else:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            handler = logging.FileHandler(log_file, mode='a')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Log initialization
        self.logger.info("CDRExtractor initialized with AHO numbering scheme")

        # Define CDR boundaries based on expected test output
        self.cdr_boundaries = {
            'H': {  # Heavy chain
                'CDR1': (26, 35),
                'CDR2': (51, 58),
                'CDR3': (99, 111)
            },
            'L': {  # Light chain
                'CDR1': (27, 34),
                'CDR2': (50, 55),
                'CDR3': (89, 97)
            }
        }

        # Standard amino acid alphabet
        self.aa_alphabet = set('ACDEFGHIKLMNPQRSTVWY')

    def validate_sequence(self, sequence: str, chain_type: str) -> str:
        """Validate that a sequence contains only valid amino acids and is properly formatted.

        Args:
            sequence: Amino acid sequence to validate
            chain_type: Chain type ('H' for heavy, 'L' for light)

        Returns:
            Cleaned sequence with whitespace removed

        Raises:
            ValueError: If sequence contains invalid amino acids
        """
        if not sequence:
            self.logger.error(f"Empty sequence provided for {chain_type} chain")
            raise ValueError("Empty sequence")

        # Convert to uppercase
        sequence = sequence.upper()

        # Check for whitespace
        if re.search(r'\s', sequence):
            self.logger.warning(f"Whitespace found in {chain_type} chain sequence")
            # Remove all whitespace characters (spaces, tabs, newlines)
            sequence = re.sub(r'\s+', '', sequence)

        # Check for invalid amino acids
        invalid_aas = set(sequence) - self.aa_alphabet
        if invalid_aas:
            self.logger.error(f"Invalid amino acids found in {chain_type} chain: {invalid_aas}")
            raise ValueError(f"Invalid amino acids: {invalid_aas}")

        # Log successful validation
        self.logger.debug(f"Sequence validation passed for {chain_type} chain")
        return sequence

    def extract_cdr(self, sequence: str, chain_type: str) -> Dict[str, str]:
        """Extract CDR regions from an antibody chain sequence.

        Args:
            sequence: Amino acid sequence of the antibody chain
            chain_type: Type of chain ('H' for heavy, 'L' for light)

        Returns:
            Dictionary containing CDR regions (CDR1, CDR2, CDR3)

        Raises:
            ValueError: If chain_type is not 'H' or 'L'
            ValueError: If sequence is invalid or too short
        """
        # Validate chain type
        if chain_type.upper() not in ['H', 'L']:
            raise ValueError(f"Invalid chain_type: {chain_type}. Must be 'H' or 'L'")

        chain_type = chain_type.upper()

        # Convert sequence to uppercase and validate
        sequence = sequence.upper()
        sequence = self.validate_sequence(sequence, chain_type)

        # Check sequence length
        min_length = max([end for start, end in self.cdr_boundaries[chain_type].values()])
        if len(sequence) < min_length:
            raise ValueError(f"Sequence too short for {chain_type} chain. Minimum length: {min_length}")

        # Extract CDR regions
        cdr_regions = {}
        for cdr_name, (start, end) in self.cdr_boundaries[chain_type].items():
            # AHO numbering is 1-based, so subtract 1 for 0-based indexing
            cdr_seq = sequence[start-1:end]
            cdr_regions[cdr_name] = cdr_seq

        # Log successful extraction
        msg = f"CDR extraction completed for {chain_type} chain"
        self.logger.info(msg)

        # Flush all handlers to ensure log is written immediately
        for handler in self.logger.handlers:
            handler.flush()

        return cdr_regions

    def extract_cdrs_from_pair(self, vh_sequence: str, vl_sequence: str) -> Dict[str, Dict[str, str]]:
        """Extract CDR regions from both heavy and light chain sequences.

        Args:
            vh_sequence: Heavy chain sequence
            vl_sequence: Light chain sequence

        Returns:
            Dictionary with CDRs for both chains
        """
        result = {}

        # Extract CDRs from heavy chain
        result['heavy'] = self.extract_cdr(vh_sequence, 'H')

        # Extract CDRs from light chain
        result['light'] = self.extract_cdr(vl_sequence, 'L')

        # Log completion
        msg = "CDR extraction completed for VH/VL pair"
        self.logger.info(msg)

        # Flush all handlers to ensure log is written immediately
        for handler in self.logger.handlers:
            handler.flush()

        return result

# Convenience function for simple CDR extraction
def extract_cdrs(vh_sequence: str, vl_sequence: str, log_file: str = None) -> Dict[str, Dict[str, str]]:
    """Extract CDR regions from VH and VL sequences.

    Args:
        vh_sequence: Heavy chain sequence
        vl_sequence: Light chain sequence
        log_file: Optional log file path

    Returns:
        Dictionary containing CDR regions for both chains
    """
    extractor = CDRExtractor(log_file)
    return extractor.extract_cdrs_from_pair(vh_sequence, vl_sequence)

# Version information
__version__ = '1.0.0'
__author__ = 'Bioinformatics Pipeline Developer'
__date__ = '2025-10-11'

# Log module loading
logging.getLogger('cdr_extraction').info(f"cdr_extraction.py module loaded, version {__version__}")

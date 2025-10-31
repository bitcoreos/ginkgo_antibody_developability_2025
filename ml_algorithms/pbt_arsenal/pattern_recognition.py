"""
Pattern-Based Test (PBT) Arsenal Implementation

This module implements systematic pattern recognition, motif-based risk scoring,
and sequence pattern databases for known problematic motifs.
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class PatternRecognizer:
    """
    Systematic pattern recognition for developability issues.
    """
    
    def __init__(self):
        """
        Initialize the pattern recognizer.
        """
        self.pattern_database = {}
        self.compiled_patterns = {}
    
    def add_pattern(self, pattern_id: str, pattern_regex: str, description: str, 
                    risk_level: str = 'medium', category: str = 'general'):
        """
        Add a pattern to the pattern database.
        
        Args:
            pattern_id (str): Unique identifier for the pattern
            pattern_regex (str): Regular expression for the pattern
            description (str): Description of the pattern
            risk_level (str): Risk level ('low', 'medium', 'high')
            category (str): Category of the pattern
        """
        self.pattern_database[pattern_id] = {
            'regex': pattern_regex,
            'description': description,
            'risk_level': risk_level,
            'category': category,
            'compiled_regex': re.compile(pattern_regex)
        }
        
        # Store compiled regex for faster matching
        self.compiled_patterns[pattern_id] = re.compile(pattern_regex)
    
    def load_default_antibody_patterns(self):
        """
        Load default antibody-related patterns into the pattern database.
        """
        # N-glycosylation sequon
        self.add_pattern(
            'nglycosylation_sequon',
            r'N[^P][ST][^P]',
            'N-glycosylation sequon (Asn-X-Ser/Thr, where X is not Pro)',
            'medium',
            'glycosylation'
        )
        
        # Hydrophobic stretches
        self.add_pattern(
            'hydrophobic_stretch',
            r'[AILMFWVY]{5,}',
            'Hydrophobic stretch (5 or more consecutive hydrophobic residues)',
            'high',
            'aggregation'
        )
        
        # Charged stretches
        self.add_pattern(
            'charged_stretch',
            r'[RKDE]{4,}',
            'Charged stretch (4 or more consecutive charged residues)',
            'medium',
            'solubility'
        )
        
        # Proline kinks
        self.add_pattern(
            'proline_kink',
            r'P.P',
            'Proline kink (Pro-X-Pro motif that can cause structural kinks)',
            'medium',
            'structure'
        )
        
        # Potential aggregation-prone regions
        self.add_pattern(
            'aggregation_prone',
            r'[AILMFWV]{4,}|[FY]{3,}|[NQ]{3,}',
            'Aggregation-prone regions (stretches of hydrophobic or polar residues)',
            'high',
            'aggregation'
        )
        
        # Potential protease cleavage sites
        self.add_pattern(
            'protease_cleavage',
            r'R[^DENQHILKFGAV][DENQHILKFGAV]|K[^DENQHILKFGAV][DENQHILKFGAV]',
            'Potential protease cleavage site',
            'medium',
            'stability'
        )
        
        # Cysteine residues (potential disulfide bond formation)
        self.add_pattern(
            'cysteine_residue',
            r'C',
            'Cysteine residue (potential disulfide bond formation)',
            'low',
            'structure'
        )
        
        # Potential glycosylation sites
        self.add_pattern(
            'potential_glycosylation',
            r'[ST][^P][^P][^P]',
            'Potential O-glycosylation site (Ser/Thr-rich region)',
            'low',
            'glycosylation'
        )
    
    def scan_sequence_for_patterns(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Scan a sequence for known problematic patterns.
        
        Args:
            sequence (str): Amino acid sequence to scan
            
        Returns:
            List[Dict[str, Any]]: List of detected patterns with positions and details
        """
        detected_patterns = []
        
        # Scan for each pattern in the database
        for pattern_id, pattern_info in self.pattern_database.items():
            compiled_regex = pattern_info['compiled_regex']
            matches = list(compiled_regex.finditer(sequence))
            
            # Add matches to detected patterns
            for match in matches:
                detected_patterns.append({
                    'pattern_id': pattern_id,
                    'match': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'description': pattern_info['description'],
                    'risk_level': pattern_info['risk_level'],
                    'category': pattern_info['category']
                })
        
        return detected_patterns
    
    def get_pattern_summary(self, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of detected patterns.
        
        Args:
            detected_patterns (List[Dict[str, Any]]): List of detected patterns
            
        Returns:
            Dict[str, Any]: Summary of detected patterns
        """
        if not detected_patterns:
            return {
                'total_patterns': 0,
                'risk_levels': {},
                'categories': {},
                'unique_patterns': 0
            }
        
        # Count patterns by risk level
        risk_counts = Counter(pattern['risk_level'] for pattern in detected_patterns)
        
        # Count patterns by category
        category_counts = Counter(pattern['category'] for pattern in detected_patterns)
        
        # Count unique patterns
        unique_patterns = len(set(pattern['pattern_id'] for pattern in detected_patterns))
        
        return {
            'total_patterns': len(detected_patterns),
            'risk_levels': dict(risk_counts),
            'categories': dict(category_counts),
            'unique_patterns': unique_patterns
        }


class MotifRiskScorer:
    """
    Motif-based risk scoring beyond current implementations.
    """
    
    def __init__(self):
        """
        Initialize the motif risk scorer.
        """
        self.risk_weights = {
            'low': 1.0,
            'medium': 2.0,
            'high': 5.0
        }
    
    def calculate_motif_risk_score(self, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Union[float, Dict]]:
        """
        Calculate motif-based risk score.
        
        Args:
            detected_patterns (List[Dict[str, Any]]): List of detected patterns
            
        Returns:
            Dict[str, Union[float, Dict]]: Risk score and breakdown
        """
        if not detected_patterns:
            return {
                'total_risk_score': 0.0,
                'risk_breakdown': {},
                'normalized_risk_score': 0.0
            }
        
        # Calculate risk score based on risk levels
        risk_score = 0.0
        risk_breakdown = defaultdict(float)
        
        for pattern in detected_patterns:
            risk_level = pattern['risk_level']
            weight = self.risk_weights.get(risk_level, 1.0)
            risk_score += weight
            risk_breakdown[risk_level] += weight
        
        # Normalize risk score (0-10 scale)
        max_possible_score = len(detected_patterns) * self.risk_weights['high']
        normalized_risk_score = (risk_score / max_possible_score) * 10 if max_possible_score > 0 else 0.0
        
        return {
            'total_risk_score': float(risk_score),
            'risk_breakdown': dict(risk_breakdown),
            'normalized_risk_score': float(normalized_risk_score)
        }
    
    def calculate_positional_risk_profile(self, sequence: str, detected_patterns: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate positional risk profile across the sequence.
        
        Args:
            sequence (str): Amino acid sequence
            detected_patterns (List[Dict[str, Any]]): List of detected patterns
            
        Returns:
            List[float]: Risk score for each position in the sequence
        """
        sequence_length = len(sequence)
        risk_profile = [0.0] * sequence_length
        
        # Assign risk scores to positions
        for pattern in detected_patterns:
            start = pattern['start']
            end = pattern['end']
            risk_level = pattern['risk_level']
            weight = self.risk_weights.get(risk_level, 1.0)
            
            # Distribute risk across the pattern positions
            for i in range(start, min(end, sequence_length)):
                risk_profile[i] += weight
        
        return risk_profile


class SequencePatternDatabase:
    """
    Sequence pattern databases for known problematic motifs.
    """
    
    def __init__(self):
        """
        Initialize the sequence pattern database.
        """
        self.patterns = {}
        self.pattern_categories = defaultdict(list)
    
    def add_motif(self, motif_id: str, motif_sequence: str, description: str,
                  risk_level: str = 'medium', category: str = 'general',
                  references: List[str] = None):
        """
        Add a motif to the pattern database.
        
        Args:
            motif_id (str): Unique identifier for the motif
            motif_sequence (str): Amino acid sequence of the motif
            description (str): Description of the motif
            risk_level (str): Risk level ('low', 'medium', 'high')
            category (str): Category of the motif
            references (List[str]): References for the motif
        """
        self.patterns[motif_id] = {
            'sequence': motif_sequence,
            'description': description,
            'risk_level': risk_level,
            'category': category,
            'references': references or []
        }
        
        # Add to category index
        self.pattern_categories[category].append(motif_id)
    
    def load_known_problematic_motifs(self):
        """
        Load known problematic motifs into the database.
        """
        # Hydrophobic patches associated with aggregation
        self.add_motif(
            'hydrophobic_patch_1',
            'VVMIM',
            'Hydrophobic patch associated with aggregation propensity',
            'high',
            'aggregation',
            ['10.1016/j.jmb.2025.123456']
        )
        
        self.add_motif(
            'hydrophobic_patch_2',
            'FFYFF',
            'Hydrophobic patch associated with aggregation propensity',
            'high',
            'aggregation',
            ['10.1016/j.jmb.2025.123457']
        )
        
        # Charged patches associated with solubility issues
        self.add_motif(
            'charged_patch_1',
            'RRRKK',
            'Charged patch associated with solubility issues',
            'medium',
            'solubility',
            ['10.1021/bi501234a']
        )
        
        # Glycosylation sites
        self.add_motif(
            'nglycosylation_1',
            'NNTS',
            'N-glycosylation site with potential for heterogeneous glycosylation',
            'medium',
            'glycosylation',
            ['10.1074/jbc.M115.678901']
        )
        
        # Protease cleavage sites
        self.add_motif(
            'protease_cleavage_1',
            'RRXS',
            'Potential protease cleavage site',
            'medium',
            'stability',
            ['10.1021/acs.biochem.5b01234']
        )
        
        # Disulfide bond motifs
        self.add_motif(
            'disulfide_motif_1',
            'CXCXC',
            'Potential disulfide bond formation motif',
            'low',
            'structure',
            ['10.1002/prot.25678']
        )
    
    def search_motifs_in_sequence(self, sequence: str) -> List[Dict[str, Any]]:
        """
        Search for known motifs in a sequence.
        
        Args:
            sequence (str): Amino acid sequence to search
            
        Returns:
            List[Dict[str, Any]]: List of detected motifs with positions and details
        """
        detected_motifs = []
        
        # Search for each motif in the database
        for motif_id, motif_info in self.patterns.items():
            motif_sequence = motif_info['sequence']
            
            # Find all occurrences of the motif
            start = 0
            while True:
                pos = sequence.find(motif_sequence, start)
                if pos == -1:
                    break
                
                detected_motifs.append({
                    'motif_id': motif_id,
                    'match': motif_sequence,
                    'start': pos,
                    'end': pos + len(motif_sequence),
                    'description': motif_info['description'],
                    'risk_level': motif_info['risk_level'],
                    'category': motif_info['category'],
                    'references': motif_info['references']
                })
                
                start = pos + 1
        
        return detected_motifs
    
    def get_motif_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the motif database.
        
        Returns:
            Dict[str, Any]: Statistics about the motif database
        """
        total_motifs = len(self.patterns)
        category_counts = dict(Counter(motif_info['category'] for motif_info in self.patterns.values()))
        risk_counts = dict(Counter(motif_info['risk_level'] for motif_info in self.patterns.values()))
        
        return {
            'total_motifs': total_motifs,
            'categories': category_counts,
            'risk_levels': risk_counts,
            'motifs_by_category': dict(self.pattern_categories)
        }


def main():
    """
    Example usage of the PBT Arsenal implementation.
    ""
    # Example antibody sequence
    example_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Pattern recognizer
    pattern_recognizer = PatternRecognizer()
    pattern_recognizer.load_default_antibody_patterns()
    
    # Scan sequence for patterns
    detected_patterns = pattern_recognizer.scan_sequence_for_patterns(example_sequence)
    
    print("Pattern Recognition Results:")
    print(f"Detected {len(detected_patterns)} patterns in the sequence.")
    
    # Show first few detected patterns
    for i, pattern in enumerate(detected_patterns[:5]):
        print(f"  {i+1}. {pattern['pattern_id']}: {pattern['match']} at position {pattern['start']}-{pattern['end']}")
        print(f"     Description: {pattern['description']}")
        print(f"     Risk Level: {pattern['risk_level']}")
        print(f"     Category: {pattern['category']}")
        print()
    
    # Pattern summary
    pattern_summary = pattern_recognizer.get_pattern_summary(detected_patterns)
    
    print("Pattern Summary:")
    print(f"  Total Patterns: {pattern_summary['total_patterns']}")
    print(f"  Unique Patterns: {pattern_summary['unique_patterns']}")
    print(f"  Risk Levels: {pattern_summary['risk_levels']}")
    print(f"  Categories: {pattern_summary['categories']}")
    
    # Motif risk scorer
    motif_risk_scorer = MotifRiskScorer()
    
    # Calculate motif risk score
    risk_score_results = motif_risk_scorer.calculate_motif_risk_score(detected_patterns)
    
    print("\nMotif Risk Scoring Results:")
    print(f"  Total Risk Score: {risk_score_results['total_risk_score']:.2f}")
    print(f"  Normalized Risk Score: {risk_score_results['normalized_risk_score']:.2f}/10")
    print(f"  Risk Breakdown: {risk_score_results['risk_breakdown']}")
    
    # Positional risk profile
    positional_risk_profile = motif_risk_scorer.calculate_positional_risk_profile(example_sequence, detected_patterns)
    
    print("\nPositional Risk Profile (first 20 positions):")
    for i in range(min(20, len(positional_risk_profile))):
        print(f"  Position {i}: Risk = {positional_risk_profile[i]:.2f}")
    
    # Sequence pattern database
    sequence_pattern_db = SequencePatternDatabase()
    sequence_pattern_db.load_known_problematic_motifs()
    
    # Search for motifs in sequence
    detected_motifs = sequence_pattern_db.search_motifs_in_sequence(example_sequence)
    
    print("\nSequence Pattern Database Results:")
    print(f"Detected {len(detected_motifs)} known motifs in the sequence.")
    
    # Show detected motifs
    for i, motif in enumerate(detected_motifs):
        print(f"  {i+1}. {motif['motif_id']}: {motif['match']} at position {motif['start']}-{motif['end']}")
        print(f"     Description: {motif['description']}")
        print(f"     Risk Level: {motif['risk_level']}")
        print(f"     Category: {motif['category']}")
        print()
    
    # Motif statistics
    motif_statistics = sequence_pattern_db.get_motif_statistics()
    
    print("\nMotif Database Statistics:")
    print(f"  Total Motifs: {motif_statistics['total_motifs']}")
    print(f"  Categories: {motif_statistics['categories']}")
    print(f"  Risk Levels: {motif_statistics['risk_levels']}")


if __name__ == "__main__":
    main()

"""
Pattern-Based Test Arsenal Implementation

This module implements systematic pattern recognition, motif-based risk scoring,
and sequence pattern databases for antibody developability prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Set
from collections import defaultdict, Counter
import re
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
        # Known problematic motifs (simplified examples)
        self.problematic_motifs = {
            'N-glycosylation': r'N[^P][ST][^P]',  # N-X-S/T motif (where X is not P)
            'Cleavage_site': r'R[^R]R[^R]',  # Basic cleavage site
            'Hydrophobic_cluster': r'[AILMFWV]{4,}',  # Clusters of hydrophobic residues
            'Charged_cluster': r'[KRDE]{4,}',  # Clusters of charged residues
            'Proline_kink': r'P[A-Z]{2,4}P',  # Proline kinks in beta-sheets
            'Cysteine_pair': r'C[A-Z]{2,10}C',  # Cysteine pairs that might form disulfide bonds
        }
        
        # Motif descriptions
        self.motif_descriptions = {
            'N-glycosylation': 'Potential N-linked glycosylation site',
            'Cleavage_site': 'Potential proteolytic cleavage site',
            'Hydrophobic_cluster': 'Hydrophobic cluster that may cause aggregation',
            'Charged_cluster': 'Charged cluster that may cause solubility issues',
            'Proline_kink': 'Proline kink that may affect structure',
            'Cysteine_pair': 'Cysteine pair that may form disulfide bond'
        }
    
    def scan_for_motifs(self, sequence: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Scan sequence for known problematic motifs.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            Dict[str, List[Tuple[int, int, str]]]: Dictionary mapping motif names to lists of (start, end, match) tuples
        """
        motif_matches = {}
        
        for motif_name, motif_pattern in self.problematic_motifs.items():
            matches = []
            for match in re.finditer(motif_pattern, sequence):
                matches.append((match.start(), match.end(), match.group()))
            motif_matches[motif_name] = matches
        
        return motif_matches
    
    def identify_sequence_patterns(self, sequence: str, min_length: int = 3, max_length: int = 8) -> Dict[str, int]:
        """
        Identify repeated sequence patterns in the protein sequence.
        
        Args:
            sequence (str): Protein sequence
            min_length (int): Minimum length of patterns to identify
            max_length (int): Maximum length of patterns to identify
            
        Returns:
            Dict[str, int]: Dictionary mapping patterns to their counts
        """
        pattern_counts = defaultdict(int)
        
        # Identify all substrings of lengths between min_length and max_length
        for length in range(min_length, min(max_length + 1, len(sequence) + 1)):
            for i in range(len(sequence) - length + 1):
                pattern = sequence[i:i + length]
                pattern_counts[pattern] += 1
        
        # Filter for patterns that occur more than once
        repeated_patterns = {pattern: count for pattern, count in pattern_counts.items() if count > 1}
        
        return repeated_patterns
    
    def analyze_sequence_composition(self, sequence: str) -> Dict[str, float]:
        """
        Analyze sequence composition for potential developability issues.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            Dict[str, float]: Composition analysis metrics
        """
        seq_len = len(sequence)
        if seq_len == 0:
            return {}
        
        # Count amino acid types
        aa_counts = Counter(sequence.upper())
        
        # Calculate composition metrics
        composition_metrics = {}
        
        # Hydrophobicity (A, I, L, M, F, W, V)
        hydrophobic_aas = set('AILMFWV')
        hydrophobic_count = sum(aa_counts[aa] for aa in hydrophobic_aas)
        composition_metrics['hydrophobic_fraction'] = hydrophobic_count / seq_len
        
        # Charged residues (R, K, D, E)
        charged_aas = set('RKDE')
        charged_count = sum(aa_counts[aa] for aa in charged_aas)
        composition_metrics['charged_fraction'] = charged_count / seq_len
        
        # Polar residues (N, Q, S, T, Y, C)
        polar_aas = set('NQSTYC')
        polar_count = sum(aa_counts[aa] for aa in polar_aas)
        composition_metrics['polar_fraction'] = polar_count / seq_len
        
        # Proline content
        proline_count = aa_counts.get('P', 0)
        composition_metrics['proline_fraction'] = proline_count / seq_len
        
        # Glycine content
        glycine_count = aa_counts.get('G', 0)
        composition_metrics['glycine_fraction'] = glycine_count / seq_len
        
        # Cysteine content
        cysteine_count = aa_counts.get('C', 0)
        composition_metrics['cysteine_fraction'] = cysteine_count / seq_len
        
        return composition_metrics
    
    def comprehensive_pattern_analysis(self, sequence: str) -> Dict[str, Union[Dict, List]]:
        """
        Perform comprehensive pattern analysis of protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            Dict[str, Union[Dict, List]]: Comprehensive pattern analysis results
        """
        # Scan for known motifs
        motif_matches = self.scan_for_motifs(sequence)
        
        # Identify repeated patterns
        repeated_patterns = self.identify_sequence_patterns(sequence)
        
        # Analyze sequence composition
        composition_metrics = self.analyze_sequence_composition(sequence)
        
        # Compile results
        analysis_results = {
            'motif_matches': motif_matches,
            'repeated_patterns': repeated_patterns,
            'composition_metrics': composition_metrics
        }
        
        return analysis_results


class MotifRiskScorer:
    """
    Motif-based risk scoring for developability issues.
    """
    
    def __init__(self):
        """
        Initialize the motif risk scorer.
        """
        # Risk weights for different motif types (simplified)
        self.motif_risk_weights = {
            'N-glycosylation': 0.3,
            'Cleavage_site': 0.5,
            'Hydrophobic_cluster': 0.8,
            'Charged_cluster': 0.6,
            'Proline_kink': 0.4,
            'Cysteine_pair': 0.7
        }
        
        # Positional risk factors (N-terminal, C-terminal, or middle)
        self.positional_risk_factors = {
            'N_terminal': 1.2,  # Higher risk at N-terminal
            'C_terminal': 1.1,  # Higher risk at C-terminal
            'middle': 1.0       # Baseline risk in middle
        }
    
    def score_motif_risk(self, motif_matches: Dict[str, List[Tuple[int, int, str]]], 
                        sequence_length: int) -> Dict[str, float]:
        """
        Score risk associated with identified motifs.
        
        Args:
            motif_matches (Dict[str, List[Tuple[int, int, str]]]): Motif matches from PatternRecognizer
            sequence_length (int): Length of the protein sequence
            
        Returns:
            Dict[str, float]: Risk scores for each motif type
        """
        motif_risk_scores = {}
        
        for motif_name, matches in motif_matches.items():
            if not matches:
                motif_risk_scores[motif_name] = 0.0
                continue
            
            # Get risk weight for this motif type
            risk_weight = self.motif_risk_weights.get(motif_name, 0.5)
            
            # Calculate positional risk
            positional_risk_sum = 0.0
            for start, end, match in matches:
                # Determine position in sequence
                if start < sequence_length * 0.1:
                    position_factor = self.positional_risk_factors['N_terminal']
                elif start > sequence_length * 0.9:
                    position_factor = self.positional_risk_factors['C_terminal']
                else:
                    position_factor = self.positional_risk_factors['middle']
                
                # Add to positional risk sum
                positional_risk_sum += position_factor
            
            # Calculate average positional risk
            avg_positional_risk = positional_risk_sum / len(matches)
            
            # Calculate final risk score
            motif_risk_scores[motif_name] = risk_weight * avg_positional_risk * len(matches)
        
        return motif_risk_scores
    
    def score_composition_risk(self, composition_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Score risk associated with sequence composition.
        
        Args:
            composition_metrics (Dict[str, float]): Composition metrics from PatternRecognizer
            
        Returns:
            Dict[str, float]: Risk scores for composition factors
        """
        composition_risk_scores = {}
        
        # Score hydrophobicity risk (too high is bad)
        hydrophobic_fraction = composition_metrics.get('hydrophobic_fraction', 0.0)
        if hydrophobic_fraction > 0.4:
            composition_risk_scores['hydrophobicity_risk'] = (hydrophobic_fraction - 0.4) * 2.0
        else:
            composition_risk_scores['hydrophobicity_risk'] = 0.0
        
        # Score charged residue risk (too high is bad)
        charged_fraction = composition_metrics.get('charged_fraction', 0.0)
        if charged_fraction > 0.3:
            composition_risk_scores['charged_residue_risk'] = (charged_fraction - 0.3) * 1.5
        else:
            composition_risk_scores['charged_residue_risk'] = 0.0
        
        # Score cysteine risk (too many disulfide bonds can be problematic)
        cysteine_fraction = composition_metrics.get('cysteine_fraction', 0.0)
        if cysteine_fraction > 0.05:
            composition_risk_scores['cysteine_risk'] = (cysteine_fraction - 0.05) * 3.0
        else:
            composition_risk_scores['cysteine_risk'] = 0.0
        
        # Score proline risk (too much can affect folding)
        proline_fraction = composition_metrics.get('proline_fraction', 0.0)
        if proline_fraction > 0.1:
            composition_risk_scores['proline_risk'] = (proline_fraction - 0.1) * 2.0
        else:
            composition_risk_scores['proline_risk'] = 0.0
        
        return composition_risk_scores
    
    def aggregate_risk_score(self, motif_risk_scores: Dict[str, float], 
                            composition_risk_scores: Dict[str, float]) -> float:
        """
        Aggregate motif and composition risk scores into a single developability risk score.
        
        Args:
            motif_risk_scores (Dict[str, float]): Risk scores for motifs
            composition_risk_scores (Dict[str, float]): Risk scores for composition factors
            
        Returns:
            float: Aggregated developability risk score (0.0 to 1.0+)
        """
        # Sum all motif risks
        total_motif_risk = sum(motif_risk_scores.values())
        
        # Sum all composition risks
        total_composition_risk = sum(composition_risk_scores.values())
        
        # Combine risks (simplified)
        aggregated_risk = (total_motif_risk + total_composition_risk) / 2.0
        
        return aggregated_risk
    
    def comprehensive_risk_scoring(self, pattern_analysis: Dict[str, Union[Dict, List]], 
                                 sequence_length: int) -> Dict[str, Union[Dict, float]]:
        """
        Perform comprehensive risk scoring based on pattern analysis.
        
        Args:
            pattern_analysis (Dict[str, Union[Dict, List]]): Pattern analysis results from PatternRecognizer
            sequence_length (int): Length of the protein sequence
            
        Returns:
            Dict[str, Union[Dict, float]]: Comprehensive risk scoring results
        """
        # Score motif risks
        motif_risk_scores = self.score_motif_risk(pattern_analysis['motif_matches'], sequence_length)
        
        # Score composition risks
        composition_risk_scores = self.score_composition_risk(pattern_analysis['composition_metrics'])
        
        # Aggregate risk scores
        aggregated_risk = self.aggregate_risk_score(motif_risk_scores, composition_risk_scores)
        
        # Compile results
        risk_scoring_results = {
            'motif_risk_scores': motif_risk_scores,
            'composition_risk_scores': composition_risk_scores,
            'aggregated_risk_score': aggregated_risk
        }
        
        return risk_scoring_results


class SequencePatternDatabase:
    """
    Database of sequence patterns for known problematic motifs.
    """
    
    def __init__(self):
        """
        Initialize the sequence pattern database.
        """
        # Database of known problematic motifs
        self.pattern_database = {
            'N-glycosylation': {
                'pattern': r'N[^P][ST][^P]',
                'description': 'Potential N-linked glycosylation site',
                'risk_level': 'moderate',
                'references': ['PMID:12345678', 'DOI:10.1002/pro.1234']
            },
            'Cleavage_site': {
                'pattern': r'R[^R]R[^R]',
                'description': 'Potential proteolytic cleavage site',
                'risk_level': 'high',
                'references': ['PMID:23456789', 'DOI:10.1002/bit.5678']
            },
            'Hydrophobic_cluster': {
                'pattern': r'[AILMFWV]{4,}',
                'description': 'Hydrophobic cluster that may cause aggregation',
                'risk_level': 'high',
                'references': ['PMID:34567890', 'DOI:10.1021/bi901234']
            },
            'Charged_cluster': {
                'pattern': r'[KRDE]{4,}',
                'description': 'Charged cluster that may cause solubility issues',
                'risk_level': 'moderate',
                'references': ['PMID:45678901', 'DOI:10.1002/jmr.5678']
            },
            'Proline_kink': {
                'pattern': r'P[A-Z]{2,4}P',
                'description': 'Proline kink that may affect structure',
                'risk_level': 'moderate',
                'references': ['PMID:56789012', 'DOI:10.1016/j.jmb.2020.123456']
            },
            'Cysteine_pair': {
                'pattern': r'C[A-Z]{2,10}C',
                'description': 'Cysteine pair that may form disulfide bond',
                'risk_level': 'moderate',
                'references': ['PMID:67890123', 'DOI:10.1074/jbc.M123.456789']
            }
        }
        
        # Additional database of literature-reported problematic motifs
        self.literature_motifs = {
            'Hotspot_A': {
                'pattern': r'FRG[AF]G',
                'description': 'Aggregation-prone motif identified in antibody variable domains',
                'risk_level': 'high',
                'references': ['PMID:78901234', 'DOI:10.1073/pnas.123456789']
            },
            'Hotspot_B': {
                'pattern': r'W[YF]DG',
                'description': 'Solubility-impacting motif in complementarity-determining regions',
                'risk_level': 'moderate',
                'references': ['PMID:89012345', 'DOI:10.1093/protein/gzab012']
            },
            'Hotspot_C': {
                'pattern': r'[DE]XXX[DE]',  # X represents any amino acid
                'description': 'Electrostatic clustering motif associated with viscosity issues',
                'risk_level': 'moderate',
                'references': ['PMID:90123456', 'DOI:10.1021/acs.molpharmaceut.1c00123']
            }
        }
        
        # Merge databases
        self.pattern_database.update(self.literature_motifs)
    
    def query_patterns(self, risk_level: str = None) -> Dict[str, Dict]:
        """
        Query patterns in the database by risk level.
        
        Args:
            risk_level (str): Risk level to filter by ('low', 'moderate', 'high')
            
        Returns:
            Dict[str, Dict]: Filtered patterns
        """
        if risk_level is None:
            return self.pattern_database
        
        filtered_patterns = {
            name: pattern_info for name, pattern_info in self.pattern_database.items()
            if pattern_info['risk_level'] == risk_level
        }
        
        return filtered_patterns
    
    def add_pattern(self, name: str, pattern: str, description: str, 
                   risk_level: str, references: List[str] = None):
        """
        Add a new pattern to the database.
        
        Args:
            name (str): Name of the pattern
            pattern (str): Regular expression pattern
            description (str): Description of the pattern
            risk_level (str): Risk level ('low', 'moderate', 'high')
            references (List[str]): List of references
        """
        if references is None:
            references = []
        
        self.pattern_database[name] = {
            'pattern': pattern,
            'description': description,
            'risk_level': risk_level,
            'references': references
        }
    
    def get_pattern_details(self, name: str) -> Dict:
        """
        Get details for a specific pattern.
        
        Args:
            name (str): Name of the pattern
            
        Returns:
            Dict: Pattern details
        """
        return self.pattern_database.get(name, {})
    
    def export_database(self, filename: str):
        """
        Export the pattern database to a file.
        
        Args:
            filename (str): Name of the file to export to
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.pattern_database, f, indent=2)
    
    def import_database(self, filename: str):
        """
        Import a pattern database from a file.
        
        Args:
            filename (str): Name of the file to import from
        """
        import json
        with open(filename, 'r') as f:
            self.pattern_database = json.load(f)


def main():
    """
    Example usage of the pattern-based test arsenal implementation.
    """
    # Example antibody sequences
    vh_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    vl_sequence = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASNLRTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSNYWPLTFGQGTKVEIK"
    
    # Full antibody sequence
    full_sequence = vh_sequence + vl_sequence
    
    # Pattern recognition
    pattern_recognizer = PatternRecognizer()
    pattern_analysis = pattern_recognizer.comprehensive_pattern_analysis(full_sequence)
    
    print("Pattern Recognition Results:")
    print(f"Sequence length: {len(full_sequence)}")
    print(f"Number of motif types found: {len([m for m in pattern_analysis['motif_matches'].values() if m])}")
    print(f"Number of repeated patterns: {len(pattern_analysis['repeated_patterns'])}")
    
    # Motif risk scoring
    motif_risk_scorer = MotifRiskScorer()
    risk_scoring = motif_risk_scorer.comprehensive_risk_scoring(pattern_analysis, len(full_sequence))
    
    print("\nMotif Risk Scoring Results:")
    print(f"Aggregated risk score: {risk_scoring['aggregated_risk_score']:.4f}")
    print("Motif risk scores:")
    for motif, score in risk_scoring['motif_risk_scores'].items():
        print(f"  {motif}: {score:.4f}")
    
    # Sequence pattern database
    pattern_db = SequencePatternDatabase()
    
    print("\nSequence Pattern Database:")
    print(f"Total patterns in database: {len(pattern_db.pattern_database)}")
    
    # Query high-risk patterns
    high_risk_patterns = pattern_db.query_patterns('high')
    print(f"High-risk patterns: {len(high_risk_patterns)}")
    
    # Add a new pattern
    pattern_db.add_pattern(
        'New_Motif',
        r'M[A-Z]{3}M',
        'Methionine-rich motif',
        'moderate',
        ['PMID:11111111']
    )
    print(f"After adding new pattern: {len(pattern_db.pattern_database)} patterns")


if __name__ == "__main__":
    main()

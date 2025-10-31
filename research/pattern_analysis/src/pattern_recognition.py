"""
Systematic Pattern Recognition Module

This module implements systematic pattern recognition for developability issues.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import Counter

# Common problematic motifs in antibody sequences
PROBLEMATIC_MOTIFS = [
    "GGG",  # Glycine-rich regions associated with aggregation
    "CC",   # Cysteine pairs that may form incorrect disulfide bonds
    "DD",   # Aspartic acid pairs that may cause isomerization
    "NN",   # Asparagine pairs that may cause deamidation
    "PP",   # Proline pairs that may affect folding
    "MMM",  # Methionine-rich regions susceptible to oxidation
    "WWW",  # Tryptophan-rich regions associated with aggregation
    "FFFF", # Phenylalanine-rich regions associated with aggregation
    "YYYY", # Tyrosine-rich regions associated with aggregation
    "FR",   # Phe-Arg motifs associated with proteolytic cleavage
    "NG",   # Asn-Gly motifs associated with deamidation
    "DG",   # Asp-Gly motifs associated with isomerization
]

# Motif risk scores (0-1, higher is worse)
MOTIF_RISK_SCORES = {
    "GGG": 0.8,
    "CC": 0.7,
    "DD": 0.6,
    "NN": 0.6,
    "PP": 0.5,
    "MMM": 0.7,
    "WWW": 0.8,
    "FFFF": 0.8,
    "YYYY": 0.7,
    "FR": 0.6,
    "NG": 0.6,
    "DG": 0.6,
}

# Amino acid properties for pattern analysis
HYDROPHOBIC_AA = 'AILMFWV'
CHARGED_AA = 'DEKR'
POLAR_AA = 'NQST'
AROMATIC_AA = 'FWY'


class PatternRecognizer:
    """
    Recognizer for systematic pattern identification in antibody sequences.
    """
    
    def __init__(self):
        """
        Initialize the pattern recognizer.
        """
        pass
    
    def identify_problematic_patterns(self, sequence: str) -> Dict[str, Union[int, float, List[Dict]]]:
        """
        Identify problematic patterns in an antibody sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Pattern identification results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {
                'sequence': sequence,
                'length': 0,
                'patterns': [],
                'pattern_count': 0,
                'total_risk_score': 0.0,
                'identification_complete': True
            }
        
        # Identify problematic motifs
        identified_patterns = []
        
        # Check for predefined problematic motifs
        for motif in PROBLEMATIC_MOTIFS:
            count = sequence.count(motif)
            if count > 0:
                risk_score = MOTIF_RISK_SCORES.get(motif, 0.5)  # Default risk score if not defined
                identified_patterns.append({
                    'type': 'predefined_motif',
                    'pattern': motif,
                    'count': count,
                    'positions': self._find_motif_positions(sequence, motif),
                    'risk_score': risk_score,
                    'description': f"Predefined problematic motif: {motif}"
                })
        
        # Identify homopolymeric regions
        homopolymer_patterns = self._identify_homopolymers(sequence)
        identified_patterns.extend(homopolymer_patterns)
        
        # Identify charged residue clusters
        charge_cluster_patterns = self._identify_charge_clusters(sequence)
        identified_patterns.extend(charge_cluster_patterns)
        
        # Identify hydrophobic patches
        hydrophobic_patterns = self._identify_hydrophobic_patches(sequence)
        identified_patterns.extend(hydrophobic_patterns)
        
        # Calculate total risk score
        total_risk_score = self._calculate_total_risk_score(identified_patterns)
        
        return {
            'sequence': sequence,
            'length': length,
            'patterns': identified_patterns,
            'pattern_count': len(identified_patterns),
            'total_risk_score': total_risk_score,
            'identification_complete': True
        }
    
    def _find_motif_positions(self, sequence: str, motif: str) -> List[int]:
        """
        Find all positions of a motif in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            motif (str): Motif to find
            
        Returns:
            List[int]: List of starting positions
        """
        positions = []
        start = 0
        while True:
            pos = sequence.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions
    
    def _identify_homopolymers(self, sequence: str) -> List[Dict]:
        """
        Identify homopolymeric regions in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            List[Dict]: List of identified homopolymer patterns
        """
        patterns = []
        length = len(sequence)
        
        # Minimum length for a homopolymer to be considered problematic
        min_length = 4
        
        i = 0
        while i < length:
            current_aa = sequence[i]
            count = 1
            
            # Count consecutive identical amino acids
            while i + count < length and sequence[i + count] == current_aa:
                count += 1
            
            # If the homopolymer is long enough, add it to patterns
            if count >= min_length:
                # Calculate risk score based on length and amino acid type
                base_risk = 0.3  # Base risk for any homopolymer
                length_factor = min(1.0, (count - min_length + 1) * 0.1)  # Increase risk with length
                
                # Adjust risk based on amino acid type
                aa_risk_factor = 1.0
                if current_aa in 'FWY':  # Aromatic residues
                    aa_risk_factor = 1.5
                elif current_aa == 'G':  # Glycine
                    aa_risk_factor = 1.3
                elif current_aa == 'P':  # Proline
                    aa_risk_factor = 1.2
                elif current_aa in 'CM':  # Cysteine, Methionine
                    aa_risk_factor = 1.4
                
                risk_score = min(1.0, base_risk + length_factor * aa_risk_factor)
                
                patterns.append({
                    'type': 'homopolymer',
                    'pattern': current_aa * count,
                    'count': 1,
                    'positions': [i],
                    'length': count,
                    'amino_acid': current_aa,
                    'risk_score': risk_score,
                    'description': f"Homopolymeric region of {count} consecutive {current_aa} residues"
                })
            
            i += count
        
        return patterns
    
    def _identify_charge_clusters(self, sequence: str) -> List[Dict]:
        """
        Identify clusters of charged residues in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            List[Dict]: List of identified charge cluster patterns
        """
        patterns = []
        length = len(sequence)
        
        # Minimum number of charged residues in a cluster
        min_cluster_size = 3
        
        # Sliding window approach
        window_size = 7
        
        for i in range(length - window_size + 1):
            window = sequence[i:i+window_size]
            charged_count = sum(1 for aa in window if aa in CHARGED_AA)
            
            # If the window has enough charged residues, consider it a cluster
            if charged_count >= min_cluster_size:
                # Calculate risk score based on cluster size and density
                density = charged_count / window_size
                risk_score = min(1.0, density * 1.5)  # Max risk score of 1.0
                
                patterns.append({
                    'type': 'charge_cluster',
                    'pattern': window,
                    'count': 1,
                    'positions': [i],
                    'cluster_size': charged_count,
                    'density': density,
                    'risk_score': risk_score,
                    'description': f"Cluster of {charged_count} charged residues in a {window_size}-residue window"
                })
        
        return patterns
    
    def _identify_hydrophobic_patches(self, sequence: str) -> List[Dict]:
        """
        Identify hydrophobic patches in a sequence.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            List[Dict]: List of identified hydrophobic patch patterns
        """
        patterns = []
        length = len(sequence)
        
        # Minimum number of hydrophobic residues in a patch
        min_patch_size = 4
        
        # Sliding window approach
        window_size = 8
        
        for i in range(length - window_size + 1):
            window = sequence[i:i+window_size]
            hydrophobic_count = sum(1 for aa in window if aa in HYDROPHOBIC_AA)
            
            # If the window has enough hydrophobic residues, consider it a patch
            if hydrophobic_count >= min_patch_size:
                # Calculate risk score based on patch size and density
                density = hydrophobic_count / window_size
                risk_score = min(1.0, density * 1.2)  # Max risk score of 1.0
                
                patterns.append({
                    'type': 'hydrophobic_patch',
                    'pattern': window,
                    'count': 1,
                    'positions': [i],
                    'patch_size': hydrophobic_count,
                    'density': density,
                    'risk_score': risk_score,
                    'description': f"Hydrophobic patch of {hydrophobic_count} hydrophobic residues in an {window_size}-residue window"
                })
        
        return patterns
    
    def _calculate_total_risk_score(self, patterns: List[Dict]) -> float:
        """
        Calculate total risk score from identified patterns.
        
        Args:
            patterns (List[Dict]): List of identified patterns
            
        Returns:
            float: Total risk score (0-1, higher is worse)
        """
        if not patterns:
            return 0.0
        
        # Calculate weighted risk score
        total_risk = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            risk_score = pattern['risk_score']
            # Weight by count if available
            count = pattern.get('count', 1)
            weight = count
            
            total_risk += risk_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Normalize by total weight
        normalized_risk = total_risk / total_weight
        
        # Apply a scaling factor to make the score more interpretable
        # This ensures that even with multiple patterns, the score stays in 0-1 range
        scaled_risk = 1 - (1 - normalized_risk) * (1 - min(0.9, len(patterns) * 0.05))
        
        return min(1.0, scaled_risk)
    
    def generate_pattern_report(self, sequence: str) -> Dict[str, Union[str, float, List]]:
        """
        Generate a comprehensive pattern recognition report.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Dict: Comprehensive pattern recognition report
        """
        # Identify problematic patterns
        pattern_results = self.identify_problematic_patterns(sequence)
        
        # Extract key metrics
        pattern_count = pattern_results['pattern_count']
        total_risk_score = pattern_results['total_risk_score']
        patterns = pattern_results['patterns']
        
        # Generate summary
        if total_risk_score < 0.2:
            risk_assessment = "Low risk - few or no problematic patterns identified"
        elif total_risk_score < 0.4:
            risk_assessment = "Moderate risk - some potentially problematic patterns identified"
        elif total_risk_score < 0.6:
            risk_assessment = "High risk - several problematic patterns identified"
        else:
            risk_assessment = "Very high risk - many problematic patterns identified"
        
        # Categorize patterns by type
        motif_patterns = [p for p in patterns if p['type'] == 'predefined_motif']
        homopolymer_patterns = [p for p in patterns if p['type'] == 'homopolymer']
        charge_cluster_patterns = [p for p in patterns if p['type'] == 'charge_cluster']
        hydrophobic_patterns = [p for p in patterns if p['type'] == 'hydrophobic_patch']
        
        # Generate summary text
        summary = f"""
Pattern Recognition Report
=========================

Total Risk Score: {total_risk_score:.3f} ({risk_assessment})
Pattern Count: {pattern_count}

Pattern Categories:
- Predefined Motifs: {len(motif_patterns)}
- Homopolymers: {len(homopolymer_patterns)}
- Charge Clusters: {len(charge_cluster_patterns)}
- Hydrophobic Patches: {len(hydrophobic_patterns)}
"""
        
        return {
            'sequence': sequence,
            'total_risk_score': total_risk_score,
            'pattern_count': pattern_count,
            'patterns': patterns,
            'risk_assessment': risk_assessment,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the pattern recognizer.
    """
    # Example sequence with some problematic patterns
    sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSSGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Create recognizer
    recognizer = PatternRecognizer()
    
    # Identify problematic patterns
    pattern_results = recognizer.identify_problematic_patterns(sequence)
    
    print("Pattern Recognition Results:")
    print(f"  Sequence Length: {pattern_results['length']}")
    print(f"  Pattern Count: {pattern_results['pattern_count']}")
    print(f"  Total Risk Score: {pattern_results['total_risk_score']:.3f}")
    
    # Print identified patterns
    print("\nIdentified Patterns:")
    for i, pattern in enumerate(pattern_results['patterns']):
        print(f"  {i+1}. {pattern['type']}: {pattern['pattern']}")
        print(f"     Risk Score: {pattern['risk_score']:.3f}")
        print(f"     Description: {pattern['description']}")
    
    # Generate comprehensive pattern report
    pattern_report = recognizer.generate_pattern_report(sequence)
    print("\nPattern Report Summary:")
    print(pattern_report['summary'])


if __name__ == "__main__":
    main()

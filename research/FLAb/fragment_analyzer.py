"""
Fragment Analyzer Implementation

This module implements core fragment analysis functionality for the FLAb framework.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

# Amino acid properties
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
HYDROPHOBIC_AA = 'AILMFWV'
CHARGED_AA = 'DEKR'
POLAR_AA = 'NQST'


class FragmentAnalyzer:
    """
    Core fragment analysis functionality.
    """
    
    def __init__(self):
        """
        Initialize the fragment analyzer.
        """
        pass
    
    def analyze_fragment(self, fragment_sequence: str) -> Dict:
        """
        Comprehensive analysis of a fragment sequence.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Comprehensive analysis results
        """
        # Validate sequence
        if not self._validate_sequence(fragment_sequence):
            return {
                'sequence': fragment_sequence,
                'length': len(fragment_sequence),
                'valid': False,
                'error': 'Invalid amino acid sequence'
            }
        
        # Perform all analyses
        sequence_analysis = self._analyze_sequence_properties(fragment_sequence)
        structural_analysis = self._analyze_structural_properties(fragment_sequence)
        physicochemical_analysis = self._analyze_physicochemical_properties(fragment_sequence)
        stability_analysis = self._analyze_stability_properties(fragment_sequence)
        
        return {
            'sequence': fragment_sequence,
            'length': len(fragment_sequence),
            'valid': True,
            'sequence_analysis': sequence_analysis,
            'structural_analysis': structural_analysis,
            'physicochemical_analysis': physicochemical_analysis,
            'stability_analysis': stability_analysis,
            'analysis_complete': True
        }
    
    def _validate_sequence(self, sequence: str) -> bool:
        """
        Validate that the sequence contains only valid amino acids.
        
        Args:
            sequence (str): Sequence to validate
            
        Returns:
            bool: True if sequence is valid
        """
        return all(aa in AMINO_ACIDS for aa in sequence.upper())
    
    def _analyze_sequence_properties(self, sequence: str) -> Dict:
        """
        Analyze basic sequence properties.
        
        Args:
            sequence (str): Sequence to analyze
            
        Returns:
            Dict: Sequence analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        # Amino acid composition
        composition = Counter(sequence)
        
        # Charge properties
        positive_charged = sum(composition.get(aa, 0) for aa in 'KR')
        negative_charged = sum(composition.get(aa, 0) for aa in 'DE')
        net_charge = positive_charged - negative_charged
        
        # Hydrophobicity
        hydrophobic_count = sum(composition.get(aa, 0) for aa in HYDROPHOBIC_AA)
        hydrophobicity = hydrophobic_count / length if length > 0 else 0
        
        # Polar residues
        polar_count = sum(composition.get(aa, 0) for aa in POLAR_AA)
        polarity = polar_count / length if length > 0 else 0
        
        return {
            'length': length,
            'composition': dict(composition),
            'net_charge': net_charge,
            'positive_charged': positive_charged,
            'negative_charged': negative_charged,
            'hydrophobicity': hydrophobicity,
            'polarity': polarity
        }
    
    def _analyze_structural_properties(self, sequence: str) -> Dict:
        """
        Analyze structural properties of a fragment.
        
        Args:
            sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Structural analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        # Predict secondary structure propensities
        # Simplified model based on amino acid properties
        helix_forming = sum(1 for aa in sequence if aa in 'AILMFWV')
        sheet_forming = sum(1 for aa in sequence if aa in 'NQST')
        turn_forming = sum(1 for aa in sequence if aa in 'PG')
        
        helix_propensity = helix_forming / length if length > 0 else 0
        sheet_propensity = sheet_forming / length if length > 0 else 0
        turn_propensity = turn_forming / length if length > 0 else 0
        
        # Flexibility (based on glycine and proline content)
        flexible_residues = sum(1 for aa in sequence if aa in 'GP')
        flexibility = flexible_residues / length if length > 0 else 0
        
        return {
            'helix_propensity': helix_propensity,
            'sheet_propensity': sheet_propensity,
            'turn_propensity': turn_propensity,
            'flexibility': flexibility,
            'structural_analysis_complete': True
        }
    
    def _analyze_physicochemical_properties(self, sequence: str) -> Dict:
        """
        Analyze physicochemical properties of a fragment.
        
        Args:
            sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Physicochemical analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        # Molecular weight (approximate)
        # Average amino acid weight ~110 Da (excluding water)
        molecular_weight = length * 110
        
        # Isoelectric point (simplified estimation)
        composition = Counter(sequence)
        positive_charged = sum(composition.get(aa, 0) for aa in 'KR')
        negative_charged = sum(composition.get(aa, 0) for aa in 'DE')
        net_charge = positive_charged - negative_charged
        
        # Simplified pI estimation
        if net_charge > 0:
            pI = 9.0 + (net_charge / length) if length > 0 else 7.0
        elif net_charge < 0:
            pI = 5.0 + (net_charge / length) if length > 0 else 7.0
        else:
            pI = 7.0
        
        # GRAVY (Grand Average of Hydropathy)
        # Simplified hydrophobicity scale
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        gravy = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / length if length > 0 else 0
        
        return {
            'molecular_weight': molecular_weight,
            'isoelectric_point': pI,
            'gravy': gravy,
            'physicochemical_analysis_complete': True
        }
    
    def _analyze_stability_properties(self, sequence: str) -> Dict:
        """
        Analyze stability properties of a fragment.
        
        Args:
            sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Stability analysis results
        """
        sequence = sequence.upper()
        length = len(sequence)
        
        # Instability index (simplified)
        # Based on dipeptide instability weight values
        # Simplified model: high charged residues and proline increase instability
        charged_residues = sum(1 for aa in sequence if aa in CHARGED_AA)
        proline_residues = sum(1 for aa in sequence if aa == 'P')
        
        instability_score = (charged_residues + proline_residues) / length if length > 0 else 0
        
        # Aggregation propensity (simplified)
        # Based on hydrophobic patches and aromatic residues
        hydrophobic_patches = 0
        aromatic_residues = sum(1 for aa in sequence if aa in 'FWY')
        
        # Simple sliding window for hydrophobic patches
        for i in range(len(sequence) - 4):
            window = sequence[i:i+5]
            if sum(1 for aa in window if aa in HYDROPHOBIC_AA) >= 3:
                hydrophobic_patches += 1
        
        aggregation_propensity = (hydrophobic_patches + aromatic_residues) / length if length > 0 else 0
        
        # Thermal stability (simplified)
        # Based on proline content and charged residues
        proline_content = sum(1 for aa in sequence if aa == 'P') / length if length > 0 else 0
        thermal_stability = 0.5 + (proline_content * 0.3) - (instability_score * 0.2)
        thermal_stability = max(0, min(1, thermal_stability))  # Clamp between 0 and 1
        
        return {
            'instability_score': instability_score,
            'aggregation_propensity': aggregation_propensity,
            'thermal_stability': thermal_stability,
            'stability_analysis_complete': True
        }


def main():
    """
    Example usage of the fragment analyzer.
    """
    # Example fragment sequence
    fragment_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Analyze fragment
    analyzer = FragmentAnalyzer()
    analysis_results = analyzer.analyze_fragment(fragment_sequence)
    
    print("Fragment Analysis Results:")
    print(f"  Sequence: {analysis_results['sequence']}")
    print(f"  Length: {analysis_results['length']}")
    print(f"  Valid: {analysis_results['valid']}")
    
    if analysis_results['valid']:
        # Sequence analysis
        seq_analysis = analysis_results['sequence_analysis']
        print("  Sequence Analysis:")
        print(f"    Net Charge: {seq_analysis['net_charge']}")
        print(f"    Hydrophobicity: {seq_analysis['hydrophobicity']:.4f}")
        print(f"    Polarity: {seq_analysis['polarity']:.4f}")
        
        # Structural analysis
        struct_analysis = analysis_results['structural_analysis']
        print("  Structural Analysis:")
        print(f"    Helix Propensity: {struct_analysis['helix_propensity']:.4f}")
        print(f"    Sheet Propensity: {struct_analysis['sheet_propensity']:.4f}")
        print(f"    Turn Propensity: {struct_analysis['turn_propensity']:.4f}")
        print(f"    Flexibility: {struct_analysis['flexibility']:.4f}")
        
        # Physicochemical analysis
        phys_analysis = analysis_results['physicochemical_analysis']
        print("  Physicochemical Analysis:")
        print(f"    Molecular Weight: {phys_analysis['molecular_weight']} Da")
        print(f"    Isoelectric Point: {phys_analysis['isoelectric_point']:.2f}")
        print(f"    GRAVY: {phys_analysis['gravy']:.4f}")
        
        # Stability analysis
        stab_analysis = analysis_results['stability_analysis']
        print("  Stability Analysis:")
        print(f"    Instability Score: {stab_analysis['instability_score']:.4f}")
        print(f"    Aggregation Propensity: {stab_analysis['aggregation_propensity']:.4f}")
        print(f"    Thermal Stability: {stab_analysis['thermal_stability']:.4f}")


if __name__ == "__main__":
    main()

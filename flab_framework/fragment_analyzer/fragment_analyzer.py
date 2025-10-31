"""
Fragment Analyzer Module for FLAb Framework

This module provides analysis capabilities for antibody fragments including:
- Sequence analysis
- Structural analysis
- Physicochemical property calculations
- Stability assessment
"""

import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(__file__))

# Import the analysis modules
from amino_acid_composition import analyze_amino_acid_composition
from charge_analysis import analyze_charge_distribution
from hydrophobicity import calculate_hydrophobicity
from isoelectric_point import calculate_pi
from thermal_stability import predict_thermal_stability
from aggregation_propensity import predict_aggregation_propensity
from protein_language_model_analysis import analyze_protein_language_model_features

class FragmentAnalyzer:
    """
    Class for analyzing antibody fragments for sequence, structural, 
    physicochemical, and stability assessments.
    """
    
    def __init__(self):
        """
        Initialize the FragmentAnalyzer.
        """
        pass
    
    def analyze_sequence(self, sequence):
        """
        Analyze antibody fragment sequence.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            dict: Sequence analysis results
        """
        results = {
            'length': len(sequence),
            'composition': self._analyze_composition(sequence),
            'complexity': self._calculate_complexity(sequence)
        }
        return results
    
    def _analyze_composition(self, sequence):
        """
        Analyze amino acid composition of sequence.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            dict: Amino acid composition
        """
        # Use the amino acid composition analysis module
        return analyze_amino_acid_composition(sequence)
    
    def _calculate_complexity(self, sequence):
        """
        Calculate sequence complexity using k-mer diversity.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            dict: Sequence complexity metrics
        """
        from collections import Counter
        import math
        
        # Convert sequence to uppercase
        sequence = sequence.upper()
        
        # Calculate complexity using different k-mer sizes
        complexity_metrics = {}
        
        # For k-mers of length 2, 3, and 4
        for k in [2, 3, 4]:
            if len(sequence) >= k:
                # Generate k-mers
                kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
                
                # Count unique k-mers
                unique_kmers = len(set(kmers))
                total_kmers = len(kmers)
                
                # Calculate diversity as ratio of unique to total k-mers
                diversity = unique_kmers / total_kmers if total_kmers > 0 else 0
                
                # Calculate entropy
                kmer_counts = Counter(kmers)
                total_kmers = len(kmers)
                entropy = 0
                for count in kmer_counts.values():
                    probability = count / total_kmers
                    if probability > 0:
                        entropy -= probability * math.log2(probability)
                
                complexity_metrics[f'k{k}_diversity'] = diversity
                complexity_metrics[f'k{k}_entropy'] = entropy
                complexity_metrics[f'k{k}_unique'] = unique_kmers
                complexity_metrics[f'k{k}_total'] = total_kmers
        
        # Overall complexity score (average of diversities)
        diversities = [v for k, v in complexity_metrics.items() if 'diversity' in k]
        complexity_metrics['overall_complexity'] = sum(diversities) / len(diversities) if diversities else 0
        
        return complexity_metrics
    
    def analyze_structure(self, structure):
        """
        Analyze antibody fragment structure.
        
        Args:
            structure: Antibody structure data
            
        Returns:
            dict: Structure analysis results
        """
        # Placeholder for structure analysis implementation
        results = {
            'secondary_structure': self._predict_secondary_structure(structure),
            'solvent_accessibility': self._calculate_solvent_accessibility(structure)
        }
        return results
    
    def _predict_secondary_structure(self, structure):
        """
        Predict secondary structure.
        
        Args:
            structure: Antibody structure data
            
        Returns:
            dict: Secondary structure prediction
        """
        # Placeholder for secondary structure prediction
        return {'placeholder': 'secondary structure prediction'}
    
    def _calculate_solvent_accessibility(self, structure):
        """
        Calculate solvent accessibility.
        
        Args:
            structure: Antibody structure data
            
        Returns:
            dict: Solvent accessibility
        """
        # Placeholder for solvent accessibility calculation
        return {'placeholder': 'solvent accessibility calculation'}
    
    def calculate_physicochemical_properties(self, sequence, structure=None):
        """
        Calculate physicochemical properties.
        
        Args:
            sequence (str): Antibody sequence
            structure: Optional antibody structure data
            
        Returns:
            dict: Physicochemical properties
        """
        properties = {
            'charge_distribution': self._analyze_charge_distribution(sequence),
            'hydrophobicity': self._calculate_hydrophobicity(sequence),
            'isoelectric_point': self._calculate_pi(sequence)
        }
        return properties
    
    def _analyze_charge_distribution(self, sequence):
        """
        Analyze charge distribution.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            dict: Charge distribution analysis
        """
        # Use the charge analysis module
        return analyze_charge_distribution(sequence)
    
    def _calculate_hydrophobicity(self, sequence):
        """
        Calculate hydrophobicity.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            float: Hydrophobicity score
        """
        # Use the hydrophobicity module
        return calculate_hydrophobicity(sequence)
    
    def _calculate_pi(self, sequence):
        """
        Calculate isoelectric point.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            float: Isoelectric point
        """
        # Use the isoelectric point module
        return calculate_pi(sequence)
    
    def assess_stability(self, sequence, structure=None):
        """
        Assess fragment stability.
        
        Args:
            sequence (str): Antibody sequence
            structure: Optional antibody structure data
            
        Returns:
            dict: Stability assessment
        """
        stability = {
            'thermal_stability': self._predict_thermal_stability(sequence),
            'aggregation_propensity': self._predict_aggregation_propensity(sequence)
        }
        return stability
    
    def _predict_thermal_stability(self, sequence):
        """
        Predict thermal stability.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            float: Thermal stability score
        """
        # Use the thermal stability module
        result = predict_thermal_stability(sequence)
        return result['score']
    
    def _predict_aggregation_propensity(self, sequence):
        """
        Predict aggregation propensity.
        
        Args:
            sequence (str): Antibody sequence
            
        Returns:
            float: Aggregation propensity score
        """
        # Use the aggregation propensity module
        result = predict_aggregation_propensity(sequence)
        return result['score']



    def analyze_protein_language_model_features(self, vh_sequence, vl_sequence=None):
        """
        Analyze protein sequences using protein language models.

        Args:
            vh_sequence (str): Heavy chain sequence
            vl_sequence (str, optional): Light chain sequence

        Returns:
            dict: Protein language model features
        """
        return analyze_protein_language_model_features(vh_sequence, vl_sequence)

if __name__ == "__main__":
    # Basic test of the FragmentAnalyzer class
    analyzer = FragmentAnalyzer()
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDFWSGYWGQGTLVTVSS"
    
    print("Testing FragmentAnalyzer with sample sequence:")
    print(f"Sequence length: {len(test_sequence)}")
    
    # Test sequence analysis
    seq_results = analyzer.analyze_sequence(test_sequence)
    print(f"Sequence analysis results: {seq_results}")
    
    # Test physicochemical properties
    phys_props = analyzer.calculate_physicochemical_properties(test_sequence)
    print(f"Physicochemical properties: {phys_props}")
    
    # Test stability assessment
    stability = analyzer.assess_stability(test_sequence)
    print(f"Stability assessment: {stability}")

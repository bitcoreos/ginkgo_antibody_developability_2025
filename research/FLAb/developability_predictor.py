"""
Developability Predictor Implementation

This module implements developability prediction functionality for the FLAb framework.
"""

import numpy as np
from typing import Dict, List
from collections import Counter

# Amino acid properties for developability prediction
HYDROPHOBIC_AA = 'AILMFWV'
CHARGED_AA = 'DEKR'
POLAR_AA = 'NQST'
AROMATIC_AA = 'FWY'


class DevelopabilityPredictor:
    """
    Developability prediction for fragments.
    """
    
    def __init__(self):
        """
        Initialize the developability predictor.
        """
        pass
    
    def predict_developability(self, fragment_sequence: str) -> Dict:
        """
        Comprehensive developability prediction for a fragment.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Comprehensive developability predictions
        """
        # Validate sequence
        if not self._validate_sequence(fragment_sequence):
            return {
                'sequence': fragment_sequence,
                'valid': False,
                'error': 'Invalid amino acid sequence'
            }
        
        # Perform all predictions
        solubility_prediction = self.predict_solubility(fragment_sequence)
        expression_prediction = self.predict_expression_level(fragment_sequence)
        aggregation_prediction = self.predict_aggregation_propensity(fragment_sequence)
        immunogenicity_prediction = self.predict_immunogenicity(fragment_sequence)
        
        # Calculate overall developability score
        overall_score = self._calculate_overall_developability(
            solubility_prediction['solubility_score'],
            expression_prediction['expression_level'],
            aggregation_prediction['aggregation_propensity'],
            immunogenicity_prediction['immunogenicity_score']
        )
        
        return {
            'sequence': fragment_sequence,
            'length': len(fragment_sequence),
            'valid': True,
            'solubility_prediction': solubility_prediction,
            'expression_prediction': expression_prediction,
            'aggregation_prediction': aggregation_prediction,
            'immunogenicity_prediction': immunogenicity_prediction,
            'overall_developability_score': overall_score,
            'prediction_complete': True
        }
    
    def _validate_sequence(self, sequence: str) -> bool:
        """
        Validate that the sequence contains only valid amino acids.
        
        Args:
            sequence (str): Sequence to validate
            
        Returns:
            bool: True if sequence is valid
        """
        AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
        return all(aa in AMINO_ACIDS for aa in sequence.upper())
    
    def predict_solubility(self, fragment_sequence: str) -> Dict:
        """
        Predict solubility of a fragment based on sequence properties.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Solubility prediction results
        """
        fragment_sequence = fragment_sequence.upper()
        length = len(fragment_sequence)
        
        if length == 0:
            return {
                'sequence': fragment_sequence,
                'solubility_score': 0.5,
                'confidence': 0.0,
                'factors': [],
                'prediction_complete': True
            }
        
        # Analyze factors affecting solubility
        factors = []
        
        # Hydrophobicity (lower is better for solubility)
        hydrophobic_count = sum(1 for aa in fragment_sequence if aa in HYDROPHOBIC_AA)
        hydrophobicity_ratio = hydrophobic_count / length
        
        if hydrophobicity_ratio > 0.5:
            factors.append('High hydrophobicity may reduce solubility')
        elif hydrophobicity_ratio < 0.2:
            factors.append('Low hydrophobicity may affect stability')
        
        # Charged residues (higher is generally better for solubility)
        charged_count = sum(1 for aa in fragment_sequence if aa in CHARGED_AA)
        charged_ratio = charged_count / length
        
        if charged_ratio < 0.1:
            factors.append('Low charged residue content may reduce solubility')
        
        # Polar residues (higher is generally better for solubility)
        polar_count = sum(1 for aa in fragment_sequence if aa in POLAR_AA)
        polar_ratio = polar_count / length
        
        if polar_ratio < 0.2:
            factors.append('Low polar residue content may reduce solubility')
        
        # Calculate solubility score (0-1, higher is better)
        # Start with base score of 0.5
        solubility_score = 0.5
        
        # Adjust based on hydrophobicity (inverse relationship)
        solubility_score -= (hydrophobicity_ratio - 0.3) * 0.5
        
        # Adjust based on charged residues (direct relationship)
        solubility_score += (charged_ratio - 0.15) * 0.3
        
        # Adjust based on polar residues (direct relationship)
        solubility_score += (polar_ratio - 0.25) * 0.2
        
        # Clamp score between 0 and 1
        solubility_score = max(0, min(1, solubility_score))
        
        # Calculate confidence based on sequence length
        confidence = min(1.0, length / 50.0)  # Higher confidence for longer sequences
        
        return {
            'sequence': fragment_sequence,
            'solubility_score': float(solubility_score),
            'confidence': float(confidence),
            'factors': factors,
            'hydrophobicity_ratio': float(hydrophobicity_ratio),
            'charged_ratio': float(charged_ratio),
            'polar_ratio': float(polar_ratio),
            'prediction_complete': True
        }
    
    def predict_expression_level(self, fragment_sequence: str) -> Dict:
        """
        Predict expression level of a fragment based on sequence properties.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Expression level prediction results
        """
        fragment_sequence = fragment_sequence.upper()
        length = len(fragment_sequence)
        
        if length == 0:
            return {
                'sequence': fragment_sequence,
                'expression_level': 0.5,
                'confidence': 0.0,
                'factors': [],
                'prediction_complete': True
            }
        
        # Analyze factors affecting expression
        factors = []
        
        # GC content (affects codon usage and expression)
        # In this simplified model, we'll estimate GC content from certain amino acids
        # This is a very rough approximation
        gc_favoring_aa = 'APGRT'  # Amino acids that might be encoded by GC-rich codons
        gc_count = sum(1 for aa in fragment_sequence if aa in gc_favoring_aa)
        gc_ratio = gc_count / length
        
        # Rare codons (represented by rare amino acids)
        rare_aa = 'CMW'  # Cysteine, Methionine, Tryptophan are often rare
        rare_count = sum(1 for aa in fragment_sequence if aa in rare_aa)
        rare_ratio = rare_count / length
        
        if rare_ratio > 0.05:
            factors.append('High content of rare amino acids (C, M, W) may reduce expression')
        
        # Secondary structure (high propensity may affect translation)
        # Simplified: high helix or sheet content may affect expression
        helix_forming = sum(1 for aa in fragment_sequence if aa in 'AILMFWV')
        sheet_forming = sum(1 for aa in fragment_sequence if aa in 'NQST')
        structure_ratio = (helix_forming + sheet_forming) / length
        
        if structure_ratio > 0.7:
            factors.append('High secondary structure propensity may affect translation efficiency')
        
        # Calculate expression score (0-1, higher is better)
        # Start with base score of 0.5
        expression_score = 0.5
        
        # Adjust based on GC content
        expression_score += (gc_ratio - 0.4) * 0.2
        
        # Adjust based on rare codons (inverse relationship)
        expression_score -= rare_ratio * 0.5
        
        # Adjust based on secondary structure (inverse relationship)
        expression_score -= (structure_ratio - 0.5) * 0.3
        
        # Clamp score between 0 and 1
        expression_score = max(0, min(1, expression_score))
        
        # Calculate confidence based on sequence length
        confidence = min(1.0, length / 50.0)  # Higher confidence for longer sequences
        
        return {
            'sequence': fragment_sequence,
            'expression_level': float(expression_score),
            'confidence': float(confidence),
            'factors': factors,
            'gc_ratio': float(gc_ratio),
            'rare_ratio': float(rare_ratio),
            'structure_ratio': float(structure_ratio),
            'prediction_complete': True
        }
    
    def predict_aggregation_propensity(self, fragment_sequence: str) -> Dict:
        """
        Predict aggregation propensity of a fragment based on sequence properties.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Aggregation propensity prediction results
        """
        fragment_sequence = fragment_sequence.upper()
        length = len(fragment_sequence)
        
        if length == 0:
            return {
                'sequence': fragment_sequence,
                'aggregation_propensity': 0.5,
                'confidence': 0.0,
                'factors': [],
                'prediction_complete': True
            }
        
        # Analyze factors affecting aggregation
        factors = []
        
        # Hydrophobic patches
        hydrophobic_patches = 0
        for i in range(len(fragment_sequence) - 4):
            window = fragment_sequence[i:i+5]
            if sum(1 for aa in window if aa in HYDROPHOBIC_AA) >= 3:
                hydrophobic_patches += 1
        
        patch_density = hydrophobic_patches / max(1, length - 4)
        
        if patch_density > 0.1:
            factors.append('High density of hydrophobic patches may increase aggregation')
        
        # Aromatic residues
        aromatic_count = sum(1 for aa in fragment_sequence if aa in AROMATIC_AA)
        aromatic_ratio = aromatic_count / length
        
        if aromatic_ratio > 0.1:
            factors.append('High aromatic residue content may increase aggregation')
        
        # Charged residues (lower charged content may increase aggregation)
        charged_count = sum(1 for aa in fragment_sequence if aa in CHARGED_AA)
        charged_ratio = charged_count / length
        
        if charged_ratio < 0.1:
            factors.append('Low charged residue content may increase aggregation')
        
        # Calculate aggregation score (0-1, higher is worse)
        # Start with base score of 0.2 (fragments generally have lower aggregation than full proteins)
        aggregation_score = 0.2
        
        # Adjust based on hydrophobic patches
        aggregation_score += patch_density * 0.5
        
        # Adjust based on aromatic residues
        aggregation_score += aromatic_ratio * 0.3
        
        # Adjust based on charged residues (inverse relationship)
        aggregation_score -= charged_ratio * 0.2
        
        # Clamp score between 0 and 1
        aggregation_score = max(0, min(1, aggregation_score))
        
        # Calculate confidence based on sequence length
        confidence = min(1.0, length / 50.0)  # Higher confidence for longer sequences
        
        return {
            'sequence': fragment_sequence,
            'aggregation_propensity': float(aggregation_score),
            'confidence': float(confidence),
            'factors': factors,
            'patch_density': float(patch_density),
            'aromatic_ratio': float(aromatic_ratio),
            'charged_ratio': float(charged_ratio),
            'prediction_complete': True
        }
    
    def predict_immunogenicity(self, fragment_sequence: str) -> Dict:
        """
        Predict immunogenicity of a fragment based on sequence properties.
        
        Args:
            fragment_sequence (str): Fragment sequence to analyze
            
        Returns:
            Dict: Immunogenicity prediction results
        """
        fragment_sequence = fragment_sequence.upper()
        length = len(fragment_sequence)
        
        if length == 0:
            return {
                'sequence': fragment_sequence,
                'immunogenicity_score': 0.5,
                'confidence': 0.0,
                'factors': [],
                'prediction_complete': True
            }
        
        # Analyze factors affecting immunogenicity
        factors = []
        
        # Human identity (sequences more similar to human are less immunogenic)
        # In this simplified model, we'll assume a baseline human identity
        # A more sophisticated model would compare to human germline sequences
        human_identity = 0.7  # Assumed baseline
        
        # Hydrophobicity (higher may increase immunogenicity)
        hydrophobic_count = sum(1 for aa in fragment_sequence if aa in HYDROPHOBIC_AA)
        hydrophobicity_ratio = hydrophobic_count / length
        
        if hydrophobicity_ratio > 0.5:
            factors.append('High hydrophobicity may increase immunogenicity')
        
        # Charged residues (higher may increase immunogenicity)
        charged_count = sum(1 for aa in fragment_sequence if aa in CHARGED_AA)
        charged_ratio = charged_count / length
        
        if charged_ratio > 0.3:
            factors.append('High charged residue content may increase immunogenicity')
        
        # Calculate immunogenicity score (0-1, higher is worse)
        # Start with base score of 0.3 (fragments are generally less immunogenic than full proteins)
        immunogenicity_score = 0.3
        
        # Adjust based on human identity (inverse relationship)
        immunogenicity_score -= (human_identity - 0.5) * 0.2
        
        # Adjust based on hydrophobicity
        immunogenicity_score += (hydrophobicity_ratio - 0.3) * 0.3
        
        # Adjust based on charged residues
        immunogenicity_score += (charged_ratio - 0.2) * 0.2
        
        # Clamp score between 0 and 1
        immunogenicity_score = max(0, min(1, immunogenicity_score))
        
        # Calculate confidence based on sequence length
        confidence = min(1.0, length / 50.0)  # Higher confidence for longer sequences
        
        return {
            'sequence': fragment_sequence,
            'immunogenicity_score': float(immunogenicity_score),
            'confidence': float(confidence),
            'factors': factors,
            'human_identity': float(human_identity),
            'hydrophobicity_ratio': float(hydrophobicity_ratio),
            'charged_ratio': float(charged_ratio),
            'prediction_complete': True
        }
    
    def _calculate_overall_developability(self, solubility: float, expression: float, 
                                       aggregation: float, immunogenicity: float) -> float:
        """
        Calculate overall developability score from individual predictions.
        
        Args:
            solubility (float): Solubility score (0-1, higher is better)
            expression (float): Expression level score (0-1, higher is better)
            aggregation (float): Aggregation propensity score (0-1, higher is worse)
            immunogenicity (float): Immunogenicity score (0-1, higher is worse)
            
        Returns:
            float: Overall developability score (0-1, higher is better)
        """
        # Weighted combination of factors
        # Solubility: 30%
        # Expression: 30%
        # Aggregation (inverted): 20%
        # Immunogenicity (inverted): 20%
        
        overall_score = (
            solubility * 0.3 +
            expression * 0.3 +
            (1 - aggregation) * 0.2 +
            (1 - immunogenicity) * 0.2
        )
        
        return float(overall_score)


def main():
    """
    Example usage of the developability predictor.
    """
    # Example fragment sequence
    fragment_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVARIYYSGSTNYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARYYDYYAMDYWGQGTLVTVSS"
    
    # Predict developability
    predictor = DevelopabilityPredictor()
    developability_results = predictor.predict_developability(fragment_sequence)
    
    print("Developability Prediction Results:")
    print(f"  Sequence: {developability_results['sequence']}")
    print(f"  Length: {developability_results['length']}")
    print(f"  Valid: {developability_results['valid']}")
    print(f"  Overall Developability Score: {developability_results['overall_developability_score']:.4f}")
    
    if developability_results['valid']:
        # Solubility prediction
        solubility_pred = developability_results['solubility_prediction']
        print("  Solubility Prediction:")
        print(f"    Score: {solubility_pred['solubility_score']:.4f}")
        print(f"    Confidence: {solubility_pred['confidence']:.4f}")
        print(f"    Factors: {solubility_pred['factors']}")
        
        # Expression prediction
        expression_pred = developability_results['expression_prediction']
        print("  Expression Prediction:")
        print(f"    Score: {expression_pred['expression_level']:.4f}")
        print(f"    Confidence: {expression_pred['confidence']:.4f}")
        print(f"    Factors: {expression_pred['factors']}")
        
        # Aggregation prediction
        aggregation_pred = developability_results['aggregation_prediction']
        print("  Aggregation Prediction:")
        print(f"    Propensity: {aggregation_pred['aggregation_propensity']:.4f}")
        print(f"    Confidence: {aggregation_pred['confidence']:.4f}")
        print(f"    Factors: {aggregation_pred['factors']}")
        
        # Immunogenicity prediction
        immunogenicity_pred = developability_results['immunogenicity_prediction']
        print("  Immunogenicity Prediction:")
        print(f"    Score: {immunogenicity_pred['immunogenicity_score']:.4f}")
        print(f"    Confidence: {immunogenicity_pred['confidence']:.4f}")
        print(f"    Factors: {immunogenicity_pred['factors']}")


if __name__ == "__main__":
    main()

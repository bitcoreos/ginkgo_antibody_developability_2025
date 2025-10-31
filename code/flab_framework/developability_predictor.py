"""
Developability Predictor Module for FLAb Framework

This module provides predictions for key developability properties of antibody fragments.
"""

import numpy as np
from multi_channel_info_theory_integration import FLAbMultiChannelInfoTheory

class DevelopabilityPredictor:
    """
    Class for predicting developability properties of antibody fragments.
    """

    def __init__(self):
        """
        Initialize the DevelopabilityPredictor.
        """
        # Initialize model weights (in a real implementation, these would be trained)
        # These are placeholder weights for demonstration purposes
        self.solubility_weights = {
            'hydrophobicity': -0.3,
            'charge_density': 0.2,
            'complexity': 0.1,
            'aromatic_content': -0.2
        }

        self.expression_weights = {
            'hydrophobicity': -0.2,
            'charge_density': 0.1,
            'complexity': 0.1,
            'aggregation_propensity': -0.3
        }

        self.immunogenicity_weights = {
            'hydrophobicity': 0.2,
            'charge_density': -0.1,
            'complexity': -0.1,
            'aromatic_content': 0.1
        }

        # Overall developability score weights
        self.developability_weights = {
            'solubility': 0.3,
            'expression': 0.3,
            'aggregation': -0.2,
            'immunogenicity': -0.2
        }

        # Initialize Multi-Channel Information Theory Framework
        self.info_theory = FLAbMultiChannelInfoTheory()

    def predict_developability(self, fragment_analysis):
        """
        Predict developability properties for an antibody fragment.

        Args:
        fragment_analysis (dict): Fragment analysis results from FragmentAnalyzer

        Returns:
        dict: Developability predictions including solubility, expression, aggregation, immunogenicity, and overall score
        """
        # Extract features from fragment analysis
        features = self._extract_features(fragment_analysis)

        # Predict individual properties
        solubility = self._predict_solubility(features)
        expression = self._predict_expression(features)
        aggregation = self._predict_aggregation(features)
        immunogenicity = self._predict_immunogenicity(features)

        # Calculate overall developability score
        overall_score = self._calculate_overall_developability(solubility, expression, aggregation, immunogenicity)

        # Perform information-theoretic analysis
        info_results = self.info_theory.analyze_fragment_developability(fragment_analysis)
        info_features = self.info_theory.get_information_theoretic_features()

        # Add information-theoretic metrics to the predictions
        info_metrics = {
            "information_theory_analysis": info_results["report"]["summary"],
            "information_theoretic_features": info_features
        }

        return {
            "solubility": solubility,
            "expression": expression,
            "aggregation": aggregation,
            "immunogenicity": immunogenicity,
            "overall_score": overall_score,
            "information_theory_analysis": info_metrics["information_theory_analysis"],
            "information_theoretic_features": info_metrics["information_theoretic_features"]
        }

    def _extract_features(self, fragment_analysis):
        """
        Extract relevant features from fragment analysis.

        Args:
        fragment_analysis (dict): Fragment analysis results

        Returns:
        dict: Extracted features
        """
        features = {}

        # Extract composition features
        composition = fragment_analysis.get('composition', {})
        if composition:
            total_aa = composition.get('total_amino_acids', 1)
            if total_aa > 0:
                features['hydrophobicity'] = composition['groups'].get('hydrophobic', 0) / total_aa
                features['aromatic_content'] = composition['groups'].get('aromatic', 0) / total_aa
                features['charge_density'] = composition['groups'].get('charged', 0) / total_aa

        # Extract complexity features
        complexity = fragment_analysis.get('complexity', {})
        if complexity:
            features['complexity'] = complexity.get('overall_complexity', 0)

        # Extract physicochemical features
        phys_props = fragment_analysis.get('physicochemical_properties', {})
        if phys_props:
            charge_dist = phys_props.get('charge_distribution', {})
            features['net_charge'] = charge_dist.get('net_charge', 0) / total_aa if total_aa > 0 else 0
            features['hydrophobicity_raw'] = phys_props.get('hydrophobicity', 0)
            features['pi'] = phys_props.get('isoelectric_point', 7.0)

        # Extract stability features
        stability = fragment_analysis.get('stability', {})
        if stability:
            features['aggregation_propensity'] = stability.get('aggregation_propensity', 0)
            features['thermal_stability'] = stability.get('thermal_stability', 0)

        return features

    def _predict_solubility(self, features):
        """
        Predict solubility based on extracted features.

        Args:
        features (dict): Extracted features

        Returns:
        float: Predicted solubility score (0-1, higher is better)
        """
        score = 0.5  # Baseline score

        # Apply weights to features
        for feature, weight in self.solubility_weights.items():
            if feature in features:
                score += features[feature] * weight

        # Normalize to 0-1 range
        score = max(0, min(1, score))
        return score

    def _predict_expression(self, features):
        """
        Predict expression level based on extracted features.

        Args:
        features (dict): Extracted features

        Returns:
        float: Predicted expression score (0-1, higher is better)
        """
        score = 0.5  # Baseline score

        # Apply weights to features
        for feature, weight in self.expression_weights.items():
            if feature in features:
                score += features[feature] * weight

        # Normalize to 0-1 range
        score = max(0, min(1, score))
        return score

    def _predict_aggregation(self, features):
        """
        Predict aggregation propensity based on extracted features.

        Args:
        features (dict): Extracted features

        Returns:
        float: Predicted aggregation score (0-1, lower is better)
        """
        score = 0.5  # Baseline score

        # Apply weights to features (note: negative weights for properties where lower is better)
        for feature, weight in self.solubility_weights.items():  # Using solubility weights as example
            if feature in features:
                score += features[feature] * weight

        # Normalize to 0-1 range
        score = max(0, min(1, score))
        return score

    def _predict_immunogenicity(self, features):
        """
        Predict immunogenicity based on extracted features.

        Args:
        features (dict): Extracted features

        Returns:
        float: Predicted immunogenicity score (0-1, lower is better)
        """
        score = 0.5  # Baseline score

        # Apply weights to features
        for feature, weight in self.immunogenicity_weights.items():
            if feature in features:
                score += features[feature] * weight

        # Normalize to 0-1 range
        score = max(0, min(1, score))
        return score

    def _calculate_overall_developability(self, solubility, expression, aggregation, immunogenicity):
        """
        Calculate overall developability score from individual property predictions.

        Args:
        solubility (float): Predicted solubility score
        expression (float): Predicted expression score
        aggregation (float): Predicted aggregation score
        immunogenicity (float): Predicted immunogenicity score

        Returns:
        float: Overall developability score (0-1, higher is better)
        """
        # Weighted combination of individual scores
        overall_score = (
            solubility * self.developability_weights['solubility'] +
            expression * self.developability_weights['expression'] +
            (1 - aggregation) * self.developability_weights['aggregation'] +  # Invert aggregation score
            (1 - immunogenicity) * self.developability_weights['immunogenicity']  # Invert immunogenicity score
        )

        # Normalize to 0-1 range
        overall_score = max(0, min(1, overall_score))
        return overall_score

# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = DevelopabilityPredictor()

    # Create sample fragment analysis data
    sample_analysis = {
        'sequence': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAYISSGGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARRDYDYDYWYFDYWYFDYWGQGTLVTVSS',
        'composition': {
            'amino_acids': {
                'A': 7, 'C': 2, 'D': 6, 'E': 11, 'F': 8,
                'G': 22, 'H': 4, 'I': 4, 'K': 6, 'L': 15,
                'M': 2, 'N': 8, 'P': 4, 'Q': 6, 'R': 10,
                'S': 15, 'T': 11, 'V': 12, 'W': 6, 'Y': 12
            },
            'groups': {
                'hydrophobic': 47,
                'aromatic': 26,
                'charged': 37,
                'polar': 50
            },
            'total_amino_acids': 162
        },
        'complexity': {
            'overall_complexity': 0.72,
            'shannon_entropy': 3.82,
            'simpson_diversity': 0.91
        },
        'physicochemical_properties': {
            'charge_distribution': {
                'net_charge': 2
            },
            'hydrophobicity': -0.2632,
            'isoelectric_point': 7.88
        },
        'stability': {
            'aggregation_propensity': 0.5744,
            'thermal_stability': 0.5767
        }
    }

    # Predict developability
    developability = predictor.predict_developability(sample_analysis)

    print(f"Solubility: {developability['solubility']:.4f}")
    print(f"Expression: {developability['expression']:.4f}")
    print(f"Aggregation: {developability['aggregation']:.4f}")
    print(f"Immunogenicity: {developability['immunogenicity']:.4f}")
    print(f"Overall developability score: {developability['overall_score']:.4f}")

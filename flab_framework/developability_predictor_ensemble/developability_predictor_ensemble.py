"""
Enhanced Developability Predictor Module for FLAb Framework

This module provides enhanced predictions for key developability properties 
of antibody fragments using ensemble methods and calibration techniques.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any

# Add paths for ensemble methods

# Import ensemble methods
from ensemble_calibration import EnsembleStrategy, CalibrationGuardrails, DiversityMeasures
from ensemble_strategies import EnsembleStrategy as EnsembleStrategy2, CalibrationGuardrail, EnsembleDiversityMeasure
from dynamic_ensemble_fusion import DynamicEnsembleFusion

class EnhancedDevelopabilityPredictor:
    """
    Enhanced class for predicting developability properties of antibody fragments 
    using ensemble methods and calibration techniques.
    """
    
    def __init__(self):
        """
        Initialize the EnhancedDevelopabilityPredictor.
        """
        # Initialize ensemble strategies
        self.ensemble_strategy = EnsembleStrategy()
        self.ensemble_strategy2 = EnsembleStrategy2()
        self.calibration_guardrails = CalibrationGuardrails()
        self.calibration_guardrail = CalibrationGuardrail()
        self.diversity_measures = DiversityMeasures()
        self.ensemble_diversity = EnsembleDiversityMeasure()
        
        # Initialize dynamic ensemble fusion
        self.dynamic_ensemble = DynamicEnsembleFusion(
            model_names=[
                'solubility_model_1', 'solubility_model_2', 'solubility_model_3',
                'expression_model_1', 'expression_model_2', 'expression_model_3',
                'aggregation_model_1', 'aggregation_model_2', 'aggregation_model_3',
                'immunogenicity_model_1', 'immunogenicity_model_2', 'immunogenicity_model_3'
            ]
        )
        
        # Model weights for different ensemble methods
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
        
        # Model performance tracking
        self.model_performance = {}
    
    def predict_developability(self, fragment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict developability properties for an antibody fragment using ensemble methods.
        
        Args:
            fragment_analysis (dict): Fragment analysis results from FragmentAnalyzer
            
        Returns:
            dict: Developability predictions including solubility, expression, aggregation, immunogenicity, and overall score
        """
        # Extract features from fragment analysis
        features = self._extract_features(fragment_analysis)
        
        # Predict individual properties using ensemble methods
        solubility_predictions = self._predict_solubility_ensemble(features)
        expression_predictions = self._predict_expression_ensemble(features)
        aggregation_predictions = self._predict_aggregation_ensemble(features)
        immunogenicity_predictions = self._predict_immunogenicity_ensemble(features)
        
        # Combine predictions using ensemble methods
        solubility = self._combine_predictions(solubility_predictions, 'solubility')
        expression = self._combine_predictions(expression_predictions, 'expression')
        aggregation = self._combine_predictions(aggregation_predictions, 'aggregation')
        immunogenicity = self._combine_predictions(immunogenicity_predictions, 'immunogenicity')
        
        # Calculate overall developability score
        overall_score = self._calculate_overall_developability(solubility, expression, aggregation, immunogenicity)
        
        # Calculate ensemble diversity metrics
        diversity_metrics = self._calculate_ensemble_diversity([
            solubility_predictions, expression_predictions, 
            aggregation_predictions, immunogenicity_predictions
        ])
        
        # Update dynamic ensemble with performance
        self._update_dynamic_ensemble([
            solubility_predictions, expression_predictions, 
            aggregation_predictions, immunogenicity_predictions
        ])
        
        return {
            'solubility': solubility,
            'expression': expression,
            'aggregation': aggregation,
            'immunogenicity': immunogenicity,
            'overall_score': overall_score,
            'ensemble_diversity': diversity_metrics,
            'model_weights': self.dynamic_ensemble.get_weights()
        }
    
    def _extract_features(self, fragment_analysis: Dict[str, Any]) -> Dict[str, float]:
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
    
    def _predict_solubility_ensemble(self, features: Dict[str, float]) -> List[float]:
        """
        Predict solubility using multiple models.
        
        Args:
            features (dict): Fragment features
            
        Returns:
            List[float]: List of solubility predictions from different models
        """
        predictions = []
        
        # Model 1: Weighted sum approach (baseline)
        score1 = 0.5  # Baseline score
        for feature, weight in self.solubility_weights.items():
            if feature in features:
                score1 += features[feature] * weight
        score1 = max(0.0, min(1.0, score1))
        predictions.append(score1)
        
        # Model 2: Alternative weighting scheme
        score2 = 0.5  # Baseline score
        alt_weights = {
            'hydrophobicity': -0.4,
            'charge_density': 0.3,
            'complexity': 0.05,
            'aromatic_content': -0.1
        }
        for feature, weight in alt_weights.items():
            if feature in features:
                score2 += features[feature] * weight
        score2 = max(0.0, min(1.0, score2))
        predictions.append(score2)
        
        # Model 3: Non-linear combination
        score3 = 0.5  # Baseline score
        if 'hydrophobicity' in features:
            score3 += (1 - features['hydrophobicity']) * 0.3  # Inverse relationship
        if 'charge_density' in features:
            score3 += features['charge_density'] * 0.2
        if 'complexity' in features:
            score3 += (1 - features['complexity']) * 0.1  # Lower complexity may improve solubility
        score3 = max(0.0, min(1.0, score3))
        predictions.append(score3)
        
        return predictions
    
    def _predict_expression_ensemble(self, features: Dict[str, float]) -> List[float]:
        """
        Predict expression level using multiple models.
        
        Args:
            features (dict): Fragment features
            
        Returns:
            List[float]: List of expression predictions from different models
        """
        predictions = []
        
        # Model 1: Weighted sum approach (baseline)
        score1 = 0.5  # Baseline score
        for feature, weight in self.expression_weights.items():
            if feature in features:
                score1 += features[feature] * weight
        score1 = max(0.0, min(1.0, score1))
        predictions.append(score1)
        
        # Model 2: Alternative weighting scheme
        score2 = 0.5  # Baseline score
        alt_weights = {
            'hydrophobicity': -0.1,
            'charge_density': 0.2,
            'complexity': 0.15,
            'aggregation_propensity': -0.4
        }
        for feature, weight in alt_weights.items():
            if feature in features:
                score2 += features[feature] * weight
        score2 = max(0.0, min(1.0, score2))
        predictions.append(score2)
        
        # Model 3: Threshold-based approach
        score3 = 0.5  # Baseline score
        if 'hydrophobicity' in features:
            if features['hydrophobicity'] < 0.4:
                score3 += 0.2  # Low hydrophobicity improves expression
            elif features['hydrophobicity'] > 0.6:
                score3 -= 0.2  # High hydrophobicity reduces expression
        if 'aggregation_propensity' in features:
            if features['aggregation_propensity'] < 0.3:
                score3 += 0.1  # Low aggregation propensity improves expression
            elif features['aggregation_propensity'] > 0.7:
                score3 -= 0.2  # High aggregation propensity reduces expression
        score3 = max(0.0, min(1.0, score3))
        predictions.append(score3)
        
        return predictions
    
    def _predict_aggregation_ensemble(self, features: Dict[str, float]) -> List[float]:
        """
        Predict aggregation propensity using multiple models.
        
        Args:
            features (dict): Fragment features
            
        Returns:
            List[float]: List of aggregation predictions from different models
        """
        predictions = []
        
        # Model 1: Direct use of aggregation propensity from features
        if 'aggregation_propensity' in features:
            score1 = features['aggregation_propensity']
        else:
            score1 = 0.5  # Default score
        predictions.append(score1)
        
        # Model 2: Weighted sum approach
        score2 = 0.5  # Baseline score
        weights = {
            'hydrophobicity': 0.4,
            'aromatic_content': 0.3,
            'complexity': 0.2
        }
        for feature, weight in weights.items():
            if feature in features:
                score2 += features[feature] * weight
        score2 = max(0.0, min(1.0, score2))
        predictions.append(score2)
        
        # Model 3: Inverse relationship with thermal stability
        score3 = 0.5  # Baseline score
        if 'thermal_stability' in features:
            score3 += (1 - features['thermal_stability']) * 0.3  # Lower thermal stability may increase aggregation
        if 'hydrophobicity' in features:
            score3 += features['hydrophobicity'] * 0.4
        score3 = max(0.0, min(1.0, score3))
        predictions.append(score3)
        
        return predictions
    
    def _predict_immunogenicity_ensemble(self, features: Dict[str, float]) -> List[float]:
        """
        Predict immunogenicity using multiple models.
        
        Args:
            features (dict): Fragment features
            
        Returns:
            List[float]: List of immunogenicity predictions from different models
        """
        predictions = []
        
        # Model 1: Weighted sum approach (baseline)
        score1 = 0.5  # Baseline score
        for feature, weight in self.immunogenicity_weights.items():
            if feature in features:
                score1 += features[feature] * weight
        score1 = max(0.0, min(1.0, score1))
        predictions.append(score1)
        
        # Model 2: Alternative weighting scheme
        score2 = 0.5  # Baseline score
        alt_weights = {
            'hydrophobicity': 0.3,
            'charge_density': -0.2,
            'complexity': -0.15,
            'aromatic_content': 0.2
        }
        for feature, weight in alt_weights.items():
            if feature in features:
                score2 += features[feature] * weight
        score2 = max(0.0, min(1.0, score2))
        predictions.append(score2)
        
        # Model 3: Threshold-based approach
        score3 = 0.5  # Baseline score
        if 'hydrophobicity' in features:
            if features['hydrophobicity'] > 0.6:
                score3 += 0.2  # High hydrophobicity increases immunogenicity
            elif features['hydrophobicity'] < 0.3:
                score3 -= 0.1  # Low hydrophobicity decreases immunogenicity
        if 'complexity' in features:
            if features['complexity'] > 0.8:
                score3 += 0.1  # High complexity increases immunogenicity
        score3 = max(0.0, min(1.0, score3))
        predictions.append(score3)
        
        return predictions
    
    def _combine_predictions(self, predictions: List[float], property_name: str) -> float:
        """
        Combine predictions using ensemble methods.
        
        Args:
            predictions (List[float]): List of predictions from different models
            property_name (str): Name of the property being predicted
            
        Returns:
            float: Combined prediction
        """
        # Use dynamic ensemble fusion for combining predictions
        model_names = [f"{property_name}_model_{i+1}" for i in range(len(predictions))]
        model_predictions = {name: np.array([pred]) for name, pred in zip(model_names, predictions)}
        
        # Get current weights from dynamic ensemble
        weights = self.dynamic_ensemble.get_weights()
        relevant_weights = [weights.get(name, 1.0/len(predictions)) for name in model_names]
        
        # Normalize weights
        weight_sum = sum(relevant_weights)
        if weight_sum > 0:
            relevant_weights = [w/weight_sum for w in relevant_weights]
        
        # Weighted average
        combined_prediction = sum(pred * weight for pred, weight in zip(predictions, relevant_weights))
        
        # Clamp between 0 and 1
        combined_prediction = max(0.0, min(1.0, combined_prediction))
        
        return combined_prediction
    
    def _calculate_overall_developability(self, solubility: float, expression: float, 
                                         aggregation: float, immunogenicity: float) -> float:
        """
        Calculate overall developability score.
        
        Args:
            solubility (float): Solubility score
            expression (float): Expression level score
            aggregation (float): Aggregation propensity score
            immunogenicity (float): Immunogenicity score
            
        Returns:
            float: Overall developability score (0-1)
        """
        # Calculate weighted sum
        score = (
            solubility * self.developability_weights['solubility'] +
            expression * self.developability_weights['expression'] +
            aggregation * self.developability_weights['aggregation'] +
            immunogenicity * self.developability_weights['immunogenicity']
        )
        
        # Calculate theoretical min and max scores based on weights
        # For scores between 0 and 1, min score occurs when positive weights are 0 and negative weights are 1
        # Max score occurs when positive weights are 1 and negative weights are 0
        min_score = (
            0 * self.developability_weights['solubility'] +
            0 * self.developability_weights['expression'] +
            1 * self.developability_weights['aggregation'] +
            1 * self.developability_weights['immunogenicity']
        )
        
        max_score = (
            1 * self.developability_weights['solubility'] +
            1 * self.developability_weights['expression'] +
            0 * self.developability_weights['aggregation'] +
            0 * self.developability_weights['immunogenicity']
        )
        
        # Normalize to 0-1 range
        if max_score != min_score:
            normalized_score = (score - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.5  # Default if all weights are equal
        
        # Clamp score between 0 and 1
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        return normalized_score
    
    def _calculate_ensemble_diversity(self, prediction_sets: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate ensemble diversity metrics.
        
        Args:
            prediction_sets (List[List[float]]): List of prediction sets for different properties
            
        Returns:
            dict: Diversity metrics
        """
        # Flatten all predictions
        all_predictions = []
        for pred_set in prediction_sets:
            all_predictions.extend(pred_set)
        
        # Convert to numpy arrays for each prediction set
        np_prediction_sets = [np.array(pred_set) for pred_set in prediction_sets]
        
        # Calculate diversity measures
        diversity_metrics = {}
        
        # Pairwise diversity
        pairwise_diversity = self.diversity_measures.calculate_pairwise_diversity(np_prediction_sets)
        diversity_metrics['pairwise'] = pairwise_diversity
        
        # Entropy diversity
        entropy_diversity = self.diversity_measures.calculate_entropy_diversity(np_prediction_sets)
        diversity_metrics['entropy'] = entropy_diversity
        
        # Disagreement rate
        try:
            disagreement_rate = self.diversity_measures.calculate_disagreement_rate(np_prediction_sets)
            diversity_metrics['disagreement_rate'] = disagreement_rate
        except:
            diversity_metrics['disagreement_rate'] = 0.0
        
        # Comprehensive diversity analysis
        comprehensive_diversity = self.ensemble_diversity.comprehensive_diversity_analysis(np_prediction_sets)
        diversity_metrics['comprehensive'] = comprehensive_diversity
        
        return diversity_metrics
    
    def _update_dynamic_ensemble(self, prediction_sets: List[List[float]]) -> None:
        """
        Update dynamic ensemble with performance metrics.
        
        Args:
            prediction_sets (List[List[float]]): List of prediction sets for different properties
        """
        # For demonstration purposes, we'll use dummy performance scores
        # In a real implementation, these would be based on validation data
        performance_scores = {
            'solubility_model_1': 0.85,
            'solubility_model_2': 0.78,
            'solubility_model_3': 0.82,
            'expression_model_1': 0.79,
            'expression_model_2': 0.81,
            'expression_model_3': 0.77,
            'aggregation_model_1': 0.88,
            'aggregation_model_2': 0.85,
            'aggregation_model_3': 0.90,
            'immunogenicity_model_1': 0.75,
            'immunogenicity_model_2': 0.72,
            'immunogenicity_model_3': 0.78
        }
        
        # Update dynamic ensemble
        self.dynamic_ensemble.update_performance(performance_scores)

# Test function
def test_enhanced_developability_predictor():
    """
    Test the EnhancedDevelopabilityPredictor class.
    """
    print("Testing EnhancedDevelopabilityPredictor:")
    
    # Create a predictor instance
    predictor = EnhancedDevelopabilityPredictor()
    
    # Create sample fragment analysis data
    sample_analysis = {
        'composition': {
            'total_amino_acids': 117,
            'groups': {
                'hydrophobic': 65,
                'charged': 20,
                'polar': 32,
                'aromatic': 18
            }
        },
        'complexity': {
            'overall_complexity': 0.9253
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
    print(f"Model weights: {developability['model_weights']}")
    
    # Print diversity metrics
    print("\nEnsemble Diversity Metrics:")
    print(f"Pairwise diversity: {developability['ensemble_diversity']['pairwise']}")
    print(f"Entropy diversity: {developability['ensemble_diversity']['entropy']}")

if __name__ == "__main__":
    test_enhanced_developability_predictor()

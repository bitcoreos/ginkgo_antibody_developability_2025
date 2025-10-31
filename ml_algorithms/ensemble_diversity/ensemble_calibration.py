"""
Ensemble Diversity & Calibration Guardrails Implementation

This module implements model ensemble strategies, calibration techniques,
and diversity measures for ensemble components.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict, Counter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


class EnsembleStrategy:
    """
    Model ensemble strategies for improved robustness.
    """
    
    def __init__(self):
        """
        Initialize the ensemble strategy.
        """
        pass
    
    def simple_average_ensemble(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Simple average ensemble of predictions.
        
        Args:
            predictions (List[np.ndarray]): List of prediction arrays from different models
            
        Returns:
            np.ndarray: Averaged predictions
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Stack predictions and calculate mean
        stacked_predictions = np.stack(predictions, axis=0)
        ensemble_predictions = np.mean(stacked_predictions, axis=0)
        
        return ensemble_predictions
    
    def weighted_average_ensemble(self, predictions: List[np.ndarray], 
                               weights: List[float]) -> np.ndarray:
        """
        Weighted average ensemble of predictions.
        
        Args:
            predictions (List[np.ndarray]): List of prediction arrays from different models
            weights (List[float]): List of weights for each model
            
        Returns:
            np.ndarray: Weighted averaged predictions
        """
        if not predictions or not weights:
            raise ValueError("No predictions or weights provided")
        
        if len(predictions) != len(weights):
            raise ValueError("Number of predictions must match number of weights")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        ensemble_predictions = np.zeros_like(predictions[0])
        for i, (pred, weight) in enumerate(zip(predictions, weights)):
            ensemble_predictions += pred * weight
        
        return ensemble_predictions
    
    def majority_vote_ensemble(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Majority vote ensemble for classification predictions.
        
        Args:
            predictions (List[np.ndarray]): List of prediction arrays from different models
            
        Returns:
            np.ndarray: Majority vote predictions
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Stack predictions
        stacked_predictions = np.stack(predictions, axis=0)
        
        # Calculate majority vote
        ensemble_predictions = np.round(np.mean(stacked_predictions, axis=0))
        
        return ensemble_predictions
    
    def rank_ensemble(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Rank-based ensemble of predictions.
        
        Args:
            predictions (List[np.ndarray]): List of prediction arrays from different models
            
        Returns:
            np.ndarray: Rank-based ensemble predictions
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Convert predictions to ranks
        ranks = []
        for pred in predictions:
            # Rank predictions (higher values get higher ranks)
            rank = np.argsort(np.argsort(pred))
            ranks.append(rank)
        
        # Average ranks
        stacked_ranks = np.stack(ranks, axis=0)
        avg_ranks = np.mean(stacked_ranks, axis=0)
        
        # Convert back to prediction scale (0-1)
        ensemble_predictions = avg_ranks / (len(predictions[0]) - 1)
        
        return ensemble_predictions


class CalibrationGuardrails:
    """
    Calibration techniques for reliable probability estimates.
    """
    
    def __init__(self):
        """
        Initialize the calibration guardrails.
        """
        self.calibrators = {}
    
    def fit_platt_scaling(self, predictions: np.ndarray, 
                       true_labels: np.ndarray, 
                       model_id: str = 'default') -> Dict[str, float]:
        """
        Fit Platt scaling calibration.
        
        Args:
            predictions (np.ndarray): Model predictions (probabilities)
            true_labels (np.ndarray): True labels
            model_id (str): Identifier for the model
            
        Returns:
            Dict[str, float]: Calibration parameters
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Convert probabilities to log-odds
        log_odds = np.log(predictions / (1 - predictions))
        
        # Fit logistic regression
        calibrator = LogisticRegression()
        calibrator.fit(log_odds.reshape(-1, 1), true_labels)
        
        # Store calibrator
        self.calibrators[model_id] = {
            'type': 'platt',
            'calibrator': calibrator
        }
        
        # Return calibration parameters
        coef = calibrator.coef_[0][0]
        intercept = calibrator.intercept_[0]
        
        return {
            'coef': float(coef),
            'intercept': float(intercept)
        }
    
    def apply_platt_scaling(self, predictions: np.ndarray, 
                         model_id: str = 'default') -> np.ndarray:
        """
        Apply Platt scaling calibration.
        
        Args:
            predictions (np.ndarray): Model predictions (probabilities)
            model_id (str): Identifier for the model
            
        Returns:
            np.ndarray: Calibrated predictions
        """
        if model_id not in self.calibrators or self.calibrators[model_id]['type'] != 'platt':
            raise ValueError(f"Platt scaling calibrator for model {model_id} not found")
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Convert probabilities to log-odds
        log_odds = np.log(predictions / (1 - predictions))
        
        # Apply calibration
        calibrator = self.calibrators[model_id]['calibrator']
        calibrated_log_odds = calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        
        return calibrated_log_odds
    
    def fit_isotonic_calibration(self, predictions: np.ndarray, 
                              true_labels: np.ndarray, 
                              model_id: str = 'default') -> None:
        """
        Fit isotonic calibration.
        
        Args:
            predictions (np.ndarray): Model predictions (probabilities)
            true_labels (np.ndarray): True labels
            model_id (str): Identifier for the model
        """
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(predictions, true_labels)
        
        # Store calibrator
        self.calibrators[model_id] = {
            'type': 'isotonic',
            'calibrator': calibrator
        }
    
    def apply_isotonic_calibration(self, predictions: np.ndarray, 
                               model_id: str = 'default') -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            predictions (np.ndarray): Model predictions (probabilities)
            model_id (str): Identifier for the model
            
        Returns:
            np.ndarray: Calibrated predictions
        """
        if model_id not in self.calibrators or self.calibrators[model_id]['type'] != 'isotonic':
            raise ValueError(f"Isotonic calibrator for model {model_id} not found")
        
        # Apply calibration
        calibrator = self.calibrators[model_id]['calibrator']
        calibrated_predictions = calibrator.predict(predictions)
        
        return calibrated_predictions
    
    def evaluate_calibration(self, predictions: np.ndarray, 
                          true_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate calibration quality.
        
        Args:
            predictions (np.ndarray): Model predictions (probabilities)
            true_labels (np.ndarray): True labels
            
        Returns:
            Dict[str, float]: Calibration metrics
        """
        # Calculate calibration metrics
        accuracy = accuracy_score(true_labels, np.round(predictions))
        logloss = log_loss(true_labels, predictions)
        brier_score = brier_score_loss(true_labels, predictions)
        
        # Expected calibration error (simplified)
        # Bin predictions into 10 bins
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(predictions, bins) - 1
        
        ece = 0.0
        for i in range(10):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(true_labels[bin_mask])
                bin_confidence = np.mean(predictions[bin_mask])
                bin_size = np.sum(bin_mask)
                ece += np.abs(bin_accuracy - bin_confidence) * bin_size / len(predictions)
        
        return {
            'accuracy': float(accuracy),
            'log_loss': float(logloss),
            'brier_score': float(brier_score),
            'expected_calibration_error': float(ece)
        }


class DiversityMeasures:
    """
    Diversity measures for ensemble components.
    """
    
    def __init__(self):
        """
        Initialize the diversity measures.
        """
        pass
    
    def calculate_pairwise_diversity(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate pairwise diversity measures.
        
        Args:
            predictions (List[np.ndarray]): List of prediction arrays from different models
            
        Returns:
            Dict[str, float]: Diversity measures
        """
        if len(predictions) < 2:
            raise ValueError("At least two predictions required for diversity calculation")
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(corr)
        
        # Calculate diversity metrics
        mean_correlation = np.mean(correlations)
        diversity_score = 1.0 - mean_correlation  # Higher diversity when correlation is lower
        
        # Calculate Q-statistic (simplified)
        q_statistics = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                pred1 = np.round(predictions[i])
                pred2 = np.round(predictions[j])
                
                # Calculate Q-statistic
                n00 = np.sum((pred1 == 0) & (pred2 == 0))
                n01 = np.sum((pred1 == 0) & (pred2 == 1))
                n10 = np.sum((pred1 == 1) & (pred2 == 0))
                n11 = np.sum((pred1 == 1) & (pred2 == 1))
                
                if (n00 + n01 + n10 + n11) > 0:
                    q_stat = (n00 * n11 - n01 * n10) / (n00 * n11 + n01 * n10 + 1e-8)
                    q_statistics.append(q_stat)
        
        mean_q_statistic = np.mean(q_statistics) if q_statistics else 0.0
        
        return {
            'mean_pairwise_correlation': float(mean_correlation),
            'diversity_score': float(diversity_score),
            'mean_q_statistic': float(mean_q_statistic)
        }
    
    def calculate_entropy_diversity(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate entropy-based diversity measures.
        
        Args:
            predictions (List[np.ndarray]): List of prediction arrays from different models
            
        Returns:
            Dict[str, float]: Entropy-based diversity measures
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Stack predictions
        stacked_predictions = np.stack(predictions, axis=0)
        
        # Calculate entropy for each sample
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        stacked_predictions = np.clip(stacked_predictions, epsilon, 1 - epsilon)
        
        # Calculate entropy
        entropy = -np.sum(stacked_predictions * np.log(stacked_predictions), axis=0)
        mean_entropy = np.mean(entropy)
        
        # Calculate disagreement (variance across models)
        variance = np.var(stacked_predictions, axis=0)
        mean_variance = np.mean(variance)
        
        return {
            'mean_entropy': float(mean_entropy),
            'mean_variance': float(mean_variance)
        }
    
    def calculate_disagreement_rate(self, predictions: List[np.ndarray]) -> float:
        """
        Calculate disagreement rate among ensemble members.
        
        Args:
            predictions (List[np.ndarray]): List of prediction arrays from different models
            
        Returns:
            float: Disagreement rate
        """
        if len(predictions) < 2:
            raise ValueError("At least two predictions required for disagreement calculation")
        
        # Convert to binary predictions
        binary_predictions = [np.round(pred) for pred in predictions]
        
        # Stack binary predictions
        stacked_predictions = np.stack(binary_predictions, axis=0)
        
        # Calculate disagreement rate
        # Disagreement when models don't all agree
        agreement = np.all(stacked_predictions == stacked_predictions[0], axis=0)
        disagreement_rate = 1.0 - np.mean(agreement)
        
        return float(disagreement_rate)


def main():
    """
    Example usage of the ensemble diversity and calibration guardrails implementation.
    """
    # Generate example predictions from different models
    np.random.seed(42)
    n_samples = 1000
    
    # True labels
    true_labels = np.random.randint(0, 2, n_samples)
    
    # Model predictions (with different biases)
    model1_pred = np.random.beta(2, 5, n_samples) * 0.3 + np.random.beta(5, 2, n_samples) * 0.7 * true_labels
    model2_pred = np.random.beta(3, 4, n_samples) * 0.4 + np.random.beta(4, 3, n_samples) * 0.6 * true_labels
    model3_pred = np.random.beta(4, 3, n_samples) * 0.5 + np.random.beta(3, 4, n_samples) * 0.5 * true_labels
    
    predictions = [model1_pred, model2_pred, model3_pred]
    
    # Ensemble strategies
    ensemble_strategy = EnsembleStrategy()
    
    # Simple average ensemble
    avg_ensemble = ensemble_strategy.simple_average_ensemble(predictions)
    print("Simple Average Ensemble Results:")
    print(f"  Ensemble shape: {avg_ensemble.shape}")
    print(f"  Mean prediction: {np.mean(avg_ensemble):.4f}")
    print(f"  Prediction range: [{np.min(avg_ensemble):.4f}, {np.max(avg_ensemble):.4f}]")
    
    # Weighted average ensemble
    weights = [0.5, 0.3, 0.2]
    weighted_ensemble = ensemble_strategy.weighted_average_ensemble(predictions, weights)
    print("\nWeighted Average Ensemble Results:")
    print(f"  Ensemble shape: {weighted_ensemble.shape}")
    print(f"  Mean prediction: {np.mean(weighted_ensemble):.4f}")
    
    # Majority vote ensemble
    majority_ensemble = ensemble_strategy.majority_vote_ensemble(predictions)
    print("\nMajority Vote Ensemble Results:")
    print(f"  Ensemble shape: {majority_ensemble.shape}")
    print(f"  Mean prediction: {np.mean(majority_ensemble):.4f}")
    
    # Rank ensemble
    rank_ensemble = ensemble_strategy.rank_ensemble(predictions)
    print("\nRank Ensemble Results:")
    print(f"  Ensemble shape: {rank_ensemble.shape}")
    print(f"  Mean prediction: {np.mean(rank_ensemble):.4f}")
    
    # Calibration guardrails
    calibration_guardrails = CalibrationGuardrails()
    
    # Fit and apply Platt scaling
    platt_params = calibration_guardrails.fit_platt_scaling(model1_pred, true_labels, 'model1')
    calibrated_model1 = calibration_guardrails.apply_platt_scaling(model1_pred, 'model1')
    
    print("\nPlatt Scaling Calibration Results:")
    print(f"  Calibration parameters: coef={platt_params['coef']:.4f}, intercept={platt_params['intercept']:.4f}")
    print(f"  Calibrated predictions range: [{np.min(calibrated_model1):.4f}, {np.max(calibrated_model1):.4f}]")
    
    # Fit and apply isotonic calibration
    calibration_guardrails.fit_isotonic_calibration(model2_pred, true_labels, 'model2')
    calibrated_model2 = calibration_guardrails.apply_isotonic_calibration(model2_pred, 'model2')
    
    print("\nIsotonic Calibration Results:")
    print(f"  Calibrated predictions range: [{np.min(calibrated_model2):.4f}, {np.max(calibrated_model2):.4f}]")
    
    # Evaluate calibration
    calibration_metrics = calibration_guardrails.evaluate_calibration(model1_pred, true_labels)
    print("\nCalibration Evaluation Metrics:")
    print(f"  Accuracy: {calibration_metrics['accuracy']:.4f}")
    print(f"  Log Loss: {calibration_metrics['log_loss']:.4f}")
    print(f"  Brier Score: {calibration_metrics['brier_score']:.4f}")
    print(f"  Expected Calibration Error: {calibration_metrics['expected_calibration_error']:.4f}")
    
    # Diversity measures
    diversity_measures = DiversityMeasures()
    
    # Pairwise diversity
    pairwise_diversity = diversity_measures.calculate_pairwise_diversity(predictions)
    print("\nPairwise Diversity Measures:")
    print(f"  Mean Pairwise Correlation: {pairwise_diversity['mean_pairwise_correlation']:.4f}")
    print(f"  Diversity Score: {pairwise_diversity['diversity_score']:.4f}")
    print(f"  Mean Q-Statistic: {pairwise_diversity['mean_q_statistic']:.4f}")
    
    # Entropy diversity
    entropy_diversity = diversity_measures.calculate_entropy_diversity(predictions)
    print("\nEntropy-Based Diversity Measures:")
    print(f"  Mean Entropy: {entropy_diversity['mean_entropy']:.4f}")
    print(f"  Mean Variance: {entropy_diversity['mean_variance']:.4f}")
    
    # Disagreement rate
    disagreement_rate = diversity_measures.calculate_disagreement_rate(predictions)
    print("\nDisagreement Rate:")
    print(f"  Disagreement Rate: {disagreement_rate:.4f}")


if __name__ == "__main__":
    main()

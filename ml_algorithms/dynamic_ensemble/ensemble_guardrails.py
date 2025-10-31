"""
Ensemble Diversity & Calibration Guardrails Implementation

This module implements ensemble diversity measures, calibration techniques,
and robustness strategies for antibody developability prediction.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import List, Dict, Tuple, Union
from scipy.stats import entropy


class EnsembleDiversityAnalyzer:
    """
    Analyzer for measuring diversity in ensemble components.
    """
    
    def __init__(self):
        """
        Initialize the ensemble diversity analyzer.
        """
        pass
    
    def calculate_pairwise_correlation(self, predictions: np.ndarray) -> float:
        """
        Calculate average pairwise correlation between ensemble members.
        
        Args:
            predictions (np.ndarray): Array of shape (n_models, n_samples) with predictions
            
        Returns:
            float: Average pairwise correlation
        """
        if predictions.shape[0] < 2:
            return 0.0
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(predictions)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Return average correlation
        return float(np.mean(upper_triangle))
    
    def calculate_q_statistic(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate the Q-statistic for ensemble diversity.
        
        Args:
            predictions (np.ndarray): Array of shape (n_models, n_samples) with predictions
            targets (np.ndarray): Array of shape (n_samples,) with true targets
            
        Returns:
            float: Q-statistic (lower values indicate higher diversity)
        """
        n_models = predictions.shape[0]
        if n_models < 2:
            return 0.0
        
        # Calculate individual model errors
        errors = np.abs(predictions - targets.reshape(1, -1))
        
        # Calculate pairwise error correlations
        error_corr_matrix = np.corrcoef(errors)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = error_corr_matrix[np.triu_indices_from(error_corr_matrix, k=1)]
        
        # Return average correlation
        return float(np.mean(upper_triangle))
    
    def calculate_disagreement_measure(self, predictions: np.ndarray) -> float:
        """
        Calculate the disagreement measure among ensemble members.
        
        Args:
            predictions (np.ndarray): Array of shape (n_models, n_samples) with predictions
            
        Returns:
            float: Disagreement measure (higher values indicate higher diversity)
        """
        if predictions.shape[0] < 2:
            return 0.0
        
        # Calculate standard deviation across models for each sample
        std_predictions = np.std(predictions, axis=0)
        
        # Return average standard deviation
        return float(np.mean(std_predictions))
    
    def calculate_entropy_diversity(self, predictions: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate diversity based on entropy of predictions.
        
        Args:
            predictions (np.ndarray): Array of shape (n_models, n_samples) with predictions
            n_bins (int): Number of bins for histogram calculation
            
        Returns:
            float: Entropy-based diversity measure (higher values indicate higher diversity)
        """
        if predictions.shape[0] < 2:
            return 0.0
        
        diversity_scores = []
        
        # Calculate entropy for each sample
        for i in range(predictions.shape[1]):
            # Create histogram of predictions for this sample
            hist, _ = np.histogram(predictions[:, i], bins=n_bins, range=(0, 1), density=True)
            
            # Avoid log(0) by adding small epsilon
            hist = hist + 1e-10
            hist = hist / np.sum(hist)
            
            # Calculate entropy
            sample_entropy = entropy(hist)
            diversity_scores.append(sample_entropy)
        
        # Return average entropy
        return float(np.mean(diversity_scores))
    
    def analyze_ensemble_diversity(self, predictions: np.ndarray, targets: np.ndarray = None) -> Dict[str, float]:
        """
        Comprehensive diversity analysis of ensemble predictions.
        
        Args:
            predictions (np.ndarray): Array of shape (n_models, n_samples) with predictions
            targets (np.ndarray): Array of shape (n_samples,) with true targets (optional)
            
        Returns:
            Dict[str, float]: Diversity metrics
        """
        diversity_metrics = {
            'pairwise_correlation': self.calculate_pairwise_correlation(predictions),
            'disagreement_measure': self.calculate_disagreement_measure(predictions),
            'entropy_diversity': self.calculate_entropy_diversity(predictions)
        }
        
        if targets is not None:
            diversity_metrics['q_statistic'] = self.calculate_q_statistic(predictions, targets)
        
        return diversity_metrics


class ProbabilityCalibrator:
    """
    Calibrator for improving probability estimates from ensemble models.
    """
    
    def __init__(self):
        """
        Initialize the probability calibrator.
        """
        self.calibrators = {}
        self.calibration_methods = {}
    
    def fit_platt_scaling(self, predictions: np.ndarray, targets: np.ndarray, model_name: str = 'ensemble') -> None:
        """
        Fit Platt scaling calibration.
        
        Args:
            predictions (np.ndarray): Array of shape (n_samples,) with predictions
            targets (np.ndarray): Array of shape (n_samples,) with true targets
            model_name (str): Name of the model to calibrate
        """
        # Avoid log(0) and log(1) by clipping predictions
        clipped_predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # Calculate log-odds
        log_odds = np.log(clipped_predictions / (1 - clipped_predictions))
        
        # Fit linear regression to map log-odds to targets
        # This is a simplified version of Platt scaling
        from sklearn.linear_model import LogisticRegression
        
        # Reshape for sklearn
        X = log_odds.reshape(-1, 1)
        y = targets
        
        # Fit logistic regression
        calibrator = LogisticRegression()
        calibrator.fit(X, y)
        
        # Store calibrator
        self.calibrators[model_name] = calibrator
        self.calibration_methods[model_name] = 'platt_scaling'
    
    def fit_isotonic_regression(self, predictions: np.ndarray, targets: np.ndarray, model_name: str = 'ensemble') -> None:
        """
        Fit isotonic regression calibration.
        
        Args:
            predictions (np.ndarray): Array of shape (n_samples,) with predictions
            targets (np.ndarray): Array of shape (n_samples,) with true targets
            model_name (str): Name of the model to calibrate
        """
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(predictions, targets)
        
        # Store calibrator
        self.calibrators[model_name] = calibrator
        self.calibration_methods[model_name] = 'isotonic_regression'
    
    def calibrate_probabilities(self, predictions: np.ndarray, model_name: str = 'ensemble') -> np.ndarray:
        """
        Calibrate probabilities using fitted calibrator.
        
        Args:
            predictions (np.ndarray): Array of shape (n_samples,) with predictions
            model_name (str): Name of the model to calibrate
            
        Returns:
            np.ndarray: Calibrated probabilities
        """
        if model_name not in self.calibrators:
            raise ValueError(f"No calibrator found for model '{model_name}'. Please fit a calibrator first.")
        
        # Apply calibration
        calibrator = self.calibrators[model_name]
        calibrated_predictions = calibrator.predict(predictions)
        
        # Ensure probabilities are in [0, 1] range
        calibrated_predictions = np.clip(calibrated_predictions, 0, 1)
        
        return calibrated_predictions
    
    def evaluate_calibration(self, predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate calibration quality using reliability diagram and Brier score.
        
        Args:
            predictions (np.ndarray): Array of shape (n_samples,) with predictions
            targets (np.ndarray): Array of shape (n_samples,) with true targets
            n_bins (int): Number of bins for reliability diagram
            
        Returns:
            Dict: Calibration metrics
        """
        # Calculate Brier score
        brier_score = brier_score_loss(targets, predictions)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            targets, predictions, n_bins=n_bins
        )
        
        # Calculate calibration error (ECE - Expected Calibration Error)
        bin_counts, _ = np.histogram(predictions, bins=n_bins, range=(0, 1))
        bin_accuracies, _ = np.histogram(predictions[targets == 1], bins=n_bins, range=(0, 1))
        
        # Avoid division by zero
        bin_accuracies = np.divide(bin_accuracies, bin_counts, out=np.zeros_like(bin_accuracies), where=bin_counts!=0)
        
        # Calculate ECE
        ece = np.sum(np.abs(fraction_of_positives - mean_predicted_value) * bin_counts) / len(predictions)
        
        return {
            'brier_score': float(brier_score),
            'expected_calibration_error': float(ece),
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }


class EnsembleRobustnessGuard:
    """
    Guard for ensuring ensemble robustness and reliability.
    """
    
    def __init__(self, diversity_threshold: float = 0.1, calibration_threshold: float = 0.1):
        """
        Initialize the ensemble robustness guard.
        
        Args:
            diversity_threshold (float): Minimum diversity threshold
            calibration_threshold (float): Maximum calibration error threshold
        """
        self.diversity_threshold = diversity_threshold
        self.calibration_threshold = calibration_threshold
        self.diversity_analyzer = EnsembleDiversityAnalyzer()
        self.calibrator = ProbabilityCalibrator()
    
    def check_ensemble_robustness(self, predictions: np.ndarray, targets: np.ndarray = None) -> Dict[str, Union[bool, float]]:
        """
        Check if ensemble meets robustness criteria.
        
        Args:
            predictions (np.ndarray): Array of shape (n_models, n_samples) with predictions
            targets (np.ndarray): Array of shape (n_samples,) with true targets (optional)
            
        Returns:
            Dict: Robustness assessment
        """
        # Analyze diversity
        diversity_metrics = self.diversity_analyzer.analyze_ensemble_diversity(predictions, targets)
        
        # Check diversity criteria
        sufficient_diversity = diversity_metrics['disagreement_measure'] > self.diversity_threshold
        
        # Initialize calibration check
        well_calibrated = True
        calibration_error = 0.0
        
        # If targets are provided, check calibration
        if targets is not None:
            # Average predictions across models
            avg_predictions = np.mean(predictions, axis=0)
            
            # Evaluate calibration
            calibration_metrics = self.calibrator.evaluate_calibration(avg_predictions, targets)
            calibration_error = calibration_metrics['expected_calibration_error']
            well_calibrated = calibration_error < self.calibration_threshold
        
        # Overall robustness
        is_robust = sufficient_diversity and well_calibrated
        
        return {
            'is_robust': is_robust,
            'sufficient_diversity': sufficient_diversity,
            'well_calibrated': well_calibrated,
            'diversity_metrics': diversity_metrics,
            'calibration_error': calibration_error
        }
    
    def apply_ensemble_guardrails(self, predictions: np.ndarray, targets: np.ndarray = None) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Apply ensemble guardrails to ensure robustness.
        
        Args:
            predictions (np.ndarray): Array of shape (n_models, n_samples) with predictions
            targets (np.ndarray): Array of shape (n_samples,) with true targets (optional)
            
        Returns:
            Dict: Guardrail results
        """
        # Check robustness
        robustness_assessment = self.check_ensemble_robustness(predictions, targets)
        
        # Initialize results
        calibrated_predictions = None
        diversity_improved = False
        
        # If targets are provided and calibration is poor, apply calibration
        if targets is not None and not robustness_assessment['well_calibrated']:
            # Average predictions across models
            avg_predictions = np.mean(predictions, axis=0)
            
            # Fit isotonic regression calibrator
            self.calibrator.fit_isotonic_regression(avg_predictions, targets)
            
            # Calibrate predictions
            calibrated_predictions = self.calibrator.calibrate_probabilities(avg_predictions)
        
        # If diversity is poor, suggest improvement strategies
        if not robustness_assessment['sufficient_diversity']:
            # In practice, this would trigger ensemble retraining or model addition
            diversity_improved = False  # Placeholder
        
        return {
            'robustness_assessment': robustness_assessment,
            'calibrated_predictions': calibrated_predictions,
            'diversity_improved': diversity_improved
        }


def main():
    """
    Example usage of the ensemble guardrails implementation.
    """
    # Generate example data
    np.random.seed(42)
    n_models = 5
    n_samples = 1000
    
    # Generate diverse predictions
    predictions = np.random.beta(2, 2, (n_models, n_samples))
    targets = np.random.binomial(1, 0.3, n_samples)
    
    # Analyze ensemble diversity
    diversity_analyzer = EnsembleDiversityAnalyzer()
    diversity_metrics = diversity_analyzer.analyze_ensemble_diversity(predictions, targets)
    
    print("Ensemble Diversity Metrics:")
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate calibration
    avg_predictions = np.mean(predictions, axis=0)
    calibrator = ProbabilityCalibrator()
    calibration_metrics = calibrator.evaluate_calibration(avg_predictions, targets)
    
    print("\nCalibration Metrics:")
    print(f"Brier Score: {calibration_metrics['brier_score']:.4f}")
    print(f"Expected Calibration Error: {calibration_metrics['expected_calibration_error']:.4f}")
    
    # Check ensemble robustness
    robustness_guard = EnsembleRobustnessGuard()
    robustness_assessment = robustness_guard.check_ensemble_robustness(predictions, targets)
    
    print("\nRobustness Assessment:")
    print(f"Is Robust: {robustness_assessment['is_robust']}")
    print(f"Sufficient Diversity: {robustness_assessment['sufficient_diversity']}")
    print(f"Well Calibrated: {robustness_assessment['well_calibrated']}")
    
    # Apply guardrails
    guardrail_results = robustness_guard.apply_ensemble_guardrails(predictions, targets)
    
    if guardrail_results['calibrated_predictions'] is not None:
        print("\nCalibration applied to improve probability estimates.")


if __name__ == "__main__":
    main()

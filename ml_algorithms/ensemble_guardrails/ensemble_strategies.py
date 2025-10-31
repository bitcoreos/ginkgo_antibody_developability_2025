"""
Ensemble Diversity & Calibration Guardrails Implementation

This module implements model ensemble strategies, calibration techniques,
and diversity measures for improved robustness in antibody developability prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
        self.ensemble_models = []
        self.ensemble_weights = []
        self.ensemble_method = None
    
    def create_diverse_ensemble(self, X: np.ndarray, y: np.ndarray, 
                              n_models: int = 5, model_types: List[str] = None) -> List[Any]:
        """
        Create a diverse ensemble of models with different algorithms and hyperparameters.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            n_models (int): Number of models in ensemble
            model_types (List[str]): List of model types to include
            
        Returns:
            List[Any]: List of trained models
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting']
        
        models = []
        
        # Create diverse models with different algorithms and hyperparameters
        for i in range(n_models):
            # Select model type in round-robin fashion
            model_type = model_types[i % len(model_types)]
            
            # Create model with different hyperparameters
            if model_type == 'random_forest':
                # Vary number of estimators and max depth
                n_estimators = 50 + (i * 25)
                max_depth = 3 + (i % 5)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=i)
            elif model_type == 'gradient_boosting':
                # Vary number of estimators and learning rate
                n_estimators = 50 + (i * 25)
                learning_rate = 0.05 + (i * 0.05)
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=i)
            else:
                # Default to Random Forest
                model = RandomForestClassifier(n_estimators=100, random_state=i)
            
            # Train model
            model.fit(X, y)
            models.append(model)
        
        self.ensemble_models = models
        self.ensemble_method = 'diverse_ensemble'
        
        return models
    
    def create_bagging_ensemble(self, X: np.ndarray, y: np.ndarray, 
                              n_models: int = 5, model_type: str = 'random_forest') -> List[Any]:
        """
        Create a bagging ensemble of models.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            n_models (int): Number of models in ensemble
            model_type (str): Type of base model
            
        Returns:
            List[Any]: List of trained models
        """
        models = []
        n_samples = X.shape[0]
        
        # Create bagging ensemble
        for i in range(n_models):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Create and train model
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=i)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(n_estimators=100, random_state=i)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=i)
            
            model.fit(X_bootstrap, y_bootstrap)
            models.append(model)
        
        self.ensemble_models = models
        self.ensemble_method = 'bagging_ensemble'
        
        return models
    
    def create_stacking_ensemble(self, X: np.ndarray, y: np.ndarray, 
                               base_models: List[Any] = None, 
                               meta_model: Any = None) -> Tuple[List[Any], Any]:
        """
        Create a stacking ensemble of models.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            base_models (List[Any]): List of base models
            meta_model (Any): Meta model for stacking
            
        Returns:
            Tuple[List[Any], Any]: Tuple of base models and trained meta model
        """
        if base_models is None:
            # Create default base models
            base_models = [
                RandomForestClassifier(n_estimators=100, random_state=0),
                GradientBoostingClassifier(n_estimators=100, random_state=1)
            ]
        
        # Train base models
        for model in base_models:
            model.fit(X, y)
        
        # Generate predictions for meta model training
        meta_features = []
        for model in base_models:
            predictions = model.predict_proba(X)[:, 1]  # Probability of positive class
            meta_features.append(predictions)
        
        # Stack predictions as meta features
        meta_X = np.column_stack(meta_features)
        
        # Create and train meta model
        if meta_model is None:
            meta_model = LogisticRegression(random_state=42)
        
        meta_model.fit(meta_X, y)
        
        self.ensemble_models = base_models
        self.ensemble_method = 'stacking_ensemble'
        
        return base_models, meta_model
    
    def predict_ensemble(self, X: np.ndarray, method: str = 'average', 
                        meta_model: Any = None, weights: List[float] = None) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X (np.ndarray): Feature matrix
            method (str): Ensemble method ('average', 'weighted', 'stacking')
            meta_model (Any): Meta model for stacking
            weights (List[float]): Weights for weighted averaging
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.ensemble_models:
            raise ValueError("No ensemble models available. Create ensemble first.")
        
        # Get predictions from all models
        predictions = []
        for model in self.ensemble_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]  # Probability of positive class
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        # Combine predictions based on method
        if method == 'average':
            # Simple average
            ensemble_predictions = np.mean(predictions, axis=0)
        elif method == 'weighted':
            # Weighted average
            if weights is None:
                weights = [1.0 / len(predictions)] * len(predictions)
            ensemble_predictions = np.average(predictions, axis=0, weights=weights)
        elif method == 'stacking':
            # Stacking with meta model
            if meta_model is None:
                raise ValueError("Meta model required for stacking method.")
            meta_X = np.column_stack(predictions)
            ensemble_predictions = meta_model.predict_proba(meta_X)[:, 1]
        else:
            raise ValueError("Invalid ensemble method. Use 'average', 'weighted', or 'stacking'.")
        
        return ensemble_predictions
    
    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray, method: str = 'average',
                         meta_model: Any = None, weights: List[float] = None) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            method (str): Ensemble method ('average', 'weighted', 'stacking')
            meta_model (Any): Meta model for stacking
            weights (List[float]): Weights for weighted averaging
            
        Returns:
            Dict[str, float]: Ensemble performance metrics
        """
        # Get ensemble predictions
        ensemble_predictions = self.predict_ensemble(X, method, meta_model, weights)
        
        # Convert probabilities to binary predictions
        binary_predictions = (ensemble_predictions > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, binary_predictions),
            'precision': precision_score(y, binary_predictions, zero_division=0),
            'recall': recall_score(y, binary_predictions, zero_division=0),
            'f1': f1_score(y, binary_predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, ensemble_predictions) if len(np.unique(y)) > 1 else float('nan')
        }
        
        return metrics


class CalibrationGuardrail:
    """
    Calibration techniques for reliable probability estimates.
    """
    
    def __init__(self):
        """
        Initialize the calibration guardrail.
        """
        self.calibrated_models = []
        self.calibration_method = None
    
    def calibrate_model_platt(self, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Calibrate model using Platt scaling.
        
        Args:
            model (Any): Trained model
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Any: Calibrated model
        """
        # Use CalibratedClassifierCV with sigmoid method (Platt scaling)
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X, y)
        
        self.calibrated_models.append(calibrated_model)
        self.calibration_method = 'platt_scaling'
        
        return calibrated_model
    
    def calibrate_model_isotonic(self, model: Any, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Calibrate model using isotonic regression.
        
        Args:
            model (Any): Trained model
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Any: Calibrated model
        """
        # Use CalibratedClassifierCV with isotonic method
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated_model.fit(X, y)
        
        self.calibrated_models.append(calibrated_model)
        self.calibration_method = 'isotonic_regression'
        
        return calibrated_model
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate calibration metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            n_bins (int): Number of bins for calibration curve
            
        Returns:
            Dict[str, Union[float, np.ndarray]]: Calibration metrics
        """
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # Calculate calibration error (ECE - Expected Calibration Error)
        bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
        bin_true, _ = np.histogram(y_prob * y_true, bins=n_bins, range=(0, 1))
        
        # Avoid division by zero
        bin_accuracies = np.divide(bin_true, bin_counts, out=np.zeros_like(bin_true, dtype=float), where=bin_counts!=0)
        
        # Calculate ECE
        ece = np.sum(np.abs(bin_accuracies - mean_predicted_value) * bin_counts) / len(y_prob)
        
        # Calculate MCE (Maximum Calibration Error)
        mce = np.max(np.abs(bin_accuracies - mean_predicted_value))
        
        return {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
        """
        Plot calibration curve (placeholder function).
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            n_bins (int): Number of bins for calibration curve
        """
        # This is a placeholder function as we can't actually plot in this environment
        # In practice, you would use matplotlib to create the plot
        print("Calibration curve plotting function called.")
        print(f"Would plot calibration curve with {n_bins} bins.")


class EnsembleDiversityMeasure:
    """
    Diversity measures for ensemble components.
    """
    
    def __init__(self):
        """
        Initialize the ensemble diversity measure.
        """
        pass
    
    def calculate_pairwise_diversity(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate pairwise diversity measures between ensemble members.
        
        Args:
            predictions (List[np.ndarray]): List of predictions from ensemble members
            
        Returns:
            Dict[str, float]: Pairwise diversity measures
        """
        if len(predictions) < 2:
            return {'pairwise_diversity': 0.0}
        
        # Convert to binary predictions
        binary_predictions = [(pred > 0.5).astype(int) for pred in predictions]
        
        # Calculate pairwise disagreement
        disagreements = []
        for i in range(len(binary_predictions)):
            for j in range(i + 1, len(binary_predictions)):
                # Calculate disagreement as fraction of samples where predictions differ
                disagreement = np.mean(binary_predictions[i] != binary_predictions[j])
                disagreements.append(disagreement)
        
        # Calculate average pairwise disagreement
        avg_disagreement = np.mean(disagreements)
        
        # Calculate Q-statistic (diversity measure)
        q_statistics = []
        for i in range(len(binary_predictions)):
            for j in range(i + 1, len(binary_predictions)):
                pred_i = binary_predictions[i]
                pred_j = binary_predictions[j]
                
                # Calculate Q-statistic
                n00 = np.sum((pred_i == 0) & (pred_j == 0))
                n01 = np.sum((pred_i == 0) & (pred_j == 1))
                n10 = np.sum((pred_i == 1) & (pred_j == 0))
                n11 = np.sum((pred_i == 1) & (pred_j == 1))
                
                # Avoid division by zero
                denominator = (n00 + n01) * (n10 + n11) + (n00 + n10) * (n01 + n11)
                if denominator > 0:
                    q_stat = (n00 * n11 - n01 * n10) / denominator
                else:
                    q_stat = 0.0
                
                q_statistics.append(q_stat)
        
        # Calculate average Q-statistic
        avg_q_stat = np.mean(q_statistics)
        
        return {
            'average_pairwise_disagreement': float(avg_disagreement),
            'average_q_statistic': float(avg_q_stat),
            'number_of_pairs': len(disagreements)
        }
    
    def calculate_entropy_diversity(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate entropy-based diversity measures.
        
        Args:
            predictions (List[np.ndarray]): List of predictions from ensemble members
            
        Returns:
            Dict[str, float]: Entropy-based diversity measures
        """
        if not predictions:
            return {'entropy_diversity': 0.0}
        
        # Stack predictions
        pred_matrix = np.column_stack(predictions)
        
        # Calculate entropy for each sample
        entropies = []
        for i in range(pred_matrix.shape[0]):
            # Get predictions for this sample
            sample_preds = pred_matrix[i, :]
            
            # Calculate entropy
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-15
            sample_preds = np.clip(sample_preds, epsilon, 1 - epsilon)
            
            # Convert to binary probabilities (simplified approach)
            # For each prediction, we consider it as probability of class 1
            # and (1 - prediction) as probability of class 0
            # Then calculate entropy
            
            # Simplified entropy calculation based on variance of predictions
            mean_pred = np.mean(sample_preds)
            var_pred = np.var(sample_preds)
            
            # Entropy-like measure based on variance
            entropy = var_pred  # Higher variance means higher diversity
            entropies.append(entropy)
        
        # Calculate average entropy
        avg_entropy = np.mean(entropies)
        
        return {
            'average_entropy_diversity': float(avg_entropy),
            'max_entropy_diversity': float(np.max(entropies)),
            'min_entropy_diversity': float(np.min(entropies))
        }
    
    def comprehensive_diversity_analysis(self, predictions: List[np.ndarray]) -> Dict[str, Union[float, Dict]]:
        """
        Perform comprehensive diversity analysis.
        
        Args:
            predictions (List[np.ndarray]): List of predictions from ensemble members
            
        Returns:
            Dict[str, Union[float, Dict]]: Comprehensive diversity analysis
        """
        # Calculate pairwise diversity
        pairwise_diversity = self.calculate_pairwise_diversity(predictions)
        
        # Calculate entropy diversity
        entropy_diversity = self.calculate_entropy_diversity(predictions)
        
        # Compile results
        comprehensive_analysis = {
            'pairwise_diversity': pairwise_diversity,
            'entropy_diversity': entropy_diversity
        }
        
        return comprehensive_analysis


def main():
    """
    Example usage of the ensemble guardrails implementation.
    """
    # Generate example data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.3, n_samples)
    
    # Split data into train and test sets
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Ensemble strategy
    ensemble_strategy = EnsembleStrategy()
    
    # Create diverse ensemble
    models = ensemble_strategy.create_diverse_ensemble(X_train, y_train, n_models=5)
    
    # Evaluate ensemble
    ensemble_metrics = ensemble_strategy.evaluate_ensemble(X_test, y_test)
    
    print("Ensemble Strategy Results:")
    print(f"Ensemble accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"Ensemble ROC AUC: {ensemble_metrics['roc_auc']:.4f}")
    
    # Calibration guardrail
    calibration_guardrail = CalibrationGuardrail()
    
    # Calibrate one of the models
    base_model = models[0]
    calibrated_model = calibration_guardrail.calibrate_model_platt(base_model, X_train, y_train)
    
    # Get predictions
    if hasattr(calibrated_model, 'predict_proba'):
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = calibrated_model.predict(X_test)
    
    # Calculate calibration metrics
    calibration_metrics = calibration_guardrail.calculate_calibration_metrics(y_test, y_prob)
    
    print("\nCalibration Guardrail Results:")
    print(f"Expected Calibration Error: {calibration_metrics['expected_calibration_error']:.4f}")
    print(f"Maximum Calibration Error: {calibration_metrics['maximum_calibration_error']:.4f}")
    
    # Ensemble diversity measure
    ensemble_diversity = EnsembleDiversityMeasure()
    
    # Get predictions from all models
    predictions = []
    for model in models:
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_test)[:, 1]
        else:
            pred = model.predict(X_test)
        predictions.append(pred)
    
    # Calculate diversity measures
    diversity_analysis = ensemble_diversity.comprehensive_diversity_analysis(predictions)
    
    print("\nEnsemble Diversity Measure Results:")
    print(f"Average pairwise disagreement: {diversity_analysis['pairwise_diversity']['average_pairwise_disagreement']:.4f}")
    print(f"Average Q-statistic: {diversity_analysis['pairwise_diversity']['average_q_statistic']:.4f}")
    print(f"Average entropy diversity: {diversity_analysis['entropy_diversity']['average_entropy_diversity']:.4f}")


if __name__ == "__main__":
    main()

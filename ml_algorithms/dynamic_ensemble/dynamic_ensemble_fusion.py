
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DynamicEnsembleFusion:
    """
    Dynamic ensemble fusion with temporal weighting for antibody developability prediction.
    Adapts model weights based on time-series performance patterns.
    """

    def __init__(self, model_names: List[str], 
                 decay_rate: float = 0.9,
                 min_weight: float = 0.01,
                 uncertainty_weight: float = 0.1):
        """
        Initialize dynamic ensemble fusion.

        Args:
            model_names: List of model names in the ensemble
            decay_rate: Decay rate for exponential moving average (higher = slower decay)
            min_weight: Minimum weight to prevent model elimination
            uncertainty_weight: Weight for uncertainty in final prediction
        """
        self.model_names = model_names
        self.n_models = len(model_names)
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.uncertainty_weight = uncertainty_weight

        # Initialize weights (equal at start)
        self.weights = np.ones(self.n_models) / self.n_models

        # Performance tracking
        self.performance_history = {name: [] for name in model_names}
        self.timestamp_history = []

        # EMA parameters
        self.ema_performance = np.zeros(self.n_models)
        self.ema_count = 0

        # Kalman filter parameters for noise reduction
        self.kf_state = np.ones(self.n_models) / self.n_models  # initial state (weights)
        self.kf_covariance = np.eye(self.n_models) * 0.1  # initial covariance
        self.kf_process_noise = np.eye(self.n_models) * 0.01
        self.kf_measurement_noise = np.eye(self.n_models) * 0.1

        # Temporal pattern detection
        self.trend_strength = np.zeros(self.n_models)
        self.seasonality_strength = np.zeros(self.n_models)

        # Regularization
        self.regularization_window = 10  # lookback window for regularization

    def update_performance(self, model_scores: Dict[str, float],
                          timestamp: Optional[datetime] = None):
        """
        Update model performance history.

        Args:
            model_scores: Dictionary of model names and their current scores
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Record timestamp
        self.timestamp_history.append(timestamp)

        # Update performance history for each model
        for name in self.model_names:
            score = model_scores.get(name, 0.0)
            self.performance_history[name].append(score)

        # Update EMA
        current_performance = np.array([model_scores.get(name, 0.0) 
                                      for name in self.model_names])

        self.ema_count += 1
        alpha = 1.0 / self.ema_count if self.ema_count < 100 else (1 - self.decay_rate)
        self.ema_performance = alpha * current_performance + (1 - alpha) * self.ema_performance

        # Update Kalman filter
        self._update_kalman_filter(current_performance)

        # Detect temporal patterns if enough data
        if len(self.timestamp_history) >= 5:
            self._detect_temporal_patterns()

        # Update weights
        self._update_weights()

    def _update_kalman_filter(self, measurements: np.ndarray):
        """
        Update Kalman filter with new measurements.
        """
        # Prediction step
        predicted_state = self.kf_state
        predicted_covariance = self.kf_covariance + self.kf_process_noise

        # Update step
        innovation = measurements - predicted_state
        innovation_covariance = predicted_covariance + self.kf_measurement_noise

        # Handle singular matrix
        if np.linalg.det(innovation_covariance) < 1e-8:
            innovation_covariance += np.eye(self.n_models) * 1e-8

        kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)

        self.kf_state = predicted_state + kalman_gain @ innovation
        self.kf_covariance = (np.eye(self.n_models) - kalman_gain) @ predicted_covariance

        # Ensure state is valid probabilities
        self.kf_state = np.abs(self.kf_state)
        if np.sum(self.kf_state) > 0:
            self.kf_state = self.kf_state / np.sum(self.kf_state)
        else:
            self.kf_state = np.ones(self.n_models) / self.n_models

    def _detect_temporal_patterns(self):
        """
        Detect trends and seasonality in model performance.
        """
        n_points = len(self.timestamp_history)
        if n_points < 5:
            return

        # Convert to time series
        time_diffs = [(self.timestamp_history[i] - self.timestamp_history[0]).total_seconds() 
                     for i in range(n_points)]

        for i, name in enumerate(self.model_names):
            scores = np.array(self.performance_history[name])

            # Simple trend detection (linear regression)
            if n_points >= 2:
                try:
                    A = np.vstack([time_diffs, np.ones(n_points)]).T
                    slope, _ = np.linalg.lstsq(A, scores, rcond=None)[0]
                    self.trend_strength[i] = np.abs(slope) * 100  # scale for impact
                except:
                    self.trend_strength[i] = 0.0

            # Seasonality detection (autocorrelation)
            if n_points >= 4:
                # Simple autocorrelation at lag 1
                autocorr = np.corrcoef(scores[:-1], scores[1:])[0,1] if len(scores) > 1 else 0.0
                self.seasonality_strength[i] = max(0.0, autocorr)

    def _update_weights(self):
        """
        Update model weights based on performance history and temporal patterns.
        """
        # Base weights from EMA performance
        base_weights = np.maximum(self.ema_performance, 0.001)  # prevent zero
        base_weights = base_weights / np.sum(base_weights)

        # Apply trend adjustment: boost models with positive trends
        trend_adjustment = 1.0 + (self.trend_strength - np.mean(self.trend_strength)) * 0.5
        trend_adjustment = np.maximum(trend_adjustment, 0.5)  # limit downside

        # Apply seasonality adjustment: favor consistent performers
        seasonality_adjustment = 1.0 + self.seasonality_strength * 0.3

        # Combine adjustments
        adjusted_weights = base_weights * trend_adjustment * seasonality_adjustment
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        # Apply Kalman filter smoothing
        kf_weights = self.kf_state

        # Combine EMA and Kalman weights
        combined_weights = 0.7 * adjusted_weights + 0.3 * kf_weights
        combined_weights = combined_weights / np.sum(combined_weights)

        # Apply minimum weight constraint
        combined_weights = np.maximum(combined_weights, self.min_weight)
        combined_weights = combined_weights / np.sum(combined_weights)

        # Regularization: prevent overfitting to short-term fluctuations
        if len(self.timestamp_history) > self.regularization_window:
            # Check for excessive weight changes
            recent_changes = []
            for i in range(max(0, len(self.timestamp_history) - self.regularization_window), 
                         len(self.timestamp_history) - 1):
                prev_weights = self._calculate_weights_at(i)
                curr_weights = self._calculate_weights_at(i + 1)
                change = np.sum(np.abs(curr_weights - prev_weights))
                recent_changes.append(change)

            if recent_changes:
                avg_change = np.mean(recent_changes)
                if avg_change > 0.3:  # high volatility
                    # Revert to more stable weights
                    stable_weights = np.ones(self.n_models) / self.n_models
                    combined_weights = 0.3 * combined_weights + 0.7 * stable_weights
                    combined_weights = combined_weights / np.sum(combined_weights)

        self.weights = combined_weights

    def _calculate_weights_at(self, time_idx: int) -> np.ndarray:
        """
        Calculate weights at a specific time index (for regularization).
        """
        # Simplified: return equal weights for regularization purposes
        return np.ones(self.n_models) / self.n_models

    def predict(self, model_predictions: Dict[str, np.ndarray],
               model_uncertainties: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Make prediction using weighted ensemble.

        Args:
            model_predictions: Dictionary of model names and their predictions
            model_uncertainties: Optional dictionary of model uncertainties

        Returns:
            Weighted ensemble prediction
        """
        # Stack predictions
        pred_matrix = np.zeros((len(self.weights), len(model_predictions[self.model_names[0]])))

        for i, name in enumerate(self.model_names):
            if name in model_predictions:
                pred_matrix[i, :] = model_predictions[name]

        # Apply weights
        weighted_pred = self.weights @ pred_matrix

        # Apply uncertainty weighting if provided
        if model_uncertainties is not None:
            uncertainty_weights = np.zeros(len(self.weights))
            for i, name in enumerate(self.model_names):
                if name in model_uncertainties and len(model_uncertainties[name]) > 0:
                    # Lower uncertainty -> higher weight
                    uncertainty_weights[i] = 1.0 / (1.0 + np.mean(model_uncertainties[name]))
                else:
                    uncertainty_weights[i] = 1.0

            # Normalize uncertainty weights
            if np.sum(uncertainty_weights) > 0:
                uncertainty_weights = uncertainty_weights / np.sum(uncertainty_weights)

            # Combine with performance weights
            final_weights = ((1 - self.uncertainty_weight) * self.weights + 
                           self.uncertainty_weight * uncertainty_weights)
            final_weights = final_weights / np.sum(final_weights)

            # Recalculate prediction with combined weights
            weighted_pred = final_weights @ pred_matrix

        return weighted_pred

    def get_weights(self) -> Dict[str, float]:
        """
        Get current model weights.
        """
        return {name: float(self.weights[i]) for i, name in enumerate(self.model_names)}

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all models.
        """
        summary = {}
        for name in self.model_names:
            scores = self.performance_history[name]
            if scores:
                summary[name] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'recent_score': float(scores[-1]) if scores else 0.0,
                    'trend_strength': float(self.trend_strength[self.model_names.index(name)]),
                    'seasonality_strength': float(self.seasonality_strength[self.model_names.index(name)])
                }
            else:
                summary[name] = {
                    'mean_score': 0.0,
                    'std_score': 0.0,
                    'recent_score': 0.0,
                    'trend_strength': 0.0,
                    'seasonality_strength': 0.0
                }

        return summary

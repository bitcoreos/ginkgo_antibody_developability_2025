"""
Ensemble Diversity and Calibration Module

This module implements ensemble diversity measurement and calibration guardrails.
"""

import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score


class EnsembleDiversityCalibration:
    """
    Ensemble diversity and calibration analyzer.
    """
    
    def __init__(self):
        """
        Initialize the ensemble diversity and calibration analyzer.
        """
        pass
    
    def create_ensemble(self, n_estimators: int = 5, max_depth: int = None, random_state: int = 42) -> Dict[str, Union[List, bool]]:
        """
        Create an ensemble of models.
        
        Args:
            n_estimators (int): Number of estimators in the ensemble
            max_depth (int): Maximum depth of the trees
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict: Ensemble of models
        """
        # Create ensemble of regressors and classifiers
        regressor_ensemble = []
        classifier_ensemble = []
        
        for i in range(n_estimators):
            # Create regressor with different random state
            regressor = RandomForestRegressor(
                n_estimators=10,
                max_depth=max_depth,
                random_state=random_state + i,
                bootstrap=True
            )
            regressor_ensemble.append(regressor)
            
            # Create classifier with different random state
            classifier = RandomForestClassifier(
                n_estimators=10,
                max_depth=max_depth,
                random_state=random_state + i,
                bootstrap=True
            )
            classifier_ensemble.append(classifier)
        
        return {
            'regressor_ensemble': regressor_ensemble,
            'classifier_ensemble': classifier_ensemble,
            'ensemble_created': True
        }
    
    def measure_ensemble_diversity(self, ensemble: List, X: np.ndarray, y: np.ndarray) -> Dict[str, Union[float, bool]]:
        """
        Measure diversity of an ensemble.
        
        Args:
            ensemble (List): List of models in the ensemble
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            Dict: Diversity measurements
        """
        if len(ensemble) < 2:
            return {
                'diversity_score': 0.0,
                'average_pairwise_diversity': 0.0,
                'measurement_complete': True
            }
        
        # Calculate pairwise diversity between ensemble members
        pairwise_diversity = []
        
        # For each pair of models in the ensemble
        for i in range(len(ensemble)):
            for j in range(i + 1, len(ensemble)):
                # Get predictions from both models
                if hasattr(ensemble[i], 'predict'):
                    pred_i = ensemble[i].predict(X)
                    pred_j = ensemble[j].predict(X)
                    
                    # Calculate diversity as 1 - correlation between predictions
                    if len(pred_i) > 1 and len(pred_j) > 1:
                        correlation = np.corrcoef(pred_i, pred_j)[0, 1]
                        diversity = 1.0 - abs(correlation) if not np.isnan(correlation) else 0.0
                        pairwise_diversity.append(diversity)
        
        # Calculate average pairwise diversity
        if pairwise_diversity:
            average_pairwise_diversity = np.mean(pairwise_diversity)
        else:
            average_pairwise_diversity = 0.0
        
        # Calculate overall diversity score (normalized)
        diversity_score = average_pairwise_diversity
        
        return {
            'diversity_score': diversity_score,
            'average_pairwise_diversity': average_pairwise_diversity,
            'measurement_complete': True
        }
    
    def apply_calibration(self, model, X: np.ndarray, y: np.ndarray, method: str = 'sigmoid') -> Dict[str, Union[object, float, bool]]:
        """
        Apply calibration to a model.
        
        Args:
            model: Model to calibrate
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            method (str): Calibration method ('sigmoid' or 'isotonic')
            
        Returns:
            Dict: Calibrated model and calibration score
        """
        # Create calibrated model
        calibrated_model = CalibratedClassifierCV(model, method=method, cv=3)
        
        # Fit the calibrated model
        calibrated_model.fit(X, y)
        
        # Calculate calibration score (using cross-validation)
        try:
            calibration_scores = cross_val_score(calibrated_model, X, y, cv=3, scoring='neg_log_loss')
            calibration_score = -np.mean(calibration_scores)
        except:
            calibration_score = 0.0
        
        return {
            'calibrated_model': calibrated_model,
            'calibration_method': method,
            'calibration_score': calibration_score,
            'calibration_complete': True
        }
    
    def evaluate_ensemble_performance(self, ensemble: List, X: np.ndarray, y: np.ndarray, task_type: str = 'regression') -> Dict[str, Union[float, bool]]:
        """
        Evaluate ensemble performance.
        
        Args:
            ensemble (List): List of models in the ensemble
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            task_type (str): Type of task ('regression' or 'classification')
            
        Returns:
            Dict: Performance metrics
        """
        if not ensemble:
            return {
                'accuracy': 0.0,
                'mse': 0.0,
                'ensemble_performance_complete': True
            }
        
        # Get predictions from all models in the ensemble
        predictions = []
        for model in ensemble:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)
        
        # Calculate ensemble prediction (average for regression, majority vote for classification)
        if predictions:
            if task_type == 'regression':
                ensemble_prediction = np.mean(predictions, axis=0)
                mse = mean_squared_error(y, ensemble_prediction)
                accuracy = 0.0
            else:
                # For classification, take majority vote
                predictions_array = np.array(predictions)
                ensemble_prediction = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), 
                    axis=0, 
                    arr=predictions_array
                )
                accuracy = accuracy_score(y, ensemble_prediction)
                mse = 0.0
        else:
            ensemble_prediction = np.zeros(len(y))
            accuracy = 0.0
            mse = 0.0
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'ensemble_performance_complete': True
        }
    
    def generate_ensemble_report(self, ensemble: List, X: np.ndarray, y: np.ndarray, task_type: str = 'regression') -> Dict[str, Union[str, float, bool]]:
        """
        Generate a comprehensive ensemble report.
        
        Args:
            ensemble (List): List of models in the ensemble
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            task_type (str): Type of task ('regression' or 'classification')
            
        Returns:
            Dict: Comprehensive ensemble report
        """
        # Measure ensemble diversity
        diversity_metrics = self.measure_ensemble_diversity(ensemble, X, y)
        
        # Evaluate ensemble performance
        performance_metrics = self.evaluate_ensemble_performance(ensemble, X, y, task_type)
        
        # Generate summary
        summary = f"""
Ensemble Diversity and Performance Report
=========================================

Diversity Metrics:
- Diversity Score: {diversity_metrics['diversity_score']:.3f}
- Average Pairwise Diversity: {diversity_metrics['average_pairwise_diversity']:.3f}

Performance Metrics:
- Accuracy: {performance_metrics['accuracy']:.3f}
- MSE: {performance_metrics['mse']:.3f}
"""
        
        return {
            'ensemble_size': len(ensemble),
            'diversity_metrics': diversity_metrics,
            'performance_metrics': performance_metrics,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the ensemble diversity and calibration analyzer.
    """
    # Generate example data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y_regression = np.random.rand(100)
    y_classification = np.random.randint(0, 2, 100)
    
    # Create analyzer
    analyzer = EnsembleDiversityCalibration()
    
    # Create ensemble
    ensemble_dict = analyzer.create_ensemble(n_estimators=5, max_depth=3)
    regressor_ensemble = ensemble_dict['regressor_ensemble']
    classifier_ensemble = ensemble_dict['classifier_ensemble']
    
    # Train ensemble members (simplified)
    for model in regressor_ensemble:
        model.fit(X, y_regression)
    
    for model in classifier_ensemble:
        model.fit(X, y_classification)
    
    # Measure ensemble diversity
    diversity_metrics = analyzer.measure_ensemble_diversity(regressor_ensemble, X, y_regression)
    print("Ensemble Diversity Metrics:")
    print(f"  Diversity Score: {diversity_metrics['diversity_score']:.3f}")
    print(f"  Average Pairwise Diversity: {diversity_metrics['average_pairwise_diversity']:.3f}")
    
    # Evaluate ensemble performance
    performance_metrics = analyzer.evaluate_ensemble_performance(regressor_ensemble, X, y_regression, 'regression')
    print("\nEnsemble Performance Metrics:")
    print(f"  MSE: {performance_metrics['mse']:.3f}")
    
    # Evaluate classifier ensemble performance
    classifier_performance_metrics = analyzer.evaluate_ensemble_performance(classifier_ensemble, X, y_classification, 'classification')
    print(f"  Accuracy: {classifier_performance_metrics['accuracy']:.3f}")
    
    # Generate comprehensive ensemble report
    ensemble_report = analyzer.generate_ensemble_report(regressor_ensemble, X, y_regression, 'regression')
    print("\nEnsemble Report Summary:")
    print(ensemble_report['summary'])
    
    # Example of calibration
    print("\nCalibration Example:")
    if classifier_ensemble:
        # Calibrate first model in ensemble
        calibration_result = analyzer.apply_calibration(classifier_ensemble[0], X, y_classification, 'sigmoid')
        print(f"  Calibration Method: {calibration_result['calibration_method']}")
        print(f"  Calibration Score: {calibration_result['calibration_score']:.3f}")


if __name__ == "__main__":
    main()

"""
AbLEF (Antibody Language Ensemble Fusion) Implementation

This module implements a simplified version of AbLEF, an ensemble fusion method 
combining multiple language models for antibody sequence analysis.
"""

import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error


class AbLEF:
    """
    Antibody Language Ensemble Fusion (AbLEF) implementation.
    """
    
    def __init__(self):
        """
        Initialize the AbLEF ensemble.
        """
        # Create ensemble of different model types
        self.models = [
            RandomForestRegressor(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(random_state=42, probability=True)
        ]
        
        # Model weights (initialized to equal weights)
        self.model_weights = [1.0/len(self.models)] * len(self.models)
        
        # Model performance history
        self.performance_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, task_type: str = 'regression'):
        """
        Fit the AbLEF ensemble.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            task_type (str): Type of task ('regression' or 'classification')
        """
        # Convert task type if needed
        if task_type == 'classification':
            # Convert regression targets to classification if needed
            if y.dtype in [np.float32, np.float64]:
                y = (y > np.median(y)).astype(int)
        
        # Fit each model in the ensemble
        for model in self.models:
            try:
                model.fit(X, y)
            except Exception as e:
                # Handle models that can't handle the data type
                print(f"Warning: Model {type(model).__name__} failed to fit: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the AbLEF ensemble.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        # Get predictions from each model
        predictions = []
        
        for model in self.models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {type(model).__name__} failed to predict: {e}")
        
        # Weighted average of predictions
        if predictions:
            # Convert to numpy array for easier manipulation
            predictions = np.array(predictions)
            
            # Apply weights
            weighted_predictions = np.average(predictions, axis=0, weights=self.model_weights[:len(predictions)])
            
            return weighted_predictions
        else:
            # Return zeros if no models could predict
            return np.zeros(X.shape[0])
    
    def update_weights(self, X_val: np.ndarray, y_val: np.ndarray, task_type: str = 'regression'):
        """
        Update model weights based on validation performance.
        
        Args:
            X_val (np.ndarray): Validation feature matrix
            y_val (np.ndarray): Validation target vector
            task_type (str): Type of task ('regression' or 'classification')
        """
        # Get predictions from each model
        predictions = []
        
        for model in self.models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val)
                    predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {type(model).__name__} failed to predict: {e}")
                predictions.append(np.zeros(len(y_val)))
        
        # Calculate performance for each model
        performances = []
        
        for i, pred in enumerate(predictions):
            try:
                if task_type == 'regression':
                    performance = -mean_squared_error(y_val, pred)  # Negative MSE (higher is better)
                else:
                    # Convert predictions to classes if needed
                    if pred.dtype in [np.float32, np.float64]:
                        pred_classes = (pred > 0.5).astype(int)
                    else:
                        pred_classes = pred
                    performance = accuracy_score(y_val, pred_classes)
                performances.append(performance)
            except Exception as e:
                print(f"Warning: Failed to calculate performance for model {i}: {e}")
                performances.append(0.0)
        
        # Store performance history
        self.performance_history.append(performances)
        
        # Update weights based on performance (softmax weighting)
        if performances:
            # Convert to numpy array
            perf_array = np.array(performances)
            
            # Apply softmax to performances (shift by max to prevent overflow)
            shifted_perf = perf_array - np.max(perf_array)
            exp_perf = np.exp(shifted_perf)
            
            # Handle case where all performances are very negative
            if np.sum(exp_perf) > 0:
                self.model_weights[:len(performances)] = exp_perf / np.sum(exp_perf)
            else:
                # If all performances are very negative, use equal weights
                self.model_weights[:len(performances)] = [1.0/len(performances)] * len(performances)
    
    def get_model_weights(self) -> List[float]:
        """
        Get current model weights.
        
        Returns:
            List[float]: Current model weights
        """
        return self.model_weights[:len(self.models)]
    
    def get_performance_history(self) -> List[List[float]]:
        """
        Get model performance history.
        
        Returns:
            List[List[float]]: Model performance history
        """
        return self.performance_history
    
    def generate_report(self) -> Dict[str, Union[str, List[float], bool]]:
        """
        Generate a comprehensive AbLEF report.
        
        Returns:
            Dict: Comprehensive AbLEF report
        """
        # Get current model weights
        weights = self.get_model_weights()
        
        # Get model names
        model_names = [type(model).__name__ for model in self.models]
        
        # Generate summary
        summary = "AbLEF (Antibody Language Ensemble Fusion) Report\n"
        summary += "=========================================\n\n"
        summary += "Model Weights:\n"
        
        for name, weight in zip(model_names, weights):
            summary += f"- {name}: {weight:.3f}\n"
        
        # Add performance history if available
        if self.performance_history:
            summary += "\nPerformance History:\n"
            for i, performances in enumerate(self.performance_history):
                summary += f"- Validation Round {i+1}: "
                for name, perf in zip(model_names, performances):
                    summary += f"{name}={perf:.3f} "
                summary += "\n"
        
        return {
            'model_names': model_names,
            'model_weights': weights,
            'performance_history': self.performance_history,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the AbLEF implementation.
    """
    # Generate example data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y_regression = np.random.rand(100)
    y_classification = np.random.randint(0, 2, 100)
    
    # Split data for validation
    X_train, X_val = X[:80], X[80:]
    y_train_reg, y_val_reg = y_regression[:80], y_regression[80:]
    y_train_clf, y_val_clf = y_classification[:80], y_classification[80:]
    
    # Create AbLEF ensemble
    ablef = AbLEF()
    
    # Example 1: Regression task
    print("AbLEF Example - Regression Task:")
    
    # Fit ensemble
    ablef.fit(X_train, y_train_reg, 'regression')
    
    # Update weights based on validation performance
    ablef.update_weights(X_val, y_val_reg, 'regression')
    
    # Make predictions
    predictions = ablef.predict(X_val)
    mse = mean_squared_error(y_val_reg, predictions)
    print(f"  MSE: {mse:.3f}")
    
    # Get model weights
    weights = ablef.get_model_weights()
    print(f"  Model Weights: {weights}")
    
    # Example 2: Classification task
    print("\nAbLEF Example - Classification Task:")
    
    # Create new ensemble for classification
    ablef_clf = AbLEF()
    
    # Fit ensemble
    ablef_clf.fit(X_train, y_train_clf, 'classification')
    
    # Update weights based on validation performance
    ablef_clf.update_weights(X_val, y_val_clf, 'classification')
    
    # Make predictions
    predictions = ablef_clf.predict(X_val)
    # Convert to classes for accuracy calculation
    pred_classes = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_val_clf, pred_classes)
    print(f"  Accuracy: {accuracy:.3f}")
    
    # Get model weights
    weights = ablef_clf.get_model_weights()
    print(f"  Model Weights: {weights}")
    
    # Generate comprehensive report
    print("\nAbLEF Report Summary:")
    report = ablef_clf.generate_report()
    print(report['summary'])


if __name__ == "__main__":
    main()

"""
PROPERMAB Implementation

This module implements a simplified version of PROPERMAB, an integrative framework 
for in silico prediction of developability properties.
"""

import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class PROPERMAB:
    """
    PROPERMAB (Predicting Oprional Properties of Engineered Recombinant MAbs) implementation.
    """
    
    def __init__(self):
        """
        Initialize the PROPERMAB framework.
        """
        # Define the developability properties to predict
        self.properties = [
            'solubility',
            'expression_level',
            'aggregation_propensity',
            'thermal_stability',
            'immunogenicity'
        ]
        
        # Create models for each property
        self.models = {
            prop: RandomForestRegressor(n_estimators=10, random_state=42) 
            for prop in self.properties
        }
        
        # Feature importance tracking
        self.feature_importance = {prop: None for prop in self.properties}
        
        # Model performance tracking
        self.performance = {prop: {} for prop in self.properties}
    
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], feature_names: List[str] = None):
        """
        Fit the PROPERMAB framework.
        
        Args:
            X (np.ndarray): Feature matrix
            y_dict (Dict[str, np.ndarray]): Dictionary of target vectors for each property
            feature_names (List[str]): Names of features (optional)
        """
        # Fit a model for each property
        for prop in self.properties:
            if prop in y_dict:
                try:
                    # Fit model
                    self.models[prop].fit(X, y_dict[prop])
                    
                    # Store feature importance if available
                    if hasattr(self.models[prop], 'feature_importances_'):
                        self.feature_importance[prop] = self.models[prop].feature_importances_
                    
                    # Store feature names if provided
                    if feature_names and len(feature_names) == X.shape[1]:
                        self.feature_names = feature_names
                except Exception as e:
                    print(f"Warning: Failed to fit model for {prop}: {e}")
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions for all developability properties.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            Dict[str, np.ndarray]: Predictions for each property
        """
        predictions = {}
        
        # Make predictions for each property
        for prop in self.properties:
            try:
                predictions[prop] = self.models[prop].predict(X)
            except Exception as e:
                print(f"Warning: Failed to predict {prop}: {e}")
                predictions[prop] = np.zeros(X.shape[0])
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the PROPERMAB framework.
        
        Args:
            X (np.ndarray): Feature matrix
            y_dict (Dict[str, np.ndarray]): Dictionary of target vectors for each property
            
        Returns:
            Dict[str, Dict[str, float]]: Performance metrics for each property
        """
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate performance metrics for each property
        performance = {}
        
        for prop in self.properties:
            if prop in y_dict and prop in predictions:
                try:
                    y_true = y_dict[prop]
                    y_pred = predictions[prop]
                    
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    performance[prop] = {
                        'mse': mse,
                        'r2': r2
                    }
                    
                    # Store performance
                    self.performance[prop] = performance[prop]
                except Exception as e:
                    print(f"Warning: Failed to evaluate {prop}: {e}")
                    performance[prop] = {
                        'mse': 0.0,
                        'r2': 0.0
                    }
            else:
                performance[prop] = {
                    'mse': 0.0,
                    'r2': 0.0
                }
        
        return performance
    
    def get_feature_importance(self, property_name: str = None) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Get feature importance for one or all properties.
        
        Args:
            property_name (str): Name of specific property (optional)
            
        Returns:
            Union[Dict[str, np.ndarray], np.ndarray]: Feature importance
        """
        if property_name:
            return self.feature_importance.get(property_name, None)
        else:
            return self.feature_importance
    
    def get_performance(self, property_name: str = None) -> Union[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Get performance metrics for one or all properties.
        
        Args:
            property_name (str): Name of specific property (optional)
            
        Returns:
            Union[Dict[str, Dict[str, float]], Dict[str, float]]: Performance metrics
        """
        if property_name:
            return self.performance.get(property_name, {})
        else:
            return self.performance
    
    def generate_report(self) -> Dict[str, Union[str, Dict, bool]]:
        """
        Generate a comprehensive PROPERMAB report.
        
        Returns:
            Dict: Comprehensive PROPERMAB report
        """
        # Generate summary
        summary = "PROPERMAB (Predicting Oprional Properties of Engineered Recombinant MAbs) Report\n"
        summary += "=========================================================================\n\n"
        
        # Add performance metrics
        summary += "Performance Metrics:\n"
        for prop in self.properties:
            if prop in self.performance and self.performance[prop]:
                perf = self.performance[prop]
                summary += f"- {prop}: MSE={perf['mse']:.3f}, R2={perf['r2']:.3f}\n"
            else:
                summary += f"- {prop}: No performance data\n"
        
        # Add feature importance if available and feature names are provided
        if hasattr(self, 'feature_names'):
            summary += "\nTop Feature Importances:\n"
            for prop in self.properties:
                if self.feature_importance[prop] is not None:
                    # Get top 3 features
                    importances = self.feature_importance[prop]
                    top_indices = np.argsort(importances)[-3:][::-1]
                    top_features = [(self.feature_names[i], importances[i]) for i in top_indices]
                    
                    summary += f"- {prop}:\n"
                    for feature, importance in top_features:
                        summary += f"  * {feature}: {importance:.3f}\n"
        
        return {
            'properties': self.properties,
            'performance': self.performance,
            'feature_importance': self.feature_importance,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the PROPERMAB implementation.
    """
    # Generate example data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Generate target variables for each property
    y_dict = {
        'solubility': np.random.rand(100),
        'expression_level': np.random.rand(100),
        'aggregation_propensity': np.random.rand(100),
        'thermal_stability': np.random.rand(100),
        'immunogenicity': np.random.rand(100)
    }
    
    # Split data for training and testing
    X_train, X_test = X[:80], X[80:]
    y_train_dict = {prop: y_dict[prop][:80] for prop in y_dict}
    y_test_dict = {prop: y_dict[prop][80:] for prop in y_dict}
    
    # Create PROPERMAB framework
    propemab = PROPERMAB()
    
    # Fit the framework
    print("Fitting PROPERMAB framework:")
    propemab.fit(X_train, y_train_dict, feature_names)
    
    # Make predictions
    print("\nMaking predictions:")
    predictions = propemab.predict(X_test)
    for prop, pred in predictions.items():
        print(f"  {prop}: {pred[:3]}...")  # Show first 3 predictions
    
    # Evaluate the framework
    print("\nEvaluating PROPERMAB framework:")
    performance = propemab.evaluate(X_test, y_test_dict)
    for prop, perf in performance.items():
        print(f"  {prop}: MSE={perf['mse']:.3f}, R2={perf['r2']:.3f}")
    
    # Get feature importance
    print("\nFeature importance for solubility:")
    importance = propemab.get_feature_importance('solubility')
    if importance is not None:
        for i, imp in enumerate(importance):
            print(f"  {feature_names[i]}: {imp:.3f}")
    
    # Generate comprehensive report
    print("\nPROPERMAB Report Summary:")
    report = propemab.generate_report()
    print(report['summary'])


if __name__ == "__main__":
    main()

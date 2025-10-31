"""
FLAb Framework Integration for Validation QA Systems
"""

import sys
import os

# Add paths for FLAb framework and validation QA systems
sys.path.append('/a0/bitcore/workspace/flab_framework')
sys.path.append('/a0/bitcore/workspace/ml_algorithms/validation_qa')

from validation_qa import ValidationQASystems


class FLAbValidationQAIntegrator:
    """
    FLAb Framework Integration for Validation QA Systems
    """
    
    def __init__(self):
        """
        Initialize FLAb Validation QA Integrator
        """
        self.validation_qa_systems = ValidationQASystems()
    
    def validate_model(self, X_train, y_train, X_test, y_test, model, task_type='regression'):
        """
        Validate a model using systematic validation
        
        Parameters:
        X_train (array-like): Training features
        y_train (array-like): Training targets
        X_test (array-like): Test features
        y_test (array-like): Test targets
        model (object): Trained model object
        task_type (str): Type of task ('regression' or 'classification')
        
        Returns:
        dict: Validation results
        """
        return self.validation_qa_systems.systematic_validation(X_train, y_train, X_test, y_test, model, task_type)
    
    def detect_drift(self, X_reference, X_current, threshold=0.1):
        """
        Detect concept drift between reference and current data
        
        Parameters:
        X_reference (array-like): Reference data
        X_current (array-like): Current data
        threshold (float): Threshold for drift detection
        
        Returns:
        dict: Drift detection results
        """
        return self.validation_qa_systems.detect_concept_drift(X_reference, X_current, threshold)
    
    def qa_submission(self, predictions, expected_range=None):
        """
        Perform quality assurance on submission predictions
        
        Parameters:
        predictions (array-like): Model predictions
        expected_range (tuple): Expected range of predictions (min, max)
        
        Returns:
        dict: QA results
        """
        return self.validation_qa_systems.submission_qa(predictions, expected_range)
    
    def generate_validation_report(self, X_train, y_train, X_test, y_test, model, task_type='regression'):
        """
        Generate a comprehensive validation report
        
        Parameters:
        X_train (array-like): Training features
        y_train (array-like): Training targets
        X_test (array-like): Test features
        y_test (array-like): Test targets
        model (object): Trained model object
        task_type (str): Type of task ('regression' or 'classification')
        
        Returns:
        dict: Validation report
        """
        return self.validation_qa_systems.generate_validation_report(X_train, y_train, X_test, y_test, model, task_type)
    
    def generate_drift_report(self, X_reference, X_current, threshold=0.1):
        """
        Generate a comprehensive drift detection report
        
        Parameters:
        X_reference (array-like): Reference data
        X_current (array-like): Current data
        threshold (float): Threshold for drift detection
        
        Returns:
        dict: Drift report
        """
        return self.validation_qa_systems.generate_drift_report(X_reference, X_current, threshold)
    
    def generate_qa_report(self, predictions, expected_range=None):
        """
        Generate a comprehensive QA report
        
        Parameters:
        predictions (array-like): Model predictions
        expected_range (tuple): Expected range of predictions (min, max)
        
        Returns:
        dict: QA report
        """
        return self.validation_qa_systems.generate_qa_report(predictions, expected_range)


def main():
    """
    Main function for testing FLAb Validation QA Integrator
    """
    print("Testing FLAb Validation QA Integrator...")
    
    # Import necessary modules
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    # Generate example data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train_regression = np.random.rand(100)
    y_train_classification = np.random.randint(0, 2, 100)
    
    X_test = np.random.rand(50, 5)
    y_test_regression = np.random.rand(50)
    y_test_classification = np.random.randint(0, 2, 50)
    
    # Create some drift in current data
    X_current = np.random.rand(50, 5) + 0.5  # Shifted distribution
    
    # Example predictions for QA
    predictions = np.random.rand(100)
    
    # Create integrator
    integrator = FLAbValidationQAIntegrator()
    
    # Example models for validation
    regressor = RandomForestRegressor(n_estimators=10, random_state=42)
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Validate models
    validation_results_reg = integrator.validate_model(X_train, y_train_regression, 
                                                      X_test, y_test_regression, 
                                                      regressor, 'regression')
    print(f"Regression Validation - MSE: {validation_results_reg['mse']:.3f}")
    
    validation_results_clf = integrator.validate_model(X_train, y_train_classification, 
                                                     X_test, y_test_classification, 
                                                     classifier, 'classification')
    print(f"Classification Validation - Accuracy: {validation_results_clf['accuracy']:.3f}")
    
    # Detect drift
    drift_results = integrator.detect_drift(X_train[:50], X_current)
    print(f"Drift Detected: {'Yes' if drift_results['drift_detected'] else 'No'}")
    
    # QA submission
    qa_results = integrator.qa_submission(predictions)
    print(f"QA Score: {qa_results['qa_score']:.3f}")
    
    # Generate reports
    validation_report = integrator.generate_validation_report(X_train, y_train_regression, 
                                                            X_test, y_test_regression, 
                                                            regressor, 'regression')
    print("\nValidation Report:")
    print(validation_report['summary'])
    
    drift_report = integrator.generate_drift_report(X_train[:50], X_current)
    print("\nDrift Report:")
    print(drift_report['summary'])
    
    qa_report = integrator.generate_qa_report(predictions)
    print("\nQA Report:")
    print(qa_report['summary'])
    
    print("\nFLAb Validation QA Integrator test completed successfully!")


if __name__ == '__main__':
    main()

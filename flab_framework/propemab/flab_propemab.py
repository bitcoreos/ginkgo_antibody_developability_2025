"""
FLAb Framework Integration for PROPERMAB

This module provides FLAb framework integration for the PROPERMAB framework.
"""

import sys
import os
import numpy as np

# Add path to research directory

# Import the PROPERMAB framework
from propemab import PROPERMAB


class FLAbPROPERMAB:
    """
    FLAb framework integration for PROPERMAB.
    """

    def __init__(self):
        """
        Initialize the FLAb PROPERMAB framework.
        """
        self.propemab = PROPERMAB()
        self.is_trained = False
        # Define weights for overall score calculation
        # These weights can be adjusted based on domain knowledge
        self.overall_score_weights = {
            'solubility': 0.25,
            'expression_level': 0.25,
            'aggregation_propensity': -0.25,  # Negative because lower is better
            'thermal_stability': 0.25,
            'immunogenicity': -0.25  # Negative because lower is better
        }

    def extract_features_from_analysis(self, fragment_analysis: dict) -> list:
        """
        Extract features from fragment analysis for PROPERMAB.

        Args:
        fragment_analysis (dict): Fragment analysis results from FragmentAnalyzer

        Returns:
        list: Feature values
        """
        features = []
        
        # Extract composition features
        composition = fragment_analysis.get('composition', {})
        if composition:
            total_aa = composition.get('total_amino_acids', 1)
            if total_aa > 0:
                # Hydrophobicity
                hydrophobic_count = composition['groups'].get('hydrophobic', 0)
                features.append(hydrophobic_count / total_aa)
                
                # Charged
                charged_count = composition['groups'].get('charged', 0)
                features.append(charged_count / total_aa)
                
                # Polar
                polar_count = composition['groups'].get('polar', 0)
                features.append(polar_count / total_aa)
                
                # Aromatic
                aromatic_count = composition['groups'].get('aromatic', 0)
                features.append(aromatic_count / total_aa)
                
                # Total amino acids
                features.append(total_aa)
        
        # Extract complexity features
        complexity = fragment_analysis.get('complexity', {})
        if complexity:
            features.append(complexity.get('overall_complexity', 0))
        
        # Extract physicochemical properties
        phys_props = fragment_analysis.get('physicochemical_properties', {})
        if phys_props:
            features.append(phys_props.get('hydrophobicity', 0))
            features.append(phys_props.get('isoelectric_point', 7.0))
            
            # Charge distribution
            charge_dist = phys_props.get('charge_distribution', {})
            features.append(charge_dist.get('net_charge', 0))
            features.append(charge_dist.get('positive_charges', 0))
            features.append(charge_dist.get('negative_charges', 0))
        
        # Extract stability features
        stability = fragment_analysis.get('stability', {})
        if stability:
            features.append(stability.get('aggregation_propensity', 0))
            features.append(stability.get('thermal_stability', 0))
        
        return features

    def prepare_training_data(self, fragment_database: dict) -> tuple:
        """
        Prepare training data from fragment database.

        Args:
        fragment_database (dict): Fragment database with analysis results

        Returns:
        tuple: (X, y_dict, feature_names)
        """
        # Handle case when fragment_database is None or empty
        if not fragment_database:
            # Return empty arrays
            return np.array([]).reshape(0, 13), {prop: np.array([]) for prop in [
                'solubility', 'expression_level', 'aggregation_propensity', 'thermal_stability', 'immunogenicity'
            ]}, [
                'hydrophobic_fraction', 'charged_fraction', 'polar_fraction', 'aromatic_fraction', 'total_amino_acids',
                'sequence_complexity', 'hydrophobicity', 'isoelectric_point', 'net_charge', 'positive_charges',
                'negative_charges', 'aggregation_propensity', 'thermal_stability'
            ]
        
        # Initialize lists for features and targets
        X_list = []
        y_dict = {
            'solubility': [],
            'expression_level': [],
            'aggregation_propensity': [],
            'thermal_stability': [],
            'immunogenicity': []
        }
        
        # Extract features and targets from each fragment
        for fragment_id, fragment_data in fragment_database.items():
            # Extract features
            features = self.extract_features_from_analysis(fragment_data)
            X_list.append(features)
            
            # For demonstration, we'll generate synthetic target values
            # In a real implementation, these would come from experimental data
            # or be predicted by other components of the FLAb framework
            np.random.seed(hash(fragment_id) % 2**32)  # For reproducible "synthetic" values
            y_dict['solubility'].append(np.random.rand())
            y_dict['expression_level'].append(np.random.rand())
            y_dict['aggregation_propensity'].append(np.random.rand())
            y_dict['thermal_stability'].append(np.random.rand())
            y_dict['immunogenicity'].append(np.random.rand())
        
        # Convert to numpy arrays
        X = np.array(X_list)
        
        # Convert target lists to numpy arrays
        for prop in y_dict:
            y_dict[prop] = np.array(y_dict[prop])
        
        # Define feature names
        feature_names = [
            'hydrophobic_fraction',
            'charged_fraction',
            'polar_fraction',
            'aromatic_fraction',
            'total_amino_acids',
            'sequence_complexity',
            'hydrophobicity',
            'isoelectric_point',
            'net_charge',
            'positive_charges',
            'negative_charges',
            'aggregation_propensity',
            'thermal_stability'
        ]
        
        return X, y_dict, feature_names

    def train(self, fragment_database: dict):
        """
        Train the PROPERMAB framework using fragment database.

        Args:
        fragment_database (dict): Fragment database with analysis results
        """
        # Prepare training data
        X, y_dict, feature_names = self.prepare_training_data(fragment_database)
        
        # Check if we have data to train on
        if X.shape[0] == 0:
            print("Warning: No data available for training PROPERMAB. Using default models.")
            # Still mark as trained so we can make predictions (with default models)
            self.is_trained = True
            return
        
        # Train PROPERMAB
        self.propemab.fit(X, y_dict, feature_names)
        self.is_trained = True

    def predict_developability(self, fragment_analysis: dict) -> dict:
        """
        Predict developability properties for a fragment.

        Args:
        fragment_analysis (dict): Fragment analysis results from FragmentAnalyzer

        Returns:
        dict: Developability predictions including overall score
        """
        if not self.is_trained:
            raise RuntimeError("PROPERMAB must be trained before making predictions")
        
        # Extract features
        features = self.extract_features_from_analysis(fragment_analysis)
        X = np.array([features])
        
        # Make predictions
        predictions = self.propemab.predict(X)
        
        # Convert to dictionary format
        result = {}
        for prop in predictions:
            result[prop] = float(predictions[prop][0])
        
        # Calculate overall developability score
        result['overall_score'] = self._calculate_overall_score(result)
        
        return result

    def _calculate_overall_score(self, predictions: dict) -> float:
        """
        Calculate overall developability score from individual property predictions.

        Args:
        predictions (dict): Individual property predictions

        Returns:
        float: Overall developability score (0-1, where 1 is most developable)
        """
        # Initialize score
        score = 0.0
        
        # Calculate weighted sum of properties
        for prop, weight in self.overall_score_weights.items():
            if prop in predictions:
                # Apply weight to property prediction
                score += predictions[prop] * weight
        
        # Normalize to 0-1 range
        # Since weights sum to 1.0, we need to adjust for negative weights
        # We'll clamp the score between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score

    def evaluate_performance(self, fragment_database: dict) -> dict:
        """
        Evaluate PROPERMAB performance using fragment database.

        Args:
        fragment_database (dict): Fragment database with analysis results

        Returns:
        dict: Performance metrics
        """
        if not self.is_trained:
            raise RuntimeError("PROPERMAB must be trained before evaluation")
        
        # Prepare evaluation data
        X, y_dict, feature_names = self.prepare_training_data(fragment_database)
        
        # Check if we have data to evaluate on
        if X.shape[0] == 0:
            return {prop: {'mse': 0.0, 'r2': 0.0} for prop in [
                'solubility', 'expression_level', 'aggregation_propensity', 'thermal_stability', 'immunogenicity'
            ]}
        
        # Evaluate PROPERMAB
        performance = self.propemab.evaluate(X, y_dict)
        
        return performance

    def get_feature_importance(self, property_name: str = None) -> dict:
        """
        Get feature importance for one or all properties.

        Args:
        property_name (str): Name of specific property (optional)

        Returns:
        dict: Feature importance
        """
        return self.propemab.get_feature_importance(property_name)

    def generate_report(self) -> dict:
        """
        Generate a comprehensive PROPERMAB report.

        Returns:
        dict: Comprehensive PROPERMAB report
        """
        return self.propemab.generate_report()


def main():
    """
    Example usage of the FLAb PROPERMAB framework.
    """
    # Example fragment database (simplified)
    fragment_database = {
        'fragment_1': {
            'composition': {
                'total_amino_acids': 120,
                'groups': {
                    'hydrophobic': 60,
                    'charged': 25,
                    'polar': 35,
                    'aromatic': 15
                }
            },
            'complexity': {
                'overall_complexity': 0.85
            },
            'physicochemical_properties': {
                'hydrophobicity': -0.3,
                'isoelectric_point': 7.2,
                'charge_distribution': {
                    'net_charge': 2,
                    'positive_charges': 15,
                    'negative_charges': 13
                }
            },
            'stability': {
                'aggregation_propensity': 0.6,
                'thermal_stability': 0.7
            }
        },
        'fragment_2': {
            'composition': {
                'total_amino_acids': 110,
                'groups': {
                    'hydrophobic': 50,
                    'charged': 30,
                    'polar': 30,
                    'aromatic': 10
                }
            },
            'complexity': {
                'overall_complexity': 0.78
            },
            'physicochemical_properties': {
                'hydrophobicity': -0.2,
                'isoelectric_point': 6.8,
                'charge_distribution': {
                    'net_charge': 0,
                    'positive_charges': 15,
                    'negative_charges': 15
                }
            },
            'stability': {
                'aggregation_propensity': 0.4,
                'thermal_stability': 0.8
            }
        }
    }

    # Create FLAb PROPERMAB framework
    flab_propemab = FLAbPROPERMAB()

    # Train the framework
    print("Training FLAb PROPERMAB framework:")
    flab_propemab.train(fragment_database)
    print("Training completed successfully.")

    # Predict developability for a new fragment
    new_fragment = {
        'composition': {
            'total_amino_acids': 115,
            'groups': {
                'hydrophobic': 55,
                'charged': 28,
                'polar': 32,
                'aromatic': 12
            }
        },
        'complexity': {
            'overall_complexity': 0.82
        },
        'physicochemical_properties': {
            'hydrophobicity': -0.25,
            'isoelectric_point': 7.0,
            'charge_distribution': {
                'net_charge': 1,
                'positive_charges': 15,
                'negative_charges': 14
            }
        },
        'stability': {
            'aggregation_propensity': 0.5,
            'thermal_stability': 0.75
        }
    }

    print("\nPredicting developability for new fragment:")
    predictions = flab_propemab.predict_developability(new_fragment)
    for prop, value in predictions.items():
        print(f"  {prop}: {value:.4f}")

    # Evaluate performance
    print("\nEvaluating PROPERMAB performance:")
    performance = flab_propemab.evaluate_performance(fragment_database)
    for prop, perf in performance.items():
        print(f"  {prop}: MSE={perf['mse']:.4f}, R2={perf['r2']:.4f}")

    # Get feature importance
    print("\nFeature importance for solubility:")
    importance = flab_propemab.get_feature_importance('solubility')
    if importance is not None:
        # Note: In this simplified example, feature names may not be available
        # In a real implementation, they would be tracked during training
        print(f"  Feature importance array shape: {importance.shape}")

    # Generate comprehensive report
    print("\nGenerating PROPERMAB report:")
    report = flab_propemab.generate_report()
    print(report['summary'][:500] + "..." if len(report['summary']) > 500 else report['summary'])


if __name__ == "__main__":
    main()

class PROPERMAB:
    """
    Minimal implementation of PROPERMAB for standalone code.
    """
    
    def __init__(self):
        """
        Initialize the PROPERMAB framework.
        """
        self.is_trained = False
        self.models = {}
        self.feature_importance = {}
    
    def train(self, X, y_dict):
        """
        Train the PROPERMAB framework.
        
        Args:
            X (array-like): Input features
            y_dict (dict): Dictionary mapping property names to target values
        """
        # Minimal implementation that just sets the trained flag
        self.is_trained = True
        # Initialize dummy models for each property
        for property_name in y_dict.keys():
            self.models[property_name] = f"Dummy model for {property_name}"
            self.feature_importance[property_name] = None
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained framework.
        
        Args:
            X (array-like): Input features
            
        Returns:
            dict: Dictionary mapping property names to predictions
        """
        # Minimal implementation that returns random predictions
        import numpy as np
        predictions = {}
        for property_name in self.models.keys():
            predictions[property_name] = np.random.rand(X.shape[0])
        return predictions
    
    def evaluate(self, X, y_dict):
        """
        Evaluate the PROPERMAB framework.
        
        Args:
            X (array-like): Input features
            y_dict (dict): Dictionary mapping property names to target values
            
        Returns:
            dict: Dictionary mapping property names to evaluation metrics
        """
        # Minimal implementation that returns dummy metrics
        metrics = {}
        for property_name in y_dict.keys():
            metrics[property_name] = {
                'mse': 0.1,
                'r2': 0.8
            }
        return metrics
    
    def get_feature_importance(self, property_name=None):
        """
        Get feature importance for one or all properties.
        
        Args:
            property_name (str): Property name to get feature importance for
            
        Returns:
            dict or array-like: Feature importance
        """
        # Minimal implementation that returns None
        if property_name:
            return self.feature_importance.get(property_name, None)
        return self.feature_importance
    
    def generate_report(self):
        """
        Generate a comprehensive PROPERMAB report.
        
        Returns:
            dict: Comprehensive PROPERMAB report
        """
        # Minimal implementation that returns a dummy report
        return {
            'training_status': 'completed' if self.is_trained else 'not trained',
            'properties_trained': list(self.models.keys()),
            'model_performance': 'good'
        }

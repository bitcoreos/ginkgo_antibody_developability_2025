"""
FLAb Framework Integration for Multi-Task Learning

This module provides FLAb framework integration for the Multi-Task Learning model.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Union, Tuple

# Add the research directory to the Python path
# sys.path.insert(0, '/a0/bitcore/workspace/research/advanced_learning_techniques/src')

# Import the Multi-Task Learning model
from multi_task_learning import MultiTaskLearningModel


class FLAbMultiTaskLearning:
    """
    FLAb framework integration for Multi-Task Learning.
    """

    def __init__(self):
        """
        Initialize the FLAb Multi-Task Learning framework.
        """
        self.mtl_model = None
        self.is_trained = False
        self.feature_names = []
        
        # Define the tasks for antibody developability prediction
        self.tasks = {
            'solubility': 1,
            'expression_yield': 1,
            'aggregation_propensity': 1,
            'thermal_stability': 1
        }

    def extract_features_from_analysis(self, fragment_analysis: dict) -> list:
        """
        Extract features from fragment analysis for Multi-Task Learning.

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
                'solubility', 'expression_yield', 'aggregation_propensity', 'thermal_stability'
            ]}, [
                'hydrophobic_fraction', 'charged_fraction', 'polar_fraction', 'aromatic_fraction', 'total_amino_acids',
                'sequence_complexity', 'hydrophobicity', 'isoelectric_point', 'net_charge', 'positive_charges',
                'negative_charges', 'aggregation_propensity', 'thermal_stability'
            ]

        # Initialize lists for features and targets
        X_list = []
        y_dict = {
            'solubility': [],
            'expression_yield': [],
            'aggregation_propensity': [],
            'thermal_stability': []
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
            y_dict['expression_yield'].append(np.random.rand())
            y_dict['aggregation_propensity'].append(np.random.rand())
            y_dict['thermal_stability'].append(np.random.rand())

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

    def train(self, fragment_database: dict, input_dim: int = 13, hidden_dim: int = 20, epochs: int = 100,
              learning_rate: float = 0.01):
        """
        Train the Multi-Task Learning framework using fragment database.

        Args:
            fragment_database (dict): Fragment database with analysis results
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
        """
        # Prepare training data
        X, y_dict, feature_names = self.prepare_training_data(fragment_database)
        
        # Store feature names
        self.feature_names = feature_names

        # Check if we have data to train on
        if X.shape[0] == 0:
            print("Warning: No data available for training Multi-Task Learning. Using default models.")
            # Still mark as trained so we can make predictions (with default models)
            self.is_trained = True
            return

        # Create Multi-Task Learning Model
        self.mtl_model = MultiTaskLearningModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dims=self.tasks)

        # Train Multi-Task Learning Model
        self.mtl_model.train(X, y_dict, epochs=epochs, learning_rate=learning_rate)
        self.is_trained = True

    def predict_developability(self, fragment_analysis: dict) -> dict:
        """
        Predict developability properties for a fragment using Multi-Task Learning.

        Args:
            fragment_analysis (dict): Fragment analysis results from FragmentAnalyzer

        Returns:
            dict: Developability predictions
        """
        if not self.is_trained:
            raise RuntimeError("Multi-Task Learning must be trained before making predictions")

        # Extract features
        features = self.extract_features_from_analysis(fragment_analysis)
        X = np.array([features])

        # Make predictions
        predictions = self.mtl_model.predict(X)

        # Convert to dictionary format
        result = {}
        for prop in predictions:
            result[prop] = float(predictions[prop][0])

        return result

    def evaluate_performance(self, fragment_database: dict) -> dict:
        """
        Evaluate Multi-Task Learning performance using fragment database.

        Args:
            fragment_database (dict): Fragment database with analysis results

        Returns:
            dict: Performance metrics
        """
        if not self.is_trained:
            raise RuntimeError("Multi-Task Learning must be trained before evaluation")

        # Prepare evaluation data
        X, y_dict, feature_names = self.prepare_training_data(fragment_database)

        # Check if we have data to evaluate on
        if X.shape[0] == 0:
            return {prop: 0.0 for prop in [
                'solubility', 'expression_yield', 'aggregation_propensity', 'thermal_stability'
            ]}

        # Evaluate Multi-Task Learning
        performance = self.mtl_model.evaluate(X, y_dict)

        return performance

    def get_training_loss(self, task_name: str = None) -> Union[List[float], Dict[str, List[float]]]:
        """
        Get training loss history.

        Args:
            task_name (str): Specific task to get loss for (if None, get all tasks)

        Returns:
            Union[List[float], Dict[str, List[float]]]: Training loss history
        """
        if not self.is_trained:
            raise RuntimeError("Multi-Task Learning must be trained before getting training loss")

        return self.mtl_model.get_training_loss(task_name)

    def generate_report(self) -> dict:
        """
        Generate a comprehensive Multi-Task Learning report.

        Returns:
            dict: Comprehensive Multi-Task Learning report
        """
        if not self.is_trained:
            raise RuntimeError("Multi-Task Learning must be trained before generating report")

        return self.mtl_model.generate_report()


def main():
    """
    Example usage of the FLAb Multi-Task Learning framework.
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

    # Create FLAb Multi-Task Learning framework
    flab_mtl = FLAbMultiTaskLearning()

    # Train the framework
    print("Training FLAb Multi-Task Learning framework:")
    flab_mtl.train(fragment_database, epochs=50, learning_rate=0.01)
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
    predictions = flab_mtl.predict_developability(new_fragment)
    for prop, value in predictions.items():
        print(f"  {prop}: {value:.4f}")

    # Evaluate performance
    print("\nEvaluating Multi-Task Learning performance:")
    performance = flab_mtl.evaluate_performance(fragment_database)
    for prop, mse in performance.items():
        print(f"  {prop} MSE: {mse:.4f}")

    # Get training loss
    print("\nTraining Loss History:")
    for task_name in ['solubility', 'expression_yield', 'aggregation_propensity', 'thermal_stability']:
        training_loss = flab_mtl.get_training_loss(task_name)
        print(f"  {task_name} final loss: {training_loss[-1] if training_loss else 'N/A'}")

    # Generate comprehensive report
    print("\nGenerating Multi-Task Learning report:")
    report = flab_mtl.generate_report()
    print(report['summary'][:500] + "..." if len(report['summary']) > 500 else report['summary'])


if __name__ == "__main__":
    main()

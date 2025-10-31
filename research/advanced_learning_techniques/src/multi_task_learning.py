"""
Multi-Task Learning Implementation

This module implements a simplified version of Multi-Task Learning
for antibody developability prediction.
"""

import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler


class MultiTaskLearningModel:
    """
    Simplified Multi-Task Learning implementation for antibody developability prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dims: Dict[str, int]):
        """
        Initialize the Multi-Task Learning Model.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dims (Dict[str, int]): Dictionary mapping task names to output dimensions
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims
        self.task_names = list(output_dims.keys())

        # Initialize shared hidden layer weights
        self.shared_weights = {
            'w1': np.random.randn(input_dim, hidden_dim) * 0.1,
            'b1': np.random.randn(hidden_dim) * 0.1
        }

        # Initialize task-specific output layer weights
        self.task_weights = {}
        for task_name, output_dim in output_dims.items():
            self.task_weights[task_name] = {
                'w2': np.random.randn(hidden_dim, output_dim) * 0.1,
                'b2': np.random.randn(output_dim) * 0.1
            }

        # Track training status
        self.is_trained = False
        self.training_loss = {task_name: [] for task_name in self.task_names}

    def forward_pass(self, x: np.ndarray, task_name: str = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through the neural network.

        Args:
            x (np.ndarray): Input features
            task_name (str): Specific task to predict (if None, predict all tasks)

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Output predictions for specified task or all tasks
        """
        # Shared hidden layer
        hidden = x @ self.shared_weights['w1'] + self.shared_weights['b1']
        # ReLU activation
        hidden = np.maximum(0, hidden)

        if task_name is not None:
            # Specific task output
            output = hidden @ self.task_weights[task_name]['w2'] + self.task_weights[task_name]['b2']
            # If output dimension is 1, flatten the result
            if self.output_dims[task_name] == 1:
                output = output.flatten()
            return output
        else:
            # All tasks output
            outputs = {}
            for task in self.task_names:
                outputs[task] = hidden @ self.task_weights[task]['w2'] + self.task_weights[task]['b2']
                # If output dimension is 1, flatten the result
                if self.output_dims[task] == 1:
                    outputs[task] = outputs[task].flatten()
            return outputs

    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute loss between predictions and targets.

        Args:
            predictions (np.ndarray): Predictions
            targets (np.ndarray): Targets

        Returns:
            float: Loss value
        """
        # Mean squared error
        loss = np.mean((predictions - targets) ** 2)
        return loss

    def backward_pass(self, x: np.ndarray, targets: Dict[str, np.ndarray],
                       predictions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """
        Backward pass to compute gradients.

        Args:
            x (np.ndarray): Input features
            targets (Dict[str, np.ndarray]): Target values for each task
            predictions (Dict[str, np.ndarray]): Predictions for each task

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]: Shared gradients, task-specific gradients
        """
        # Shared hidden layer
        hidden = x @ self.shared_weights['w1'] + self.shared_weights['b1']
        hidden = np.maximum(0, hidden)  # ReLU

        # Initialize gradients
        shared_gradients = {
            'dw1': np.zeros_like(self.shared_weights['w1']),
            'db1': np.zeros_like(self.shared_weights['b1'])
        }

        task_gradients = {}

        # Compute gradients for each task
        for task_name in self.task_names:
            # Ensure predictions and targets have the correct shape
            pred = predictions[task_name]
            targ = targets[task_name]

            # Reshape if necessary for single output tasks
            if self.output_dims[task_name] == 1:
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                if targ.ndim == 1:
                    targ = targ.reshape(-1, 1)

            # Output layer gradients
            output_error = pred - targ

            # Gradients for output layer
            dw2 = hidden.T @ output_error / x.shape[0]
            db2 = np.mean(output_error, axis=0)

            task_gradients[task_name] = {
                'dw2': dw2,
                'db2': db2
            }

            # Gradients for shared hidden layer
            # Reshape output_error if necessary for matrix multiplication
            if output_error.ndim == 1:
                output_error = output_error.reshape(-1, 1)

            hidden_error = output_error @ self.task_weights[task_name]['w2'].T
            hidden_error[hidden <= 0] = 0  # ReLU derivative

            # Accumulate shared gradients
            shared_gradients['dw1'] += x.T @ hidden_error / x.shape[0]
            shared_gradients['db1'] += np.mean(hidden_error, axis=0)

        # Average shared gradients across tasks
        shared_gradients['dw1'] /= len(self.task_names)
        shared_gradients['db1'] /= len(self.task_names)

        return shared_gradients, task_gradients

    def update_weights(self, shared_gradients: Dict[str, np.ndarray],
                       task_gradients: Dict[str, Dict[str, np.ndarray]],
                       learning_rate: float = 0.01) -> None:
        """
        Update weights using gradients.

        Args:
            shared_gradients (Dict[str, np.ndarray]): Shared layer gradients
            task_gradients (Dict[str, Dict[str, np.ndarray]]): Task-specific gradients
            learning_rate (float): Learning rate
        """
        # Update shared weights
        self.shared_weights['w1'] -= learning_rate * shared_gradients['dw1']
        self.shared_weights['b1'] -= learning_rate * shared_gradients['db1']

        # Update task-specific weights
        for task_name in self.task_names:
            self.task_weights[task_name]['w2'] -= learning_rate * task_gradients[task_name]['dw2']
            self.task_weights[task_name]['b2'] -= learning_rate * task_gradients[task_name]['db2']

    def predict(self, x: np.ndarray, task_name: str = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions using the trained model.

        Args:
            x (np.ndarray): Input data
            task_name (str): Specific task to predict (if None, predict all tasks)

        Returns:
            Union[np.ndarray, Dict[str, np.ndarray]]: Predictions for specified task or all tasks
        """
        if not self.is_trained:
            print("Warning: Model is not trained yet.")
            if task_name is not None:
                return np.zeros((x.shape[0], self.output_dims[task_name]))
            else:
                return {task: np.zeros((x.shape[0], dim)) for task, dim in self.output_dims.items()}

        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        return self.forward_pass(x, task_name)

    def train(self, x: np.ndarray, y: Dict[str, np.ndarray], epochs: int = 100,
              learning_rate: float = 0.01):
        """
        Train the multi-task model.

        Args:
            x (np.ndarray): Input data
            y (Dict[str, np.ndarray]): Target values for each task
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
        """
        print(f"Training multi-task model for {epochs} epochs...")

        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward_pass(x)

            # Compute total loss
            total_loss = 0
            for task_name in self.task_names:
                task_loss = self.compute_loss(predictions[task_name], y[task_name])
                self.training_loss[task_name].append(task_loss)
                total_loss += task_loss

            # Backward pass
            shared_gradients, task_gradients = self.backward_pass(x, y, predictions)

            # Update weights
            self.update_weights(shared_gradients, task_gradients, learning_rate)

            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.6f}")
                for task_name in self.task_names:
                    print(f"  {task_name} Loss: {self.training_loss[task_name][-1]:.6f}")

        self.is_trained = True
        print("Multi-task model training completed.")

    def evaluate(self, x: np.ndarray, y: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the Multi-Task Learning Model.

        Args:
            x (np.ndarray): Input data
            y (Dict[str, np.ndarray]): Target values for each task

        Returns:
            Dict[str, float]: Evaluation metrics for each task
        """
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # Make predictions
        predictions = self.forward_pass(x)

        # Compute metrics for each task
        metrics = {}
        for task_name in self.task_names:
            mse = mean_squared_error(y[task_name], predictions[task_name])
            metrics[task_name] = mse

        return metrics

    def get_training_loss(self, task_name: str = None) -> Union[List[float], Dict[str, List[float]]]:
        """
        Get training loss history.

        Args:
            task_name (str): Specific task to get loss for (if None, get all tasks)

        Returns:
            Union[List[float], Dict[str, List[float]]]: Training loss history
        """
        if task_name is not None:
            return self.training_loss[task_name]
        else:
            return self.training_loss

    def generate_report(self) -> Dict[str, Union[str, Dict, bool]]:
        """
        Generate a comprehensive Multi-Task Learning report.

        Returns:
            Dict: Comprehensive Multi-Task Learning report
        """
        # Generate summary
        summary = "Multi-Task Learning Model Report\n"
        summary += "===============================\n\n"

        # Add configuration
        summary += f"Configuration:\n"
        summary += f"- Input Dimension: {self.input_dim}\n"
        summary += f"- Hidden Dimension: {self.hidden_dim}\n"
        summary += f"- Tasks: {', '.join(self.task_names)}\n"
        summary += f"- Trained: {self.is_trained}\n"

        # Add training loss if available
        if self.is_trained:
            summary += "\nFinal Training Loss:\n"
            for task_name in self.task_names:
                if self.training_loss[task_name]:
                    summary += f"- {task_name}: {self.training_loss[task_name][-1]:.6f}\n"

        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'tasks': self.task_names,
            'output_dims': self.output_dims,
            'is_trained': self.is_trained,
            'training_loss': {task: losses[-1] if losses else None for task, losses in self.training_loss.items()},
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the Multi-Task Learning implementation.
    """
    # Generate example data
    np.random.seed(42)
    samples = 1000
    input_dim = 10

    # Define tasks and their output dimensions
    tasks = {
        'solubility': 1,
        'expression_yield': 1,
        'aggregation_propensity': 1,
        'thermal_stability': 1
    }

    # Generate data
    x = np.random.rand(samples, input_dim)
    y = {task: np.random.rand(samples, dim) for task, dim in tasks.items()}

    # Create Multi-Task Learning Model
    mtl_model = MultiTaskLearningModel(input_dim=input_dim, hidden_dim=20, output_dims=tasks)

    # Print initial information
    print("Multi-Task Learning Example")
    print("==========================")
    print(f"Samples: {samples}")
    print(f"Input dimension: {input_dim}")
    print(f"Tasks: {', '.join(tasks.keys())}")

    # Train the model
    mtl_model.train(x, y, epochs=100, learning_rate=0.01)

    # Make predictions for all tasks
    print("\nMaking predictions for all tasks:")
    test_x = np.random.rand(20, input_dim)
    all_predictions = mtl_model.predict(test_x)
    for task_name, predictions in all_predictions.items():
        print(f"  {task_name} predictions shape: {predictions.shape}")
        print(f"  First 3 {task_name} predictions: {predictions[:3].flatten()}")

    # Make predictions for a specific task
    print("\nMaking predictions for specific task (solubility):")
    solubility_predictions = mtl_model.predict(test_x, task_name='solubility')
    print(f"Solubility predictions shape: {solubility_predictions.shape}")
    print(f"First 3 solubility predictions: {solubility_predictions[:3].flatten()}")

    # Evaluate the model
    print("\nEvaluating Multi-Task Learning Model:")
    test_y = {task: np.random.rand(20, dim) for task, dim in tasks.items()}
    metrics = mtl_model.evaluate(test_x, test_y)
    for task_name, mse in metrics.items():
        print(f"  {task_name} MSE: {mse:.6f}")

    # Get training loss
    print("\nTraining Loss History:")
    for task_name in tasks.keys():
        training_loss = mtl_model.get_training_loss(task_name)
        print(f"  {task_name} final loss: {training_loss[-1] if training_loss else 'N/A'}")

    # Generate comprehensive report
    print("\nMulti-Task Learning Model Report Summary:")
    report = mtl_model.generate_report()
    print(report['summary'])


if __name__ == "__main__":
    main()

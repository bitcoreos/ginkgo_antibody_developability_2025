"""
Active Learning Implementation

This module implements a simplified version of Active Learning 
for antibody developability prediction.
"""

import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler


class ActiveLearningModel:
    """
    Simplified Active Learning implementation for antibody developability prediction.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 acquisition_strategy: str = 'uncertainty'):
        """
        Initialize the Active Learning Model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension
            acquisition_strategy (str): Strategy for selecting samples ('uncertainty', 'random', 'margin')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.acquisition_strategy = acquisition_strategy
        
        # Initialize model weights
        self.weights = {
            'w1': np.random.randn(input_dim, hidden_dim) * 0.1,
            'b1': np.random.randn(hidden_dim) * 0.1,
            'w2': np.random.randn(hidden_dim, output_dim) * 0.1,
            'b2': np.random.randn(output_dim) * 0.1
        }
        
        # Track training status
        self.is_trained = False
        self.training_loss = []
        self.labeled_indices = []
        self.unlabeled_indices = []
    
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        
        Args:
            x (np.ndarray): Input features
            
        Returns:
            np.ndarray: Output predictions
        """
        # First layer
        hidden = x @ self.weights['w1'] + self.weights['b1']
        # ReLU activation
        hidden = np.maximum(0, hidden)
        
        # Output layer
        output = hidden @ self.weights['w2'] + self.weights['b2']
        
        return output
    
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
    
    def backward_pass(self, x: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients.
        
        Args:
            x (np.ndarray): Input features
            targets (np.ndarray): Target values
            predictions (np.ndarray): Predictions
            
        Returns:
            Dict[str, np.ndarray]: Gradients
        """
        # Compute output layer gradients
        output_error = predictions - targets
        hidden = x @ self.weights['w1'] + self.weights['b1']
        hidden = np.maximum(0, hidden)  # ReLU
        
        # Gradients for output layer
        dw2 = hidden.T @ output_error / x.shape[0]
        db2 = np.mean(output_error, axis=0)
        
        # Gradients for hidden layer
        hidden_error = output_error @ self.weights['w2'].T
        hidden_error[hidden <= 0] = 0  # ReLU derivative
        
        dw1 = x.T @ hidden_error / x.shape[0]
        db1 = np.mean(hidden_error, axis=0)
        
        return {
            'dw1': dw1,
            'db1': db1,
            'dw2': dw2,
            'db2': db2
        }
    
    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float = 0.01) -> None:
        """
        Update weights using gradients.
        
        Args:
            gradients (Dict[str, np.ndarray]): Gradients
            learning_rate (float): Learning rate
        """
        self.weights['w1'] -= learning_rate * gradients['dw1']
        self.weights['b1'] -= learning_rate * gradients['db1']
        self.weights['w2'] -= learning_rate * gradients['dw2']
        self.weights['b2'] -= learning_rate * gradients['db2']
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            print("Warning: Model is not trained yet.")
            return np.zeros((x.shape[0], self.output_dim))
        
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        return self.forward_pass(x)
    
    def uncertainty_sampling(self, x: np.ndarray, num_samples: int = 10) -> List[int]:
        """
        Select samples with highest prediction uncertainty.
        
        Args:
            x (np.ndarray): Unlabeled data
            num_samples (int): Number of samples to select
            
        Returns:
            List[int]: Indices of selected samples
        """
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        # For regression, we can use prediction variance as uncertainty
        # In this simplified implementation, we'll use the absolute value of predictions as a proxy
        predictions = self.forward_pass(x)
        uncertainty = np.abs(predictions)
        
        # Select samples with highest uncertainty
        selected_indices = np.argsort(uncertainty.flatten())[-num_samples:]
        
        return selected_indices.tolist()
    
    def random_sampling(self, x: np.ndarray, num_samples: int = 10) -> List[int]:
        """
        Randomly select samples.
        
        Args:
            x (np.ndarray): Unlabeled data
            num_samples (int): Number of samples to select
            
        Returns:
            List[int]: Indices of selected samples
        """
        # Randomly select samples
        selected_indices = np.random.choice(len(x), size=min(num_samples, len(x)), replace=False)
        
        return selected_indices.tolist()
    
    def margin_sampling(self, x: np.ndarray, num_samples: int = 10) -> List[int]:
        """
        Select samples with smallest margin between top two predictions.
        For regression, we'll use a simplified approach based on prediction confidence.
        
        Args:
            x (np.ndarray): Unlabeled data
            num_samples (int): Number of samples to select
            
        Returns:
            List[int]: Indices of selected samples
        """
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        # For regression, we'll use the absolute value of predictions as confidence
        predictions = self.forward_pass(x)
        confidence = np.abs(predictions)
        
        # Select samples with lowest confidence (smallest margin)
        selected_indices = np.argsort(confidence.flatten())[:num_samples]
        
        return selected_indices.tolist()
    
    def select_samples(self, x: np.ndarray, num_samples: int = 10) -> List[int]:
        """
        Select samples for labeling based on acquisition strategy.
        
        Args:
            x (np.ndarray): Unlabeled data
            num_samples (int): Number of samples to select
            
        Returns:
            List[int]: Indices of selected samples
        """
        if self.acquisition_strategy == 'uncertainty':
            return self.uncertainty_sampling(x, num_samples)
        elif self.acquisition_strategy == 'random':
            return self.random_sampling(x, num_samples)
        elif self.acquisition_strategy == 'margin':
            return self.margin_sampling(x, num_samples)
        else:
            print(f"Unknown acquisition strategy: {self.acquisition_strategy}")
            return self.random_sampling(x, num_samples)
    
    def active_learning_loop(self, x_pool: np.ndarray, y_pool: np.ndarray, 
                           x_labeled: np.ndarray, y_labeled: np.ndarray,
                           num_iterations: int = 5, samples_per_iteration: int = 10,
                           epochs_per_iteration: int = 50, learning_rate: float = 0.01):
        """
        Perform active learning loop.
        
        Args:
            x_pool (np.ndarray): Pool of unlabeled data
            y_pool (np.ndarray): True labels for pool data (for evaluation)
            x_labeled (np.ndarray): Initially labeled data
            y_labeled (np.ndarray): Labels for initially labeled data
            num_iterations (int): Number of active learning iterations
            samples_per_iteration (int): Number of samples to select per iteration
            epochs_per_iteration (int): Number of training epochs per iteration
            learning_rate (float): Learning rate
        """
        print(f"Starting Active Learning Loop with {num_iterations} iterations...")
        
        # Initialize labeled and unlabeled sets
        x_current_labeled = x_labeled.copy()
        y_current_labeled = y_labeled.copy()
        unlabeled_indices = list(range(len(x_pool)))
        
        # Standardize pool data
        pool_scaler = StandardScaler()
        x_pool_scaled = pool_scaler.fit_transform(x_pool)
        
        for iteration in range(num_iterations):
            print(f"\nActive Learning Iteration {iteration + 1}/{num_iterations}")
            
            # Train model on current labeled data
            print(f"  Training on {len(x_current_labeled)} labeled samples...")
            
            # Standardize current labeled data
            labeled_scaler = StandardScaler()
            x_current_labeled_scaled = labeled_scaler.fit_transform(x_current_labeled)
            
            # Train for specified epochs
            for epoch in range(epochs_per_iteration):
                # Forward pass
                predictions = self.forward_pass(x_current_labeled_scaled)
                
                # Compute loss
                loss = self.compute_loss(predictions, y_current_labeled)
                
                # Backward pass
                gradients = self.backward_pass(x_current_labeled_scaled, y_current_labeled, predictions)
                
                # Update weights
                self.update_weights(gradients, learning_rate)
            
            self.is_trained = True
            print(f"  Training completed. Final loss: {loss:.6f}")
            
            # Evaluate model on a test set (using some of the pool data)
            test_indices = np.random.choice(len(x_pool_scaled), size=min(100, len(x_pool_scaled)), replace=False)
            x_test = x_pool_scaled[test_indices]
            y_test = y_pool[test_indices]
            test_predictions = self.forward_pass(x_test)
            test_mse = mean_squared_error(y_test, test_predictions)
            print(f"  Test MSE: {test_mse:.6f}")
            
            # Select new samples for labeling
            if len(unlabeled_indices) > 0:
                print(f"  Selecting {samples_per_iteration} new samples for labeling...")
                
                # Get unlabeled data
                x_unlabeled = x_pool_scaled[unlabeled_indices]
                
                # Select samples
                selected_indices_local = self.select_samples(x_unlabeled, samples_per_iteration)
                selected_indices_global = [unlabeled_indices[i] for i in selected_indices_local]
                
                # Add selected samples to labeled set
                x_current_labeled = np.vstack([x_current_labeled, x_pool[selected_indices_global]])
                y_current_labeled = np.vstack([y_current_labeled, y_pool[selected_indices_global]])
                
                # Remove selected samples from unlabeled set
                for idx in sorted(selected_indices_local, reverse=True):
                    unlabeled_indices.pop(idx)
                
                print(f"  Selected samples: {selected_indices_global}")
            else:
                print("  No more unlabeled samples available.")
        
        print("\nActive learning loop completed.")
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the Active Learning Model.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): Target values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        # Make predictions
        predictions = self.forward_pass(x)
        
        # Compute metrics
        mse = mean_squared_error(y, predictions)
        
        return {
            'mse': mse
        }
    
    def get_training_loss(self) -> List[float]:
        """
        Get training loss history.
        
        Returns:
            List[float]: Training loss history
        """
        return self.training_loss
    
    def generate_report(self) -> Dict[str, Union[str, Dict, bool]]:
        """
        Generate a comprehensive Active Learning report.
        
        Returns:
            Dict: Comprehensive Active Learning report
        """
        # Generate summary
        summary = "Active Learning Model Report\n"
        summary += "==========================\n\n"
        
        # Add configuration
        summary += f"Configuration:\n"
        summary += f"- Input Dimension: {self.input_dim}\n"
        summary += f"- Hidden Dimension: {self.hidden_dim}\n"
        summary += f"- Output Dimension: {self.output_dim}\n"
        summary += f"- Acquisition Strategy: {self.acquisition_strategy}\n"
        summary += f"- Trained: {self.is_trained}\n"
        
        # Add training loss if available
        if self.training_loss:
            summary += f"- Final Training Loss: {self.training_loss[-1]:.6f}\n"
        
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'acquisition_strategy': self.acquisition_strategy,
            'is_trained': self.is_trained,
            'training_loss': self.training_loss[-1] if self.training_loss else None,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the Active Learning implementation.
    """
    # Generate example data
    np.random.seed(42)
    total_samples = 1000
    labeled_samples = 50
    input_dim = 10
    output_dim = 1
    
    # Generate all data
    x_all = np.random.rand(total_samples, input_dim)
    y_all = np.random.rand(total_samples, output_dim)
    
    # Split into labeled and unlabeled sets
    x_labeled = x_all[:labeled_samples]
    y_labeled = y_all[:labeled_samples]
    x_pool = x_all[labeled_samples:]
    y_pool = y_all[labeled_samples:]
    
    # Create Active Learning Model
    al_model = ActiveLearningModel(input_dim=input_dim, hidden_dim=20, output_dim=output_dim, 
                                 acquisition_strategy='uncertainty')
    
    # Print initial information
    print("Active Learning Example")
    print("=======================")
    print(f"Total samples: {total_samples}")
    print(f"Initially labeled samples: {labeled_samples}")
    print(f"Unlabeled samples: {len(x_pool)}")
    
    # Perform active learning loop
    al_model.active_learning_loop(x_pool, y_pool, x_labeled, y_labeled,
                                num_iterations=5, samples_per_iteration=10,
                                epochs_per_iteration=50, learning_rate=0.01)
    
    # Make predictions
    print("\nMaking predictions:")
    test_x = np.random.rand(20, input_dim)
    predictions = al_model.predict(test_x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 3 predictions: {predictions[:3].flatten()}")
    
    # Evaluate the model
    print("\nEvaluating Active Learning Model:")
    test_y = np.random.rand(20, output_dim)
    metrics = al_model.evaluate(test_x, test_y)
    print(f"MSE: {metrics['mse']:.6f}")
    
    # Generate comprehensive report
    print("\nActive Learning Model Report Summary:")
    report = al_model.generate_report()
    print(report['summary'])


if __name__ == "__main__":
    main()

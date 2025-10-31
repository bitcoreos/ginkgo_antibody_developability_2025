"""
Transfer Learning Implementation

This module implements a simplified version of Transfer Learning 
for antibody developability prediction.
"""

import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler


class TransferLearningModel:
    """
    Simplified Transfer Learning implementation for antibody developability prediction.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 source_task_dims: Tuple[int, int] = (20, 1)):
        """
        Initialize the Transfer Learning Model.
        
        Args:
            input_dim (int): Input feature dimension for target task
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension for target task
            source_task_dims (Tuple[int, int]): Source task input and output dimensions
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.source_input_dim, self.source_output_dim = source_task_dims
        
        # Initialize source task model weights
        self.source_weights = {
            'w1': np.random.randn(self.source_input_dim, hidden_dim) * 0.1,
            'b1': np.random.randn(hidden_dim) * 0.1,
            'w2': np.random.randn(hidden_dim, self.source_output_dim) * 0.1,
            'b2': np.random.randn(self.source_output_dim) * 0.1
        }
        
        # Initialize target task model weights
        self.target_weights = {
            'w1': np.random.randn(input_dim, hidden_dim) * 0.1,
            'b1': np.random.randn(hidden_dim) * 0.1,
            'w2': np.random.randn(hidden_dim, output_dim) * 0.1,
            'b2': np.random.randn(output_dim) * 0.1
        }
        
        # Track training status
        self.source_trained = False
        self.target_trained = False
        self.source_training_loss = []
        self.target_training_loss = []
    
    def forward_pass(self, x: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Forward pass through the neural network.
        
        Args:
            x (np.ndarray): Input features
            weights (Dict[str, np.ndarray]): Network weights
            
        Returns:
            np.ndarray: Output predictions
        """
        # First layer
        hidden = x @ weights['w1'] + weights['b1']
        # ReLU activation
        hidden = np.maximum(0, hidden)
        
        # Output layer
        output = hidden @ weights['w2'] + weights['b2']
        
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
    
    def backward_pass(self, x: np.ndarray, targets: np.ndarray, predictions: np.ndarray, 
                      weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients.
        
        Args:
            x (np.ndarray): Input features
            targets (np.ndarray): Target values
            predictions (np.ndarray): Predictions
            weights (Dict[str, np.ndarray]): Network weights
            
        Returns:
            Dict[str, np.ndarray]: Gradients
        """
        # Compute output layer gradients
        output_error = predictions - targets
        hidden = x @ weights['w1'] + weights['b1']
        hidden = np.maximum(0, hidden)  # ReLU
        
        # Gradients for output layer
        dw2 = hidden.T @ output_error / x.shape[0]
        db2 = np.mean(output_error, axis=0)
        
        # Gradients for hidden layer
        hidden_error = output_error @ weights['w2'].T
        hidden_error[hidden <= 0] = 0  # ReLU derivative
        
        dw1 = x.T @ hidden_error / x.shape[0]
        db1 = np.mean(hidden_error, axis=0)
        
        return {
            'dw1': dw1,
            'db1': db1,
            'dw2': dw2,
            'db2': db2
        }
    
    def update_weights(self, weights: Dict[str, np.ndarray], 
                       gradients: Dict[str, np.ndarray], 
                       learning_rate: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Update weights using gradients.
        
        Args:
            weights (Dict[str, np.ndarray]): Current weights
            gradients (Dict[str, np.ndarray]): Gradients
            learning_rate (float): Learning rate
            
        Returns:
            Dict[str, np.ndarray]: Updated weights
        """
        updated_weights = {}
        updated_weights['w1'] = weights['w1'] - learning_rate * gradients['dw1']
        updated_weights['b1'] = weights['b1'] - learning_rate * gradients['db1']
        updated_weights['w2'] = weights['w2'] - learning_rate * gradients['dw2']
        updated_weights['b2'] = weights['b2'] - learning_rate * gradients['db2']
        
        return updated_weights
    
    def train_source_task(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, 
                         learning_rate: float = 0.01):
        """
        Train the model on the source task.
        
        Args:
            x (np.ndarray): Source task input data
            y (np.ndarray): Source task target values
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
        """
        print(f"Training source task model for {epochs} epochs...")
        
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward_pass(x, self.source_weights)
            
            # Compute loss
            loss = self.compute_loss(predictions, y)
            self.source_training_loss.append(loss)
            
            # Backward pass
            gradients = self.backward_pass(x, y, predictions, self.source_weights)
            
            # Update weights
            self.source_weights = self.update_weights(self.source_weights, gradients, learning_rate)
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        self.source_trained = True
        print("Source task training completed.")
    
    def transfer_knowledge(self, transfer_layers: List[str] = ['w1', 'b1']):
        """
        Transfer knowledge from source task to target task.
        
        Args:
            transfer_layers (List[str]): List of layer names to transfer
        """
        if not self.source_trained:
            print("Warning: Source task model is not trained yet.")
            return
        
        print("Transferring knowledge from source task to target task...")
        
        # Transfer specified layers
        for layer in transfer_layers:
            if layer in self.source_weights and layer in self.target_weights:
                # Transfer weights, handling dimension differences
                source_shape = self.source_weights[layer].shape
                target_shape = self.target_weights[layer].shape
                
                if source_shape == target_shape:
                    # Same dimensions, direct transfer
                    self.target_weights[layer] = self.source_weights[layer].copy()
                    print(f"  Transferred {layer} directly")
                elif len(source_shape) == 2 and len(target_shape) == 2:
                    # Both are weight matrices, transfer compatible parts
                    min_rows = min(source_shape[0], target_shape[0])
                    min_cols = min(source_shape[1], target_shape[1])
                    self.target_weights[layer][:min_rows, :min_cols] = \
                        self.source_weights[layer][:min_rows, :min_cols]
                    print(f"  Partially transferred {layer} ({min_rows}x{min_cols})")
                elif len(source_shape) == 1 and len(target_shape) == 1:
                    # Both are bias vectors, transfer compatible parts
                    min_size = min(source_shape[0], target_shape[0])
                    self.target_weights[layer][:min_size] = \
                        self.source_weights[layer][:min_size]
                    print(f"  Partially transferred {layer} ({min_size})")
                else:
                    print(f"  Skipping {layer} due to incompatible dimensions")
            else:
                print(f"  Skipping {layer} as it doesn't exist in both models")
    
    def fine_tune_target_task(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, 
                            learning_rate: float = 0.01, freeze_layers: List[str] = []):
        """
        Fine-tune the model on the target task.
        
        Args:
            x (np.ndarray): Target task input data
            y (np.ndarray): Target task target values
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            freeze_layers (List[str]): List of layer names to freeze during fine-tuning
        """
        print(f"Fine-tuning target task model for {epochs} epochs...")
        
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward_pass(x, self.target_weights)
            
            # Compute loss
            loss = self.compute_loss(predictions, y)
            self.target_training_loss.append(loss)
            
            # Backward pass
            gradients = self.backward_pass(x, y, predictions, self.target_weights)
            
            # Freeze specified layers
            for layer in freeze_layers:
                if layer in gradients:
                    gradients[layer] = np.zeros_like(gradients[layer])
            
            # Update weights
            self.target_weights = self.update_weights(self.target_weights, gradients, learning_rate)
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        self.target_trained = True
        print("Target task fine-tuning completed.")
    
    def predict(self, x: np.ndarray, task: str = 'target') -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            x (np.ndarray): Input data
            task (str): Task to predict for ('source' or 'target')
            
        Returns:
            np.ndarray: Predictions
        """
        if task == 'source' and not self.source_trained:
            print("Warning: Source task model is not trained yet.")
            return np.zeros((x.shape[0], self.source_output_dim))
        
        if task == 'target' and not self.target_trained:
            print("Warning: Target task model is not trained yet.")
            return np.zeros((x.shape[0], self.output_dim))
        
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        if task == 'source':
            return self.forward_pass(x, self.source_weights)
        else:  # target
            return self.forward_pass(x, self.target_weights)
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, task: str = 'target') -> Dict[str, float]:
        """
        Evaluate the Transfer Learning Model.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): Target values
            task (str): Task to evaluate ('source' or 'target')
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Standardize input
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        # Make predictions
        if task == 'source':
            predictions = self.forward_pass(x, self.source_weights)
        else:  # target
            predictions = self.forward_pass(x, self.target_weights)
        
        # Compute metrics
        mse = mean_squared_error(y, predictions)
        
        return {
            'mse': mse
        }
    
    def get_source_training_loss(self) -> List[float]:
        """
        Get source task training loss history.
        
        Returns:
            List[float]: Training loss history
        """
        return self.source_training_loss
    
    def get_target_training_loss(self) -> List[float]:
        """
        Get target task training loss history.
        
        Returns:
            List[float]: Training loss history
        """
        return self.target_training_loss
    
    def generate_report(self) -> Dict[str, Union[str, Dict, bool]]:
        """
        Generate a comprehensive Transfer Learning report.
        
        Returns:
            Dict: Comprehensive Transfer Learning report
        """
        # Generate summary
        summary = "Transfer Learning Model Report\n"
        summary += "=============================\n\n"
        
        # Add configuration
        summary += f"Configuration:\n"
        summary += f"- Source Input Dimension: {self.source_input_dim}\n"
        summary += f"- Source Output Dimension: {self.source_output_dim}\n"
        summary += f"- Target Input Dimension: {self.input_dim}\n"
        summary += f"- Target Output Dimension: {self.output_dim}\n"
        summary += f"- Hidden Dimension: {self.hidden_dim}\n"
        summary += f"- Source Trained: {self.source_trained}\n"
        summary += f"- Target Trained: {self.target_trained}\n"
        
        # Add training loss if available
        if self.source_training_loss:
            summary += f"- Final Source Training Loss: {self.source_training_loss[-1]:.6f}\n"
        if self.target_training_loss:
            summary += f"- Final Target Training Loss: {self.target_training_loss[-1]:.6f}\n"
        
        return {
            'source_input_dim': self.source_input_dim,
            'source_output_dim': self.source_output_dim,
            'target_input_dim': self.input_dim,
            'target_output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'source_trained': self.source_trained,
            'target_trained': self.target_trained,
            'source_training_loss': self.source_training_loss[-1] if self.source_training_loss else None,
            'target_training_loss': self.target_training_loss[-1] if self.target_training_loss else None,
            'summary': summary,
            'report_complete': True
        }


def main():
    """
    Example usage of the Transfer Learning implementation.
    """
    # Generate example data for source and target tasks
    np.random.seed(42)
    source_samples = 1000
    target_samples = 100
    source_input_dim = 20
    target_input_dim = 15
    output_dim = 1
    
    # Source task data (e.g., predicting solubility)
    source_x = np.random.rand(source_samples, source_input_dim)
    source_y = np.random.rand(source_samples, output_dim)
    
    # Target task data (e.g., predicting expression yield, related but different)
    target_x = np.random.rand(target_samples, target_input_dim)
    target_y = np.random.rand(target_samples, output_dim)
    
    # Create Transfer Learning Model
    tl_model = TransferLearningModel(input_dim=target_input_dim, hidden_dim=20, output_dim=output_dim, 
                                    source_task_dims=(source_input_dim, output_dim))
    
    # Print initial information
    print("Transfer Learning Example")
    print("=========================")
    print(f"Source task samples: {source_samples}")
    print(f"Target task samples: {target_samples}")
    print(f"Source input dimension: {source_input_dim}")
    print(f"Target input dimension: {target_input_dim}")
    
    # Train source task
    tl_model.train_source_task(source_x, source_y, epochs=100, learning_rate=0.01)
    
    # Transfer knowledge
    tl_model.transfer_knowledge(transfer_layers=['w1', 'b1'])
    
    # Fine-tune on target task
    tl_model.fine_tune_target_task(target_x, target_y, epochs=100, learning_rate=0.01)
    
    # Make predictions
    print("\nMaking predictions:")
    source_predictions = tl_model.predict(source_x[:20], task='source')
    target_predictions = tl_model.predict(target_x[:20], task='target')
    print(f"Source predictions shape: {source_predictions.shape}")
    print(f"Target predictions shape: {target_predictions.shape}")
    print(f"First 3 source predictions: {source_predictions[:3].flatten()}")
    print(f"First 3 target predictions: {target_predictions[:3].flatten()}")
    
    # Evaluate the model
    print("\nEvaluating Transfer Learning Model:")
    source_metrics = tl_model.evaluate(source_x[:20], source_y[:20], task='source')
    target_metrics = tl_model.evaluate(target_x[:20], target_y[:20], task='target')
    print(f"Source MSE: {source_metrics['mse']:.6f}")
    print(f"Target MSE: {target_metrics['mse']:.6f}")
    
    # Get training loss
    source_training_loss = tl_model.get_source_training_loss()
    target_training_loss = tl_model.get_target_training_loss()
    print(f"Final source training loss: {source_training_loss[-1] if source_training_loss else 'N/A'}")
    print(f"Final target training loss: {target_training_loss[-1] if target_training_loss else 'N/A'}")
    
    # Generate comprehensive report
    print("\nTransfer Learning Model Report Summary:")
    report = tl_model.generate_report()
    print(report['summary'])


if __name__ == "__main__":
    main()

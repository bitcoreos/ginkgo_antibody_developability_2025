# Neural-ODEs Documentation

## Overview

This document provides detailed documentation for the Neural-ODEs (Neural Ordinary Differential Equations) implementation. Neural-ODEs are used for modeling temporal dynamics of developability risks, implementing continuous-depth models that can capture complex temporal patterns in antibody developability.

## Features

1. **Continuous Dynamics Modeling**: Models temporal dynamics using continuous differential equations
2. **Neural Network Dynamics**: Uses neural networks to parameterize the dynamics function
3. **ODE Solving**: Solves ordinary differential equations using numerical integration
4. **Temporal Prediction**: Predicts future states based on learned dynamics
5. **Training and Evaluation**: Comprehensive training and evaluation capabilities
6. **Performance Tracking**: Maintains training loss history

## Implementation Details

### NeuralODE Class

The `NeuralODE` class is the core of the implementation:

```python
neural_ode = NeuralODE(hidden_layers=[10, 10], learning_rate=0.01)
```

#### Methods

- `dynamics_function(state, t)`: Dynamics function that defines the ODE
- `fit(time_points, states)`: Fit the Neural-ODE to training data
- `solve(time_points, initial_state)`: Solve the ODE to get state trajectories
- `predict_future(future_time_points, initial_state)`: Predict future states using the trained Neural-ODE
- `get_training_loss()`: Get training loss history
- `generate_report()`: Generate a comprehensive Neural-ODE report

### Continuous Dynamics Modeling

The implementation models temporal dynamics using continuous differential equations:

1. **State Representation**: Represents system state as a vector of variables
2. **Dynamics Function**: Neural network that defines how state changes over time
3. **ODE Formulation**: Models dynamics as ds/dt = f(s, t) where s is state and f is dynamics function

```python
def dynamics_function(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
    # Predict derivatives using the neural network
    state_reshaped = state.reshape(1, -1)
    derivatives = self.dynamics_network.predict(state_reshaped)
    return derivatives.flatten()
```

### Neural Network Dynamics

The implementation uses a neural network to parameterize the dynamics function:

1. **MLP Regressor**: Multi-layer perceptron for learning dynamics
2. **Configurable Architecture**: Customizable hidden layer sizes
3. **Training Process**: Fits network to state derivatives calculated from training data

```python
self.dynamics_network = MLPRegressor(
    hidden_layer_sizes=hidden_layers,
    learning_rate_init=learning_rate,
    max_iter=1000,
    random_state=42
)
```

### ODE Solving

The implementation solves ODEs using numerical integration:

1. **Scipy Integration**: Uses scipy's odeint for solving ODEs
2. **Initial Value Problems**: Solves with given initial state
3. **Time Point Specification**: Solves at specified time points

```python
solution = odeint(self.dynamics_function, y0, time_points)
```

### Temporal Prediction

The implementation can predict future states based on learned dynamics:

1. **Future Time Points**: Specify future time points for prediction
2. **State Trajectories**: Generate complete state trajectories
3. **Initial State Specification**: Use custom initial state for prediction

```python
future_states = neural_ode.predict_future(future_time_points)
```

## Usage Example

```python
from src.neural_odes import NeuralODE
import numpy as np

# Generate example temporal data
np.random.seed(42)
time_points = np.linspace(0, 10, 100)

# Generate example state trajectories
state_1 = np.exp(-0.1 * time_points) + 0.1 * np.random.randn(len(time_points))
state_2 = np.exp(-0.05 * time_points) + 0.1 * np.random.randn(len(time_points))

# Combine states
states = np.column_stack([state_1, state_2])

# Split data for training and testing
train_time = time_points[:80]
train_states = states[:80]
test_time = time_points[80:]
test_states = states[80:]

# Create Neural-ODE
neural_ode = NeuralODE(hidden_layers=[20, 20], learning_rate=0.01)

# Fit the Neural-ODE
neural_ode.fit(train_time, train_states)

# Solve ODE on test time points
predicted_states = neural_ode.solve(test_time)

# Calculate prediction error
mse = mean_squared_error(test_states, predicted_states)
print(f"Prediction MSE: {mse:.6f}")

# Predict future states
future_time = np.linspace(10, 15, 20)
future_states = neural_ode.predict_future(future_time)
print(f"Future states shape: {future_states.shape}")

# Generate comprehensive report
report = neural_ode.generate_report()
print(report['summary'])
```

## Integration with FLAb Framework

This implementation can be integrated with the existing FLAb framework by:

1. Using Neural-ODEs in the DevelopabilityPredictor for temporal risk modeling
2. Incorporating temporal dynamics into the OptimizationRecommender for time-aware optimization
3. Using state trajectories in the FragmentAnalyzer for dynamic property analysis
4. Generating Neural-ODE reports as part of the analysis pipeline

## Future Enhancements

1. **Advanced ODE Solvers**: Integration with more sophisticated ODE solvers
2. **Deep Learning Dynamics**: Using deep neural networks for dynamics functions
3. **Stochastic Dynamics**: Modeling stochastic temporal processes
4. **Multi-State Systems**: Handling complex multi-state dynamical systems
5. **Real-Time Prediction**: Implementing real-time state prediction
6. **Parameter Estimation**: Estimating model parameters from data
7. **Control Systems**: Adding control inputs to the dynamical system

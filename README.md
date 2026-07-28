
---

## 🐍 `Lab-1.py` — Copy & Paste

```python
"""
Deep Learning Lab 1 - Perceptron using NumPy
=============================================

A simple implementation of the Perceptron Algorithm to simulate
the AND Logic Gate using NumPy.

Author: Divyansh
"""

import numpy as np


def step_function(x: float) -> int:
    """
    Step activation function.
    
    Returns 1 if x >= 0, otherwise returns 0.
    
    Args:
        x: Weighted sum of inputs and bias
        
    Returns:
        Binary output (0 or 1)
    """
    return 1 if x >= 0 else 0


def train_perceptron(
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 10,
    verbose: bool = True
) -> tuple[np.ndarray, float, list]:
    """
    Train a single-layer perceptron using the perceptron learning rule.
    
    Args:
        inputs: Input dataset of shape (n_samples, n_features)
        targets: Expected outputs of shape (n_samples,)
        learning_rate: Step size for weight updates (default: 0.1)
        epochs: Number of training iterations (default: 10)
        verbose: Whether to print training progress (default: True)
        
    Returns:
        Tuple of (trained_weights, trained_bias, training_history)
    """
    n_samples, n_features = inputs.shape
    
    # Initialize weights and bias with small random values
    np.random.seed(42)  # For reproducibility
    weights = np.random.randn(n_features) * 0.1
    bias = 0.0
    
    history = []
    
    if verbose:
        print("=" * 50)
        print("      PERCEPTRON TRAINING STARTED")
        print("=" * 50)
        print(f"\nHyperparameters:")
        print(f"  Learning Rate : {learning_rate}")
        print(f"  Epochs        : {epochs}")
        print(f"  Samples       : {n_samples}")
        print(f"\nInitial Weights: {weights}")
        print(f"Initial Bias   : {bias:.4f}")
        print("\n" + "-" * 50)
    
    # Training loop
    for epoch in range(epochs):
        total_error = 0
        
        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
        
        for i in range(n_samples):
            # Forward pass: Compute weighted sum
            weighted_sum = np.dot(inputs[i], weights) + bias
            
            # Apply activation function
            prediction = step_function(weighted_sum)
            
            # Calculate error
            error = targets[i] - prediction
            total_error += abs(error)
            
            # Update weights and bias (Perceptron Learning Rule)
            weights += learning_rate * error * inputs[i]
            bias += learning_rate * error
            
            if verbose:
                status = "✓ Correct" if error == 0 else f"✗ Error={error}"
                print(f"  Sample {i+1}: {inputs[i]} | "
                      f"Target={targets[i]}, Pred={prediction} | {status}")
        
        history.append(total_error)
        
        if verbose:
            print(f"  Total Errors in Epoch {epoch + 1}: {total_error}")
            
        # Early stopping if perfectly classified
        if total_error == 0:
            if verbose:
                print(f"\n🎉 Converged at Epoch {epoch + 1}!")
            break
    
    return weights, bias, history


def predict(inputs: np.ndarray, weights: np.ndarray, bias: float) -> list[int]:
    """
    Make predictions using trained perceptron.
    
    Args:
        inputs: Input dataset
        weights: Trained weights
        bias: Trained bias
        
    Returns:
        List of predictions
    """
    predictions = []
    for x in inputs:
        weighted_sum = np.dot(x, weights) + bias
        predictions.append(step_function(weighted_sum))
    return predictions


def main():
    """Main execution function."""
    
    # ============================================
    # AND Gate Dataset
    # ============================================
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    targets = np.array([0, 0, 0, 1])
    
    # ============================================
    # Train the Perceptron
    # ============================================
    weights, bias, history = train_perceptron(
        inputs=inputs,
        targets=targets,
        learning_rate=0.1,
        epochs=10,
        verbose=True
    )
    
    # ============================================
    # Display Results
    # ============================================
    print("\n" + "=" * 50)
    print("      PERCEPTRON TRAINING COMPLETE")
    print("=" * 50)
    print(f"\nFinal Weights: {weights}")
    print(f"Final Bias   : {bias:.4f}")
    
    # ============================================
    # Make Predictions
    # ============================================
    predictions = predict(inputs, weights, bias)
    
    print("\n" + "-" * 40)
    print("           PREDICTIONS")
    print("-" * 40)
    
    correct = 0
    for i in range(len(inputs)):
        status = "✓" if predictions[i] == targets[i] else "✗"
        print(f"Input: {inputs[i]} -> Output: {predictions[i]}  {status}")
        if predictions[i] == targets[i]:
            correct += 1
    
    accuracy = (correct / len(inputs)) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    
    # ============================================
    # Decision Boundary Visualization (Text)
    # ============================================
    print("\n" + "=" * 50)
    print("      DECISION BOUNDARY ANALYSIS")
    print("=" * 50)
    print(f"\nDecision Boundary Equation:")
    print(f"  {weights[0]:.2f}*x₁ + {weights[1]:.2f}*x₂ + {bias:.2f} = 0")
    print(f"\n  => x₂ = {-weights[0]/weights[1]:.2f}*x₁ + {-bias/weights[1]:.2f}")
    print("\nThis line separates Class 0 from Class 1 in the 2D input space.")


if __name__ == "__main__":
    main()

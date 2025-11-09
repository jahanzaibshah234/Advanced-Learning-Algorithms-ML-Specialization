"""Project: Backprop on a Tiny Neural Net
We'll build and train (manually, no TensorFlow yet) a network:

Structure

Inputs: 2 features (x₁, x₂)

Hidden layer: 2 neurons (with ReLU)

Output: 1 neuron (with sigmoid → binary classification)

Goal

Implement forward pass

Compute loss (binary cross-entropy)

Do backpropagation manually with derivatives

Update weights"""

import numpy as np

# Sample Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]]) # OR - truth table

# Initialize Parameters
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.1
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1) * 0.1 
b2 = np.zeros((1, 1))

# Activations Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Training loop
learning_rate = 0.1
for epoch in range(1000):
    # Forward Pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Loss (binary cross-entropy)
    loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

    # Backpropagation
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / len(X)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)


    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1) / len(X)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
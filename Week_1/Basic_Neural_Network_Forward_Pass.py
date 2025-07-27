import numpy as np

# ReLU Activation
def relu(z):
    return np.maximum(0, z)

# Input (3 Features)
X = np.random.randn(3, 1)

# Initialize Parameters
W1 = np.random.randn(4, 3)
b1 = np.random.randn(4, 1)
W2 = np.random.randn(2, 4)
b2 = np.random.randn(2, 1)
W3 = np.random.randn(1, 2)
b3 = np.random.randn(1, 1)

# Forward Pass
# First Hidden Layer (4 neurons)
Z1 = np.dot(W1, X) + b1
A1 = relu(Z1)

# Second Hidden Layer (2 neurons)
Z2 = np.dot(W2, A1) + b2
A2 = relu(Z2)

# Output Layer (1 neurons)
Z3 = np.dot(W3, A2) + b3
A3 = Z3 # No activation(output: regression) or use sigmoid if classification

print("Output:", A3)
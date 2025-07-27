import numpy as np

# ReLU Activation
def relu(z):
    return np.maximum(0, z)

# Input (3 Features)
X = np.array([1.0, 2.0, 3.0])

# First hidden layer (4 neurons)
W1 = np.random.randn(4, 3)
b1 = np.random.randn(4)
Z1 = np.dot(W1, X) + b1
A1 = relu(Z1)

# Second hidden layer (3 neurons)
W2 = np.random.randn(3, 4)
b2 = np.random.randn(3)
Z2 = np.dot(W2, A1) + b2
A2 = relu(Z2)

# Output layer (1 neuron)
W3 = np.random.randn(1, 3)
b3 = np.random.randn(1)
Z3 = np.dot(W3, A2) + b3
A3 = relu(Z3)

print("Output:", A3)
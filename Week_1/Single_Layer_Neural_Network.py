import numpy as np

# Input Vector (3-Features)
X = np.array([1.0, 2.0, 3.0])

# Weights: 2 neurons, each with 3 weights
W = np.array([
    [0.2, 0.4, 0.6], # Weight for Neuron 1
    [0.5, 0.3, 0.1]  # Weight for Neuron 2
])


# Biases for 2 Neuron
b = np.array([0.1, -0.2])

# ReLU activation
def relu(z):
    return np.maximum(0, z)

# Layer Output
Z = np.dot(W, X) + b
R = relu(Z)

print("Z (Weighted sum):", Z)
print("A (After ReLU):", R)




import numpy as np

def relu(z):
    return np.maximum(0, z)

def neuron_layer(inputs, weights, bias):
    z = np.dot(weights, inputs) + bias
    return relu(z)

inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.5, 0.1]
bias = 0.4

output = neuron_layer(inputs, weights, bias)
print("Neuron Output:", output)
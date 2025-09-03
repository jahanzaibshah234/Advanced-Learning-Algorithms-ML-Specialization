import numpy as np

def softmax(z):
    stable_z = z - np.max(z)
    ez = np.exp(stable_z)
    sm = ez / np.sum(ez)
    return sm

def neuron_layer(inputs, weights, bias):
    z = np.dot(weights, inputs) + bias
    return softmax(z)

inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([[0.2, 0.5, 0.1],
                   [0.4, 0.6, 0.1],
                   [0.1, 0.3, 0.2]])
bias = np.array([0.4, 0.1, 0.2])

output = neuron_layer(inputs, weights, bias)
print("Neuron Output:", output)
print("Sum of Output:", np.sum(output))


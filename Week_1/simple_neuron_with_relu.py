def relu(z):
    return max(0, z)

def simple_neuron(inputs, weights, bias):
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    return relu(z)

inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.5, 0.1]
bias = 0.4

output = simple_neuron(inputs, weights, bias)
print("Neuron Output:", output)
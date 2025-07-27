import numpy as np

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction Function
def predict_dog_or_cat(image_pixels, weights, bias):
    z = sum(w * x for w, x in zip(weights, image_pixels)) + bias
    return sigmoid(z)


# Example with 3 pixels only (for simplicity)
image = [0.2, 0.5, 0.9]  # Example pixel values (normalized 0–1)
weight = [0.6, -0.3, 0.8] # Random Weights
bias = -0.4 # Random Bias

output = predict_dog_or_cat(image, weight, bias)

# Classify
prediction = 1 if output > 0.5 else 0
print(f"Predicted Class: {'Cat' if prediction == 1 else 'Dog'}")
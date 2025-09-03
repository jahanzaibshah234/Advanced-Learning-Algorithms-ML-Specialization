""" Project: MNIST Multiclass Classification with Softmax

-----------------------------------------------------
Trains a neural network to recognize handwritten digits 0 to 9
from the MNIST dataset."""

# Hides all TF logs except errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
(train_images, train_label), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() 

# Normalize 
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_images, train_label, epochs=5, validation_split=0.1, shuffle=True)

# Evaluate Model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Accuracy: {accuracy*100:.2f}")

# Prediction
probs = model.predict(test_images[:1])
print("Probabilities:", probs[0])
print("Predicted class:", np.argmax(probs[0]))
print("True label:", test_labels[0])

# Show the image
plt.imshow(test_images[0], cmap='gray')
plt.title(f"True: {test_labels[0]}, Predicted: {np.argmax(probs)}")
plt.show()
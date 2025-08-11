""" Project: MNIST Binary Digit Classification (0 vs 1) using TensorFlow

-----------------------------------------------------
Trains a neural network to recognize handwritten digits 0 and 1
from the MNIST dataset, and visualizes performance."""

# Hides all TF logs except errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import requried libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Filter Digits 0 and 1 Only
train_filter = np.where((train_labels == 0) | (train_labels == 1))
test_filter = np.where((test_labels == 0) | (test_labels == 1))

train_images, train_labels = train_images[train_filter], train_labels[train_filter]
test_images, test_labels = test_images[test_filter], test_labels[test_filter]

# Normalize Data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Visualize Few Images
indices = np.random.choice(len(train_images), 5, replace=False)
plt.figure(figsize=(5, 5))

for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(f"Label: {train_labels[idx]}")
    plt.axis("off")
plt.show()

# Build Neural Network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(), # Flatten 28x28 → 784
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

#Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model & Save History
history = model.fit(train_images, train_labels, epochs=10, verbose=1, validation_data=(test_images, test_labels))

# Evaluate Model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {accuracy * 100:.2f}%")



# Make Predictions
predictions = model.predict(test_images[:10])
for i, prob in enumerate(predictions):
    print(f"Image {i} - Prob(1): {prob[0]:.4f} - Actual: {test_labels[i]}")


# Plot Accuracy & Loss
plt.figure(figsize=(10, 4))

# Accuracy 
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)


# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Show Predictions on Test Images
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    prob = predictions[i][0]
    plt.title(f"{prob:.2f}")
    plt.axis('off')
plt.suptitle("Predicted Probabilities for '1'")
plt.show()



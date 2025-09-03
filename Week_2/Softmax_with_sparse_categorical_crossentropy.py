import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

# Example data (3 classes)
X = np.array([[1, 2], [2, 1], [3, 3]], dtype=float)
y_sparse = np.array([0, 1, 2])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\nTraining with sparse_categorical_crossentropy:")
model.fit(X, y_sparse, epochs=5, verbose=1)

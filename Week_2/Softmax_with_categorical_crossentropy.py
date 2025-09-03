import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

# Example data (3 classes)
X = np.array([[1, 2], [2, 1], [3, 3]], dtype=float)
y_onehot = np.array([
    [1,0,0],   # class 0
    [0,1,0],   # class 1
    [0,0,1]    # class 2
])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Training with categorical_crossentropy:")
model.fit(X, y_onehot, epochs=5, verbose=1)

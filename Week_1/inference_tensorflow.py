import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np


# Define a simple model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model with x and y data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)
model.fit(x, y, epochs=100, verbose=0)

# Inference(Prediction)
print(model.predict(np.array([6.0])))


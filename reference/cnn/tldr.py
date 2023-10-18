import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

"""
This script shows how to generally build and use a CNN model in Keras.
It should not be used as a reference for any specific details. For that, see the other scripts in this directory.
"""

input_shape = (28, 28, 1)  # TODO
x_train = None  # TODO
y_train = None  # TODO

model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=15,
)

history_df = pd.DataFrame(history.history)

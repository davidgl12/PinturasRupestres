import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_and_resize

DATA_DIR = "./data/train"
SAVE_PATH = "./models/autoencoder.h5"


def build_autoencoder():
    input_img = layers.Input(shape=(256, 256, 3))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder


def load_training_data():
    imgs = []
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".jpg"):
            imgs.append(load_and_resize(os.path.join(DATA_DIR, filename)))
    imgs = np.array(imgs) / 255.0
    return imgs


if __name__ == "__main__":
    X = load_training_data()

    autoencoder = build_autoencoder()
    autoencoder.fit(X, X, epochs=20, batch_size=8, shuffle=True)

    autoencoder.save(SAVE_PATH)
    print("Modelo guardado en:", SAVE_PATH)

import cv2
import numpy as np
import tensorflow as tf
from utils import *
import os

MODEL_PATH = "models/autoencoder.h5"
DATA_DIR = "data/diario"
REF_PATH = "data/referencia.jpg"

autoencoder = tf.keras.models.load_model(MODEL_PATH)


def detect_change(img_ref, img_today):
    img_ref_aligned = align_images(img_ref, img_today)

    inp_ref = img_ref_aligned / 255.0
    inp_today = img_today / 255.0

    rec_ref = autoencoder.predict(np.expand_dims(inp_ref, 0))[0]
    rec_today = autoencoder.predict(np.expand_dims(inp_today, 0))[0]

    error_map = np.mean(np.abs(rec_today - rec_ref), axis=2)
    error_map = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return error_map


if __name__ == "__main__":
    img_ref = load_and_resize(REF_PATH)

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".jpg"):
            continue

        img_today = load_and_resize(os.path.join(DATA_DIR, filename))

        error_map = detect_change(img_ref, img_today)

        heatmap = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            load_and_resize(os.path.join(DATA_DIR, filename)), 0.6, heatmap, 0.4, 0
        )

        out_path = f"output/heatmaps/{filename}"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if np.mean(error_map) > 40:
            alert_path = f"output/alerts/{filename}"
            cv2.imwrite(alert_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print("⚠️ Alerta de deterioro en:", filename)

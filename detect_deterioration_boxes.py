import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
import os


def normalize_light(test, reference):
    return match_histograms(test, reference, channel_axis=-1)


def detect_deterioration_with_boxes(reference_path, test_path, output_path=None):
    # Cargar imágen de referencia e imágen diaria
    ref = cv2.imread(reference_path)
    test = cv2.imread(test_path)

    if ref is None or test is None:
        raise ValueError("No se pudieron cargar las imágenes")

    # Cortar las imágenes al mismo tamaño
    ref = cv2.resize(ref, (test.shape[1], test.shape[0]))

    # Convertir a escala de grises
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # Normalizar la iluminación
    test_gray = normalize_light(test_gray, ref_gray)

    cv2.imwrite("./test_gray.jpg", test_gray)
    cv2.imwrite("./ref_gray.jpg", ref_gray)

    # Calcular SSIM
    score, diff = ssim(ref_gray, test_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold para encontrar diferencias
    thresh = cv2.threshold(diff, 230, 255, cv2.THRESH_BINARY_INV)[1]

    # encontrar los contornos de las areas diferentes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    marked = test.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Ignorar ruido muy pequeño
        if w * h < 200:
            continue

        boxes.append((x, y, w, h))
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Guardar imagen
    if output_path:
        cv2.imwrite(output_path, marked)

    return score, boxes, marked


if __name__ == "__main__":
    score, boxes, marked_img = detect_deterioration_with_boxes(
        "./data/referencia.jpg",
        "./data/comparar.jpg",
        output_path="PINTURAMARCADA.jpg",
    )

    print("SSIM:", score)
    print("Zonas con deterioro detectadas:")

    for b in boxes:
        print("x, y, w, h =", b)

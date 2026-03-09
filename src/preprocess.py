# src/preprocess.py

import cv2
import numpy as np


def load_image(path):
    """
    Carga una imagen en escala de grises desde el path
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")

    return img


def split_digits(image):
    """
    Divide una matrícula en 4 segmentos iguales.
    Cada segmento se redimensiona a 28x28 y se normaliza.

    Devuelve una lista de 4 imágenes listas para la CNN.
    """
    # Ancho estimado de cada dígito
    width = image.shape[1] // 4

    digits = []

    for i in range(4):
        # Recortar la parte correspondiente al dígito i
        digit = image[:, i * width:(i + 1) * width]

        # Redimensionar a tamaño MNIST
        digit = cv2.resize(digit, (28, 28))

        # Normalizar a [0,1]
        digit = digit.astype(np.float32) / 255.0

        # Añadir a la lista
        digits.append(digit)

    return digits
import numpy as np
import cv2

# FUNCIÓN PARA CARGAR UNA IMAGEN DESDE DISCO
def load_image(path):

    # Leer la imagen
    # cv2.IMREAD_GRAYSCALE convierte la imagen a escala de grises
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Si la imagen no se pudo cargar:
    # se lanza un error para evitar continuar con datos inválidos
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")

    # Devolver la imagen cargada
    return img

# FUNCIÓN PARA DIVIDIR UNA MATRÍCULA EN 4 DÍGITOS
def split_digits(image):

    # Calcular el ancho de cada dígito
    # La matrícula completa tiene 4 dígitos
    width = image.shape[1] // 4

    # Lista donde se almacenarán los dígitos extraídos
    digits = []

    # Recorrer las 4 posiciones de la matrícula
    for i in range(4):

        # Extraer la región correspondiente al dígito i
        digit = image[:, i*width:(i+1)*width]

        # Redimensionar el dígito a 28x28 píxeles (MNIST)
        digit = cv2.resize(digit, (28,28))

        # Convertir la imagen a un vector de 784 valores (28x28)
        digit = digit.flatten()

        # Normalizar los valores de los píxeles a rango [0,1]
        digit = digit / 255.0

        # Añadir el dígito procesado a la lista
        digits.append(digit)

    # Devolver la lista con los 4 dígitos preparados para el clasificador
    return digits
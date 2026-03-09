# src/lector_matriculas.py

import numpy as np

from src.preprocess import split_digits
from src.classifier import predict_digit
from src.utils import validar_matricula


def leer_digitos(imagen):
    """
    Recibe una imagen de matrícula y devuelve:
    - plate: string con la matrícula leída
    - valido: True/False según si la matrícula es válida
    """

    # 1)Comprobaciones de imagen

    # El sistema espera una imagen de 28x112
    # (4 dígitos de 28x28 colocados horizontalmente)
    if imagen is None or imagen.shape != (28, 112):
        return "", False

    # Imagen completamente oscura
    if np.mean(imagen) < 5:
        return "", False

    # Imagen completamente blanca / saturada
    if np.mean(imagen) > 250:
        return "", False

    # Imagen con muy poco contraste
    if np.std(imagen) < 10:
        return "", False

    # 2) Segmentación en 4

    digits = split_digits(imagen)

    if len(digits) != 4:
        return "", False

    plate = ""

    # 3) Clasificación dígito a dígito

    for d in digits:
        pred, conf, entropy, is_invalid = predict_digit(d)

        # Si el modelo cree que no es un dígito válido
        if is_invalid:
            return "", False

        # Si la confianza es baja, rechazamos
        if conf < 0.80:
            return "", False

        # Si la incertidumbre es alta, rechazamos
        if entropy > 1.20:
            return "", False

        # Si pasa todos los filtros, añadimos el dígito
        plate += str(pred)

    # 4) Validación

    valido = validar_matricula(plate)

    return plate, valido
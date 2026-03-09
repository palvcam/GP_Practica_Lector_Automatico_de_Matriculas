# src/classifier.py

import os
import numpy as np
import tensorflow as tf

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta del modelo entrenado
MODEL_PATH = os.path.join(BASE_DIR, "models", "models", "digit_cnn_invalid.keras")

# Clase especial para inválido
INVALID_CLASS = 10

# Cargar el modelo una sola vez
model = tf.keras.models.load_model(MODEL_PATH)


def predictive_entropy(probs, eps=1e-12):
    """
    Calcula la entropía de la distribución de probabilidades.
    Cuanto mayor es, más incertidumbre tiene el modelo.
    """
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def predict_digit(digit):
    """
    Recibe una imagen 28x28 ya normalizada y devuelve:
    - predicción
    - confianza
    - entropía
    - si es inválido
    """
    # Asegurar formato correcto para la CNN
    digit = np.array(digit, dtype=np.float32).reshape(1, 28, 28, 1)

    # Predicción del modelo
    probs = model.predict(digit, verbose=0)[0]

    # Clase predicha
    pred = int(np.argmax(probs))

    # Confianza = probabilidad máxima
    confidence = float(np.max(probs))

    # Incertidumbre
    entropy = predictive_entropy(probs)

    # ¿El modelo lo considera inválido?
    is_invalid = (pred == INVALID_CLASS)

    return pred, confidence, entropy, is_invalid
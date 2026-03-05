import joblib
import os
import numpy as np

# __file__ : ruta del archivo actual
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "digit_model.pkl")

# CARGA DEL MODELO ENTRENADO
# Así se evita tener que entrenarlo cada vez que se ejecuta el sistema
model = joblib.load(MODEL_PATH)

# FUNCIÓN PARA PREDECIR UN DÍGITO
def predict_digit(digit):

    # Convertir el dígito a un array de numpy
    # y asegurar que tenga forma (1, 784)
    # ya que el modelo espera una matriz de muestras
    digit = np.array(digit).reshape(1, -1)

    # Obtener las probabilidades de pertenecer a cada clase (0-9)
    probs = model.predict_proba(digit)[0]

    # Obtener la confianza de la predicción
    # es decir, la probabilidad más alta
    confidence = max(probs)

    # Obtener el dígito predicho
    # se selecciona la clase con mayor probabilidad
    pred = model.classes_[np.argmax(probs)]

    # Devolver:
    # -el dígito predicho
    # -la confianza del modelo
    return pred, confidence
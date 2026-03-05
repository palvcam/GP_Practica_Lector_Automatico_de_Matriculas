from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

print("Loading MNIST")

# Descarga del dataset MNIST desde OpenML
# MNIST contiene imágenes de dígitos escritos a mano (0-9) de tamaño 28x28
# Cada imagen se representa como un vector de 784 píxeles (28x28)
X, y = fetch_openml(
    "mnist_784",    # nombre del dataset
    version=1,            # versión del dataset
    return_X_y=True,      # devuelve directamente datos (X) y etiquetas (y)
    as_frame=False        # devuelve arrays de numpy en lugar de dataframes
)

# Normalización de los valores de los píxeles
# Los píxeles originalmente están entre 0 y 255
# Se escalan a valores entre 0 y 1 para mejorar el entrenamiento del modelo
X = X / 255.0

print("Training model")

# Creación del modelo de clasificación
# Se utiliza Regresión Logística
model = LogisticRegression(max_iter=1000)

# Entrenamiento del modelo utilizando el dataset MNIST
# El modelo aprende a asociar patrones de píxeles con los dígitos correspondientes
model.fit(X, y)

# Guardado del modelo entrenado en un archivo
# Esto permite reutilizar el modelo sin necesidad de volver a entrenarlo
joblib.dump(model, "digit_model.pkl")

print("Model saved")
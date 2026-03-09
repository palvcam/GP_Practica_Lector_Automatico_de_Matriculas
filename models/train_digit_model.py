# train_digit_model.py

import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Semillas para que el experimento sea reproducible
np.random.seed(42)
tf.random.set_seed(42)

# Tamaño esperado de cada dígito
IMG_SIZE = 28

# Clase extra para "dígito inválido"
INVALID_CLASS = 10


# =========================================================
# FUNCIONES PARA CREAR EJEMPLOS INVALID
# =========================================================

def add_occlusion(img):
    """
    Tapa una región rectangular aleatoria de la imagen.
    Esto simula un dígito parcialmente oculto.
    """
    img = img.copy()
    h, w = img.shape

    # Tamaño aleatorio de la oclusión
    occ_h = np.random.randint(6, 16)
    occ_w = np.random.randint(6, 16)

    # Posición aleatoria
    y = np.random.randint(0, h - occ_h + 1)
    x = np.random.randint(0, w - occ_w + 1)

    # La zona tapada puede ser negra o blanca
    value = np.random.choice([0.0, 1.0])
    img[y:y + occ_h, x:x + occ_w] = value

    return img


def add_noise(img):
    """
    Añade ruido gaussiano a la imagen.
    Esto simula una captura de mala calidad.
    """
    img = img.copy()

    # Ruido gaussiano
    noise = np.random.normal(0, 0.35, img.shape)

    img = img + noise

    # Limitar valores al rango [0,1]
    img = np.clip(img, 0.0, 1.0)

    return img


def shift_digit(img):
    """
    Desplaza el dígito unos píxeles en horizontal y vertical.
    Esto simula que el dígito esté descentrado.
    """
    img = img.copy()

    # Desplazamientos aleatorios
    dx = np.random.randint(-6, 7)
    dy = np.random.randint(-6, 7)

    # Transformación afín
    shifted = tf.keras.preprocessing.image.apply_affine_transform(
        img[..., np.newaxis],
        tx=dy,
        ty=dx,
        fill_mode="constant",
        cval=0.0
    )

    return shifted[..., 0]


def make_blank():
    """
    Genera una imagen completamente negra o blanca.
    """
    value = np.random.choice([0.0, 1.0])
    return np.full((IMG_SIZE, IMG_SIZE), value, dtype=np.float32)


def make_random_noise():
    """
    Genera una imagen aleatoria sin estructura de dígito.
    """
    return np.random.rand(IMG_SIZE, IMG_SIZE).astype(np.float32)


def build_invalid_samples(X_digits, n_invalid):
    """
    Genera ejemplos artificiales para la clase INVALID.

    Estrategias usadas:
    - dígito tapado
    - dígito con ruido
    - dígito desplazado
    - imagen completamente vacía
    - imagen aleatoria
    """
    invalids = []
    n_source = len(X_digits)

    for _ in range(n_invalid):
        mode = np.random.choice([
            "occluded",
            "noisy",
            "shifted",
            "blank",
            "random"
        ])

        # Partimos de un dígito real si el modo lo necesita
        if mode in ["occluded", "noisy", "shifted"]:
            img = X_digits[np.random.randint(0, n_source)].copy()

            if mode == "occluded":
                img = add_occlusion(img)
            elif mode == "noisy":
                img = add_noise(img)
            elif mode == "shifted":
                img = shift_digit(img)

        elif mode == "blank":
            img = make_blank()

        else:
            img = make_random_noise()

        invalids.append(img)

    return np.array(invalids, dtype=np.float32)


# =========================================================
# CONSTRUCCIÓN DE LA CNN
# =========================================================

def build_model():
    """
    Crea una CNN sencilla para clasificar:
    0-9 + invalid
    """
    model = tf.keras.Sequential([
        # Entrada: imagen 28x28 en escala de grises
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # Primera capa convolucional
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Segunda capa convolucional
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Tercera capa convolucional
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),

        # Aplanar
        tf.keras.layers.Flatten(),

        # Capa densa
        tf.keras.layers.Dense(128, activation="relu"),

        # Dropout para reducir overfitting
        tf.keras.layers.Dropout(0.3),

        # Salida de 11 clases:
        # 0-9 y la clase 10 = invalid
        tf.keras.layers.Dense(11, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================================================
# CARGA DE DATOS
# =========================================================

print("Loading MNIST...")

# Carga del dataset MNIST
# X: imágenes
# y: etiquetas (0-9)
X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False
)

# Convertir a float y normalizar a [0,1]
X = X.astype("float32") / 255.0

# Convertir etiquetas a enteros
y = y.astype("int64")

# Cambiar forma de (n, 784) a (n, 28, 28)
X = X.reshape(-1, 28, 28)

print("Creating invalid samples...")

# Crear un número de inválidos artificiales
# Aquí usamos la mitad del tamaño del dataset original
n_invalid = len(X) // 2

X_invalid = build_invalid_samples(X, n_invalid)

# Etiqueta 10 para todos los inválidos
y_invalid = np.full((n_invalid,), INVALID_CLASS, dtype=np.int64)

# Unir válidos + inválidos
X_all = np.concatenate([X, X_invalid], axis=0)
y_all = np.concatenate([y, y_invalid], axis=0)

# Añadir el canal para Keras: (28,28) -> (28,28,1)
X_all = X_all[..., np.newaxis]

# Separación train/validación
X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.1,
    random_state=42,
    stratify=y_all
)

print("Training CNN...")

model = build_model()

# Early stopping para detener entrenamiento si no mejora
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# Entrenamiento
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
    callbacks=callbacks
)

# Crear carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Guardar el modelo
model.save("models/digit_cnn_invalid.keras")

print("Model saved in models/digit_cnn_invalid.keras")
# generate_dataset.py
import os
import gzip
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2

# URL base desde donde se descargará el dataset MNIST
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# Diccionario con los archivos necesarios del dataset MNIST
data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
}

# Carpeta donde se guardará el dataset descargado
data_dir = "mnist"

# Si la carpeta no existe, se crea
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Descarga de los archivos del dataset MNIST si no existen localmente
for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading", fname)
        r = requests.get(base_url + fname)
        with open(fpath, "wb") as f:
            f.write(r.content)

# CARGA DEL DATASET MNIST
# Cargar las imágenes de entrenamiento
# Cada imagen MNIST tiene tamaño 28x28
with gzip.open(os.path.join(data_dir, data_sources["training_images"]), "rb") as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

# Cargar las etiquetas correspondientes (0-9)
with gzip.open(os.path.join(data_dir, data_sources["training_labels"]), "rb") as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)

# Crear carpetas para almacenar las matrículas generadas
# valid: imágenes correctas
# invalid: imágenes defectuosas
os.makedirs("data/valid", exist_ok=True)
os.makedirs("data/invalid", exist_ok=True)

# FUNCIÓN PARA CREAR UNA MATRÍCULA DE 4 DÍGITOS
def create_plate(indices):

    # Crear una imagen vacía donde se colocarán los 4 dígitos
    # Cada dígito es 28x28, por lo que la imagen final será 28x112
    final_image = np.zeros((28*4,28),dtype=np.uint8).flatten()

    # Copiar los píxeles de cada dígito MNIST
    for i,index in enumerate(indices):

        # Obtener la imagen del dataset y convertirla a 28x28
        img = images[index].reshape(28,28).transpose().flatten()

        # Insertar el dígito en la posición correspondiente
        final_image[784*i:784*(i+1)] = img

    # Reorganizar el array para formar la matrícula horizontal
    final_image = final_image.reshape(28*4,28).transpose()

    # Construir el número de matrícula a partir de las etiquetas
    number = ""
    for i in indices:
        number += str(labels[i])

    return final_image, number

# GENERACIÓN DE MATRÍCULAS VÁLIDAS
for i in range(200):

    # Seleccionar 4 dígitos aleatorios del dataset MNIST
    indices = random.sample(range(len(images)),4)

    # Crear la matrícula
    img,number = create_plate(indices)

    # Guardar la imagen en la carpeta valid
    plt.imsave(f"data/valid/{number}_{i}.png",img,cmap="gray")


# GENERACIÓN DE MATRÍCULAS DEFECTUOSAS
for i in range(100):

    # Seleccionar 4 dígitos aleatorios
    indices = random.sample(range(len(images)),4)

    # Crear matrícula base
    img,number = create_plate(indices)

    # Seleccionar aleatoriamente un tipo de defecto
    defect = random.choice([
        "black",        # imagen completamente negra
        "white",        # imagen completamente blanca
        "noise",        # ruido fuerte
        "blur",         # imagen borrosa
        "occlusion",    # parte de la matrícula tapada
        "low_contrast"  # contraste muy bajo
    ])

    # APLICACIÓN DEL DEFECTO
    # Imagen completamente negra (simula fallo de cámara)
    if defect == "black":
        img[:] = 0

    # Imagen completamente blanca (sensor saturado)
    elif defect == "white":
        img[:] = 255

    # Añadir ruido gaussiano fuerte
    elif defect == "noise":
        noise = np.random.normal(0,80,img.shape)
        img = img + noise

    # Aplicar desenfoque gaussiano (imagen borrosa)
    elif defect == "blur":
        img = cv2.GaussianBlur(img,(9,9),0)

    # Tapar parte de la matrícula (oclusiones)
    elif defect == "occlusion":
        img[:,40:80] = 0

    # Reducir contraste de la imagen
    elif defect == "low_contrast":
        img = img * 0.3

    # Guardar la imagen defectuosa
    plt.imsave(f"data/invalid/{defect}_{i}.png",img,cmap="gray")
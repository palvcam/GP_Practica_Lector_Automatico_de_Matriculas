import os
from src.preprocess import load_image
from src.lector_matriculas import leer_digitos

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TEST DE MATRÍCULAS VÁLIDAS
def test_valid():

    # Ruta donde se encuentran las imágenes válidas
    path = os.path.join(BASE_DIR, "data", "valid")

    # Se prueban algunas imágenes (las primeras 5)
    for img in os.listdir(path)[:5]:

        # Cargar la imagen
        image = load_image(os.path.join(path, img))

        # Ejecutar el sistema de lectura de matrícula
        plate, valid = leer_digitos(image)

        # Comprobar que el sistema considera válida la matrícula
        assert valid == True

# TEST DE MATRÍCULAS DEFECTUOSAS
def test_invalid():

    # Ruta donde se encuentran las imágenes defectuosas
    path = os.path.join(BASE_DIR, "data", "invalid")

    # Variable para comprobar si al menos una imagen inválida
    # es detectada correctamente por el sistema
    detected_invalid = False

    # Se prueban varias imágenes defectuosas
    for img in os.listdir(path)[:10]:

        # Cargar la imagen
        image = load_image(os.path.join(path, img))

        # Ejecutar el sistema
        plate, valid = leer_digitos(image)

        # Si el sistema detecta una matrícula inválida,
        # marcamos el test como correcto
        if valid == False:
            detected_invalid = True
            break

    # El test pasa si al menos una imagen defectuosa
    # ha sido detectada correctamente
    assert detected_invalid
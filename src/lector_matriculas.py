# lector_matriculas.py
import os
import numpy as np

from src.preprocess import load_image, split_digits
from src.classifier import predict_digit
from src.utils import validar_matricula

# ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def leer_digitos(imagen):

    # 1. Comprobación de formato/dimensiones
    # La imagen esperada debe tener tamaño (28,112)
    # (4 dígitos MNIST de 28x28 colocados horizontalmente)
    if imagen.shape != (28, 112):
        return "", False


    # 2. Detección de imágenes defectuosas
    # Imagen completamente negra (fallo de cámara)
    if np.mean(imagen) < 5:
        return "", False
    # Imagen completamente blanca / saturada
    if np.mean(imagen) > 250:
        return "", False
    # Imagen con contraste muy bajo (borrosa o mala calidad)
    if np.std(imagen) < 10:
        return "", False

    # 3. Segmentación de la matrícula en 4 dígitos
    digits = split_digits(imagen)

    # Comprobar que se han obtenido exactamente 4 dígitos
    if len(digits) != 4:
        return "", False

    plate = ""

    # 4. Clasificación de cada dígito
    for d in digits:

        pred, conf = predict_digit(d)

        # Si la confianza del modelo es baja,
        # consideramos que la matrícula no es fiable
        if conf < 0.3:
            return "", False

        plate += str(pred)

    # 5. Validación final del formato de matrícula
    valido = validar_matricula(plate)

    return plate, valido


# Ejecución de prueba del sistema
if __name__ == "__main__":

    valid_dir = os.path.join(BASE_DIR, "data", "valid")
    invalid_dir = os.path.join(BASE_DIR, "data", "invalid")

    def probar_primera_imagen(directorio, etiqueta):
        files = sorted(os.listdir(directorio))
        if not files:
            print(f"[{etiqueta}] No hay imágenes en {directorio}")
            return

        file = files[0]
        image_path = os.path.join(directorio, file)

        img = load_image(image_path)
        plate, ok = leer_digitos(img)

        print(f"\n[{etiqueta}] Imagen: {file}")
        print(f"[{etiqueta}] Matrícula: {plate}")
        print(f"[{etiqueta}] Válida: {ok}")

    probar_primera_imagen(valid_dir, "VALID")
    probar_primera_imagen(invalid_dir, "INVALID")
from flask import Flask, request, jsonify
import numpy as np
import cv2

from src.lector_matriculas import leer_digitos

app = Flask(__name__)


def file_to_gray_image(file_storage):
    file_bytes = file_storage.read()

    if not file_bytes:
        raise ValueError("El archivo está vacío")

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    return img


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Falta el archivo en el campo 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No se ha seleccionado ningún archivo"}), 400

    try:
        img = file_to_gray_image(file)

        # Redimensionar para adaptarlo al formato esperado
        img = cv2.resize(img, (112, 28))

        plate, ok = leer_digitos(img)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 500

    return jsonify({
        "matricula": plate,
        "valida": bool(ok)
    })


@app.route("/")
def home():
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Lector de matrículas</title>
</head>
<body>
  <h1>Lector de matrículas</h1>

  <form id="uploadForm">
    <input type="file" id="image" name="image" accept="image/*" />
    <button type="submit">Predecir</button>
  </form>

  <h2>Respuesta</h2>
  <pre id="prediction">None</pre>

  <script>
    const form = document.getElementById("uploadForm");

    form.addEventListener("submit", async function(e) {
      e.preventDefault();

      const fileInput = document.getElementById("image");
      const result = document.getElementById("prediction");

      if (!fileInput.files.length) {
        result.innerText = "Selecciona una imagen";
        return;
      }

      const formData = new FormData();
      formData.append("image", fileInput.files[0]);

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData
        });

        const data = await response.json();
        result.innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        result.innerText = "Error: " + err;
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
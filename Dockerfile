# Imagen base
FROM python:3.10

# Instalar librerías necesarias para OpenCV
RUN apt-get update && apt-get install -y libgl1

# Directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar proyecto
COPY . .

# Añadir proyecto al PYTHONPATH
ENV PYTHONPATH=/app

# Ejecutar el sistema
CMD ["python", "src/lector_matriculas.py"]
# Lector Automático de Matrículas (4 Dígitos)

## Descripción

Este proyecto implementa un sistema de aprendizaje automático para leer matrículas numéricas de cuatro dígitos a partir de una imagen.

El sistema:
- recibe una imagen que contiene 4 dígitos
- separa la imagen en cuatro segmentos
- clasifica cada dígito con una CNN
- detecta casos inválidos (dígitos tapados, ruido, imágenes incorrectas)
- devuelve:
  - la matrícula detectada
  - si es válida o no

ESTRUCTURA DEL PROYECTO

GP_Practica/

src/
- server.py              # API Flask
- lector_matriculas.py   # Lógica principal del sistema
- classifier.py          # Clasificador CNN
- preprocess.py          # Procesamiento de imágenes
- utils.py               # Validación de matrículas

models/models/
- digit_cnn_invalid.keras   # Modelo entrenado

data/
- valid/      # Ejemplos de matrículas válidas
- invalid/    # Ejemplos de matrículas inválidas

tests/
- test_unit.py
- test_integration.py
- test_system.py

train_digit_model.py
requirements.txt
Dockerfile
README.md

## Ejecución

La forma más sencilla de ejecutar el sistema es usando Docker.

1. Construir la imagen Docker

Desde la carpeta del proyecto:

docker build -t lector-matriculas .

2. Ejecutar el contenedor (en local para verificar que funciona)

docker run -p 8080:8080 lector-matriculas

El servidor se iniciará y aparecerá algo similar a:

Running on http://0.0.0.0:8080

## Uso del sistema

Abrir en el navegador:

http://localhost:8080

Subir una imagen con una matrícula.

El sistema devolverá algo como:

{
  "matricula": "1234",
  "valida": true
}

## Uso del sistema en AWS

1. Crear una instancia de EC2
2. Instalar docker en esta instancia
3. Conectar nuestro entorno en local a AWS: ssh -i key.pem user@IP_PUBLICA_EC2
4. Crear la imagen .tar en local
5. Pasar la imagen .tar a la instancia de EC2
6. Ejecutar el docker
7. Abrir en el navegador con la ip pública

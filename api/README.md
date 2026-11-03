# API de predicciones

Esta carpeta contiene los archivos necesarios para desarrollar una API con FastAPI que permita utilizar el modelo de Machine Learning entrenado.

La API permite comprobar su funcionamiento, consultar la información del modelo y realizar predicciones.

También incorpora una gestión básica de errores y validación de los datos recibidos.

## Archivo principal

- `main.py`: contiene la aplicación FastAPI, carga el modelo campeón registrado en MLflow y define los endpoints.

## Requisitos previos

Antes de iniciar la API, es necesario:

1. Instalar las dependencias del proyecto.
2. Ejecutar el pipeline para entrenar y registrar el modelo en MLflow.
3. Comprobar que el modelo tiene asignado el alias `champion`.

## Cómo iniciar la API

Desde la carpeta principal del proyecto, ejecutar:

```bash
uvicorn api.main:app --reload
```

Una vez iniciada, la documentación interactiva está disponible en:

http://127.0.0.1:8000/docs

## Endpoints disponibles

La API dispone de tres endpoints:

- `GET /health`: comprueba que la API está activa y que el modelo se ha cargado correctamente.
- `GET /model-info`: devuelve el nombre, el alias y la versión del modelo registrado.
- `POST /predict`: recibe los datos de entrada y devuelve una predicción.

## Ejemplo de petición

Para realizar una predicción mediante `POST /predict`, se deben enviar las diez variables de entrada:

```json
{
  "age": 0.05,
  "sex": 0.04,
  "bmi": 0.07,
  "bp": 0.02,
  "s1": -0.03,
  "s2": -0.04,
  "s3": -0.01,
  "s4": -0.02,
  "s5": 0.04,
  "s6": 0.01
}
```

La respuesta contiene el valor de la predicción:

```json
{
  "prediction": 230.09
}
```

## Gestión de errores

La API contempla diferentes situaciones:

- Código `503`: cuando el modelo no está disponible.
- Código `500`: cuando se produce un error al consultar el modelo o realizar una predicción.
- Código `422`: cuando los datos enviados no cumplen el formato esperado.

La API está diseñada para realizar pruebas y obtener predicciones, no para realizar diagnósticos médicos.

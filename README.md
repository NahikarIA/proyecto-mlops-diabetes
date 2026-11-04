# Proyecto MLOps: Predicción de la progresión de la diabetes

## ¿En qué consiste el proyecto?

Este proyecto consiste en desarrollar un proceso básico de MLOps para trabajar con un modelo de Machine Learning capaz de realizar predicciones relacionadas con la progresión de la diabetes.

El proceso incluye la obtención, validación y limpieza de los datos, el entrenamiento y la evaluación de varios modelos, y el registro del modelo seleccionado en MLflow.

También incluye una API con FastAPI para consultar el modelo y realizar predicciones.

## Datos utilizados

Para realizar el proyecto se utiliza el conjunto de datos `load_diabetes` de Scikit-learn.

Este conjunto contiene:

- 442 registros.
- 10 variables de entrada.
- Una variable objetivo llamada `target`.

Las variables están transformadas, por lo que sus valores no corresponden directamente a medidas habituales como la edad en años o el peso en kilos.

## Organización del proyecto

El proyecto está organizado en diferentes carpetas para separar las distintas tareas:

```text
proyecto-mlops-diabetes/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── main.py
│   └── README.md
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── notebooks/
│   ├── Proyecto_MLOps.ipynb
│   └── README.md
│
├── pipeline/
│   ├── run_pipeline.py
│   └── README.md
│
├── src/
│   ├── data_extraction.py
│   ├── data_validation.py
│   ├── data_cleaning.py
│   ├── model_training.py
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Herramientas utilizadas

Las principales herramientas utilizadas en el proyecto son:

- **Python:** lenguaje de programación.
- **Pandas y NumPy:** para trabajar con los datos.
- **Scikit-learn:** para entrenar y evaluar los modelos.
- **MLflow:** para registrar los entrenamientos y los modelos.
- **FastAPI:** para desarrollar la API de predicciones.
- **Pydantic:** para validar los datos recibidos por la API.
- **Uvicorn:** para iniciar la API.
- **GitHub Actions:** para automatizar la ejecución del pipeline.
- **Google Colab:** entorno de trabajo utilizado durante el desarrollo.

## Instalación

Para descargar el repositorio:

```bash
git clone https://github.com/NahikarIA/proyecto-mlops-diabetes.git
```

Entrar en la carpeta del proyecto:

```bash
cd proyecto-mlops-diabetes
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución del pipeline

El archivo `pipeline/run_pipeline.py` permite ejecutar todas las fases del proyecto de forma consecutiva.

Para iniciar el proceso, desde la carpeta principal del proyecto, ejecutar:

```bash
python -m pipeline.run_pipeline
```

El pipeline realiza las siguientes tareas:

1. Extracción de los datos originales.
2. Validación de los datos.
3. Limpieza y preparación de los datos.
4. Entrenamiento y evaluación de los modelos.
5. Registro de los modelos en MLflow y selección del modelo campeón.

## Entrenamiento y resultados

Se han utilizado tres modelos de regresión:

- Linear Regression.
- Random Forest.
- Gradient Boosting.

Para evaluar los resultados se utilizan las métricas RMSE, MAE y R².

## Criterios de selección

Para seleccionar el modelo campeón se establecen dos condiciones:

- RMSE inferior a 55.
- R² superior a 0.40.

Entre los modelos que cumplen estas condiciones, se selecciona el que obtiene el RMSE más bajo.

Resultados obtenidos:

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 53.85 | 42.79 | 0.45 |
| Random Forest | 54.33 | 44.05 | 0.44 |
| Gradient Boosting | 53.84 | 44.60 | 0.45 |

El modelo seleccionado es Gradient Boosting, ya que cumple los criterios establecidos y obtiene el RMSE más bajo entre los modelos evaluados en el pipeline.

## MLflow y registro del modelo

MLflow se utiliza para guardar los parámetros, las métricas y los modelos obtenidos durante los entrenamientos.

El modelo seleccionado se registra con el nombre:

`Diabetes_Champion_Model`

También se le asigna el alias:

`champion`

Este alias permite que la API pueda cargar el modelo seleccionado sin tener que indicar manualmente su versión.

## API de predicciones

El proyecto incluye una API desarrollada con FastAPI para consultar el estado del servicio y realizar predicciones.

Antes de iniciar la API, es necesario ejecutar el pipeline para registrar el modelo en MLflow.

Para iniciar la API:

```bash
uvicorn api.main:app --reload
```

La documentación interactiva está disponible en:

http://127.0.0.1:8000/docs

La API dispone de tres endpoints:

- `GET /health`: comprueba que la API está activa y que el modelo se ha cargado correctamente.
- `GET /model-info`: muestra información sobre el modelo registrado, incluyendo su nombre, alias y versión.
- `POST /predict`: recibe los datos de entrada y devuelve una predicción junto con la versión del modelo utilizada.

## Ejemplo de respuesta de predicción

```json
{
  "prediction": 237.35568496718219,
  "model_version": "2"
}

### Gestión de errores

La API contempla los siguientes códigos de respuesta:

- `200`: petición realizada correctamente.
- `422`: los datos enviados son incorrectos o faltan campos obligatorios.
- `503`: el modelo no está disponible.
- `500`: se ha producido un error durante la predicción.

## GitHub Actions

El proyecto incluye un flujo de trabajo de GitHub Actions que ejecuta automáticamente el pipeline cuando se realizan cambios en la rama `main`.

El proceso realiza las siguientes tareas:

1. Descarga el repositorio.
2. Prepara el entorno de Python.
3. Instala las dependencias.
4. Ejecuta el pipeline completo.

La ejecución se ha comprobado y finaliza correctamente.

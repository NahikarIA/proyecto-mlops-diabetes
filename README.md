# Proyecto MLOps: Predicción de la progresión de la diabetes

## 1. ¿En qué consiste el proyecto?

En este proyecto he trabajado en la creación de un proceso básico de MLOps utilizando un conjunto de datos relacionado con la diabetes.

La idea es realizar las diferentes fases necesarias para trabajar con un modelo de Machine Learning: obtener los datos, comprobar que están bien, limpiarlos, entrenar varios modelos, comparar sus resultados y guardar el que haya conseguido mejores métricas.

También he creado una API con FastAPI para poder utilizar el modelo y obtener predicciones.

## 2. Datos utilizados

Para realizar el proyecto he utilizado el conjunto de datos `load_diabetes`, disponible en la biblioteca Scikit-learn.

Este conjunto contiene:

- 442 registros.
- 10 variables de entrada.
- Una variable objetivo llamada `target`.

El objetivo es predecir un valor relacionado con la progresión de la diabetes.

Las variables están transformadas, por lo que sus valores no corresponden directamente a medidas habituales como la edad en años o el peso en kilos.

## 3. Organización del proyecto

He organizado el proyecto en diferentes carpetas para tener cada parte del trabajo separada:

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
└── requirements.txt
```

## 4. Herramientas utilizadas

Estas son las principales herramientas que he utilizado durante el proyecto:

- Python: lenguaje de programación.
- Pandas y NumPy: para trabajar con los datos.
- Scikit-learn: para entrenar y evaluar los modelos.
- MLflow: para guardar los resultados y registrar los modelos.
- FastAPI: para crear la API de predicciones.
- Pydantic: para validar los datos que recibe la API.
- Uvicorn: para iniciar la API.
- GitHub Actions: para automatizar la ejecución del pipeline.
- Google Colab: entorno utilizado para trabajar con Python.

## 5. Instalación

Primero hay que descargar el repositorio:

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

## 6. Ejecución del pipeline

He creado un archivo llamado `run_pipeline.py` para poder ejecutar todas las fases del proyecto desde un único sitio.

Para ponerlo en marcha hay que ejecutar:

```bash
python -m pipeline.run_pipeline
```

El proceso realiza los siguientes pasos:

1. **Extracción:** obtiene los datos originales.
2. **Validación:** comprueba que los datos tienen las columnas esperadas y que no faltan valores obligatorios.
3. **Limpieza:** elimina posibles valores nulos y registros duplicados.
4. **Entrenamiento:** entrena varios modelos y compara sus resultados.
5. **Registro:** guarda en MLflow el modelo que cumple los criterios de calidad.

## 7. Entrenamiento y resultados

Para comparar los resultados he utilizado tres modelos:

- Linear Regression (Regresión lineal).
- Random Forest (Bosque aleatorio).
- Gradient Boosting.

Para evaluar su funcionamiento he utilizado tres métricas:

- **RMSE:** mide el error de las predicciones.
- **MAE:** calcula el error absoluto medio.
- **R²:** indica cuánto explica el modelo de la variación de los datos.

### Criterio para elegir el modelo

He establecido dos condiciones para seleccionar el modelo:

- RMSE inferior a 55.
- R² superior a 0.40.

De los modelos que cumplen estas condiciones, se selecciona el que tiene el RMSE más bajo.

En las pruebas realizadas he obtenido estos resultados:

- **Linear Regression:** RMSE 53.85, MAE 42.79 y R² 0.45.
- **Random Forest:** RMSE 54.33, MAE 44.05 y R² 0.44.
- **Gradient Boosting:** RMSE 53.84, MAE 44.60 y R² 0.45.

El modelo seleccionado ha sido Gradient Boosting, porque ha obtenido el RMSE más bajo.

## 8. MLflow y registro del modelo

He utilizado MLflow para guardar la información de los entrenamientos y poder consultar los resultados de los diferentes modelos.

También he registrado el modelo seleccionado con el nombre:

`Diabetes_Champion_Model`

Para identificarlo he utilizado el alias:

`champion`

De esta manera, la API puede cargar el modelo seleccionado sin tener que indicar manualmente su versión.

## 9. API con FastAPI

Otra parte del proyecto ha sido crear una API que permite consultar el estado del servicio y realizar predicciones utilizando el modelo registrado en MLflow.

Antes de iniciar la API por primera vez, hay que ejecutar el pipeline para que el modelo quede registrado en MLflow.

Para iniciar la API, desde la carpeta principal del proyecto, hay que ejecutar:

```bash
uvicorn api.main:app --reload
```

Una vez iniciada, se puede acceder a la documentación interactiva desde:

http://127.0.0.1:8000/docs

La API dispone de tres endpoints:

- **GET /health:** comprueba que la API está activa y que el modelo se ha cargado correctamente.
- **GET /model-info:** muestra información sobre el modelo utilizado.
- **POST /predict:** recibe los datos y devuelve una predicción.

### Ejemplo de datos para realizar una predicción

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

La respuesta de la API es un valor numérico que corresponde a la predicción del modelo.

## 10. GitHub Actions

También he configurado GitHub Actions para que el proyecto pueda comprobarse automáticamente cuando se suben cambios a la rama `main`.

El flujo realiza estas tareas:

1. Descarga el repositorio.
2. Prepara el entorno de Python.
3. Instala las dependencias.
4. Ejecuta el pipeline completo.

He comprobado que la ejecución termina correctamente, por lo que la automatización está funcionando.

## 11. Limitaciones del proyecto

Este proyecto se ha realizado con fines de aprendizaje, utilizando un conjunto de datos pequeño y modelos básicos de Machine Learning.

Los resultados dependen de los datos utilizados y de la configuración de los modelos.

La API sirve para probar el funcionamiento del modelo y obtener predicciones, pero no está pensada para realizar diagnósticos médicos ni para tomar decisiones clínicas.

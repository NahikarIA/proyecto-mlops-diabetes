# Código fuente del proyecto

Esta carpeta contiene los scripts de Python que desarrollan las distintas fases del flujo de trabajo de Machine Learning.

Cada archivo se encarga de una tarea concreta:

## Archivos principales

- `data_extraction.py`: obtiene el conjunto de datos de diabetes de Scikit-learn y lo guarda en formato CSV en la carpeta `data/raw`.

- `data_validation.py`: comprueba que el archivo de datos existe, que contiene las columnas esperadas y que los valores tienen el formato adecuado.

- `data_cleaning.py`: realiza la limpieza de los datos, eliminando valores nulos y filas duplicadas, y guarda el resultado en `data/processed`.

- `model_training.py`: prepara los datos, entrena y evalúa distintos modelos de regresión, registra los parámetros, métricas y modelos en MLflow y selecciona el modelo campeón según los criterios de calidad establecidos.

## Flujo de trabajo

Los scripts se ejecutan siguiendo este orden:

1. Extracción de datos.
2. Validación de datos.
3. Limpieza de datos.
4. Entrenamiento, evaluación y registro de modelos.

Estas tareas se coordinan desde el archivo `pipeline/run_pipeline.py`.

# Datos originales

Esta carpeta contiene los datos originales utilizados en el proyecto MLOps de predicción de la progresión de la diabetes.

## Origen de los datos

El conjunto de datos se obtiene mediante la función `load_diabetes()` de la biblioteca Scikit-learn.

Contiene 442 registros, 10 variables de entrada y una variable objetivo (`target`) relacionada con la progresión de la diabetes.

## Archivo generado

- `diabetes.csv`: contiene los datos originales extraídos y guardados en formato CSV.

Los datos se conservan sin modificar para mantener una copia del conjunto de datos de partida. La extracción se realiza mediante el script `src/data_extraction.py`.

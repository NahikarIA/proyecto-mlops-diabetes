# Datos procesados

Esta carpeta contiene los datos después de los procesos de validación y limpieza.

## Archivo generado

- `diabetes_clean.csv`: contiene el conjunto de datos preparado para el entrenamiento y la evaluación de los modelos de Machine Learning.

## Proceso de limpieza

El script `src/data_cleaning.py` realiza las siguientes operaciones:

- Comprueba si existen valores nulos y elimina las filas que los contienen.
- Elimina los registros duplicados.
- Comprueba que todas las columnas contienen valores numéricos.

Los datos procesados se utilizan posteriormente para entrenar y evaluar los modelos de Machine Learning.

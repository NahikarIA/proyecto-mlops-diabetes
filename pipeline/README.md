# Organización del flujo de trabajo

Esta carpeta contiene los archivos necesarios para coordinar y ejecutar las diferentes etapas del proyecto MLOps.

El pipeline conecta las fases de extracción, validación y limpieza de datos con el entrenamiento, la evaluación y el registro de los modelos en MLflow.

## Archivo principal

- `run_pipeline.py`: coordina la ejecución de todas las fases del proyecto, llamando a las funciones definidas en los scripts de la carpeta `src`.

## Orden de ejecución

El pipeline realiza las siguientes tareas:

1. **Extracción:** obtiene los datos de diabetes de Scikit-learn y los guarda en formato CSV.
2. **Validación:** comprueba que los datos existen y cumplen los requisitos establecidos.
3. **Limpieza:** elimina valores nulos y registros duplicados.
4. **Entrenamiento:** entrena y evalúa los modelos de Machine Learning.
5. **Registro:** guarda los parámetros, métricas y modelos en MLflow, y asigna el alias `champion` al modelo que cumple los criterios de calidad.

## Cómo ejecutar el pipeline

Desde la carpeta principal del proyecto, ejecutar:

```bash
python -m pipeline.run_pipeline
```

Al finalizar correctamente, se mostrará un mensaje indicando que el pipeline se ha completado.

El proceso permite ejecutar todas las fases de forma consecutiva, sin tener que iniciar cada script por separado.

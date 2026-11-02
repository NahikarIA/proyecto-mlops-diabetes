# Importar librerías necesarias
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_diabetes

# Localizar la carpeta principal del proyecto
PROJECT_ROOT = Path(_file_).resolve().parent.parent

# Indicar dónde vamos a guardar los datos originales
RAW_DATA_PATH = PROJECT_ROOT/"data"/"raw"/"diabetes.csv"


def extract_data():
  # Cargar el conjunto de datos de diabetes de scikit_learn
  diabetes = load_diabetes(as_frame=True)

  # Convertir los datos a un DataFrame de pandas
  df = diabetes.frame

  # Crear la carpeta de destino si todavía no existe
  RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

  # Guardar los datos originales en un archivo CSV
  df.to_csv(RAW_DATA_PATH, index=False)

  # Mostrar dónde se han guardado los datos
  print(f"Datos extraídos correctamente: {RAW_DATA_PATH}")

# Ejecutar la extracción cuando se inicia este archivo
if _name_ == "_main_":
    extract_data()


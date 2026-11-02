# Importar librerías necesarias
from pathlib import Path
import pandas as pd

# Localizar la carpeta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Indicar dónde se encuentran los datos originales
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"

# Definir las columnas que esperamos encontrar
EXPECTED_COLUMNS = [
  "age", "sex", "bmi", "bp", "s1", 
  "s2", "s3", "s4", "s5", "s6", "target"
]


def validate_data():
  # Comprobar que el archivo existe
  if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
      f"No se ha encontrado el archivo: {RAW_DATA_PATH}"
    )

# Cargar los datos
df = pd.read_csv(RAW_DATA_PATH)

# Comprobar que el archivo no está vacío
if df.empty:
    raise ValueError("El archivo de datos está vacío")

# Comprobar que están todas las columnas
missing_columns = [
  column for column in EXPECTED_COLUMNS
  if column not in df.columns
]

if missing_columns:
    raise ValueError(
      f"Faltan columnas obligatorias: {missing_columns}"
    )

# Comprobar que todas las columnas contienen datos numéricos
if not all(
   pd.api.types.is_numeric_dtype(df[column])
   for column in EXPECTED_COLUMNS
):
      raise ValueError("Hay columnas que no contienen datos numéricos")

# Comprobar que la variable objetivo existe y no tiene valores nulos
if df["target"].isna().any():
  raise ValueError("La variable objetivo contiene valores nulos")

# Mostrar el resultado de la validación
print("Validación completada correctamente")
print(f"Número de filas: {len(df)}")
print(f"Número de columnas: {len(df.columns)}")

return df

# Ejecutar la validación si iniciamos este archivo directamente
if __name__ == "__main__":
  validate_data()

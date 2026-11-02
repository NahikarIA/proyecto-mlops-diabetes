# Importar librerías necesarias
from pathlib import Path
import pandas as pd

# Localizar la carpeta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Indicar las rutas de los archivos de datos
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_clean.csv"


def clean_data():
  # Comprobar que existe el archivo original
  if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
      f"No se ha encontrado el archivo: {RAW_DATA_PATH}"
    )

# Cargar los datos originales
df = pd.read_csv(RAW_DATA_PAHT)

# Comprobar si existen valores nulos
print(f"Valores nulos antes de limpiar: {df.isnull().sum().sum()}")

# Eliminar las filas que contienen valores nulos
df = df.dropna()

# Eliminar las filas duplicadas
df = df.drop_duplicates()

# Comprobar que todas las columnas son numéricas
for column in df.columns:
  df[column] = pd.to_numeric(df[column], errors="raise")

# Crear la carpeta de destino si no existe
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# Guardar los datos limpios
df.to_csv(PROCESSED_DATA_PATH, index=False)

# Mostrar el resultado de limpieza
print("Limpieza de datos completada")
print(f"Filas finales: {len(df)}")
print(f"Datos guardados en: {PROCESSED_DATA_PATH}")

return df

# Ejecutar limpieza si iniciamos archivo directamente
if __name__ == "__main__":
  clean_data()

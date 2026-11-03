# Importar las funciones de cada fase del proyecto
from src.data_extraction import extract_data
from src.data_validation import validate_data
from src.data_cleaning import clean_data
from src.model_training import train_models


# Ejecutar el proceo completo de MLOps
def run_pipeline():

    print("\n--- FASE 1: EXTRACCIÓN DE DATOS ---")
    extract_data()

    print("\n--- FASE 2: VALIDACIÓN DE DATOS ---")
    validate_data()

    print("\n--- FASE 3: LIMPIEZA DE DATOS ---")
    clean_data()

    print("\n--- FASE 4: ENTRENAMIENTO Y REGISTRO ---")
    train_models()

    print("\n--- PIPELINE COMPLETADO CORRECTAMENTE ---")


# Ejecutar el pipeline al iniciar este archivo
if __name__ == "__main__":
      run_pipeline()

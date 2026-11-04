# Importar librerías necesarias
from pathlib import Path
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Localizar la carpeta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configurar la conexión con la base de datos de MLflow
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_registry_uri(f"sqlite:///{MLFLOW_DB}")

# Nombre y alias del modelo
MODEL_NAME = "Diabetes_Champion_Model"
MODEL_ALIAS = "champion"

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de predicción de diabetes",
    description="API para realizar predicciones con un modelo de Machine Learning",
    version="1.0.0"
)

# Inicializar las variables del modelo
model = None
model_version = None

# Cargar el modelo campeón desde MLflow
try:
    client = mlflow.MlflowClient()

    # Obtener la versión asociada al alias champion
    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS
    )

    # Guardar la versión del modelo
    model_version = str(champion.version)

    # Cargar esa versión concreta
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{model_version}"
    )

    print(f"Modelo campeón cargado correctamente. Versión: {model_version}")

except Exception as error:
    model = None
    model_version = None
    print(f"No se ha podido cargar el modelo: {error}")


# Definir los datos de entrada
class DiabetesData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


# Comprobar el estado de la API
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    return {
        "status": "ok",
        "model_loaded": True
    }


# Consultar información del modelo
@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": model_version,
        "status": "ready"
    }


# Realizar una predicción
@app.post("/predict")
def predict(data: DiabetesData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    try:
        # Preparar los datos de entrada
        input_data = [[
            data.age,
            data.sex,
            data.bmi,
            data.bp,
            data.s1,
            data.s2,
            data.s3,
            data.s4,
            data.s5,
            data.s6
        ]]

        # Obtener la predicción
        prediction = model.predict(input_data)

        # Devolver la predicción y la versión utilizada
        return {
            "prediction": float(prediction[0]),
            "model_version": model_version
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(error)}"
        )

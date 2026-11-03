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

# Nombre y alias del modelo que se va utilizar
MODEL_NAME = "Diabetes_Champion_Model"
MODEL_ALIAS = "champion"

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de predicción de diabetes",
    description="API para realizar predicciones con un modelo de Machine Learning",
    version="1.0.0"
)

# Cargar el modelo campeón desde el registro de MLflow
try:
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )
    print("Modelo campeón cargado correctamente")

except Exception as error:
    model = None
    print(f"No se ha podido cargar el modelo: {error}")


# Definir los datos que recibirá la API para hacer una predicción
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

    # Comprobar si el modelo está disponible
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    # Devolver el estado de la API
    return {
        "status": "ok",
        "model_loaded": True
    }

# Consultar información del modelo utilizado por la API
@app.get("/model-info")
def model_info():

    # Comprobar si el modelo está disponible
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    try:
        # Consultar la versión asociada al alias champion
        client = mlflow.MlflowClient()

        champion = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=MODEL_ALIAS
        )

        # Devolver la información del modelo
        return {
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_version": champion.version,
            "status": "ready"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al consultar el modelo: {str(error)}"
        )
        
# Realizar una predicción con el modelo
@app.post("/predict")
def predict(data: DiabetesData):

    # Comprobar si el modelo está disponible
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible"
        )

    try:
        # Convertir los datos recibidos en una lista
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

        # Obtener la predicción del modelo
        prediction = model.predict(input_data)

        # Devolver el resultado
        return {
            "prediction": float(prediction[0])
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(error)}"
        )
     
            
          

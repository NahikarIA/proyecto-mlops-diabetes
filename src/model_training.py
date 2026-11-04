# Importar librerías necesarias
from pathlib import Path
import pandas as pd
import mlflow
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Localizar la carpeta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configurar la base de datos de MLflow
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_registry_uri(f"sqlite:///{MLFLOW_DB}")

# Indicar dónde están guardados los datos limpios
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_clean.csv"

# Definir el nombre del experimento en MLflow
mlflow.set_experiment("Diabetes_ML_Experiments")

# Definir los criterios mínimos de calidad del modelo
MAX_RMSE = 55
MIN_R2 = 0.40


def load_data():
    # Cargar los datos limpios
    df = pd.read_csv(DATA_PATH)

    # Separar las variables de entrada y la variable objetivo
    X = df.drop("target", axis=1)
    y = df["target"]

    # Dividir los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):

    # Definir los modelos que vamos a entrenar
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    # Guardar los resultados de cada modelo
    results = {}

    # Entrenar y evaluar cada modelo
    for name, model in models.items():

        # Iniciar un registro en MLflow para cada modelo
        with mlflow.start_run(run_name=name):

            # Entrenar el modelo
            model.fit(X_train, y_train)

            # Realizar las predicciones
            predictions = model.predict(X_test)

            # Calcular las métricas de evaluación
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Registrar los parámetros del modelo en MLflow
            mlflow.log_params(model.get_params())

            # Registrar las métricas en MLflow
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # Guardar el modelo como artefacto en MLflow
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            # Guardar los resultados de cada modelo
            results[name] = {
                "model": model,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "run_id": mlflow.active_run().info.run_id
            }

            print(f"\nModelo: {name}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R2: {r2:.2f}")

    # Seleccionar los modelos que cumplen los criterios de calidad
    approved_models = {
        name: result
        for name, result in results.items()
        if result["RMSE"] < MAX_RMSE and result["R2"] > MIN_R2
    }

    # Comprobar si hay algún modelo que cumpla los criterios
    if not approved_models:
        print("Ningún modelo ha superado los criterios de calidad")
        return results

    # Seleccionar el modelo con menor RMSE
    champion_name = min(
        approved_models,
        key=lambda name: approved_models[name]["RMSE"]
    )

    print(f"\nModelo campeón seleccionado: {champion_name}")    

    # Registrar el modelo campeón en MLflow
    champion_model = approved_models[champion_name]["model"]
    champion_run_id = approved_models[champion_name]["run_id"]

    model_name = "Diabetes_Champion_Model"

    model_info = mlflow.sklearn.log_model(
        champion_model,
        name="champion_model",
        registered_model_name=model_name,
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # Asignar el alias champion a la versión registrada
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_info.registered_model_version
    )

    print(f"Modelo registrado: {model_name}")
    print("Alias asignado: champion")
    
    return results


# Ejecutar el proceso de entrenamiento
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    results = train_models(X_train, X_test, y_train, y_test)

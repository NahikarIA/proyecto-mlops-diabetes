# Importar librerías necesarias
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow


# Localizar la carpeta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Indicar dónde están guardados los datos limpios
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_clean.csv"

# Definir el nombre del experimento en MLflow
mlflow.set_experiment("Diabetes_ML_Experiments")


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

            # Guardar el modelo y sus métricas
            results[name] = {
                "model": model,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            }

            print(f"\nModelo: {name}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R2: {r2:.2f}")

    return results
  
# Ejecutar el proceso de train
if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_data()
  results = train_models(X_train, X_test, y_train, y_test)

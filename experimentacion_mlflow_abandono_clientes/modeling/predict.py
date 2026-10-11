# Predicciones con el modelo entrenado
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluar_umbral(y_test, y_proba):
    # El umbral por defecto de 0.5 no necesariamente es el mejor para el negocio
    resultados = []
    for umbral in [0.3, 0.4, 0.5, 0.6]:
        pred = (y_proba >= umbral).astype(int)
        resultados.append(
            {
                "umbral": umbral,
                "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
            }
        )
    return pd.DataFrame(resultados).round(3)


def cargar_modelo(run_id, artifact_path="modelo_baseline"):
    # Regresa el modelo que quedo guardado en mlflow
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact_path}")

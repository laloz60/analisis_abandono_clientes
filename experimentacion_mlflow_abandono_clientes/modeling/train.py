# Entrenamiento del modelo base y registro del experimento en mlflow
# Se corre con: python -m experimentacion_mlflow_abandono_clientes.modeling.train
# o con: make train
import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from experimentacion_mlflow_abandono_clientes.config import (
    EXPERIMENT_NAME,
    INTERIM_DIR,
    MODEL_PARAMS,
    REPORTS_DIR,
    RUN_NAME,
    SEED,
    TEST_SIZE,
    TRACKING_URI,
)
from experimentacion_mlflow_abandono_clientes.dataset import bitacora, cargar_datos, limpiar_datos
from experimentacion_mlflow_abandono_clientes.features import (
    construir_preprocesador,
    dividir_datos,
)


def entrenar(X_train, y_train, params=None):
    params = params or MODEL_PARAMS
    preprocesador = construir_preprocesador(X_train)

    modelo = Pipeline(
        [
            ("preprocesamiento", preprocesador),
            ("clasificador", LogisticRegression(**params)),
        ]
    )
    modelo.fit(X_train, y_train)

    # Modelo Dummy de referencia, siempre predice la clase mas frecuente
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return modelo, dummy


def evaluar(modelo, X, y):
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo base de churn y registra el run en mlflow"
    )
    parser.add_argument(
        "-C",
        "--C",
        type=float,
        default=MODEL_PARAMS["C"],
        help="Inverso de la regularizacion (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MODEL_PARAMS["max_iter"],
        help="Maximas iteraciones del optimizador (default: %(default)s)",
    )
    parser.add_argument(
        "--solver",
        default=MODEL_PARAMS["solver"],
        help="Algoritmo de optimizacion, p. ej. lbfgs o liblinear (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla aleatoria del split y del modelo (default: %(default)s)",
    )
    args = parser.parse_args()

    params = {
        **MODEL_PARAMS,
        "C": args.C,
        "max_iter": args.max_iter,
        "solver": args.solver,
        "random_state": args.seed,
    }

    churn_df = limpiar_datos(cargar_datos())
    X_train, X_test, y_train, y_test = dividir_datos(churn_df, seed=args.seed)
    modelo, dummy = entrenar(X_train, y_train, params)

    metricas_dummy = evaluar(dummy, X_test, y_test)
    metricas_train = evaluar(modelo, X_train, y_train)
    metricas_test = evaluar(modelo, X_test, y_test)

    y_pred = modelo.predict(X_test)

    # Reporte de clasificacion en su carpeta
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(
            classification_report(y_test, y_pred, target_names=["No churn", "Churn"], digits=3)
        )
    bitacora().to_csv(INTERIM_DIR / "bitacora_limpieza.csv", index=False)

    # Conexion al servidor de mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experimento = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experimento is None:
        descripcion = (
            "Proyecto de prediccion de abandono de clientes (churn) de una compania de telecomunicaciones. "
            "Este experimento contiene el modelo base (baseline) contra el que se compararan los siguientes modelos."
        )
        tags = {
            "project_name": "telco-churn",
            "dataset": "blastchar/telco-customer-churn",
            "problem": "binary_classification",
            "author": "A01797668",
            "mlflow.note.content": descripcion,
        }
        client.create_experiment(name=EXPERIMENT_NAME, tags=tags)
        print("Experimento creado")
    mlflow.set_experiment(EXPERIMENT_NAME)

    nombres = modelo.named_steps["preprocesamiento"].get_feature_names_out()

    with mlflow.start_run(run_name=RUN_NAME) as run:
        for nombre, valor in params.items():
            mlflow.log_param(f"model_{nombre}", valor)

        extras = {
            "model_type": "LogisticRegression",
            "test_size": TEST_SIZE,
            "seed": args.seed,
            "stratify": True,
            "scaler": "StandardScaler",
            "encoder": "OneHotEncoder(drop=if_binary)",
            "n_rows": churn_df.shape[0],
            "n_features_raw": X_train.shape[1],
            "n_features_encoded": len(nombres),
            "totalcharges_imputation": "0 when tenure==0",
        }
        for nombre, valor in extras.items():
            mlflow.log_param(nombre, valor)

        for nombre, valor in metricas_test.items():
            mlflow.log_metric(f"test_{nombre}", valor)
        for nombre, valor in metricas_train.items():
            mlflow.log_metric(f"train_{nombre}", valor)
        for nombre, valor in metricas_dummy.items():
            mlflow.log_metric(f"dummy_test_{nombre}", valor)

        mlflow.set_tags({"stage": "baseline", "model_family": "linear"})

        mlflow.log_artifact(INTERIM_DIR / "bitacora_limpieza.csv", artifact_path="datos")
        mlflow.log_artifact(REPORTS_DIR / "classification_report.txt", artifact_path="reportes")

        # Modelo con firma y un ejemplo pequeño de entrada
        firma = infer_signature(X_train, modelo.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=modelo, name="modelo_baseline", signature=firma, input_example=X_train.head(5)
        )

    print("ROC-AUC prueba:", round(metricas_test["roc_auc"], 4))
    print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()

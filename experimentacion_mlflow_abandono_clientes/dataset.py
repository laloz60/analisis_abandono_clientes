# Carga y limpieza de los datos del churn
# Se corre con: python -m experimentacion_mlflow_abandono_clientes.dataset
import numpy as np
import pandas as pd

from experimentacion_mlflow_abandono_clientes.config import (
    INTERIM_DIR,
    NUM_COLS,
    PROCESSED_DIR,
    RAW_FILE,
    TARGET,
)


def cargar_datos():
    # El customerID es unico por registro, por eso se vuelve el indice
    churn_df = pd.read_csv(RAW_FILE)
    churn_df.set_index("customerID", inplace=True)
    return churn_df


def limpiar_datos(churn_df):
    df = churn_df.copy()

    # TotalCharges viene como texto y con espacios en blanco en 11 registros
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", np.nan))
    # Esos 11 clientes tienen tenure = 0, todavia no deben nada
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})

    cat_cols = [col for col in df.columns if col not in NUM_COLS + [TARGET]]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # Contract como ordinal para que tenga sentido el orden
    df["Contract"] = pd.Categorical(
        df["Contract"], categories=["Month-to-month", "One year", "Two year"], ordered=True
    )
    df[TARGET] = df[TARGET].replace({"Yes": 1, "No": 0}).astype(int)
    return df


def bitacora():
    # Tabla de decisiones de limpieza, se sube como artefacto a mlflow
    return pd.DataFrame(
        {
            "problema": [
                "TotalCharges como texto",
                "Nulos en TotalCharges",
                "SeniorCitizen 0/1",
                "Perfiles duplicados",
                "Outliers",
                "customerID",
            ],
            "decision": [
                "Convertir a float",
                "Imputar con 0 (tenure = 0)",
                "Convertir a No/Yes",
                "Se conservan (distinto customerID)",
                "No se encontraron",
                "Se usa como indice",
            ],
        }
    )


def main():
    df = limpiar_datos(cargar_datos())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "churn_limpio.csv")
    bitacora().to_csv(INTERIM_DIR / "bitacora_limpieza.csv", index=False)
    print("Datos limpios:", df.shape)


if __name__ == "__main__":
    main()

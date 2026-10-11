# Preprocesamiento: division estratificada y las transformaciones de las columnas
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from experimentacion_mlflow_abandono_clientes.config import NUM_COLS, SEED, TARGET, TEST_SIZE


def dividir_datos(churn_df, seed=SEED, test_size=TEST_SIZE):
    X = churn_df.drop(columns=TARGET)
    y = churn_df[TARGET]
    # 80% entrenamiento, 20% prueba, manteniendo la proporcion de churn
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def construir_preprocesador(X):
    # Las numericas se escalan y las categoricas pasan a one-hot
    cat_cols = [col for col in X.columns if col not in NUM_COLS]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols),
        ]
    )

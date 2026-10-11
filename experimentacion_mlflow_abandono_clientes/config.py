# Rutas y constantes del proyecto para no tener todo hardcodeado en cada archivo
import os
from pathlib import Path

from dotenv import load_dotenv

PROJ_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJ_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
INTERIM_DIR = PROJ_ROOT / "data" / "interim"
PROCESSED_DIR = PROJ_ROOT / "data" / "processed"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Semilla para que los resultados sean reproducibles
SEED = 42
TEST_SIZE = 0.2
TARGET = "Churn"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Hiperparametros del modelo base
MODEL_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": SEED,
}

EXPERIMENT_NAME = "Telco_Churn_Baseline"
RUN_NAME = "logreg_baseline"

# La direccion del servidor de mlflow vive en el .env
load_dotenv(PROJ_ROOT / ".env")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

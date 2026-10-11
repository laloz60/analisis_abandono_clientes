# Predicción de abandono de clientes (Telco Churn)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Modelo base para predicción de abandono de clientes (Telco Churn): EDA, limpieza y registro en MLflow

## Cómo ejecutar

Requisitos: Python 3.12 y las dependencias de `requirements.txt`:

```bash
pip install -r requirements.txt
```

El dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) ya está incluido en `data/raw/`.

1. Levanta el servidor de MLflow en una terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

2. Entrena el modelo base en otra terminal (o con `make train`):

```bash
python -m experimentacion_mlflow_abandono_clientes.modeling.train
```

Se pueden variar los hiperparámetros por la línea de comandos, por ejemplo:

```bash
python -m experimentacion_mlflow_abandono_clientes.modeling.train -C 0.1 --solver liblinear --max-iter 500 --seed 7
```

Los resultados se consultan en `http://127.0.0.1:5000`, experimento `Telco_Churn_Baseline`.

## Reproducibilidad

La semilla está fija en `config.py` (`SEED = 42`) y se aplica tanto al split como al modelo: dos corridas con los mismos argumentos producen las mismas métricas. Las versiones exactas de las dependencias están fijadas en `requirements.txt`.

## Project Organization

```
├── Makefile                      <- make train, make notebook, make lint
├── README.md
├── pyproject.toml
├── requirements.txt              <- versiones fijadas de las dependencias
├── data
│   └── raw
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks
│   ├── .nbautoexport
│   ├── 0.01-gc-telco-churn-baseline.ipynb   <- notebook original del curso
│   ├── 0.02-gc-eda.ipynb                    <- exploración de datos
│   └── 1.00-gc-flujo-completo.ipynb         <- flujo completo con mlflow
├── reports
│   └── figures
└── experimentacion_mlflow_abandono_clientes
    ├── __init__.py
    ├── config.py                <- semilla, rutas e hiperparámetros
    ├── dataset.py               <- carga y limpieza de datos
    ├── features.py              <- split y preprocesamiento
    └── modeling
        ├── __init__.py
        ├── predict.py           <- inferencia y evaluación con umbral
        └── train.py             <- entrenamiento con argparse y mlflow
```

--------


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
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         experimentacion_mlflow_abandono_clientes and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── experimentacion_mlflow_abandono_clientes   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes experimentacion_mlflow_abandono_clientes a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
```

--------


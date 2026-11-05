"""Preprocesamiento, estimadores y métricas de Fase 1."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datos import CATEGORICAS, NUMERICAS


UMBRAL = 0.30  # Elegido en la validación de Fase 1; prueba no ajusta este valor.
MAX_ITER = 1000


def crear_pipeline(semilla: int) -> Pipeline:
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocesador = ColumnTransformer([
        ("numericas", numericas, list(NUMERICAS)),
        ("categoricas", categoricas, list(CATEGORICAS)),
    ])
    return Pipeline([
        ("preprocesador", preprocesador),
        ("modelo", LogisticRegression(max_iter=MAX_ITER, random_state=semilla)),
    ])


def crear_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def aplicar_umbral(probabilidades: np.ndarray, umbral: float = UMBRAL) -> np.ndarray:
    return (probabilidades >= umbral).astype(int)


def calcular_metricas(reales, predichas) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(reales, predichas)),
        "precision": float(precision_score(reales, predichas, zero_division=0)),
        "recall": float(recall_score(reales, predichas, zero_division=0)),
        "f1": float(f1_score(reales, predichas, zero_division=0)),
    }

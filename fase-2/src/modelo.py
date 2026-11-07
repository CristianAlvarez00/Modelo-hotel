"""Preprocesamiento, estimadores y metricas de Fase 1."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datos import CATEGORICAS, NUMERICAS


UMBRAL = 0.30  # valor usado en fase 1
MAX_ITER = 1000  # maximo de iteraciones del modelo


def crear_pipeline(semilla: int) -> Pipeline:
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])  # prepara columnas numericas
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])  # prepara columnas categoricas
    preprocesador = ColumnTransformer([
        ("numericas", numericas, list(NUMERICAS)),
        ("categoricas", categoricas, list(CATEGORICAS)),
    ])  # aplica transformaciones por tipo de columna
    return Pipeline([
        ("preprocesador", preprocesador),
        ("modelo", LogisticRegression(max_iter=MAX_ITER, random_state=semilla)),
    ])  # arma pipeline final


def crear_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")  # usa la clase mas frecuente como referencia


def aplicar_umbral(probabilidades: np.ndarray, umbral: float = UMBRAL) -> np.ndarray:
    return (probabilidades >= umbral).astype(int)  # convierte probabilidades en 0 y 1


def calcular_metricas(reales, predichas) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(reales, predichas)),
        "precision": float(precision_score(reales, predichas, zero_division=0)),
        "recall": float(recall_score(reales, predichas, zero_division=0)),
        "f1": float(f1_score(reales, predichas, zero_division=0)),
    }  # devuelve metricas de clasificacion

"""Datos sintéticos pequeños y modelos de prueba; sin red ni el dataset completo."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Permite `import train` y `import src` aunque pytest se lance desde otra carpeta.
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos import CATEGORICAS, COLUMNAS  # noqa: E402
from train import entrenar  # noqa: E402

OBJETIVO = "is_canceled"
SEMILLA = 42

VALORES = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "arrival_date_month": ["January", "May", "July", "October"],
    "meal": ["BB", "HB", "SC"],
    "country": ["PRT", "ESP", "FRA", "GBR", "DEU"],
    "market_segment": ["Online TA", "Groups", "Direct", "Offline TA/TO"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate"],
    "reserved_room_type": ["A", "C", "D", "E"],
    "customer_type": ["Transient", "Contract", "Transient-Party"],
}


def crear_datos(filas: int = 400, semilla: int = 0) -> pd.DataFrame:
    """Reservas sintéticas con las 17 entradas y un objetivo ligado a lead_time."""
    rng = np.random.default_rng(semilla)
    datos = pd.DataFrame({
        "lead_time": rng.integers(0, 400, filas),
        "stays_in_weekend_nights": rng.integers(0, 4, filas),
        "stays_in_week_nights": rng.integers(0, 8, filas),
        "adults": rng.integers(1, 4, filas),
        "children": rng.integers(0, 3, filas).astype(float),
        "babies": rng.integers(0, 2, filas),
        "is_repeated_guest": rng.integers(0, 2, filas),
        "previous_cancellations": rng.integers(0, 3, filas),
        "previous_bookings_not_canceled": rng.integers(0, 5, filas),
    })
    for columna in CATEGORICAS:
        datos[columna] = rng.choice(VALORES[columna], filas)
    ruido = rng.normal(0, 60, filas)
    datos[OBJETIVO] = (datos["lead_time"] + ruido > 200).astype(int)
    return datos


@pytest.fixture(scope="session")
def csv_entrenamiento(tmp_path_factory) -> Path:
    ruta = tmp_path_factory.mktemp("datos") / "reservas.csv"
    crear_datos().to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def modelos(tmp_path_factory, csv_entrenamiento) -> Path:
    """Carpeta con dos versiones (v1 y v2) entrenadas una sola vez por sesión."""
    carpeta = tmp_path_factory.mktemp("models")
    entrenar(csv_entrenamiento, OBJETIVO, carpeta, "v1", SEMILLA)
    entrenar(csv_entrenamiento, OBJETIVO, carpeta, "v2", 7)
    return carpeta


@pytest.fixture
def entradas() -> pd.DataFrame:
    """Reservas nuevas, sin objetivo, con las 17 columnas."""
    return crear_datos(filas=10, semilla=99)[list(COLUMNAS)]

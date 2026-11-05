"""Lectura, esquema y partición de los datos de entrenamiento."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


NUMERICAS = (
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
)
CATEGORICAS = (
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "customer_type",
)
COLUMNAS = NUMERICAS + CATEGORICAS
TEST_SIZE = 0.20


def cargar_datos(ruta: Path, objetivo: str) -> tuple[pd.DataFrame, pd.Series]:
    """Lee un CSV etiquetado y devuelve únicamente las 17 entradas de Fase 1."""
    if not ruta.is_file():
        raise ValueError(f"No existe el CSV de entrenamiento: {ruta}")
    try:
        datos = pd.read_csv(ruta)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"No se pudo leer el CSV {ruta}: {exc}") from exc

    if not objetivo or objetivo in COLUMNAS:
        raise ValueError("El objetivo debe ser una columna binaria distinta de las 17 entradas.")
    faltantes = [col for col in (*COLUMNAS, objetivo) if col not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")
    if datos.empty:
        raise ValueError("El CSV no contiene observaciones.")

    entradas = datos.loc[:, COLUMNAS].copy()
    for columna in NUMERICAS:
        original = entradas[columna]
        convertido = pd.to_numeric(original, errors="coerce")
        if (original.notna() & convertido.isna()).any() or not np.isfinite(
            convertido.dropna().to_numpy(dtype=float)
        ).all():
            raise ValueError(f"La columna numérica '{columna}' contiene valores incompatibles.")
        entradas[columna] = convertido

    for columna in CATEGORICAS:
        if not entradas[columna].dropna().map(lambda valor: isinstance(valor, str)).all():
            raise ValueError(f"La columna categórica '{columna}' debe contener texto.")

    try:
        etiquetas = pd.to_numeric(datos[objetivo], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"El objetivo '{objetivo}' debe contener solo 0 y 1.") from exc
    if etiquetas.isna().any() or not etiquetas.isin([0, 1]).all():
        raise ValueError(f"El objetivo '{objetivo}' debe contener solo 0 y 1, sin nulos.")
    if etiquetas.nunique() != 2:
        raise ValueError("El objetivo debe contener ambas clases: 0 y 1.")

    return entradas, etiquetas.astype(int)


def separar_grupos(
    entradas: pd.DataFrame, etiquetas: pd.Series, semilla: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """Separa grupos de entradas idénticas antes de ajustar transformaciones."""
    grupos = pd.util.hash_pandas_object(entradas, index=False)
    if grupos.nunique() < 2:
        raise ValueError("Se requieren al menos dos grupos de entradas diferentes.")
    separador = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=semilla)
    indices_train, indices_test = next(separador.split(entradas, etiquetas, groups=grupos))
    x_train, x_test = entradas.iloc[indices_train].copy(), entradas.iloc[indices_test].copy()
    y_train, y_test = etiquetas.iloc[indices_train].copy(), etiquetas.iloc[indices_test].copy()
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "La partición por grupos dejó una clase fuera de entrenamiento o prueba; "
            "use más datos o cambie la semilla."
        )
    if not set(grupos.iloc[indices_train]).isdisjoint(grupos.iloc[indices_test]):
        raise RuntimeError("La partición comparte entradas idénticas entre conjuntos.")
    return x_train, x_test, y_train, y_test, int(grupos.nunique())

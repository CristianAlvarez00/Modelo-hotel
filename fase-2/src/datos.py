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
    if not ruta.is_file():  # Verifica que el archivo exista antes de abrirlo.
        raise ValueError(f"No existe el CSV de entrenamiento: {ruta}")
    try:  # Intenta abrir el archivo CSV y prepara la captura de errores.
        datos = pd.read_csv(ruta)  # Carga la tabla desde el archivo CSV.
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:  # Si falla la lectura, lanza un error claro.
        raise ValueError(f"No se pudo leer el CSV {ruta}: {exc}") from exc

    if not objetivo or objetivo in COLUMNAS:  # Verifica que el objetivo exista y no sea una variable de entrada.
        raise ValueError("El objetivo debe ser una columna binaria distinta de las 17 entradas.")
    faltantes = [col for col in (*COLUMNAS, objetivo) if col not in datos.columns]  # Busca columnas que faltan en el CSV.
    if faltantes:  # Si faltan columnas, corta la ejecución con un mensaje claro.
        raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")
    if datos.empty:  # Si el archivo está vacío, no tiene sentido seguir.
        raise ValueError("El CSV no contiene observaciones.")

    entradas = datos.loc[:, COLUMNAS].copy()  # Selecciona solo las columnas de entrada y hace una copia para no tocar el original.
    for columna in NUMERICAS:  # Recorre cada columna numérica para validarla.
        original = entradas[columna]  # Guarda la serie original antes de convertirla.
        convertido = pd.to_numeric(original, errors="coerce")  # Intenta convertir a número y deja nulos si hay valores raros.
        if (original.notna() & convertido.isna()).any() or not np.isfinite(
            convertido.dropna().to_numpy(dtype=float)
        ).all():  # Si hay valores no numéricos o no finitos, no sirve.
            raise ValueError(f"La columna numérica '{columna}' contiene valores incompatibles.")
        entradas[columna] = convertido  # Guarda la versión limpia y numérica de la columna.

    for columna in CATEGORICAS:  # Recorre cada columna categórica para revisar su formato.
        if not entradas[columna].dropna().map(lambda valor: isinstance(valor, str)).all():  # Revisa que los valores no nulos sean texto.
            raise ValueError(f"La columna categórica '{columna}' debe contener texto.")

    try:  # Intenta convertir la columna objetivo a números para validar la etiqueta.
        etiquetas = pd.to_numeric(datos[objetivo], errors="raise")
    except (TypeError, ValueError) as exc:  # Si la etiqueta no es numérica o válida, lanza error.
        raise ValueError(f"El objetivo '{objetivo}' debe contener solo 0 y 1.") from exc
    if etiquetas.isna().any() or not etiquetas.isin([0, 1]).all():  # Comprueba que no haya nulos ni valores fuera de 0 y 1.
        raise ValueError(f"El objetivo '{objetivo}' debe contener solo 0 y 1, sin nulos.")
    if etiquetas.nunique() != 2:  # Revisa que existan exactamente dos clases posibles.
        raise ValueError("El objetivo debe contener ambas clases: 0 y 1.")

    return entradas, etiquetas.astype(int)  # Devuelve las entradas limpias y la meta convertida a enteros.


def separar_grupos(
    entradas: pd.DataFrame, etiquetas: pd.Series, semilla: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """Separa grupos de entradas idénticas antes de ajustar transformaciones."""
    grupos = pd.util.hash_pandas_object(entradas, index=False)
    if grupos.nunique() < 2:
        raise ValueError("Se requieren al menos dos grupos de entradas diferentes.")
    separador = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=semilla)  # Crea el separador con una semilla fija.
    indices_train, indices_test = next(separador.split(entradas, etiquetas, groups=grupos))  # Genera índices de entrenamiento y prueba por grupo.
    x_train, x_test = entradas.iloc[indices_train].copy(), entradas.iloc[indices_test].copy()  # Separar las entradas de entrenamiento y prueba.
    y_train, y_test = etiquetas.iloc[indices_train].copy(), etiquetas.iloc[indices_test].copy()  # Separar las etiquetas de entrenamiento y prueba.
    if y_train.nunique() != 2 or y_test.nunique() != 2:  # Verifica que cada partición tenga ambas clases.
        raise ValueError(
            "La partición por grupos dejó una clase fuera de entrenamiento o prueba; "
            "use más datos o cambie la semilla."
        )
    if not set(grupos.iloc[indices_train]).isdisjoint(grupos.iloc[indices_test]):  # Si hay filas idénticas en ambos lados, hay fuga de datos.
        raise RuntimeError("La partición comparte entradas idénticas entre conjuntos.")
    return x_train, x_test, y_train, y_test, int(grupos.nunique())  # Devuelve los conjuntos y la cantidad total de grupos únicos.

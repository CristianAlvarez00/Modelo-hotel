"""Carga de modelos versionados, validación de entradas y predicción por lotes.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .datos import CATEGORICAS, COLUMNAS, NUMERICAS
from .modelo import aplicar_umbral


CLAVES_ARTEFACTO = ("pipeline", "umbral", "columnas", "objetivo", "version")


class ErrorPrediccion(ValueError):
    """Error previsible de uso: modelo o CSV inválido. El CLI lo muestra sin traza."""


def cargar_modelo(ruta: Path) -> dict:
    """Carga un artefacto desde `modelo.joblib` o desde la carpeta de una versión."""
    ruta = Path(ruta)
    if ruta.is_dir():
        ruta = ruta / "modelo.joblib"
    if not ruta.is_file():
        raise ErrorPrediccion(f"No existe el modelo: {ruta}")
    try:
        # joblib usa pickle: solo cargar modelos propios (los que crea train.py).
        artefacto = joblib.load(ruta)
    except Exception as exc:  # joblib/pickle pueden fallar de muchas formas
        raise ErrorPrediccion(f"No se pudo leer el modelo {ruta}: {exc}") from exc

    if not isinstance(artefacto, dict):
        raise ErrorPrediccion(f"El archivo {ruta} no es un artefacto de modelo válido.")
    faltantes = [clave for clave in CLAVES_ARTEFACTO if clave not in artefacto]
    if faltantes:
        raise ErrorPrediccion(f"El artefacto {ruta} no contiene: {', '.join(faltantes)}")
    if list(artefacto["columnas"]) != list(COLUMNAS):
        raise ErrorPrediccion(
            f"El modelo {ruta} usa columnas distintas a las de este código; "
            "no es compatible."
        )
    return artefacto


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee el CSV de entrada con las mismas reglas de nulos que el entrenamiento."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorPrediccion(f"No existe el CSV de entrada: {ruta}")
    try:
        return pd.read_csv(ruta)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ErrorPrediccion(f"No se pudo leer el CSV {ruta}: {exc}") from exc


def validar_entradas(datos: pd.DataFrame, objetivo: str) -> tuple[pd.DataFrame, list[str]]:
    """Devuelve las 17 entradas en el orden del modelo y una lista de advertencias.
    """
    if datos.empty:
        raise ErrorPrediccion("El CSV de entrada no contiene observaciones.")
    faltantes = [col for col in COLUMNAS if col not in datos.columns]
    if faltantes:
        raise ErrorPrediccion(f"Faltan columnas requeridas: {', '.join(faltantes)}")

    advertencias = []
    extras = [col for col in datos.columns if col not in COLUMNAS and col != objetivo]
    if extras:
        advertencias.append(f"Columnas extra ignoradas: {', '.join(map(str, extras))}")

    entradas = datos.loc[:, list(COLUMNAS)].copy()
    for columna in NUMERICAS:
        original = entradas[columna]
        convertido = pd.to_numeric(original, errors="coerce")
        if (original.notna() & convertido.isna()).any():
            raise ErrorPrediccion(f"La columna numérica '{columna}' contiene texto no numérico.")
        if not np.isfinite(convertido.dropna().to_numpy(dtype=float)).all():
            raise ErrorPrediccion(f"La columna numérica '{columna}' contiene valores infinitos.")
        entradas[columna] = convertido
    for columna in CATEGORICAS:
        if not entradas[columna].dropna().map(lambda valor: isinstance(valor, str)).all():
            raise ErrorPrediccion(f"La columna categórica '{columna}' debe contener texto.")
    return entradas, advertencias


def categorias_desconocidas(pipeline, entradas: pd.DataFrame) -> dict[str, int]:
    """Cuenta valores no nulos que el codificador nunca vio (se codifican en ceros)."""
    codificador = (
        pipeline.named_steps["preprocesador"]
        .named_transformers_["categoricas"]
        .named_steps["codificar"]
    )
    resultado = {}
    for columna, vistas in zip(CATEGORICAS, codificador.categories_):
        nuevos = int((entradas[columna].notna() & ~entradas[columna].isin(vistas)).sum())
        if nuevos:
            resultado[columna] = nuevos
    return resultado


def predecir(artefacto: dict, entradas: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline y el umbral guardados; conserva el orden de las filas."""
    probabilidades = artefacto["pipeline"].predict_proba(entradas)[:, 1]
    resultado = entradas.copy()
    resultado["prediccion"] = aplicar_umbral(probabilidades, artefacto["umbral"])
    resultado["probabilidad_cancelacion"] = probabilidades
    return resultado


def escribir_salida(resultado: pd.DataFrame, ruta: Path) -> None:
    """Escribe el CSV completo o nada: archivo temporal y renombrado atómico."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    try:
        resultado.to_csv(temporal, index=False)
        os.replace(temporal, ruta)
    except OSError as exc:
        temporal.unlink(missing_ok=True)
        raise ErrorPrediccion(f"No se pudo escribir la salida {ruta}: {exc}") from exc


def predecir_csv(modelo: Path, entrada: Path, salida: Path) -> tuple[pd.DataFrame, list[str]]:
    """Flujo completo. Todo se valida antes de crear el archivo de salida."""
    artefacto = cargar_modelo(modelo)
    entradas, advertencias = validar_entradas(leer_csv(entrada), artefacto["objetivo"])
    nuevas = categorias_desconocidas(artefacto["pipeline"], entradas)
    if nuevas:
        detalle = ", ".join(f"{col} ({n})" for col, n in nuevas.items())
        advertencias.append(f"Categorías no vistas en entrenamiento (se codifican en ceros): {detalle}")
    resultado = predecir(artefacto, entradas)
    escribir_salida(resultado, salida)
    return resultado, advertencias

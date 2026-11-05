"""Entrena y versiona el modelo de cancelación sin ejecutar el notebook."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import joblib
import pandas as pd
import sklearn

from src.datos import CATEGORICAS, COLUMNAS, NUMERICAS, TEST_SIZE, cargar_datos, separar_grupos
from src.modelo import MAX_ITER, UMBRAL, aplicar_umbral, calcular_metricas, crear_baseline, crear_pipeline


def referencia_codigo() -> str:
    """Huella de los tres archivos fuente, útil incluso antes de hacer commit."""
    base = Path(__file__).resolve().parent
    huella = hashlib.sha256()
    for ruta in (base / "train.py", base / "src/datos.py", base / "src/modelo.py"):
        huella.update(ruta.relative_to(base).as_posix().encode())
        huella.update(ruta.read_bytes())
    return huella.hexdigest()


def entrenar(data: Path, target: str, output_dir: Path, version: str, seed: int) -> Path:
    """Entrena y crea una versión nueva; jamás sustituye una existente."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("La versión debe usar letras, números, punto, guion o guion bajo.")
    destino = output_dir / version
    if destino.exists():
        raise FileExistsError(f"La versión '{version}' ya existe: {destino}")

    entradas, etiquetas = cargar_datos(data, target)
    x_train, x_test, y_train, y_test, n_grupos = separar_grupos(entradas, etiquetas, seed)

    pipeline = crear_pipeline(seed)
    pipeline.fit(x_train, y_train)  # Preprocesamiento se ajusta solo aquí.
    pred_modelo = aplicar_umbral(pipeline.predict_proba(x_test)[:, 1])

    baseline = crear_baseline()
    baseline.fit(x_train, y_train)
    pred_baseline = baseline.predict(x_test)
    metricas = {
        "conjunto": "prueba",
        "modelo": calcular_metricas(y_test, pred_modelo),
        "baseline": calcular_metricas(y_test, pred_baseline),
    }
    metadata = {
        "version": version,
        "creado_utc": datetime.now(timezone.utc).isoformat(),
        "problema": "clasificacion_binaria",
        "objetivo": target,
        "algoritmo": "LogisticRegression",
        "columnas": list(COLUMNAS),
        "tipos_esperados": {**dict.fromkeys(NUMERICAS, "numerico"),
                            **dict.fromkeys(CATEGORICAS, "texto")},
        "semilla": seed,
        "umbral": UMBRAL,
        "origen_umbral": "Fase 1: mejor F1 en validacion; fijo antes de prueba",
        "datos": {"archivo": str(data), "observaciones": len(entradas), "grupos": n_grupos},
        "particion": {
            "metodo": "GroupShuffleSplit sobre hash de las 17 entradas",
            "test_size_grupos": TEST_SIZE,
            "filas_entrenamiento": len(x_train),
            "filas_prueba": len(x_test),
        },
        "metrica_principal": "f1",
        "metricas": metricas,
        "parametros": {
            "regresion_logistica": {"max_iter": MAX_ITER, "random_state": seed},
            "baseline": {"strategy": "most_frequent"},
            "imputacion_numerica": "median",
            "imputacion_categorica": "Desconocido",
            "codificacion": "OneHotEncoder(handle_unknown='ignore')",
            "escalamiento": "StandardScaler",
        },
        "entorno": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "requirements": "fase-2/requirements.txt (pendiente Persona 3)",
        },
        "codigo_sha256": referencia_codigo(),
    }
    artefacto = {
        "pipeline": pipeline,
        "umbral": UMBRAL,
        "columnas": list(COLUMNAS),
        "objetivo": target,
        "version": version,
    }

    destino.mkdir(parents=True, exist_ok=False)
    joblib.dump(artefacto, destino / "modelo.joblib")
    for nombre, contenido in (("metrics.json", metricas), ("metadata.json", metadata)):
        (destino / nombre).write_text(
            json.dumps(contenido, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Versión {version}: {len(x_train)} filas de entrenamiento, {len(x_test)} de prueba")
    for nombre in ("modelo", "baseline"):
        valores = metricas[nombre]
        print(f"{nombre}: " + ", ".join(f"{k}={v:.3f}" for k, v in valores.items()))
    for nombre in ("modelo.joblib", "metrics.json", "metadata.json"):
        print(f"Guardado: {destino / nombre}")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="CSV etiquetado de reservas")
    parser.add_argument("--target", required=True, help="Columna objetivo binaria (0/1)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Carpeta de versiones")
    parser.add_argument("--version", required=True, help="Identificador nuevo, por ejemplo v1")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición (42)")
    args = parser.parse_args()
    try:
        entrenar(args.data, args.target, args.output_dir, args.version, args.seed)
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

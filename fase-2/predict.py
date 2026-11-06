"""Predice cancelaciones con una versión guardada; el CSV no necesita el objetivo."""

import argparse
from pathlib import Path
import sys

from src.prediccion import ErrorPrediccion, predecir_csv


MODELOS_POR_DEFECTO = Path(__file__).resolve().parent / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modelo = parser.add_mutually_exclusive_group(required=True)
    modelo.add_argument("--version", help="Versión guardada, por ejemplo v1")
    modelo.add_argument("--model", type=Path, help="Ruta a un modelo.joblib o a su carpeta")
    parser.add_argument("--models-dir", type=Path, default=MODELOS_POR_DEFECTO,
                        help="Carpeta con las versiones (models/); se usa con --version")
    parser.add_argument("--input", required=True, type=Path, help="CSV con las 17 columnas")
    parser.add_argument("--output", required=True, type=Path, help="CSV de salida")
    parser.add_argument("--overwrite", action="store_true",
                        help="Permite reemplazar el CSV de salida si ya existe")
    args = parser.parse_args()

    try:
        if args.version is not None:
            if Path(args.version).name != args.version:
                raise ErrorPrediccion("La versión no puede contener rutas.")
            ruta_modelo = args.models_dir / args.version
        else:
            ruta_modelo = args.model
        if args.output.exists() and not args.overwrite:
            raise ErrorPrediccion(
                f"La salida ya existe: {args.output} (use --overwrite para reemplazarla)"
            )
        if args.output.resolve() == args.input.resolve():
            raise ErrorPrediccion("La salida no puede ser el mismo archivo que la entrada.")
        resultado, advertencias = predecir_csv(ruta_modelo, args.input, args.output)
    except (ErrorPrediccion, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for aviso in advertencias:
        print(f"Aviso: {aviso}", file=sys.stderr)
    canceladas = int(resultado["prediccion"].sum())
    print(f"{len(resultado)} filas procesadas; {canceladas} predichas como cancelación (1).")
    print(f"Guardado: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

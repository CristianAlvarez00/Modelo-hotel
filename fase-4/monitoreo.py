"""Calcula dos métricas de operación a partir del registro de POST /predict.

    python fase-4/monitoreo.py --registro fase-4/registros/solicitudes.csv

Muestra el total de solicitudes y el porcentaje de fallidas (HTTP 400 a 599).
Estas métricas describen el servicio, no la calidad de las predicciones.
"""

import argparse
import csv
import os
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
REGISTRO_POR_DEFECTO = RAIZ / "fase-4/registros/solicitudes.csv"


class ErrorRegistro(ValueError):
    """Registro inexistente o con un formato que no permite contar."""


def contar(ruta: Path) -> tuple[int, int]:
    """Devuelve (total de solicitudes, solicitudes fallidas)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorRegistro(f"No existe el registro: {ruta}")
    total = fallidas = 0
    try:
        with ruta.open(newline="", encoding="utf-8") as archivo:
            lector = csv.DictReader(archivo)
            if lector.fieldnames is None:  # Archivo vacío: no hay solicitudes.
                return 0, 0
            if "estado_http" not in lector.fieldnames:
                raise ErrorRegistro("El registro no tiene la columna 'estado_http'.")
            for fila in lector:
                try:
                    estado = int((fila["estado_http"] or "").strip())
                except ValueError:
                    raise ErrorRegistro(
                        f"Línea {lector.line_num}: estado_http no es un número entero."
                    ) from None
                total += 1
                if 400 <= estado <= 599:
                    fallidas += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ErrorRegistro(f"No se pudo leer el registro {ruta}: {exc}") from exc
    return total, fallidas


def resumen(total: int, fallidas: int) -> str:
    if total == 0:
        porcentaje = "sin solicitudes"
    else:
        porcentaje = f"{100 * fallidas / total:.1f}".replace(".", ",") + " %"
    return (
        f"Total de solicitudes: {total}\n"
        f"Solicitudes fallidas (HTTP 400 a 599): {fallidas}\n"
        f"Porcentaje de fallos: {porcentaje}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--registro", type=Path,
        default=Path(os.getenv("REGISTRO_CSV", str(REGISTRO_POR_DEFECTO))),
        help="CSV del registro (por defecto REGISTRO_CSV o fase-4/registros/solicitudes.csv)",
    )
    args = parser.parse_args()
    try:
        total, fallidas = contar(args.registro)
    except ErrorRegistro as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(resumen(total, fallidas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

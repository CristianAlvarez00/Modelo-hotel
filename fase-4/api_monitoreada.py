"""Punto de entrada de Fase 4: la API de Fase 3 con registro de POST /predict.

    python -m uvicorn api_monitoreada:app --app-dir fase-4 --host 127.0.0.1 --port 8000

Variables de entorno: las de Fase 3 (MODELOS_DIR, MODELO_VERSION, ...) y
REGISTRO_CSV, la ruta del registro (por defecto fase-4/registros/solicitudes.csv).
"""

import os
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
# Las carpetas fase-2 y fase-3 tienen guion: se agregan al path para importar sus módulos.
for carpeta in (RAIZ / "fase-3", RAIZ / "fase-2", Path(__file__).resolve().parent):
    if str(carpeta) not in sys.path:
        sys.path.insert(0, str(carpeta))

from api import crear_app
from configuracion import Configuracion
from registro import instalar_registro

REGISTRO_POR_DEFECTO = RAIZ / "fase-4/registros/solicitudes.csv"


def crear_app_monitoreada(config: Configuracion | None = None, ruta_registro: Path | None = None):
    ruta = Path(ruta_registro or os.getenv("REGISTRO_CSV", str(REGISTRO_POR_DEFECTO)))
    return instalar_registro(crear_app(config), ruta)


app = crear_app_monitoreada()

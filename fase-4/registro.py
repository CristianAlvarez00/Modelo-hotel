"""Registro de solicitudes a POST /predict (Fase 4).

Un middleware escribe exactamente una fila por intento, cuando ya se conoce el
código HTTP: también las rechazadas por validación (422), los errores de JSON
y los 503. No guarda el cuerpo de la solicitud, solo cuatro campos.
"""

import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
from threading import Lock

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

CAMPOS = ("fecha_hora", "estado_http", "version", "prediccion")
RUTA_REGISTRADA = "/predict"
log = logging.getLogger("registro")


class RegistroCSV:
    """CSV abierto siempre en modo agregar: el historial se conserva entre ejecuciones."""

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self._candado = Lock()  # Usar un solo proceso de Uvicorn.

    def agregar(self, estado_http: int, version: str = "", prediccion: str = "") -> None:
        fila = {
            "fecha_hora": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "estado_http": estado_http,
            "version": version,
            "prediccion": prediccion,
        }
        with self._candado:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            es_nuevo = not self.ruta.exists() or self.ruta.stat().st_size == 0
            with self.ruta.open("a", newline="", encoding="utf-8") as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=CAMPOS, lineterminator="\n")
                if es_nuevo:
                    escritor.writeheader()  # Solo al crear el archivo.
                escritor.writerow(fila)


class MiddlewareRegistro(BaseHTTPMiddleware):
    def __init__(self, app, registro: RegistroCSV):
        super().__init__(app)
        self.registro = registro

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path != RUTA_REGISTRADA:
            return await call_next(request)
        try:
            respuesta = await call_next(request)
        except Exception:
            await self._registrar(500, request)  # Error no controlado: se registra y se propaga.
            raise
        await self._registrar(respuesta.status_code, request)
        return respuesta

    async def _registrar(self, estado: int, request: Request) -> None:
        version, prediccion = "", ""
        if estado < 400:  # La ruta solo deja estos datos en las respuestas exitosas.
            version = str(getattr(request.state, "version", ""))
            prediccion = str(getattr(request.state, "prediccion", ""))
        try:
            await run_in_threadpool(self.registro.agregar, estado, version, prediccion)
        except OSError as exc:  # Un fallo del registro no debe tumbar la API.
            log.error("No se pudo escribir el registro %s: %s", self.registro.ruta, exc)


def instalar_registro(app: FastAPI, ruta: Path) -> FastAPI:
    """Envuelve una app existente sin tocar sus rutas."""
    registro = RegistroCSV(ruta)
    app.state.registro = registro
    app.add_middleware(MiddlewareRegistro, registro=registro)
    return app

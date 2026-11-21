"""API del modelo hotelero; reutiliza el entrenamiento y la inferencia de Fase 2."""

from datetime import datetime, timezone
from pathlib import Path
import sys
from threading import Lock
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

# fase-2 tiene guion en el nombre: se agrega su carpeta para importar sus módulos.
FASE2 = Path(__file__).resolve().parents[1] / "fase-2"
if str(FASE2) not in sys.path:
    sys.path.insert(0, str(FASE2))

from train import entrenar
from src.prediccion import ErrorPrediccion, cargar_modelo, predecir, validar_entradas
from configuracion import Configuracion
from esquemas import Reserva, SolicitudEntrenamiento


def crear_app(config: Configuracion | None = None) -> FastAPI:
    config = config or Configuracion.desde_entorno()
    servicio = FastAPI(title="Cancelación de reservas", version="3.0")
    servicio.state.config = config
    candado_entrenamiento = Lock()

    @servicio.get("/health")
    def health():
        # Indica que el servidor responde; /predict comprueba el modelo por separado.
        return {"estado": "en ejecucion"}

    @servicio.post("/predict")
    def predict(reserva: Reserva, request: Request):
        try:
            artefacto = cargar_modelo(config.modelos / config.version)
        except ErrorPrediccion as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        datos = pd.DataFrame([{
            clave: np.nan if valor is None else valor
            for clave, valor in reserva.model_dump().items()
        }])
        try:
            entradas, _ = validar_entradas(datos, artefacto["objetivo"])
            fila = predecir(artefacto, entradas).iloc[0]
        except ErrorPrediccion as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise HTTPException(status_code=503, detail="El modelo guardado no puede predecir esta entrada.") from exc
        resultado = {
            "prediccion": int(fila["prediccion"]),
            "probabilidad_cancelacion": float(fila["probabilidad_cancelacion"]),
            "version": artefacto["version"],
        }
        # Persona 2 puede leer estos valores en su middleware, sin analizar la respuesta.
        request.state.version = resultado["version"]
        request.state.prediccion = resultado["prediccion"]
        return resultado

    @servicio.post("/train", status_code=201)
    def train(solicitud: SolicitudEntrenamiento | None = None):
        # La configuración fija los datos; el cliente puede enviar {} o cuerpo vacío.
        if not candado_entrenamiento.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Ya hay un entrenamiento en curso.")
        try:
            fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            version = f"api-{fecha}-{uuid4().hex[:12]}"
            entrenar(config.datos, config.objetivo, config.modelos, version,
                     config.semilla, config.test_size)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=503, detail=f"No se pudo entrenar: {exc}") from exc
        finally:
            candado_entrenamiento.release()
        return {"version": version}

    return servicio


app = crear_app()

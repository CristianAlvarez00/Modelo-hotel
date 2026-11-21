"""API sobre modelos pequeños; no usa red ni modifica las versiones de entrega."""
from pathlib import Path
import sys
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parents[2]
for carpeta in (RAIZ / "fase-2", RAIZ / "fase-3"):
    sys.path.insert(0, str(carpeta))

from api import crear_app
from configuracion import Configuracion
from train import entrenar
from src.prediccion import cargar_modelo, predecir, validar_entradas


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("api")
    # Muestra local pequeña, con ambas clases. No descarga ni usa el dataset completo.
    datos = pd.read_csv(RAIZ / "data/hotel_bookings.csv", nrows=1500)
    csv = carpeta / "datos.csv"
    datos.to_csv(csv, index=False)
    modelos = carpeta / "modelos"
    entrenar(csv, "is_canceled", modelos, "pequeno", 42)
    return Configuracion(modelos=modelos, version="pequeno", datos=csv)


@pytest.fixture
def reserva():
    return json.loads(pd.read_csv(RAIZ / "fase-2/data/ejemplo_prediccion.csv").head(1).to_json(orient="records"))[0]


def test_health(entorno):
    with TestClient(crear_app(entorno)) as cliente:
        r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "en ejecucion"


def test_predict(entorno, reserva):
    with TestClient(crear_app(entorno)) as cliente:
        r = cliente.post("/predict", json=reserva)
    assert r.status_code == 200
    assert r.json()["version"] == "pequeno"
    assert r.json()["prediccion"] in (0, 1)
    modelo = cargar_modelo(entorno.modelos / entorno.version)
    x, _ = validar_entradas(pd.DataFrame([reserva]), "is_canceled")
    esperado = predecir(modelo, x).iloc[0]
    assert r.json()["probabilidad_cancelacion"] == pytest.approx(esperado["probabilidad_cancelacion"])


def test_campo_ausente(entorno, reserva):
    reserva.pop("hotel")
    with TestClient(crear_app(entorno)) as cliente:
        r = cliente.post("/predict", json=reserva)
    assert r.status_code == 422
    assert "hotel" in r.text


@pytest.mark.parametrize("valor", ["20", True, "abc"])
def test_tipo_incompatible(entorno, reserva, valor):
    reserva["lead_time"] = valor
    with TestClient(crear_app(entorno)) as cliente:
        r = cliente.post("/predict", json=reserva)
    assert r.status_code == 422
    assert "lead_time" in r.text


def test_nulos_y_categoria_nueva(entorno, reserva):
    reserva.update(children=None, country=None, hotel="Hotel nuevo")
    with TestClient(crear_app(entorno)) as cliente:
        r = cliente.post("/predict", json=reserva)
    assert r.status_code == 200
    assert 0 <= r.json()["probabilidad_cancelacion"] <= 1


def test_modelo_ausente(tmp_path, reserva):
    with TestClient(crear_app(Configuracion(modelos=tmp_path))) as cliente:
        assert cliente.get("/health").status_code == 200
        r = cliente.post("/predict", json=reserva)
    assert r.status_code == 503
    assert "No existe el modelo" in r.text


def test_train_conserva_version_activa(entorno, reserva):
    ruta = entorno.modelos / "pequeno/modelo.joblib"
    antes = ruta.read_bytes()
    with TestClient(crear_app(entorno)) as cliente:
        r = cliente.post("/train", json={})
        assert r.status_code == 201
        nueva = r.json()["version"]
        for nombre in ("modelo.joblib", "metrics.json", "metadata.json"):
            assert (entorno.modelos / nueva / nombre).is_file()
        assert cargar_modelo(entorno.modelos / nueva)["version"] == nueva
        assert cliente.post("/predict", json=reserva).json()["version"] == "pequeno"
    assert ruta.read_bytes() == antes

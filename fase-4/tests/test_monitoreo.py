"""Registro de POST /predict y script de métricas; sin red ni modelos de entrega."""
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "fase-4"))

from api_monitoreada import crear_app_monitoreada  # agrega fase-2 y fase-3 al path
from api import crear_app
from configuracion import Configuracion
from train import entrenar
from monitoreo import ErrorRegistro, contar, main, resumen
from registro import CAMPOS


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("monitoreo")
    # Muestra pequeña del dataset local; no usa red ni las versiones de entrega.
    datos = pd.read_csv(RAIZ / "data/hotel_bookings.csv", nrows=1500)
    csv_datos = carpeta / "datos.csv"
    datos.to_csv(csv_datos, index=False)
    modelos = carpeta / "modelos"
    entrenar(csv_datos, "is_canceled", modelos, "pequeno", 42)
    return Configuracion(modelos=modelos, version="pequeno", datos=csv_datos)


@pytest.fixture
def reserva():
    ejemplo = pd.read_csv(RAIZ / "fase-2/data/ejemplo_prediccion.csv").head(1)
    return json.loads(ejemplo.to_json(orient="records"))[0]


def filas(ruta):
    with Path(ruta).open(newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def test_dos_validas_y_una_invalida(config, reserva, tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    sin_hotel = {k: v for k, v in reserva.items() if k != "hotel"}
    with TestClient(crear_app_monitoreada(config, ruta)) as cliente:
        assert cliente.post("/predict", json=reserva).status_code == 200
        assert cliente.post("/predict", json=reserva).status_code == 200
        assert cliente.post("/predict", json=sin_hotel).status_code == 422

    registro = filas(ruta)
    assert [f["estado_http"] for f in registro] == ["200", "200", "422"]
    assert registro[0]["version"] == "pequeno" and registro[0]["prediccion"] in ("0", "1")
    assert registro[2]["version"] == "" and registro[2]["prediccion"] == ""
    total, fallidas = contar(ruta)
    assert (total, fallidas) == (3, 1)
    assert "33,3 %" in resumen(total, fallidas)


def solicitud_sin_hotel(cliente, reserva):
    return cliente.post("/predict", json={k: v for k, v in reserva.items() if k != "hotel"})


def solicitud_tipo_malo(cliente, reserva):
    return cliente.post("/predict", json={**reserva, "lead_time": "abc"})


def solicitud_json_roto(cliente, reserva):
    return cliente.post("/predict", content=b"{no es json",
                        headers={"Content-Type": "application/json"})


def solicitud_cuerpo_vacio(cliente, reserva):
    return cliente.post("/predict")


@pytest.mark.parametrize("enviar", [
    solicitud_sin_hotel, solicitud_tipo_malo, solicitud_json_roto, solicitud_cuerpo_vacio,
])
def test_rechazo_antes_de_la_ruta_deja_una_fila(config, reserva, tmp_path, enviar):
    ruta = tmp_path / "solicitudes.csv"
    with TestClient(crear_app_monitoreada(config, ruta)) as cliente:
        assert enviar(cliente, reserva).status_code == 422
    registro = filas(ruta)
    assert len(registro) == 1
    assert registro[0]["estado_http"] == "422"
    assert registro[0]["version"] == "" and registro[0]["prediccion"] == ""


def test_modelo_no_disponible_registra_503(reserva, tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    sin_modelo = Configuracion(modelos=tmp_path / "no_hay_modelos")
    with TestClient(crear_app_monitoreada(sin_modelo, ruta)) as cliente:
        assert cliente.post("/predict", json=reserva).status_code == 503
    registro = filas(ruta)
    assert len(registro) == 1 and registro[0]["estado_http"] == "503"
    assert registro[0]["version"] == "" and registro[0]["prediccion"] == ""


def test_fecha_con_zona_horaria(config, reserva, tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    with TestClient(crear_app_monitoreada(config, ruta)) as cliente:
        cliente.post("/predict", json=reserva)
    fecha = datetime.fromisoformat(filas(ruta)[0]["fecha_hora"])
    assert fecha.tzinfo is not None


def test_solo_se_registra_post_predict(config, tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    with TestClient(crear_app_monitoreada(config, ruta)) as cliente:
        assert cliente.get("/health").status_code == 200
        assert cliente.get("/predict").status_code == 405
    assert not ruta.exists()


def test_registro_persiste_al_reiniciar(config, reserva, tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    with TestClient(crear_app_monitoreada(config, ruta)) as cliente:
        cliente.post("/predict", json=reserva)
        cliente.post("/predict", json=reserva)
    antes = filas(ruta)
    # "Reinicio": una app nueva sobre el mismo archivo.
    with TestClient(crear_app_monitoreada(config, ruta)) as cliente:
        cliente.post("/predict", json={"falta": "todo"})
    despues = filas(ruta)
    assert len(despues) == 3
    assert despues[:2] == antes  # lo anterior no se reescribe ni se cuenta dos veces
    assert ruta.read_text(encoding="utf-8").count("fecha_hora") == 1  # un solo encabezado


def test_no_guarda_el_cuerpo_de_la_solicitud(config, reserva, tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    with TestClient(crear_app_monitoreada(config, ruta)) as cliente:
        cliente.post("/predict", json=reserva)
    texto = ruta.read_text(encoding="utf-8")
    assert texto.splitlines()[0].split(",") == list(CAMPOS)
    for dato in (reserva["hotel"], reserva["country"], reserva["market_segment"]):
        assert dato not in texto


def test_el_registro_no_cambia_las_respuestas(config, reserva, tmp_path):
    casos = [reserva, {k: v for k, v in reserva.items() if k != "hotel"}]
    with TestClient(crear_app(config)) as base, \
            TestClient(crear_app_monitoreada(config, tmp_path / "r.csv")) as monitoreada:
        for caso in casos:
            esperado, obtenido = base.post("/predict", json=caso), monitoreada.post("/predict", json=caso)
            assert obtenido.status_code == esperado.status_code
            assert obtenido.json() == esperado.json()


def test_metricas_con_registro_vacio(tmp_path):
    vacio = tmp_path / "vacio.csv"
    vacio.write_text("", encoding="utf-8")
    solo_encabezado = tmp_path / "encabezado.csv"
    solo_encabezado.write_text(",".join(CAMPOS) + "\n", encoding="utf-8")
    for ruta in (vacio, solo_encabezado):
        total, fallidas = contar(ruta)
        assert (total, fallidas) == (0, 0)
        assert "sin solicitudes" in resumen(total, fallidas)


def test_metricas_rechazan_registro_invalido(tmp_path, monkeypatch, capsys):
    with pytest.raises(ErrorRegistro, match="No existe"):
        contar(tmp_path / "no_existe.csv")
    malo = tmp_path / "malo.csv"
    malo.write_text(",".join(CAMPOS) + "\nhoy,abc,,\n", encoding="utf-8")
    with pytest.raises(ErrorRegistro, match="estado_http"):
        contar(malo)
    # Por terminal: código distinto de cero y mensaje en stderr.
    monkeypatch.setattr(sys, "argv", ["monitoreo.py", "--registro", str(malo)])
    assert main() == 1
    assert "estado_http" in capsys.readouterr().err


def test_metricas_por_terminal(tmp_path, monkeypatch, capsys):
    ruta = tmp_path / "solicitudes.csv"
    ruta.write_text(",".join(CAMPOS) + "\nt,200,v1,1\nt,200,v1,0\nt,422,,\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["monitoreo.py", "--registro", str(ruta)])
    assert main() == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 3" in salida and "33,3 %" in salida

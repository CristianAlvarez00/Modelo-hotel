import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

import predict
from src.datos import COLUMNAS, cargar_datos, separar_grupos
from src.modelo import aplicar_umbral, crear_pipeline
from src.prediccion import ErrorPrediccion, cargar_modelo, predecir_csv
from train import entrenar

from conftest import OBJETIVO, SEMILLA


def huella(carpeta):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(carpeta.iterdir())
    }


# 1. Entrenamiento pequeño y creación de artefactos
def test_entrenamiento_crea_artefactos(modelos):
    carpeta = modelos / "v1"
    assert {p.name for p in carpeta.iterdir()} == {"modelo.joblib", "metrics.json", "metadata.json"}
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    metricas = json.loads((carpeta / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["version"] == "v1"
    assert metadata["columnas"] == list(COLUMNAS)
    assert metadata["umbral"] == 0.30
    assert {"modelo", "baseline"} <= set(metricas)
    artefacto = cargar_modelo(carpeta)
    assert artefacto["umbral"] == 0.30 and artefacto["version"] == "v1"


# 2. Carga/predicción por fila y consistencia antes/después de guardar
def test_prediccion_por_fila_y_consistencia(modelos, csv_entrenamiento, entradas, tmp_path):
    # Modelo "antes de guardar": mismo flujo que train.py, sin pasar por disco.
    x, y = cargar_datos(csv_entrenamiento, OBJETIVO)
    x_train, _, y_train, _, _ = separar_grupos(x, y, SEMILLA)
    en_memoria = crear_pipeline(SEMILLA).fit(x_train, y_train)
    esperadas = en_memoria.predict_proba(entradas)[:, 1]

    entradas.to_csv(tmp_path / "in.csv", index=False)
    resultado, _ = predecir_csv(modelos / "v1", tmp_path / "in.csv", tmp_path / "out.csv")
    np.testing.assert_allclose(resultado["probabilidad_cancelacion"], esperadas)
    np.testing.assert_array_equal(resultado["prediccion"], aplicar_umbral(esperadas, 0.30))

    # Una fila a la vez da lo mismo que el lote, y el orden se conserva.
    for i in range(len(entradas)):
        entradas.iloc[[i]].to_csv(tmp_path / "una.csv", index=False)
        fila, _ = predecir_csv(modelos / "v1", tmp_path / "una.csv", tmp_path / f"una{i}.csv")
        assert fila["probabilidad_cancelacion"].iloc[0] == pytest.approx(esperadas[i])
    salida = pd.read_csv(tmp_path / "out.csv")
    assert len(salida) == len(entradas)
    assert {"prediccion", "probabilidad_cancelacion"} <= set(salida.columns)


# 3. Nulos en variables admitidas
def test_nulos_en_variables_admitidas(modelos, entradas, tmp_path):
    entradas.loc[0, "lead_time"] = np.nan
    entradas.loc[1, "country"] = np.nan
    entradas.loc[2, "children"] = np.nan
    entradas.to_csv(tmp_path / "in.csv", index=False)
    resultado, _ = predecir_csv(modelos / "v1", tmp_path / "in.csv", tmp_path / "out.csv")
    assert len(resultado) == len(entradas)
    assert np.isfinite(resultado["probabilidad_cancelacion"]).all()
    assert set(resultado["prediccion"]) <= {0, 1}


# 4. Columna requerida ausente
def test_columna_requerida_ausente(modelos, entradas, tmp_path, monkeypatch, capsys):
    entradas.drop(columns=["hotel"]).to_csv(tmp_path / "in.csv", index=False)
    with pytest.raises(ErrorPrediccion, match="hotel"):
        predecir_csv(modelos / "v1", tmp_path / "in.csv", tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()

    # Por terminal: código 1, mensaje en stderr y ningún archivo de salida.
    monkeypatch.setattr(sys, "argv", [
        "predict.py", "--model", str(modelos / "v1"),
        "--input", str(tmp_path / "in.csv"), "--output", str(tmp_path / "out.csv"),
    ])
    assert predict.main() == 1
    assert "hotel" in capsys.readouterr().err
    assert not (tmp_path / "out.csv").exists()


# 5. Dos versiones y rechazo de sobrescritura
def test_dos_versiones_y_no_sobrescribir(modelos, csv_entrenamiento):
    assert cargar_modelo(modelos / "v1")["version"] == "v1"
    assert cargar_modelo(modelos / "v2")["version"] == "v2"
    antes = huella(modelos / "v1")
    with pytest.raises(FileExistsError):
        entrenar(csv_entrenamiento, OBJETIVO, modelos, "v1", 123)
    assert huella(modelos / "v1") == antes


# Contrato de entrada
def test_columnas_reordenadas_objetivo_y_extras(modelos, entradas, tmp_path):
    base = tmp_path / "base.csv"
    entradas.to_csv(base, index=False)
    esperado, _ = predecir_csv(modelos / "v1", base, tmp_path / "o1.csv")

    raro = entradas[list(reversed(COLUMNAS))].copy()
    raro[OBJETIVO] = 1  # se ignora sin aviso
    raro["comentario"] = "x"  # se descarta con aviso
    raro.to_csv(tmp_path / "raro.csv", index=False)
    resultado, avisos = predecir_csv(modelos / "v1", tmp_path / "raro.csv", tmp_path / "o2.csv")
    np.testing.assert_allclose(
        resultado["probabilidad_cancelacion"], esperado["probabilidad_cancelacion"]
    )
    assert any("comentario" in aviso for aviso in avisos)
    assert not any(OBJETIVO in aviso for aviso in avisos)


def test_categoria_nueva_avisa_y_predice(modelos, entradas, tmp_path):
    entradas.loc[0, "country"] = "XXX"
    entradas.to_csv(tmp_path / "in.csv", index=False)
    resultado, avisos = predecir_csv(modelos / "v1", tmp_path / "in.csv", tmp_path / "o.csv")
    assert len(resultado) == len(entradas)
    assert any("country" in aviso for aviso in avisos)


def test_texto_en_numerica_y_modelo_inexistente(modelos, entradas, tmp_path):
    entradas.assign(lead_time="abc").to_csv(tmp_path / "in.csv", index=False)
    with pytest.raises(ErrorPrediccion, match="lead_time"):
        predecir_csv(modelos / "v1", tmp_path / "in.csv", tmp_path / "out.csv")
    with pytest.raises(ErrorPrediccion, match="No existe el modelo"):
        cargar_modelo(modelos / "v9")
    assert not (tmp_path / "out.csv").exists()

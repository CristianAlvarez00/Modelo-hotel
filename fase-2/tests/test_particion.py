"""Corrección de Fase 2: la proporción debe afectar la partición y los metadatos."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.datos import cargar_datos, separar_grupos


def test_proporcion_configurable(csv_entrenamiento, tmp_path):
    script = Path(__file__).resolve().parents[1] / "train.py"
    proceso = subprocess.run([
        sys.executable, str(script), "--data", str(csv_entrenamiento),
        "--target", "is_canceled", "--output-dir", str(tmp_path),
        "--version", "configurable", "--test-size", "0.35",
    ], capture_output=True, text=True)
    assert proceso.returncode == 0, proceso.stderr
    metadata = json.loads((tmp_path / "configurable/metadata.json").read_text())
    assert metadata["particion"]["test_size_grupos"] == 0.35
    x, y = cargar_datos(csv_entrenamiento, "is_canceled")
    train, test, _, _, n = separar_grupos(x, y, 42, 0.35)
    assert metadata["particion"]["filas_prueba"] == len(test)
    assert len(test) > len(separar_grupos(x, y, 42, 0.20)[1])
    assert set(pd.util.hash_pandas_object(train, index=False)).isdisjoint(
        pd.util.hash_pandas_object(test, index=False))


@pytest.mark.parametrize("proporcion", [0, 1, -0.1, 1.2, float("nan")])
def test_proporcion_invalida(csv_entrenamiento, proporcion):
    x, y = cargar_datos(csv_entrenamiento, "is_canceled")
    with pytest.raises(ValueError, match="proporción"):
        separar_grupos(x, y, 42, proporcion)

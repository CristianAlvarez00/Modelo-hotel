"""Configuración del servicio. Las rutas también se pueden cambiar en Docker."""

from dataclasses import dataclass
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Configuracion:
    modelos: Path = RAIZ / "fase-2/models"
    version: str = "v1"
    datos: Path = RAIZ / "data/hotel_bookings.csv"
    objetivo: str = "is_canceled"
    semilla: int = 42
    test_size: float = 0.20

    def __post_init__(self):
        if not self.version or Path(self.version).name != self.version or self.version in (".", ".."):
            raise ValueError("MODELO_VERSION debe ser un identificador, no una ruta.")
        if not 0 < self.test_size < 1:
            raise ValueError("TEST_SIZE debe estar entre 0 y 1.")

    @classmethod
    def desde_entorno(cls):
        return cls(
            modelos=Path(os.getenv("MODELOS_DIR", str(RAIZ / "fase-2/models"))),
            version=os.getenv("MODELO_VERSION", "v1"),
            datos=Path(os.getenv("DATOS_CSV", str(RAIZ / "data/hotel_bookings.csv"))),
            objetivo=os.getenv("OBJETIVO", "is_canceled"),
            semilla=int(os.getenv("SEMILLA", "42")),
            test_size=float(os.getenv("TEST_SIZE", "0.20")),
        )

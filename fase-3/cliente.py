"""Cliente de prueba de la API: llama a /health, /predict y /train."""

import argparse
import json
from pathlib import Path
import sys

import httpx

RESERVA = Path(__file__).resolve().parent / "ejemplo_reserva.json"


def mostrar(nombre, respuesta):
    print(f"{nombre}: HTTP {respuesta.status_code}")
    print(json.dumps(respuesta.json(), indent=2, ensure_ascii=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API de cancelaciones")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--reserva", type=Path, default=RESERVA)
    parser.add_argument("--sin-train", action="store_true", help="no llamar a /train")
    args = parser.parse_args()

    reserva = json.loads(args.reserva.read_text(encoding="utf-8"))
    fallos = 0
    try:
        with httpx.Client(base_url=args.url, timeout=10) as cliente:
            r = cliente.get("/health")
            mostrar("GET /health", r)
            fallos += r.status_code != 200

            r = cliente.post("/predict", json=reserva)
            mostrar("POST /predict", r)
            fallos += r.status_code != 200

            incompleta = {k: v for k, v in reserva.items() if k != "hotel"}
            r = cliente.post("/predict", json=incompleta)
            mostrar("POST /predict (sin hotel, se espera 422)", r)
            fallos += r.status_code != 422

            if not args.sin_train:
                r = cliente.post("/train", json={}, timeout=600)
                mostrar("POST /train", r)
                fallos += r.status_code != 201
    except httpx.RequestError as exc:
        print(f"No se pudo conectar con {args.url}: {exc}")
        return 1

    print("Resultado:", "correcto" if not fallos else f"{fallos} llamada(s) con respuesta inesperada")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
# Fase 3 – API de cancelación de reservas

La API usa el entrenamiento y el pipeline guardado de Fase 2. Se implementaron
las tres rutas y sus pruebas. Docker, el cliente y el registro de Fase 4 quedan
pendientes de integración.

## Ejecutar

Desde la raíz, con Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r fase-3/requirements.txt
python -m uvicorn api:app --app-dir fase-3 --host 127.0.0.1 --port 8000
```

Con el servidor abierto, desde otra terminal en la raíz (Bash o zsh):

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @fase-3/ejemplo_reserva.json
curl -X POST http://127.0.0.1:8000/train -H 'Content-Type: application/json' -d '{}'
```

Las rutas y el esquema se pueden consultar en `http://127.0.0.1:8000/docs`.

## Rutas

- `GET /health`: HTTP 200 con `{"estado":"en ejecucion"}`.
- `POST /predict`: recibe las 17 variables y devuelve predicción, probabilidad
  y versión. Una clave ausente o un tipo incompatible produce 422 con el campo
  indicado. Si el modelo no está disponible, devuelve 503.
- `POST /train`: acepta `{}` o cuerpo vacío, entrena con la configuración del
  proyecto y devuelve 201 con la versión creada cuando sus tres archivos están
  guardados. Conserva las anteriores y sigue prediciendo con la versión activa.
  Esperar a que termine la solicitud. Usar un solo proceso de Uvicorn.

`ejemplo_reserva.json` contiene una fila del CSV de ejemplo de Fase 2. Con v1
produjo esta respuesta:

```json
{"prediccion":1,"probabilidad_cancelacion":0.9122282335433375,"version":"v1"}
```

1 significa cancelación predicha y 0 no cancelación. Las 17 claves son obligatorias;
los números se envían como números JSON y las categorías como texto. Se admite
`null` para aplicar la imputación guardada. Las categorías nuevas se codifican
en ceros y se ignoran claves adicionales. No se necesita `is_canceled`.
El esquema completo está en [Fase 2](../fase-2/README.md).

## Configuración

| Variable de entorno | Valor por defecto |
| --- | --- |
| MODELOS_DIR | fase-2/models dentro del repositorio |
| MODELO_VERSION | v1 |
| DATOS_CSV | data/hotel_bookings.csv dentro del repositorio |
| OBJETIVO | is_canceled |
| SEMILLA | 42 |
| TEST_SIZE | 0.20, proporción de grupos para prueba |


Los valores por defecto se localizan desde el código. Una ruta relativa enviada
por entorno se interpreta desde la carpeta donde se inicia el servidor.
`/train` guarda sus archivos en `MODELOS_DIR/<version>/` con un identificador
`api-<fecha>-<sufijo>`. No acepta rutas desde el JSON.

Para usar v2, detener el servidor y arrancar desde la raíz con:

```bash
MODELO_VERSION=v2 python -m uvicorn api:app --app-dir fase-3 --host 127.0.0.1 --port 8000
```

Para volver a v1, repetir con `MODELO_VERSION=v1`.

## Pruebas

Desde la raíz, con el entorno activo:

```bash
python -m pytest fase-2/tests fase-3/tests -q
```

Pasan 23 casos: 14 de Fase 2 y 9 de la API. La API cubre salud, predicción,
campo ausente, tipos, nulos, categoría nueva, modelo ausente y entrenamiento.
Los modelos de prueba se crean en directorios temporales.

Se ensayó también con Uvicorn y solicitudes HTTP: salud 200, predicción con v1
200, entrada sin hotel 422 y entrenamiento 201. Se guardó y cargó la versión
`api-20261121T011830-11cac9fa7999`; después de entrenar, la predicción siguió
usando v1. Las versiones creadas en ensayos locales no se incluyen en Git.

Se reutilizan `train.entrenar` y `src/prediccion.py` de Fase 2.
Para las pruebas se usó [TestClient de FastAPI](https://fastapi.tiangolo.com/tutorial/testing/).

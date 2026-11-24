# Fase 3: API de predicción de cancelaciones

Esta fase permite consultar el modelo y entrenar una versión mediante solicitudes
HTTP. Se reutilizan las funciones de Fase 2, las 17 variables de entrada y el
pipeline guardado, incluido el umbral de cada modelo.

## Contenido

- `api.py`: rutas de salud, predicción y entrenamiento.
- `configuracion.py`: selección de datos, modelo y partición.
- `esquemas.py`: validación de las entradas JSON.
- `cliente.py` y `ejemplo_reserva.json`: cliente y reserva de ejemplo.
- `tests/`: pruebas de la API.
- `Dockerfile`: imagen para ejecutar el servidor.

## Ejecución local

Requiere Python 3.12; se comprobó con Python 3.12.14. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r fase-3/requirements.txt
python -m uvicorn api:app --app-dir fase-3 --host 127.0.0.1 --port 8000
```

En Windows, activar el entorno con `.venv\Scripts\Activate.ps1` en PowerShell.
Con el servidor abierto, ejecutar desde otra terminal:

```bash
python fase-3/cliente.py
```

El cliente consulta la salud, envía una reserva, comprueba el rechazo de una
entrada sin `hotel` y solicita un entrenamiento. Termina con código 0 cuando
las respuestas son las esperadas. Para omitir el entrenamiento:

```bash
python fase-3/cliente.py --sin-train
```

El esquema y las rutas también están disponibles en `http://localhost:8000/docs`.

## Operaciones

| Ruta | Resultado esperado |
| --- | --- |
| `GET /health` | HTTP 200 con `{"estado": "en ejecucion"}` |
| `POST /predict` | HTTP 200 con predicción, probabilidad y versión |
| `POST /train` | HTTP 201 con el identificador de la versión guardada |

La predicción recibe un objeto con las 17 claves de `ejemplo_reserva.json`. No
necesita `is_canceled`. Se envían números JSON para las variables numéricas y
texto para las categóricas; el orden de las claves no importa. Las claves deben
estar presentes aunque su valor sea `null`: el pipeline aplica su imputación.
Las categorías desconocidas se codifican en ceros y las claves extra se ignoran.
El [README de Fase 2](../fase-2/README.md) describe las 17 entradas.

Ejemplo desde Bash o zsh:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @fase-3/ejemplo_reserva.json
```

Respuesta con `v1`:

```json
{"prediccion": 1, "probabilidad_cancelacion": 0.9122282335433375, "version": "v1"}
```

`1` significa cancelación predicha y `0` no cancelación. El preprocesamiento no
se vuelve a ajustar al predecir. Un campo ausente o un tipo incompatible produce
HTTP 422 e indica el campo; un modelo no disponible produce HTTP 503.

`/train` acepta `{}` o un cuerpo vacío y usa la configuración del proyecto.
Espera a que termine el ajuste y guarda `modelo.joblib`, `metrics.json` y
`metadata.json` en `MODELOS_DIR/api-<fecha>-<sufijo>/`. Las versiones anteriores
se conservan. La versión nueva no se activa automáticamente. Se usa un solo
proceso de Uvicorn; un segundo entrenamiento simultáneo devuelve HTTP 409.

## Configuración y selección de versión

| Variable de entorno | Valor por defecto |
| --- | --- |
| `MODELOS_DIR` | `fase-2/models` dentro del repositorio |
| `MODELO_VERSION` | `v1` |
| `DATOS_CSV` | `data/hotel_bookings.csv` dentro del repositorio |
| `OBJETIVO` | `is_canceled` |
| `SEMILLA` | `42` |
| `TEST_SIZE` | `0.20`, proporción de grupos para prueba |

Los valores por defecto se localizan desde el código. Las rutas relativas
indicadas por entorno se interpretan desde la carpeta de ejecución. Se incluyen
`v1` y `v2`; ambas conservan sus métricas, metadatos y umbral 0.30.

Para activar `v2`, detener el servidor y arrancarlo con:

```bash
MODELO_VERSION=v2 python -m uvicorn api:app --app-dir fase-3 --host 127.0.0.1 --port 8000
```

Para volver a `v1`, repetir con `MODELO_VERSION=v1`. En PowerShell, asignar primero
`$env:MODELO_VERSION = "v2"` y ejecutar el comando de Uvicorn sin el prefijo.

## Docker

Desde la raíz del repositorio, con Docker instalado y en ejecución:

```bash
docker build -f fase-3/Dockerfile -t hotel-api .
docker run --rm --name hotel-api -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  hotel-api
```

El servidor escucha en `0.0.0.0:8000`. Los datos se montan como lectura y las
versiones creadas con `/train` quedan en la carpeta de modelos del equipo.
La opción `--user` usa el usuario del anfitrión para escribir en esa carpeta.
La imagen no entrena durante su construcción e incluye una comprobación de salud.

En otra terminal, ejecutar `python fase-3/cliente.py`. Tras detener el contenedor
con `docker stop hotel-api`, los modelos siguen en `fase-2/models/`. Para elegir
otra versión en Docker, añadir `-e MODELO_VERSION=v2` al comando de ejecución.

Los ejemplos de varias líneas son para Bash/zsh. En PowerShell, escribir el
comando en una línea, usar `${PWD}` en los montajes y omitir `--user`; las carpetas
montadas deben permitir escritura al usuario del contenedor.

## Pruebas y comprobación

Desde la raíz, con el entorno activo:

```bash
python -m pytest fase-2/tests fase-3/tests fase-4/tests -q
```

La API tiene nueve casos: salud, predicción consistente con el pipeline, campo
ausente, tipos inválidos, nulos y categoría nueva, modelo ausente y entrenamiento
sin cambiar la versión activa. Los modelos de prueba se crean en carpetas temporales.

Se comprobó también el cliente con solicitudes HTTP reales, la creación y carga
de una candidata, la activación de `v2` y la vuelta a `v1`. Las dos versiones
originales permanecieron intactas. La fecha y los resultados de la demostración
Docker se indican en la comprobación final de Fase 4.

El 23 de noviembre de 2026 se construyó la imagen en Docker Engine 29.10.0
(Linux ARM64) y el cliente terminó con código 0. Se comprobó `/train` con los
modelos montados, se eliminó el contenedor y se cargó la candidata en otro nuevo.
La imagen también arrancó con v2 y sus comprobaciones de salud fueron correctas.

## Referencias

Se reutilizan `train.entrenar` y `src/prediccion.py` de Fase 2. Las pruebas usan
[TestClient de FastAPI](https://fastapi.tiangolo.com/tutorial/testing/).

## Identificación de la entrega

Etiqueta de la revisión comprobada: `entrega-fases-3-4`.

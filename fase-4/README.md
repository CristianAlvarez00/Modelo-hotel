# Fase 4: Monitoreo de la API

Esta fase registra los intentos de predicción y permite comprobar qué versión
del modelo se utilizó. El porcentaje de fallos describe el funcionamiento del
servicio; para evaluar la calidad predictiva se necesitan resultados reales.

## Contenido

- `api_monitoreada.py`: API de Fase 3 con el registro habilitado.
- `registro.py`: escritura de un evento por intento de predicción.
- `monitoreo.py`: cálculo del total y el porcentaje de fallos.
- `muestra_registro.csv`: registro usado en el reporte.
- `tests/`: pruebas del registro y las métricas.
- `Dockerfile`: imagen del servidor con monitoreo.

## Ejecución local

Desde la raíz, con Python 3.12 y el entorno de Fase 3 activo:

```bash
pip install -r fase-3/requirements.txt
MODELO_VERSION=v1 python -m uvicorn api_monitoreada:app --app-dir fase-4 --host 127.0.0.1 --port 8000
```

En PowerShell, asignar `$env:MODELO_VERSION = "v1"` antes del comando de Uvicorn.
Se usa un solo proceso. La API conserva las rutas de Fase 3 y agrega el registro
sin cambiar las respuestas de predicción.

| Variable | Valor por defecto |
| --- | --- |
| `REGISTRO_CSV` | `fase-4/registros/solicitudes.csv` dentro del repositorio |

También se mantienen las variables de [configuración de Fase 3](../fase-3/README.md).
La carpeta se crea al registrar el primer evento. Para leerla desde otra terminal:

```bash
python fase-4/monitoreo.py --registro fase-4/registros/solicitudes.csv
python fase-4/monitoreo.py --help
```

## Registro y métricas

El CSV se abre en modo agregar y su encabezado se escribe una sola vez. Se
conserva entre ejecuciones y contiene estos campos:

| Campo | Contenido |
| --- | --- |
| `fecha_hora` | Fecha y hora UTC con zona |
| `estado_http` | Código HTTP devuelto |
| `version` | Versión que predijo; puede quedar vacía en errores |
| `prediccion` | 0 o 1; vacía cuando la entrada no produjo predicción |

Se registra una fila por `POST /predict`, incluidas entradas inválidas, errores
de JSON y modelo no disponible. No se guarda el contenido de la reserva ni se
cuentan `/health` o `/train`. El script considera fallidas las respuestas de 400
a 599. Con un registro vacío muestra total 0 y «sin solicitudes».

## Docker y persistencia

Desde la raíz, en Bash o zsh, con Docker instalado y en ejecución:

```bash
docker build -f fase-4/Dockerfile -t hotel-api-monitoreada .
mkdir -p fase-4/registros
docker run --rm --name hotel-monitoreo -p 8000:8000 \
  --user "$(id -u):$(id -g)" \
  -e MODELO_VERSION=v1 \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  -v "$PWD/fase-4/registros:/app/fase-4/registros" \
  hotel-api-monitoreada
```

Los tres montajes permiten leer los datos, conservar las versiones creadas por
`/train` y continuar el registro al volver a crear el contenedor. La opción
`--user` permite escribir con el usuario del anfitrión. Para PowerShell, usar el
comando en una línea, `${PWD}` en los montajes y omitir `--user`.

Para repetir el ejemplo de tres intentos, usar un registro nuevo: añadir
`-e REGISTRO_CSV=/app/fase-4/registros/ensayo-nuevo.csv` con un nombre que aún no
exista. Con el servidor abierto, desde otra terminal en la raíz:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @fase-3/ejemplo_reserva.json
python fase-3/cliente.py --sin-train
python fase-4/monitoreo.py --registro fase-4/registros/ensayo-nuevo.csv
```

El curl envía una reserva válida. El cliente envía otra válida y una sin `hotel`:
en total son tres intentos. Para comprobar el reinicio, detener con
`docker stop hotel-monitoreo`, repetir el arranque con los mismos montajes y
confirmar que las filas anteriores se conservan antes de enviar nuevas reservas.

## Reporte

El registro publicado en `muestra_registro.csv` se obtuvo el 23 de noviembre
de 2026 (hora de Colombia) con la API en Docker. Contiene dos llamadas válidas
con la reserva de ejemplo y una tercera sin `hotel`:

| Solicitudes | Código HTTP | Resultado |
| --- | --- | --- |
| Dos entradas válidas | 200 | Predicción y versión identificadas |
| Una entrada sin `hotel` | 422 | Rechazo esperado de validación |

```bash
python fase-4/monitoreo.py --registro fase-4/muestra_registro.csv
```

Resultado: **3 solicitudes, 1 fallida y 33,3 % de fallos**. El fallo corresponde
al rechazo previsto de la entrada incompleta; no es una medida del error del
modelo. El registro conserva fecha, código, versión y predicción de cada intento.
Las dos predicciones válidas fueron 1. Las fechas del CSV están en UTC.

## Versiones y política de reentrenamiento

| Versión | Ubicación | Uso |
| --- | --- | --- |
| `v1` | `fase-2/models/v1/` | Versión activa por defecto |
| `v2` | `fase-2/models/v2/` | Alternativa seleccionable |

Las dos contienen modelo, métricas y metadatos. Se entrenaron con los mismos
datos y configuración, con F1 de prueba 0,680; v2 demuestra la selección de
versiones y no representa una mejora del algoritmo. Para cambiar, reiniciar con
`MODELO_VERSION=v2`; para volver, usar `MODELO_VERSION=v1`.

Al reunir 500 reservas nuevas con resultado real, el equipo revisará el modelo.
Una persona designada ejecutará `/train` y comparará el F1 de la candidata con
el de la versión activa sobre los mismos datos reservados, no usados para ajustar.
Solo se activará la candidata si mejora el F1; si presenta problemas, se volverá
a la anterior mediante `MODELO_VERSION`.

## Pruebas

```bash
python -m pytest fase-2/tests fase-3/tests fase-4/tests -q
```

Fase 4 tiene 14 casos: registro de válidas y rechazadas, JSON inválido, cuerpo
vacío, modelo ausente, fecha con zona, persistencia sin duplicados, conservación
de respuestas y cálculo de métricas. El registro se guarda en carpetas temporales.

## Comprobación final

Se construyeron las imágenes de Fases 3 y 4 en Docker Engine 29.10.0 sobre
Linux ARM64, usando una máquina temporal de Lima en macOS. El 23 de noviembre
de 2026 se comprobó que:

- El cliente recibe los códigos esperados y el monitoreo muestra 3 intentos y 33,3 %.
- `/train` conserva los artefactos en el montaje del anfitrión.
- Al eliminar y recrear el contenedor, la candidata se puede cargar y el registro
  conserva sus filas anteriores, con un solo encabezado.
- Los archivos originales de v1 y v2 permanecen intactos.
- Las 37 pruebas pasan también dentro de Docker. Se muestra un aviso de
  deprecación de AnyIO usado por TestClient, sin fallos de pruebas.

Etiqueta de la revisión comprobada: `entrega-fases-3-4`.

## Limitaciones

Se usa un solo proceso de Uvicorn y no hay rotación automática del registro.
La ruta registrada es `POST /predict`; una llamada a `/predict/` con barra final
puede recibir una redirección. Los errores de escritura del registro se informan
en el log del servidor.

## Referencias

Se utilizan el [middleware de FastAPI](https://fastapi.tiangolo.com/tutorial/middleware/)
y `BaseHTTPMiddleware` de Starlette.

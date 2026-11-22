# Fase 4 – Monitoreo de la API

Registra cada intento de `POST /predict` de la API de Fase 3 y calcula dos
métricas de operación: el total de solicitudes y el porcentaje de fallidas.
Fase 3 no se modifica: `api_monitoreada.py` importa su app y le agrega un
middleware.

| Archivo | Función |
| --- | --- |
| `api_monitoreada.py` | Punto de entrada: la API de Fase 3 con registro |
| `registro.py` | Middleware que escribe una fila por solicitud a `/predict` |
| `monitoreo.py` | Script que lee el registro y muestra las dos métricas |
| `muestra_registro.csv` | Registro real de la ejecución descrita en el reporte |
| `tests/test_monitoreo.py` | 14 pruebas con pytest |

## Ejecutar

Desde la raíz del repositorio, con Python 3.12 y las dependencias de Fase 3
(`pip install -r fase-3/requirements.txt`). Los comandos se ejecutaron en
PowerShell (Windows); entre paréntesis, el equivalente en Bash.

Arrancar la API con el registro y una versión del modelo identificada:

```powershell
$env:MODELO_VERSION = "v1"   # Bash: export MODELO_VERSION=v1
python -m uvicorn api_monitoreada:app --app-dir fase-4 --host 127.0.0.1 --port 8000
```

Con el servidor abierto, enviar solicitudes (por ejemplo con el JSON de
`fase-3/ejemplo_reserva.json`, como se explica en el [README de Fase 3](../fase-3/README.md))
y, desde otra terminal, calcular las métricas:

```powershell
python fase-4/monitoreo.py --registro fase-4/registros/solicitudes.csv
python fase-4/monitoreo.py --help
```

Usar un solo proceso de Uvicorn: el registro se protege con un candado dentro
del proceso.

### Configuración

Se mantienen las variables de Fase 3 (`MODELOS_DIR`, `MODELO_VERSION`, `DATOS_CSV`,
`OBJETIVO`, `SEMILLA`, `TEST_SIZE`) y se agrega una:

| Variable | Valor por defecto |
| --- | --- |
| `REGISTRO_CSV` | `fase-4/registros/solicitudes.csv` dentro del repositorio |

El script de métricas usa la misma variable si no se le pasa `--registro`.
Si la carpeta del registro no existe, se crea al primer evento. Para que el
registro persista en Docker, la carpeta que contiene `REGISTRO_CSV` debe ser un
montaje del equipo anfitrión; el comando exacto se documenta junto a la imagen
de Fase 3.

## Qué se registra

El archivo es un CSV abierto siempre en modo agregar, de modo que el historial se
conserva entre ejecuciones. El encabezado se escribe solo al crear el archivo.

| Columna | Contenido |
| --- | --- |
| `fecha_hora` | Fecha y hora en UTC con zona (`2026-11-21T03:14:50+00:00`) |
| `estado_http` | Código HTTP de la respuesta |
| `version` | Versión del modelo que predijo; vacía si hubo error |
| `prediccion` | 1 = cancelación predicha, 0 = no cancelación; vacía si hubo error |

- Se escribe **una fila por cada `POST /predict`**, incluidas las rechazadas por
  validación (422), los JSON inválidos y los 503 por modelo no disponible. El
  middleware espera la respuesta para conocer el código y por eso ve también los
  rechazos que ocurren antes de entrar a la ruta.
- No se registran otras rutas (`/health`, `/train`) ni otros métodos.
- No se guarda el cuerpo de la solicitud: ningún dato de la reserva queda en el registro.
- Si no se puede escribir el registro, se deja un mensaje en el log del servidor y
  la API sigue respondiendo.

## Reporte

Ejecución real del 21 de noviembre de 2026 (UTC), en local con Uvicorn, Python 3.12.10
y `MODELO_VERSION=v1`, sobre un registro vacío. Se enviaron tres solicitudes a
`POST /predict`:

| Solicitud | Respuesta |
| --- | --- |
| Válida 1 (`fase-3/ejemplo_reserva.json`) | 200, predicción 1, versión `v1` |
| Válida 2 (otra fila de `fase-2/data/ejemplo_prediccion.csv`) | 200, predicción 0, versión `v1` |
| Sin el campo obligatorio `hotel` | 422, «Field required» para `hotel` |

Registro resultante, guardado en [`muestra_registro.csv`](muestra_registro.csv):

```csv
fecha_hora,estado_http,version,prediccion
2026-11-21T03:14:50+00:00,200,v1,1
2026-11-21T03:14:50+00:00,200,v1,0
2026-11-21T03:14:51+00:00,422,,
```

Salida de `python fase-4/monitoreo.py --registro <registro>`:

```text
Total de solicitudes: 3
Solicitudes fallidas (HTTP 400 a 599): 1
Porcentaje de fallos: 33,3 %
```

La API procesó dos reservas con `v1` y rechazó la tercera porque faltaba el campo
`hotel`. La tasa de fallos fue 1/3 (33,3 %); el rechazo era el resultado esperado de
la prueba de validación. Estas métricas describen el funcionamiento del servicio, no
la calidad de las predicciones: medirla exigiría conocer si cada reserva se canceló
realmente.

Comprobaciones adicionales con la misma ejecución:

- Con un registro vacío o solo con encabezado, el script muestra total 0 y «sin solicitudes».
- Se detuvo el servidor y se reinició con `MODELO_VERSION=v2`; una solicitud válida
  respondió con versión `v2` y quedó como cuarta fila. Las tres anteriores no cambiaron
  ni se contaron dos veces, y el encabezado no se repitió (4 solicitudes, 25,0 % de fallos).

## Versiones del modelo

| Versión | Ubicación | Uso |
| --- | --- | --- |
| `v1` | `fase-2/models/v1/` | Versión activa por defecto |
| `v2` | `fase-2/models/v2/` | Alternativa seleccionable |

Cada carpeta contiene `modelo.joblib`, `metrics.json` y `metadata.json`. La API usa la
indicada en `MODELO_VERSION` (por defecto `v1`) y la devuelve en cada respuesta y en el
registro. Para cambiar de versión, detener el servidor y arrancarlo con la variable;
para volver a la anterior, repetir con `MODELO_VERSION=v1`.

Ambas se entrenaron con la misma configuración (semilla 42, misma partición, F1 en
prueba 0,680); `v2` sirve para demostrar el cambio de versión, no es un modelo distinto.
Las versiones creadas con `POST /train` se guardan como `api-<fecha>-<sufijo>` en la
misma carpeta y no se activan automáticamente.

## Política de reentrenamiento

Al reunir 500 reservas nuevas con su resultado real (cancelada o no), el equipo revisará
el modelo. Una persona designada del equipo ejecutará `POST /train` para crear una
candidata, y se compararán la versión activa y la candidata con el F1 sobre los mismos
datos reservados, no usados para entrenar. La candidata se activa, cambiando
`MODELO_VERSION` y reiniciando el servidor, solo si mejora el F1 de la versión activa.

## Pruebas

Desde la raíz, con el entorno activo:

```powershell
python -m pytest fase-2/tests fase-3/tests fase-4/tests -q
```

Pasan 37 casos: 23 de Fases 2 y 3 y 14 de esta fase. Las de Fase 4 comprueban, con modelos
pequeños en directorios temporales: tres solicitudes (dos válidas y una inválida) dejan
exactamente tres filas y 33,3 %; los 422 por validación, los JSON inválidos, los cuerpos
vacíos y los 503 dejan una fila cada uno; el registro persiste al reiniciar sin duplicar el
encabezado; no se guarda el cuerpo; el middleware no cambia las respuestas de la API; y el
script maneja el registro vacío y los archivos inválidos.

## Limitaciones

- El porcentaje de fallos mide el servicio, no la calidad del modelo.
- Un solo proceso de Uvicorn; con varios, las filas podrían intercalarse.
- El registro crece sin límite; no hay rotación ni borrado.
- Un `POST` a `/predict/` (con barra final) no se registra como intento de `/predict`.

## Referencias

- Middleware en FastAPI: <https://fastapi.tiangolo.com/tutorial/middleware/>
- Middleware en Starlette (`BaseHTTPMiddleware`): <https://www.starlette.io/middleware/>

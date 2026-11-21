# Fase 3 API de cancelación de reservas

Se implementaron las tres rutas, la configuración, el esquema y las pruebas.
Docker, el cliente de prueba y el registro de Fase 4 están pendientes.

## Arranque local

Desde la raíz del repositorio, con Python 3.12 (validado con 3.12.14):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r fase-3/requirements.txt
python -m uvicorn api:app --app-dir fase-3 --host 127.0.0.1 --port 8000
```

Dejar esa terminal abierta. Desde otra terminal en la raíz:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @fase-3/ejemplo_reserva.json
curl -X POST http://127.0.0.1:8000/train -H 'Content-Type: application/json' -d '{}'
```

También se pueden ver las rutas y el esquema en `http://127.0.0.1:8000/docs`.
Estos comandos son para Bash/zsh. Persona 3 debe comprobar el cliente y adaptar
la documentación para el contenedor antes de presentar la entrega completa.

## Entrada y respuestas

`/health` devuelve HTTP 200 y `{"estado":"en ejecucion"}`. Esto comprueba el
servidor; no afirma que el modelo esté disponible.

`/predict` recibe un objeto con las 17 claves de `ejemplo_reserva.json`, obtenido
de la primera fila del CSV de Fase 2. El esquema completo está en
`../fase-2/README.md`. Se exige un número JSON para variables numéricas (ni
texto ni booleanos) y texto para las categóricas. Las claves deben estar aunque
su valor sea `null`. El pipeline imputa números con la mediana y categorías con
`Desconocido`. Categorías nuevas se aceptan y codifican en ceros. Se ignoran
claves adicionales; no se necesita `is_canceled`. No se ajustan transformaciones
al predecir.

Respuesta observada con el ejemplo y v1:

```json
{"prediccion":1,"probabilidad_cancelacion":0.9122282335433375,"version":"v1"}
```

`1` significa cancelación predicha y `0` no cancelación. El umbral se toma del
artefacto (0.30 en v1/v2). Una clave ausente o un tipo incompatible produce
HTTP 422 con el campo indicado. Si el modelo no existe o no se puede cargar,
se responde HTTP 503 con un mensaje. El ejemplo sin `hotel` fue rechazado con
422. Un nulo válido pasa por el mismo pipeline que se guardó en Fase 2.

`/train` acepta `{}` o un cuerpo vacío. Usa el CSV, objetivo, semilla y proporción
configurados; no recibe rutas arbitrarias desde el JSON. Responde HTTP 201
con `{"version":"api-..."}` cuando sus tres archivos están guardados en
`MODELOS_DIR/<version>/`. El identificador incluye fecha UTC y un sufijo único.
Si ya hay un entrenamiento en ese proceso responde 409. Datos inaccesibles o
inválidos producen 503. La llamada es síncrona: esperar a que termine.
Para el uso local del curso, arrancar un solo proceso de Uvicorn.

## Configuración y vuelta a una versión anterior

| Variable de entorno | Valor por defecto |
| --- | --- |
| MODELOS_DIR | fase-2/models dentro del repositorio |
| MODELO_VERSION | v1 |
| DATOS_CSV | data/hotel_bookings.csv dentro del repositorio |
| OBJETIVO | is_canceled |
| SEMILLA | 42 |
| TEST_SIZE | 0.20, proporción de grupos para prueba |

Los valores por defecto se localizan desde el código; no dependen de dónde se
abra la terminal. Las rutas relativas proporcionadas por entorno sí se resuelven
desde la carpeta del comando. Ejemplo desde la raíz para activar v2:

```bash
MODELO_VERSION=v2 python -m uvicorn api:app --app-dir fase-3 --host 127.0.0.1 --port 8000
```

Para volver a v1, detener el servidor y repetir con `MODELO_VERSION=v1`.
Entrenar una candidata no modifica esa selección. La API lee el artefacto de
la versión configurada en cada predicción. Se conservan las versiones anteriores.

## Pruebas y evidencia

Desde la raíz, con el entorno activo:

```bash
python -m pytest fase-2/tests fase-3/tests -q
```

También se puede ejecutar `python -m pytest -q` dentro de cada carpeta. La API
cubre salud, predicción consistente con el pipeline, campo ausente, tipos,
nulos/categoría nueva, modelo ausente y entrenamiento sin cambiar versión activa.
Se entrena un modelo pequeño en un directorio temporal para esas pruebas.

En `evidencia/http.json` está la comprobación HTTP real con Uvicorn: /health
200, /predict 200 con v1, entrada sin hotel 422 y /train 201. El archivo indica
la versión creada y dónde quedaron su modelo, métricas y metadatos. Se hizo otra
predicción después de entrenar y siguió usando v1, como se esperaba.
Las fechas se registran en UTC; pueden mostrar el día siguiente al de Colombia.

El registro histórico de cada llamada que exige Fase 4 lo implementará Persona 2.
Esta evidencia es un ensayo de Fase 3, no sustituye ese registro.

## Fuentes e integración pendiente

Se reutilizan `train.entrenar` y las funciones de `src/prediccion.py` de Fase 2.
El servidor sigue los ejemplos oficiales de FastAPI para rutas y
[TestClient](https://fastapi.tiangolo.com/tutorial/testing/).
La base integrada era main `fd250f91`; los archivos nuevos requieren un PR
revisado hacia develop. Persona 3 debe identificar el commit o etiqueta final,
completar Docker con montajes y cliente, y repetir la secuencia de entrega.

# Fase 2 – Predicción de cancelación de reservas (scripts y Docker)

Convierte el modelo de Fase 1 en una app de línea de comandos: entrenamiento, predicción, pruebas y Docker. Mismas 17 variables, mismo pipeline (imputación + One-Hot + escalado + regresión logística) y umbral `0.30`. Única diferencia respecto al notebook: aquí solo hay un split train/prueba (el ajuste de umbral ya se hizo en Fase 1 y quedó fijo).

## Instalación

```bash
cd fase-2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

- Entrenamiento: `../data/hotel_bookings.csv` (el mismo de Fase 1).
- Predicción: CSV con las 17 columnas, sin `is_canceled`. Ejemplo en `data/ejemplo_prediccion.csv`. Orden de columnas libre; columnas extra se ignoran con aviso.

## Entrenar

```bash
python train.py --data ../data/hotel_bookings.csv --target is_canceled \
  --output-dir models --version v2 --seed 42
```
Crea `models/<version>/{modelo.joblib, metrics.json, metadata.json}`. Si la versión ya existe, falla sin modificar nada.

## Predecir

```bash
python predict.py --version v1 --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones.csv
```
No sobrescribe sin `--overwrite`. Errores sin traza, código de salida 1, sin archivo de salida a medias.

## Docker

```bash
docker build -t modelos-fase2:1.0 .

# entrenar
docker run --rm \
  -v "$PWD/../data:/app/data:ro" -v "$PWD/models:/app/models" \
  modelos-fase2:1.0 python train.py \
  --data data/hotel_bookings.csv --target is_canceled \
  --output-dir models --version v-docker --seed 42

# predecir
docker run --rm \
  -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --version v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```
No entrena durante el build. `models/` y `outputs/` persisten en el host (verificado tras `--rm`).

## Pruebas

```bash
python -m pytest -v
```
14 casos, sin red (datos sintéticos). Cubren: artefactos de entrenamiento, consistencia lote/fila, nulos, columna ausente, dos versiones sin sobrescritura, columnas reordenadas/extra, categoría nueva, texto inválido.

## Métricas (v1, prueba)

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline (Dummy) | 0.640 | 0.000 | 0.000 | 0.000 |
| Regresión logística | 0.731 | 0.595 | 0.794 | 0.680 |

## Notas

- Se fijó SciPy 1.14.1; el entrenamiento real de las correcciones terminó sin el aviso de `iprint`.
- Limitaciones del dataset (duplicados, valores extremos, alcance a 2 hoteles): ver README raíz.
## Correcciones de la evaluación de Fase 2

Entorno local comprobado: Python 3.12.14. El Dockerfile usa Python 3.12.
Se fijó SciPy 1.14.1 para usar una versión compatible con el optimizador de
scikit-learn 1.6.1. La instalación sigue siendo `pip install -r requirements.txt`.

La partición ahora se configura con `--test-size` (0.20 por defecto). Es la
proporción de **grupos**, por lo que la proporción de filas puede variar. Se
mantienen la separación de entradas idénticas y el umbral 0.30.

```bash
python train.py --help
python predict.py --help
python train.py --data ../data/hotel_bookings.csv --target is_canceled --output-dir models --version v-nueva --seed 42 --test-size 0.20
python predict.py --version v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones-v2.csv
```

Se conservan `models/v1/` y `models/v2/`, cada una con los tres archivos. `v2`
usa los mismos datos, semilla 42 y proporción 0.20; no representa un intento
por mejorar la métrica. Las diferencias numéricas pequeñas se pueden deber al
entorno. `v1` conserva sus metadatos históricos, incluida la referencia que
entonces decía que requirements estaba pendiente.

## Esquema de las 17 entradas

Todas las columnas son obligatorias. Una celda vacía se admite y se imputa con
las reglas del pipeline: mediana para números, `Desconocido` para categorías.
La mediana y la escala se aprenden solo con entrenamiento. Los números deben
ser finitos y las categorías texto. Una categoría nueva se codifica en ceros;
las columnas extra se ignoran con aviso. El objetivo no es necesario al predecir.

| Columna | Tipo | Significado |
| --- | --- | --- |
| lead_time | Número | Días de anticipación de la reserva |
| stays_in_weekend_nights | Número | Noches de fin de semana |
| stays_in_week_nights | Número | Noches entre semana |
| adults | Número | Adultos |
| children | Número | Niños |
| babies | Número | Bebés |
| is_repeated_guest | Número | Huésped repetido, 0 o 1 |
| previous_cancellations | Número | Cancelaciones anteriores |
| previous_bookings_not_canceled | Número | Reservas anteriores no canceladas |
| hotel | Texto | Tipo de hotel |
| arrival_date_month | Texto | Mes de llegada |
| meal | Texto | Régimen de comidas |
| country | Texto | País |
| market_segment | Texto | Segmento de mercado |
| distribution_channel | Texto | Canal de distribución |
| reserved_room_type | Texto | Habitación reservada |
| customer_type | Texto | Tipo de cliente |

La salida conserva el orden de filas, incluye `prediccion` (1 = cancelación
predicha, 0 = no cancelación) y `probabilidad_cancelacion`.

## Identificación y evidencia

La entrega evaluada fue `fd250f91b93e659703d50a786c6484a470322d76`, integrada
mediante el PR #14. Las correcciones actuales aún necesitan su propio commit/PR.
Los metadatos nuevos guardan `codigo_commit_base` y `codigo_sha256`; la huella
identifica los archivos de entrenamiento incluso si hay cambios sin commit.
También guardan la huella del CSV y las versiones de librerías usadas.
La huella de `v1` corresponde al código original; añadir comentarios después
no cambia el modelo ya guardado.

Las comprobaciones de corrección están en `evidencia/validacion.json`: creación
de v2 con el CSV completo, carga de ambas versiones, predicción con el ejemplo
y rechazo de una versión repetida sin cambios en los archivos. Para repetirlo
use otro identificador y una salida nueva. La suite original se conserva y se
agregan pruebas para la proporción configurable y sus límites.

La afirmación anterior sobre Docker corresponde a la entrega original. Docker
no está instalado en el equipo donde se hicieron estas correcciones; la Persona 3
debe repetir el build y los comandos con montajes antes del cierre siguiente.

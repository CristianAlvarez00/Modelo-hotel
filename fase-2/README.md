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
8 pruebas, sin red (datos sintéticos). Cubren: artefactos de entrenamiento, consistencia lote/fila, nulos, columna ausente, dos versiones sin sobrescritura, columnas reordenadas/extra, categoría nueva, texto inválido.

## Métricas (v1, prueba)

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline (Dummy) | 0.640 | 0.000 | 0.000 | 0.000 |
| Regresión logística | 0.731 | 0.595 | 0.794 | 0.680 |

## Notas

- Warning esperado al entrenar: `OptimizeWarning: Unknown solver options: iprint` (scikit-learn 1.6.1), no afecta las métricas.
- Limitaciones del dataset (duplicados, valores extremos, alcance a 2 hoteles): ver README raíz.
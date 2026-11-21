# Fase 2 – Predicción de cancelación de reservas

El entrenamiento y la predicción del notebook se ejecutan desde terminal.
Se conservan las 17 variables, la separación por grupos, la regresión logística
y el umbral 0.30 de Fase 1.

## Instalación

Python 3.12, comprobado con 3.12.14. Desde la raíz:

```bash
cd fase-2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenamiento y predicción

Desde `fase-2/`:

```bash
python train.py --help
python predict.py --help
python train.py --data ../data/hotel_bookings.csv --target is_canceled --output-dir models --version v-nueva --seed 42 --test-size 0.20
python predict.py --version v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

El CSV de entrenamiento está incluido en `../data/hotel_bookings.csv`.
`--test-size` indica la proporción de grupos reservados para prueba; la proporción
de filas puede variar. Se separan los datos antes de ajustar el pipeline.

Cada versión guarda `modelo.joblib`, `metrics.json` y `metadata.json` en
`models/<version>/`. Una versión existente se rechaza sin modificar sus archivos.
La salida de predicción tampoco se sobrescribe sin `--overwrite`.

## Entrada y salida

El CSV de predicción no necesita `is_canceled`. Sus columnas pueden estar en
cualquier orden y las adicionales se ignoran con aviso. Todas las columnas de
esta tabla deben existir:

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


Los números deben ser finitos y las categorías texto. Las celdas vacías se
imputan con la mediana en numéricas y `Desconocido` en categóricas. Las medianas,
la escala y las categorías se aprenden solo con entrenamiento. Las categorías
nuevas se codifican en ceros.

La salida conserva el orden y número de filas. Incluye `prediccion` (1 significa
cancelación predicha, 0 no cancelación) y `probabilidad_cancelacion`.

## Versiones y resultados

Se incluyen `models/v1/` y `models/v2/`. Ambas usan semilla 42, los mismos datos
y proporción 0.20. v2 se creó para completar el versionamiento; no se cambió
el modelo para buscar una mejora.

| Modelo en prueba | Accuracy | Precisión | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 0.640 | 0.000 | 0.000 | 0.000 |
| Regresión logística, v1 y v2 | 0.731 | 0.595 | 0.794 | 0.680 |

Se comprobó el recorrido desde una copia limpia: entrenamiento con el CSV
completo (95.003 filas de entrenamiento y 24.387 de prueba), carga y predicción
con ambas versiones y rechazo de un identificador repetido sin alterar archivos.
SciPy se fijó en 1.14.1 para mantener compatibilidad con scikit-learn 1.6.1.
Las limitaciones de los datos están explicadas en el README raíz.

La entrega de Fase 2 fue `fd250f91`; las correcciones del entrenamiento están en
`b668173`. Los metadatos nuevos guardan la huella del CSV, las versiones de las
librerías, el commit base y la huella de los archivos de entrenamiento. v1 conserva
sus metadatos originales.

## Pruebas

```bash
python -m pytest -q
```

Pasan 14 casos con datos sintéticos: entrenamiento, persistencia, predicción,
nulos, entradas inválidas, versiones y proporción configurable.

## Docker

Desde `fase-2/`, en Bash o zsh:

```bash
docker build -t modelos-fase2:1.0 .
docker run --rm \
  -v "$PWD/../data:/app/data:ro" -v "$PWD/models:/app/models" \
  modelos-fase2:1.0 python train.py \
  --data data/hotel_bookings.csv --target is_canceled \
  --output-dir models --version v-docker --seed 42

docker run --rm \
  -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --version v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

La imagen no entrena durante el build. Los montajes conservan modelos y salidas
en el equipo anfitrión. Estos comandos se conservan de Fase 2; la nueva ejecución
Docker queda pendiente para la integración, porque este equipo no tiene Docker.

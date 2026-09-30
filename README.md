# Predicción de cancelación de reservas hoteleras

**Proyecto Integrador – Fase 1: Modelo Predictivo**
Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II

## Integrantes del equipo

- Brayam Camilo Aguirre Ricaurte
- Juan Sebastián Naranjo Jiménez
- Cristian Alvarez Cuarta

## Descripción del problema

Las cancelaciones de reservas afectan la planeación de ocupación e ingresos de un hotel. Este proyecto construye un modelo de **clasificación binaria** que, con la información disponible al registrar una reserva, predice si esta será cancelada.

- **Variable objetivo:** `is_canceled` (`1` = reserva cancelada, `0` = no cancelada).
- **Tipo de problema:** aprendizaje supervisado, clasificación binaria.
- **Objetivo del modelo:** anticipar cancelaciones para poder actuar sobre ellas (por ejemplo, gestionar sobreventa o contactar al cliente).

## Fuente de datos

Conjunto de datos **Hotel Booking Demand** (reservas de un hotel urbano y un resort, previamente aprobado por el docente), incluido en `data/hotel_bookings.csv`.

| Característica | Valor |
| --- | --- |
| Observaciones | 119.390 |
| Columnas | 32 (31 predictoras candidatas + objetivo) |
| Clase "No cancelada" | 75.166 (62,96 %) |
| Clase "Cancelada" | 44.224 (37,04 %) |

**Valores faltantes:** `company` (94,31 %), `agent` (13,69 %), `country` (0,41 %, 488 filas) y `children` (4 filas).
**Limitaciones:** hay 31.994 filas idénticas (26,80 %) que no se pueden confirmar como duplicados reales por no existir un identificador de reserva; además hay valores extremos (por ejemplo `adr` hasta 5.400 y `adults` hasta 55) y el conjunto describe solo dos hoteles, por lo que la generalización a otros contextos no está garantizada.

## Organización del repositorio

```
Modelo-hotel/
├── data/
│   └── hotel_bookings.csv        # conjunto de datos
├── fase-1/
│   ├── modelo_hotel.ipynb        # EDA, preparación, entrenamiento y evaluación
│   └── modelo_hotel.joblib       # pipeline final + umbral de decisión
├── .gitignore
└── README.md
```

## Cómo ejecutar la Fase 1

El notebook está en `fase-1/modelo_hotel.ipynb`. Descarga el dataset automáticamente desde este repositorio público, por lo que necesita conexión a internet. Hay tres formas de ejecutarlo:

### Opción 1: Google Colab
Abrir el notebook en Colab y usar *Entorno de ejecución → Ejecutar todas*. No requiere instalación.

### Opción 2: Docker (recomendada, reproducible)
Requiere Docker instalado y abierto.

```bash
git clone https://github.com/CristianAlvarez00/Modelo-hotel.git
cd Modelo-hotel
git checkout develop
docker build -t modelo-hotel .
docker run --rm -p 8888:8888 modelo-hotel
```

Abrir en el navegador el enlace con `token` que aparece en la terminal, abrir `modelo_hotel.ipynb` y ejecutar todas las celdas. El modelo se guarda como `modelo_hotel.joblib` dentro del contenedor.

### Opción 3: Entorno local
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / Mac
pip install -r fase-1/requirements.txt
pip install notebook
cd fase-1
jupyter notebook modelo_hotel.ipynb
```

Las versiones de `fase-1/requirements.txt` son las del entorno de Colab con el que se entrenó el modelo. Con otras versiones de scikit-learn, el archivo `.joblib` no carga.

## Metodología

### 1. Análisis exploratorio
Dimensiones, tipos de datos, valores faltantes, duplicados, estadísticas descriptivas, distribución de la variable objetivo, tasa de cancelación por variables clave y correlación de Pearson entre variables numéricas. Hallazgos principales:

- Hay desbalance moderado de clases (63 % / 37 %).
- La cancelación aumenta con la anticipación de la reserva: 9,6 % con 0–7 días frente a 57,0 % con más de 180 días.
- City Hotel cancela más (41,7 %) que Resort Hotel (27,8 %).
- La correlación entre variables numéricas es baja en general; la más alta es entre noches de semana y de fin de semana, que se conservan ambas porque describen aspectos distintos de la estancia.

### 2. Preparación de los datos
- **Variables usadas (17):**
  - Numéricas: `lead_time`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`.
  - Categóricas: `hotel`, `arrival_date_month`, `meal`, `country`, `market_segment`, `distribution_channel`, `reserved_room_type`, `customer_type`.
- **Faltantes:** `children` se imputa con la mediana (calculada solo con entrenamiento); `country` se rellena con la categoría `Desconocido`. `agent` y `company` se excluyen por su alta proporción de nulos y porque son identificadores, no magnitudes.
- **Codificación y escalado:** One-Hot Encoding para categóricas (`handle_unknown="ignore"`) y `StandardScaler` para numéricas, todo dentro de un `Pipeline` con `ColumnTransformer`.
- **Duplicados:** no se eliminaron filas, porque eliminarlas altera de forma considerable la proporción de cancelaciones y no hay forma de confirmar que sean la misma reserva.
- **Valores extremos:** se identifican pero no se modifican en esta primera versión.

### 3. Prevención de fuga de información
- `is_canceled` no forma parte de `X`.
- Se excluyen `reservation_status` y `reservation_status_date` (describen el resultado), `assigned_room_type`, `booking_changes` y `days_in_waiting_list` (pueden reflejar información posterior al registro) y `deposit_type` y `adr` (según la documentación del dataset pueden derivarse de pagos posteriores a la reserva).
- Imputación, codificación y escalado se ajustan **solo con datos de entrenamiento**.
- Como hay filas idénticas, la separación se hace por grupos (`GroupShuffleSplit`): las filas con las mismas características seleccionadas caen siempre en el mismo conjunto. El notebook lo verifica con `assert`.
- El conjunto de prueba no se usa para elegir el modelo ni el umbral.

### 4. Separación de datos

| Conjunto | Filas | Uso |
| --- | ---: | --- |
| Entrenamiento | 95.003 (≈ 80 %) | Ajuste final del modelo |
| ↳ Ajuste | 76.402 | Entrenar y comparar modelos |
| ↳ Validación | 18.601 | Comparar modelos y elegir umbral |
| Prueba | 24.387 (≈ 20 %) | Evaluación final, una sola vez |

Método: `GroupShuffleSplit` con `RANDOM_STATE = 42` (prueba) y `43` (validación).

### 5. Modelo base y modelos comparados
La métrica principal es **F1 de la clase cancelada**; se reportan también accuracy, precisión y recall. Se eligió F1 porque las clases están desbalanceadas y interesa equilibrar detectar cancelaciones (recall) sin generar demasiadas falsas alarmas (precisión); el accuracy por sí solo sería engañoso (predecir siempre "no cancela" ya da 63 %).

Resultados en **validación** (umbral 0,5):

| Modelo | Accuracy | Precisión | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Modelo base (`DummyClassifier`, clase más frecuente) | 0,630 | 0,000 | 0,000 | 0,000 |
| **Regresión logística** | 0,770 | 0,756 | 0,559 | **0,643** |
| Árbol de decisión (`max_depth=8`, `min_samples_leaf=50`) | 0,766 | 0,828 | 0,466 | 0,597 |
| Random Forest (100 árboles, `max_depth=12`, `min_samples_leaf=20`) | 0,759 | 0,881 | 0,403 | 0,553 |

### 6. Ajuste del umbral de decisión
Sobre la regresión logística se probaron cuatro umbrales usando únicamente validación:

| Umbral | Accuracy | Precisión | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| **0,30** | 0,729 | 0,607 | 0,760 | **0,675** |
| 0,40 | 0,755 | 0,677 | 0,651 | 0,663 |
| 0,50 | 0,770 | 0,756 | 0,559 | 0,643 |
| 0,60 | 0,763 | 0,816 | 0,464 | 0,591 |

Se eligió **0,30** por tener el mayor F1. Luego el modelo se reentrenó con todo el conjunto de entrenamiento.

## Algoritmo final

**Regresión logística** (`max_iter=1000`) dentro de un `Pipeline` de scikit-learn con imputación, One-Hot Encoding y escalado, con **umbral de decisión 0,30**.

## Resultados finales (conjunto de prueba)

| Accuracy | Precisión | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0,731 | 0,595 | 0,794 | 0,680 |

Matriz de confusión:

| | Predicho: no cancela | Predicho: cancela |
| --- | ---: | ---: |
| **Real: no cancela** | 10.844 | 4.753 |
| **Real: cancela** | 1.807 | 6.983 |

## Conclusiones

### Comparación con el modelo base (Dummy)
El modelo base (`DummyClassifier`, siempre predice "no cancela") sirve de referencia mínima.

| Modelo | Conjunto | Accuracy | Precisión | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dummy (clase más frecuente) | Validación | 0,630 | 0,000 | 0,000 | 0,000 |
| Regresión logística, umbral 0,30 | Validación | 0,729 | 0,607 | 0,760 | 0,675 |
| Regresión logística, umbral 0,30 | Prueba | 0,731 | 0,595 | 0,794 | 0,680 |

- **El modelo sí mejora al base.** El Dummy nunca detecta una cancelación (recall y F1 = 0), así que su accuracy de 0,63 solo refleja la proporción de la clase mayoritaria. El modelo final detecta 6.983 de las 8.790 cancelaciones reales de prueba (recall 0,794) y obtiene F1 = 0,680.
- **Accuracy vs. F1.** En prueba el Dummy alcanzaría un accuracy de ≈ 0,640 (63,96 % de reservas no canceladas) frente a 0,731 del modelo: la mejora en accuracy es de ≈ 9 puntos, pero la diferencia real está en que el modelo identifica cancelaciones. Por eso se eligió F1 como métrica principal.
- **Estabilidad.** El F1 de prueba (0,680) es casi igual al de validación (0,675), lo que indica que el modelo generaliza y que no hay señales de fuga de información ni sobreajuste.
- **Frente a otros modelos.** La regresión logística tuvo el mayor F1 (0,643 con umbral 0,5) frente al árbol (0,597) y Random Forest (0,553). Estos dos últimos son más precisos (0,828 y 0,881) pero detectan menos cancelaciones (recall 0,466 y 0,403) con la configuración probada.
- **¿Es razonable la métrica?** Sí, para una primera versión que usa únicamente información conocida al registrar la reserva. No es un resultado excelente: con un umbral de 0,30 aproximadamente 4 de cada 10 alertas de cancelación son falsas (precisión 0,595).

### Dificultades
- **Desbalance de clases** (63 % / 37 %), que hace engañoso el accuracy.
- **Filas idénticas (26,8 %) sin identificador de reserva:** no se pudo confirmar si eran duplicados, y una separación aleatoria habría puesto copias en entrenamiento y prueba. Se resolvió con separación por grupos.
- **Riesgo de fuga de información:** variables como `deposit_type`, `adr`, `booking_changes` o `assigned_room_type` pueden reflejar información posterior a la reserva. Excluirlas reduce el rendimiento, pero mantiene la evaluación honesta (`deposit_type` es muy informativa: 99,4 % de las reservas `Non Refund` se cancelan).
- **Valores faltantes y de alta cardinalidad:** `company` (94 %) y `agent` (14 %) con muchos nulos y usados como identificadores; `country` con muchas categorías.
- **Valores extremos** (`adr` hasta 5.400, `adults` hasta 55, `lead_time` hasta 737) que no se trataron en esta versión.
- **Trade-off precisión/recall:** bajar el umbral a 0,30 mejora la detección de cancelaciones, pero genera 4.753 falsas alarmas.

### Posibles mejoras
- Tratar valores extremos (recorte o transformación) y crear variables nuevas: noches totales, número total de huéspedes, día de la semana o fecha de llegada.
- Probar modelos de boosting (Gradient Boosting, XGBoost, LightGBM) y ajustar hiperparámetros con validación cruzada por grupos, en lugar de un único conjunto de validación.
- Comparar modelos con métricas independientes del umbral (ROC-AUC, PR-AUC) y elegir el umbral según el costo real de una falsa alarma frente a una cancelación no detectada.
- Evaluar si `deposit_type` puede usarse legítimamente, según el momento en que se realice la predicción.
- Reducir la cardinalidad de `country` (agrupar países poco frecuentes) y probar una versión con menos variables para comparar.
- Analizar la importancia de variables y los errores del modelo para interpretar mejor qué reservas se confunden.

## Instrucciones para ejecutar el notebook

1. Abrir `fase-1/modelo_hotel.ipynb` en [Google Colab](https://colab.research.google.com/) (opción *Archivo → Abrir cuaderno → GitHub*).
2. Seleccionar **Entorno de ejecución → Ejecutar todas**.
3. Revisar las métricas y la matriz de confusión al final del notebook.

No hace falta subir el CSV: el notebook lo descarga desde una versión fija (commit) de este repositorio. Requiere Python 3 y `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` y `joblib` (ejecutado con pandas 2.2.3 y scikit-learn 1.6.1). El notebook puede ejecutarse de inicio a fin sin intervención manual y usa semilla aleatoria fija.

### Usar el modelo guardado

```python
import joblib

modelo = joblib.load("fase-1/modelo_hotel.joblib")
pipeline, umbral = modelo["pipeline"], modelo["umbral"]

# nuevas_reservas: DataFrame con las 17 columnas predictoras descritas arriba
probabilidad = pipeline.predict_proba(nuevas_reservas)[:, 1]
cancelara = (probabilidad >= umbral).astype(int)
```

El archivo guarda un diccionario con el `Pipeline` entrenado (preprocesamiento + modelo) y el umbral 0,30. El notebook comprueba que el modelo cargado produce las mismas predicciones que el original.

## Flujo de trabajo en Git

Se usó un Git Flow simplificado con un único repositorio:

- `main`: versiones estables de las entregas.
- `develop`: rama de integración.
- `feature/analisis-exploratorio`, `feature/tratamiento-datos`, `feature/modelo-predictivo`: una rama por tarea, integradas a `develop` mediante Pull Requests.

## Próximas fases

Conversión del notebook en scripts de entrenamiento y predicción, despliegue de una API REST con Docker y monitoreo básico.

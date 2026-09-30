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

### Interpretación
- **¿Mejora al modelo base?** Sí. El modelo base no detecta ninguna cancelación (F1 = 0); el modelo final detecta 6.983 de las 8.790 cancelaciones reales (recall 0,794).
- **¿La métrica es razonable?** El F1 de prueba (0,680) es consistente con el de validación (0,675), lo que indica que no hay sobreajuste evidente ni fuga de información. Es un resultado moderado, esperable al no usar variables posteriores a la reserva.
- **Costo del umbral:** se generan 4.753 falsas alarmas; el umbral 0,30 favorece detectar más cancelaciones a cambio de menor precisión.
- **Dificultades:** desbalance de clases, alto porcentaje de nulos en `agent`/`company`, filas idénticas sin identificador y riesgo de fuga en variables como `deposit_type` y `adr`.
- **Posibles mejoras:** tratar valores extremos, crear variables nuevas (p. ej. noches totales, fecha de llegada), evaluar modelos de boosting, ajustar hiperparámetros con validación cruzada por grupos, y estudiar si `deposit_type` puede usarse legítimamente según el momento de predicción.

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

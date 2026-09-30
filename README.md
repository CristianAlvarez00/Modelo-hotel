# Predicción de cancelación de reservas hoteleras

### Objetivo

Este proyecto desarrolla un modelo de clasificación para predecir si una reserva hotelera será cancelada. La variable objetivo es `is_canceled`: `1` indica cancelación y `0` indica que la reserva no fue cancelada.

### Organización

- `data/hotel_bookings.csv`: conjunto de datos utilizado.
- `fase-1/modelo_hotel.ipynb`: análisis exploratorio, preparación de datos, entrenamiento y evaluación.
- `fase-1/modelo_hotel.joblib`: modelo final entrenado junto con su umbral de decisión.

### Ejecución

1. Abrir `fase-1/modelo_hotel.ipynb` en Google Colab.
2. Seleccionar **Entorno de ejecución → Ejecutar todas**.
3. Revisar las métricas y la matriz de confusión al final del notebook.

No es necesario cargar manualmente el CSV en Colab: el notebook obtiene una versión fija del archivo desde este repositorio. La ejecución requiere Python 3 y las librerías `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` y `joblib`.

### Preparación y modelos

Se seleccionaron variables que pueden conocerse al registrar la reserva y se excluyeron columnas que describen el resultado o pueden contener información posterior. Las filas idénticas se conservaron porque no existe un identificador que permita confirmar que son reservas duplicadas. Al separar los datos se impidió que entradas idénticas quedaran en conjuntos distintos.

Los valores faltantes numéricos se imputan con la mediana y los categóricos con la categoría `Desconocido`. La imputación, la codificación y el escalado se ajustan dentro de un `Pipeline` usando solamente los datos de entrenamiento correspondientes.

Se compararon un modelo base, regresión logística, árbol de decisión y Random Forest. La regresión logística obtuvo el mayor F1 en validación entre las configuraciones evaluadas. Se eligió el umbral 0,30 usando únicamente validación y luego se entrenó el modelo final con todo el conjunto de entrenamiento.

### Resultado final

En el conjunto de prueba, la regresión logística con umbral 0,30 obtuvo:

| Accuracy | Precisión | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0,731 | 0,595 | 0,794 | 0,680 |

El modelo identificó 6.983 de las 8.790 cancelaciones reales. También predijo una cancelación para 4.753 reservas que no se cancelaron. El umbral elegido favorece detectar más cancelaciones, aceptando más falsas alarmas.

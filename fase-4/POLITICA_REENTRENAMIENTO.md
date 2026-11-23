# Fase 4 – Política de reentrenamiento

1. **Cuándo.** Al reunir 500 reservas nuevas con su resultado real (cancelada o no),
   el equipo revisa el modelo.
2. **Quién y cómo.** Una persona designada del equipo ejecuta `POST /train`, que crea
   una versión candidata `api-<fecha>-<sufijo>` sin tocar la versión activa.
3. **Comparación.** Se evalúan la versión activa y la candidata con el F1 de la clase
   cancelada sobre los mismos datos reservados, no usados para entrenar.
4. **Activación.** La candidata se activa solo si mejora el F1 de la versión activa:
   se cambia `MODELO_VERSION` y se reinicia el servidor.
5. **Reversión.** Si la nueva versión da problemas, se vuelve a la anterior con
   `MODELO_VERSION=<versión anterior>`; las versiones no se borran.
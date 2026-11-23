FROM python:3.12-slim
WORKDIR /app

COPY fase-2/requirements.txt fase-2/requirements.txt
COPY fase-3/requirements.txt fase-3/requirements.txt
RUN pip install --no-cache-dir -r fase-3/requirements.txt

COPY fase-2/src fase-2/src
COPY fase-2/train.py fase-2/train.py
COPY fase-2/models fase-2/models
COPY data/hotel_bookings.csv data/hotel_bookings.csv
COPY fase-3/api.py fase-3/configuracion.py fase-3/esquemas.py fase-3/
COPY fase-4/api_monitoreada.py fase-4/registro.py fase-4/monitoreo.py fase-4/

RUN useradd --create-home app \
 && mkdir -p /app/fase-4/registros \
 && chown -R app:app /app/fase-2/models /app/fase-4/registros
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "api_monitoreada:app", "--app-dir", "fase-4", "--host", "0.0.0.0", "--port", "8000"]
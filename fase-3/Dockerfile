FROM python:3.12-slim
WORKDIR /app

COPY fase-2/requirements.txt fase-2/requirements.txt
COPY fase-3/requirements.txt fase-3/requirements.txt
RUN pip install --no-cache-dir -r fase-3/requirements.txt

COPY fase-2/src fase-2/src
COPY fase-2/train.py fase-2/train.py
COPY fase-2/models fase-2/models
COPY data/hotel_bookings.csv data/hotel_bookings.csv
COPY fase-3/api.py fase-3/configuracion.py fase-3/esquemas.py fase-3/ejemplo_reserva.json fase-3/

RUN useradd --create-home app && chown -R app:app /app/fase-2/models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "api:app", "--app-dir", "fase-3", "--host", "0.0.0.0", "--port", "8000"]
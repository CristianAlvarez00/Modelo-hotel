FROM python:3.12-slim
WORKDIR /app
COPY fase-1/requirements.txt fase-1/requirements.txt
RUN pip install --no-cache-dir -r fase-1/requirements.txt
RUN pip install --no-cache-dir notebook
COPY . .
WORKDIR /app
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
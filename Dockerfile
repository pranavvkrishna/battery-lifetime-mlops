FROM python:3.11-slim

WORKDIR /app

# system deps needed by some ML libraries (xgboost, tensorflow)
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY api/main.py .

# MLflow tracking data (model registry) — copied in so the container
# has access to the same registered models without needing a live
# connection back to your laptop
COPY mlruns/ ./mlruns/
COPY mlflow.db .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
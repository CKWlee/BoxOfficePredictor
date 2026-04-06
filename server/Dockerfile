FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ .
COPY models/ ./models/
COPY data/processed/ ./data/processed/
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT

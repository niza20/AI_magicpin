FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Single worker: the bot keeps judge-pushed contexts and conversations in memory.
CMD ["sh", "-c", "uvicorn bot:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
